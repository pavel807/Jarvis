//! CTC-ветка FastConformer: линейная голова (512 → vocab+1), greedy-декодер
//! и beam search с опциональным словарём предметной области Jarvis.
//!
//! Blank-токен — последний (id = vocab_size), как в NeMo.
//!
//! Beam search: по кадрам лог-вероятностей, префиксное дерево гипотез,
//! склейка повторов через blank (стандартная CTC-семантика). Опциональный
//! словарь (`beam_bias`) — shallow fusion: гипотеза, чей префикс совпадает
//! с началом слова из словаря, получает бонус `bias` на каждый токен
//! продолжения; готовые слова словаря получают бонус `word_bias`. Это
//! повышает точность названий приложений и команд, не подменяя реальные
//! слова пользователя: бонус мал (единицы log-prob) и всегда есть шанс
//! обычного декодирования.

use super::tokenizer::SpTokenizer;
use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::VarBuilder;

pub struct CtcHead {
    weight: Tensor, // [vocab+1, d_model]
    bias: Tensor,   // [vocab+1]
}

impl CtcHead {
    pub fn load(vb: VarBuilder, d_model: usize, vocab_out: usize) -> Result<Self> {
        // NeMo хранит голову как Conv1d с ядром 1: [vocab+1, d_model, 1].
        let weight = vb.get((vocab_out, d_model, 1), "weight")?.squeeze(D::Minus1)?;
        let bias = vb.get(vocab_out, "bias")?;
        Ok(Self { weight, bias })
    }

    /// Логиты CTC: [B, T, vocab+1] из выхода энкодера [B, T, d_model].
    pub fn logits(&self, enc_out: &Tensor) -> Result<Tensor> {
        // Matmul в 2D: Metal-бэкенд отвергает 3D@2D напрямую.
        let (b, t, d) = enc_out.dims3()?;
        let vocab_out = self.weight.dim(0)?;
        let x = enc_out
            .reshape((b * t, d))?
            .matmul(&self.weight.t()?)? // [B*T, vocab+1]
            .reshape((b, t, vocab_out))?;
        Ok(x.broadcast_add(&self.bias.unsqueeze(0)?.unsqueeze(0)?)?)
    }
}

/// Greedy CTC-декодирование: argmax по кадрам, склейка повторов, выкидывание
/// blank. Возвращает последовательность id токенов.
pub fn greedy(logits: &[f32], frames: usize, vocab_out: usize, blank_id: usize) -> Vec<usize> {
    let mut ids = Vec::new();
    let mut prev = usize::MAX;
    for t in 0..frames {
        let row = &logits[t * vocab_out..(t + 1) * vocab_out];
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        if best != blank_id && best != prev {
            ids.push(best);
        }
        prev = best;
    }
    ids
}

/// Гипотеза beam search.
#[derive(Clone, Debug)]
struct Beam {
    /// Токены гипотезы (blank и склейки уже убраны).
    tokens: Vec<usize>,
    /// Сумма log-prob по кадрам (для сортировки).
    score: f64,
    /// Последний emitted-токен кадра (для склейки повторов между кадрами).
    last: usize,
}

/// Beam search по кадрам лог-вероятностей.
///
/// `bias` — опциональный словарь Jarvis: (бонус за продолжение слова из
/// словаря, бонус за завершённое слово). Токены словаря кодируются
/// тем же токенизатором; совпадение ищется по текущим токенам гипотезы.
pub fn beam_search(
    logprobs: &[f32],
    frames: usize,
    vocab_out: usize,
    blank_id: usize,
    width: usize,
    dict: Option<&JarvisDictionary>,
) -> Vec<usize> {
    let mut beams = vec![Beam {
        tokens: Vec::new(),
        score: 0.0,
        last: usize::MAX,
    }];
    for t in 0..frames {
        let row = &logprobs[t * vocab_out..(t + 1) * vocab_out];
        let mut candidates: Vec<Beam> = Vec::with_capacity(beams.len() * (vocab_out.min(width * 4)));
        for beam in &beams {
            // Топ-K токенов этого кадра: собираем индексы по убыванию.
            let mut order: Vec<usize> = (0..vocab_out).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            order.truncate(width.max(8));
            for &tok in &order {
                let mut score = beam.score + row[tok] as f64;
                let mut tokens = beam.tokens.clone();
                let mut last = beam.last;
                if tok == blank_id {
                    last = usize::MAX; // blank разрывает склейку повторов
                } else if tok == last {
                    // повтор того же токена — CTC склеивает его
                    last = tok;
                } else {
                    tokens.push(tok);
                    last = tok;
                    if let Some(dict) = dict {
                        score += dict.score_prefix(&tokens);
                    }
                }
                candidates.push(Beam { tokens, score, last });
            }
        }
        // Дедупликация: гипотезы с одинаковыми токенами и last оставляем
        // с максимальным score (склейка через blank).
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut next: Vec<Beam> = Vec::with_capacity(width);
        for cand in candidates {
            if next.len() >= width {
                break;
            }
            if let Some(existing) = next.iter_mut().find(|b| b.tokens == cand.tokens && b.last == cand.last) {
                if cand.score > existing.score {
                    existing.score = cand.score;
                }
                continue;
            }
            next.push(cand);
        }
        beams = next;
        if beams.is_empty() {
            beams.push(Beam { tokens: Vec::new(), score: 0.0, last: usize::MAX });
        }
    }
    beams.into_iter().next().map(|b| b.tokens).unwrap_or_default()
}

/// Словарь предметной области Jarvis для bias'а beam search: кодированные
/// последовательности токенов имён приложений/команд.
pub struct JarvisDictionary {
    /// Полные последовательности токенов слов словаря.
    phrases: Vec<Vec<usize>>,
    /// Бонус за токен-продолжение слова словаря.
    bias: f64,
    /// Бонус за завершённое слово словаря.
    word_bias: f64,
}

impl JarvisDictionary {
    pub fn new(phrases: &[String], tokenizer: &SpTokenizer, bias: f64, word_bias: f64) -> Result<Self> {
        let mut encoded = Vec::new();
        for phrase in phrases {
            let ids = tokenizer.encode(phrase)?;
            if !ids.is_empty() {
                encoded.push(ids);
            }
        }
        Ok(Self {
            phrases: encoded,
            bias,
            word_bias,
        })
    }

    /// Бонус гипотезы `tokens`: если она является префиксом (или равна)
    /// фразы словаря — за продолжение и за завершение.
    fn score_prefix(&self, tokens: &[usize]) -> f64 {
        let mut bonus = 0.0;
        for phrase in &self.phrases {
            if phrase.len() >= tokens.len() && phrase[..tokens.len()] == *tokens {
                bonus += self.bias * (phrase.len() - tokens.len() + 1) as f64;
                if phrase.len() == tokens.len() {
                    bonus += self.word_bias;
                }
            }
        }
        bonus
    }
}