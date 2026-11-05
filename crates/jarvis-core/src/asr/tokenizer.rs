//! Токенизатор FastConformer — оригинальный SentencePiece Unigram модели
//! (словарь 1024, unk id 0, blank добавляется NeMo как id 1024).
//!
//! Декодирование (id → текст) — прямое: склейка кусков с заменой "▁" на
//! пробел и постобработкой пробелов, идентичной эталону onnx-asr
//! (регулярка `\A\s|\s\B|(\s)\b`).
//!
//! Кодирование (текст → id) — Viterbi по Unigram-модели. Нормализация
//! соответствует spec'у SentencePiece модели: `nmt_nfkc`, add_dummy_prefix,
//! remove_extra_whitespaces, escape_whitespaces (все true). NFKC-часть
//! нормализатора воспроизводится крейтом `tokenizers` (`Normalizer::Nmt` —
//! это прямой порт charsmap nmt_nfkc из SentencePiece).

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::models::unigram::Unigram;
use tokenizers::Tokenizer;

pub struct SpTokenizer {
    /// Куски словаря по id: 0..<unk>, 1..vocab_size, blank = vocab_size
    /// (последний, из NeMo).
    vocab: Vec<String>,
    /// HF-токенизатор для кодирования текста (Unigram + Nmt + Metaspace).
    encoder: Tokenizer,
}

impl SpTokenizer {
    /// vocab_path — файл `tokenizer.vocab` (формат "piece\tscore", 1024 строк).
    pub fn load(vocab_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(vocab_path)
            .with_context(|| format!("Не удалось прочитать {vocab_path:?}"))?;
        let mut pieces: Vec<(String, f64)> = Vec::new();
        let mut vocab: Vec<String> = Vec::new();
        for line in text.lines() {
            let (piece, score) = line
                .split_once('\t')
                .with_context(|| format!("Строка токенизатора без score: {line:?}"))?;
            vocab.push(piece.to_string());
            pieces.push((piece.to_string(), score.parse().unwrap_or(0.0)));
        }
        // unk id 0 (первая строка <unk>); blank NeMo добавляет в конце —
        // в Unigram-модель он не входит (не участвует в кодировании текста).
        let model = Unigram::from(pieces, Some(0), false)
            .map_err(|e| anyhow::anyhow!("Unigram: {e}"))?;
        let mut encoder = Tokenizer::new(model);
        encoder.with_normalizer(Some(tokenizers::normalizers::Nmt));
        encoder.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::metaspace::Metaspace::default(),
        ));

        Ok(Self { vocab, encoder })
    }

    /// Кусок словаря по id (blank не входит в словарь токенизатора).
    pub fn piece(&self, id: usize) -> Option<&str> {
        self.vocab.get(id).map(String::as_str)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Декодирование id → текст (blank и служебные id отфильтрованы).
    pub fn decode(&self, ids: &[usize]) -> String {
        let joined: String = ids
            .iter()
            .filter_map(|&id| self.vocab.get(id))
            .map(|piece| piece.replace('▁', " "))
            .collect();
        postprocess_spaces(&joined)
    }

    /// Кодирование текста → id (Viterbi Unigram; unk для неизвестного).
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let encoding = self
            .encoder
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Ошибка кодирования токенизатора: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }
}

/// Постобработка пробелов, идентичная эталону onnx-asr
/// (`re.sub(r"\A\s|\s\B|(\s)\b", ...)`): убрать ведущие пробелы, пробелы
/// перед не-словесными символами (пунктуацией) и задвоенные; одиночный
/// пробел между словами оставить.
pub fn postprocess_spaces(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            // \b сразу после пробела: следующий символ — словесный, пробел
            // не ведущий. Иначе (\A\s, \s\B) — убрать.
            let next_word = i + 1 < chars.len() && is_word(chars[i + 1]);
            if i > 0 && next_word {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::postprocess_spaces;

    #[test]
    fn spaces_follow_onnx_asr_pattern() {
        // На входе — уже склеенные куски с заменённым "▁": постобработка
        // получает обычные пробелы.
        assert_eq!(postprocess_spaces(" привет мир"), "привет мир");
        // Пробел перед пунктуацией уходит; задвоенные — тоже.
        assert_eq!(postprocess_spaces("раз , два"), "раз, два");
        assert_eq!(postprocess_spaces("a  b"), "a b");
        // Замыкающий пробел уходит.
        assert_eq!(postprocess_spaces("конец "), "конец");
    }
}