//! Нейросетевая расстановка ударений для русского (ядро RUAccent,
//! Apache-2.0; ONNX-модели из TigreGotico/stressonnx-models).
//!
//! «Интеллектуальная функция» перед фонемизацией:
//! 1. словарь е→ё (85,5 тыс. слов) — «еще» → «ещё» (ё всегда ударная);
//! 2. словарь ударений (110,8 тыс. слов) — «аб+а» → ударная гласная;
//! 3. для незнакомых слов — char-level RoFormer ONNX (0,8 МБ) предсказывает
//!    ударную гласную.
//!
//! Результат — порядковый номер ударной гласной слова; движок kokoro
//! ставит по нему знак ˈ в IPA и редуцирует безударные гласные.
//! Контекстные омографы (замок/мука) требуют модель на 342 МБ — не
//! включены, такие слова остаются на совести espeak.

use anyhow::{Context, Result};
use ort::session::{Session, SessionInputs, SessionInputValue};
use ort::value::{Tensor, Value};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Гласные русского текста (для подсчёта порядкового номера).
const RU_VOWELS: &str = "аеёиоуыэюя";

const STRESS_OVERRIDES: &[(&str, &str)] = &[
    ("дела", "д+ела"),
    ("какая", "как+ая"),
    ("какие", "как+ие"),
    ("какое", "как+ое"),
    ("такая", "так+ая"),
    ("такие", "так+ие"),
    ("такое", "так+ое"),
];


/// Гласные в IPA-выходе espeak (после которого расставляем ˈ).
pub const IPA_VOWELS: &str = "aeiouyəɐɔɪʊɛɨɵ";

const UNK_ID: i64 = 1;
const BOS_ID: i64 = 2;
const EOS_ID: i64 = 3;
/// Порог уверенности нейросети (как в stressonnx).
const MIN_CONFIDENCE: f32 = 0.55;
/// Максимальная длина входа нейросети (max_length из config.json).
const MAX_TOKENS: usize = 40;

#[derive(Deserialize)]
struct AccentConfig {
    id2label: HashMap<String, String>,
}

pub struct RuStress {
    session: Mutex<Session>,
    char_vocab: HashMap<char, i64>,
    id2label: HashMap<String, String>,
    /// Словарь ударений (110 тыс. слов) — первый эшелон, точный.
    accents: HashMap<String, String>,
    yo_words: HashMap<String, String>,
}

/// Распаковывает и скачивает файлы ударителя в models/stress-ru/.
fn ensure_files() -> Result<PathBuf> {
    let dir = crate::paths::models_dir().join("stress-ru");
    std::fs::create_dir_all(&dir)?;

    let files = [
        ("ru_ruaccent/nn_accent/model.onnx", "accent.onnx"),
        ("ru_ruaccent/nn_accent/vocab.txt", "vocab.txt"),
        ("ru_ruaccent/nn_accent/config.json", "config.json"),
        ("ru_ruaccent/dictionary/yo_words.json.gz", "yo_words.json.gz"),
    ];

    let client = hf_hub::HFClientSync::new().context("Не удалось создать HuggingFace-клиент")?;
    let repo = client.model("TigreGotico".to_string(), "stressonnx-models".to_string());

    for (remote, local_name) in files {
        let dest = dir.join(local_name);
        if dest.exists() {
            continue;
        }
        println!("📥 Загрузка ударителя ({local_name})...");
        let downloaded = repo
            .download_file()
            .filename(remote.to_string())
            .send()
            .with_context(|| format!("Не удалось скачать {remote}"))?;
        std::fs::copy(&downloaded, &dest)
            .with_context(|| format!("Не удалось сохранить {local_name}"))?;
    }

    Ok(dir)
}

fn read_gzip_json(path: &Path) -> Result<HashMap<String, String>> {
    let raw = std::fs::read(path)?;
    let mut decoder = flate2::read::GzDecoder::new(&raw[..]);
    let mut json = String::new();
    decoder.read_to_string(&mut json)?;
    serde_json::from_str(&json).with_context(|| format!("Ошибка парсинга {:?}", path))
}

impl RuStress {
    /// Загружает ударитель (скачивает ~2 МБ при первом запуске).
    pub fn load() -> Result<Self> {
        let started = std::time::Instant::now();
        let dir = ensure_files()?;

        let mut char_vocab = HashMap::new();
        for (id, line) in std::fs::read_to_string(dir.join("vocab.txt"))?
            .lines()
            .enumerate()
        {
            if let Some(c) = line.chars().next() {
                char_vocab.insert(c, id as i64);
            }
        }

        let config: AccentConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;

        let accents = read_gzip_json(&dir.join("accents_nn.json.gz"))?;
        let yo_words = read_gzip_json(&dir.join("yo_words.json.gz"))?;

        let session = Session::builder()
            .context("Не удалось создать ONNX Session")?
            .commit_from_file(dir.join("accent.onnx"))
            .context("Не удалось загрузить модель ударений")?;

        println!(
            "✅ Ударитель готов (словарь {}, ё {}, нейросеть OOV) за {:.1?}",
            accents.len(),
            yo_words.len(),
            started.elapsed()
        );

        Ok(Self {
            session: Mutex::new(session),
            char_vocab,
            id2label: config.id2label,
            accents,
            yo_words,
        })
    }

    /// Восстанавливает ё по словарю (ё всегда ударная, espeak знает её).
    pub fn yo_restore(&self, word: &str) -> String {
        if !word.contains('е') && !word.contains('Е') {
            return word.to_string();
        }
        let lower = word.to_lowercase();
        match self.yo_words.get(&lower) {
            Some(restored) => {
                // Сохраняем регистр первой буквы.
                let mut chars = restored.chars();
                match chars.next() {
                    Some(first) => {
                        let upper = word.chars().next().is_some_and(|c| c.is_uppercase());
                        let head = if upper {
                            first.to_uppercase().to_string()
                        } else {
                            first.to_string()
                        };
                        format!("{head}{}", chars.as_str())
                    }
                    None => word.to_string(),
                }
            }
            None => word.to_string(),
        }
    }

    /// Порядковый номер ударной гласной слова (1-based). Эшелоны:
    /// переопределения -> словарь (110 тыс., точный) -> нейросеть (любое
    /// новое слово). ё всегда ударная; служебные слова не акцентируем.
    /// None — ни один эшелон не сработал, оставляем ударение espeak.
    pub fn stressed_vowel_ordinal(&self, word: &str) -> Option<usize> {
        let lower = word.to_lowercase();

        // Служебные слова и не-слова не акцентируем.
        if lower.chars().any(|c| !(c.is_alphabetic() || c == '-')) {
            return None;
        }
        let vowel_count = lower.chars().filter(|c| RU_VOWELS.contains(*c)).count();
        if vowel_count < 2 {
            return None;
        }

        // ё всегда ударная.
        if let Some(ord) = vowel_ordinal(&lower, |c| c == 'ё') {
            return Some(ord);
        }

        // Переопределения, затем словарь, затем нейросеть.
        if let Some((_, forced)) = STRESS_OVERRIDES.iter().find(|(w, _)| *w == lower) {
            return Self::plus_notation_ordinal(forced);
        }
        if let Some(stressed) = self.accents.get(&lower) {
            return Self::plus_notation_ordinal(stressed);
        }

        self.nn_stressed_vowel_ordinal(&lower)
    }

    /// Нотация «аб+а» -> порядковый номер ударной гласной (1-based).
    /// Несколько '+' (варианты) не разбираем — вернём None, возьмёт NN.
    fn plus_notation_ordinal(stressed: &str) -> Option<usize> {
        let mut pluses = 0usize;
        for (i, c) in stressed.char_indices() {
            if c == '+' {
                let after: String = stressed[i + 1..].to_string();
                if let Some(first) = after.chars().next() {
                    if RU_VOWELS.contains(first) {
                        let before = &stressed[..i];
                        let idx = before.chars().filter(|c| RU_VOWELS.contains(*c)).count();
                        return Some(idx + 1);
                    }
                }
                pluses += 1;
                if pluses > 1 {
                    return None;
                }
            }
        }
        None
    }

    /// Нейросеть: char-level RoFormer -> метка STRESS_PRIMARY на гласной.
    fn nn_stressed_vowel_ordinal(&self, lower: &str) -> Option<usize> {
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() + 2 > MAX_TOKENS {
            return None;
        }

        let mut ids: Vec<i64> = Vec::with_capacity(chars.len() + 2);
        ids.push(BOS_ID);
        for c in &chars {
            ids.push(self.char_vocab.get(c).copied().unwrap_or(UNK_ID));
        }
        ids.push(EOS_ID);
        let n = ids.len();

        let input_ids = Tensor::from_array(([1usize, n], ids.clone())).ok()?;
        let attn = Tensor::from_array(([1usize, n], vec![1i64; n])).ok()?;
        let tti = Tensor::from_array(([1usize, n], vec![0i64; n])).ok()?;

        let inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = vec![
            (
                Cow::Borrowed("input_ids"),
                SessionInputValue::Owned(Value::from(input_ids)),
            ),
            (
                Cow::Borrowed("attention_mask"),
                SessionInputValue::Owned(Value::from(attn)),
            ),
            (
                Cow::Borrowed("token_type_ids"),
                SessionInputValue::Owned(Value::from(tti)),
            ),
        ];

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session.run(SessionInputs::from(inputs)).ok()?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>().ok()?;
        // [1, n, num_labels]
        let num_labels = *shape.last()? as usize;
        let flat: &[f32] = logits;

        for i in 1..=chars.len() {
            let row = &flat[i * num_labels..(i + 1) * num_labels];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let argmax = exps
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, _)| k)?;
            let confidence = exps[argmax] / sum;
            if confidence < MIN_CONFIDENCE {
                continue;
            }
            let label = self.id2label.get(&argmax.to_string())?;
            if label != "STRESS_PRIMARY" {
                continue;
            }
            // Позиция i в логитах соответствует символу chars[i - 1];
            // порядковый номер ударной гласной = гласные в chars[..i].
            if !RU_VOWELS.contains(chars[i - 1]) {
                continue;
            }
            return Some(lower.chars().take(i).filter(|c| RU_VOWELS.contains(*c)).count());
        }
        None
    }
}

/// Порядковый номер первой гласной, удовлетворяющей предикату (1-based).
fn vowel_ordinal(word: &str, pred: impl Fn(char) -> bool) -> Option<usize> {
    let mut seen = 0usize;
    for c in word.chars() {
        if RU_VOWELS.contains(c) {
            seen += 1;
            if pred(c) {
                return Some(seen);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn plus_ordinal() {
        assert_eq!(RuStress::plus_notation_ordinal("аб+а"), Some(2));
        assert_eq!(RuStress::plus_notation_ordinal("д+ела"), Some(1));
        assert_eq!(RuStress::plus_notation_ordinal("как+ая"), Some(2));
    }

    #[test]
    fn vowel_ordinal_basics() {
        assert_eq!(vowel_ordinal("замок", |c| c == 'о'), Some(2));
        assert_eq!(vowel_ordinal("замок", |c| c == 'а'), Some(1));
        assert_eq!(vowel_ordinal("ещё", |c| c == 'ё'), Some(2));
        assert_eq!(vowel_ordinal("въезд", |c| c == 'ё'), None);
    }


}
