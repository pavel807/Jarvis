//! Контекстное разрешение омографов (ruaccent, Apache-2.0) на candle.
//!
//! Слова вроде «замок», «стоит», «дела», «мука», «начала» читаются по-разному
//! в зависимости от смысла фразы. Словарный ударитель такие слова угадывает
//! по частоте и потому ошибается; здесь мы берём список допустимых прочтений
//! (`omographs.json`) и выбираем подходящее по контексту маленькой
//! DistilBERT-моделью: она получает на вход пару (предложение с разметкой
//! `<w>слово</w>`, вариант ударения) и оценивает, насколько вариант подходит.
//! Как в оригинальном ruaccent, только инференс — на candle
//! (см. `nn::distilbert`).

use crate::nn::distilbert::{DistilBertClassifier, DistilBertConfig};
use anyhow::{Context, Result};
use candle_core::Device;
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Словарь омографов и модель кладём рядом с ударителем.
const OM_DIR: &str = "stress-ru/omograph";
const OMOGRAPHS: &str = "stress-ru/omographs.json.gz";

/// Ограничение длины входа модели (max_position_embeddings).
const MAX_TOKENS: usize = 512;

pub struct OmographSolver {
    model: DistilBertClassifier,
    tokenizer: Tokenizer,
    /// слово -> допустимые прочтения в нотации «аб+а» (2-3 штуки).
    omographs: HashMap<String, Vec<String>>,
}

fn read_gzip_json(path: &std::path::Path) -> Result<HashMap<String, Vec<String>>> {
    let raw = std::fs::read(path)?;
    let mut decoder = flate2::read::GzDecoder::new(&raw[..]);
    let mut json = String::new();
    decoder.read_to_string(&mut json)?;
    serde_json::from_str(&json).with_context(|| format!("Ошибка парсинга {path:?}"))
}

/// Скачивает словарь омографов, токенизатор и модель в
/// models/stress-ru/omograph/ (веса — в safetensors для candle).
fn ensure_files() -> Result<PathBuf> {
    let dir = crate::paths::models_dir().join(OM_DIR);
    std::fs::create_dir_all(&dir)?;

    let model_file = dir.join("model.safetensors");
    let tokenizer_file = dir.join("tokenizer.json");
    let omographs = crate::paths::models_dir().join(OMOGRAPHS);

    if model_file.exists() && tokenizer_file.exists() && omographs.exists() {
        return Ok(dir);
    }

    let client = hf_hub::HFClientSync::new().context("Не удалось создать HuggingFace-клиент")?;

    let accent = client.model("ruaccent".to_string(), "accentuator".to_string());
    if !tokenizer_file.exists() {
        println!("📥 Загрузка токенизатора омографов...");
        let downloaded = accent
            .download_file()
            .filename("nn/nn_omograph/tiny2/tokenizer.json".to_string())
            .send()
            .context("Не удалось скачать tokenizer.json")?;
        std::fs::copy(&downloaded, &tokenizer_file)
            .context("Не удалось сохранить tokenizer.json")?;
    }

    if !omographs.exists() {
        println!("📥 Загрузка словаря омографов...");
        let stress = client.model("TigreGotico".to_string(), "stressonnx-models".to_string());
        let downloaded = stress
            .download_file()
            .filename("ru_ruaccent/dictionary/omographs.json.gz".to_string())
            .send()
            .context("Не удалось скачать omographs.json.gz")?;
        std::fs::copy(&downloaded, &omographs).context("Не удалось сохранить omographs.json.gz")?;
    }

    // Веса для candle. ONNX — только источник для сборки safetensors,
    // поэтому держим его на диске лишь пока веса не собраны.
    if !model_file.exists() {
        let onnx = dir.join("model.onnx");
        if !onnx.exists() {
            println!("📥 Загрузка контекстного ударителя (model.onnx)...");
            let downloaded = accent
                .download_file()
                .filename("nn/nn_omograph/tiny2/model.onnx".to_string())
                .send()
                .context("Не удалось скачать model.onnx")?;
            std::fs::copy(&downloaded, &onnx).context("Не удалось сохранить model.onnx")?;
        }
        crate::nn::ensure_safetensors(&model_file)?;
    }
    Ok(dir)
}

impl OmographSolver {
    /// Загружает словарь и модель (скачивает ~11 МБ при первом запуске).
    pub fn load() -> Result<Self> {
        let started = std::time::Instant::now();
        let dir = ensure_files()?;
        let omographs = read_gzip_json(&crate::paths::models_dir().join(OMOGRAPHS))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Ошибка загрузки токенизатора омографов: {e}"))?;
        let model = DistilBertClassifier::load(
            &dir.join("model.safetensors"),
            DistilBertConfig::default(),
            &Device::Cpu,
        )?;

        println!(
            "✅ Омографы готовы (слов {}, модель {:.1} МБ) за {:.1?}",
            omographs.len(),
            std::fs::metadata(dir.join("model.safetensors"))
                .map(|m| m.len() as f64 / 1048576.0)
                .unwrap_or(0.0),
            started.elapsed()
        );

        Ok(Self {
            model,
            tokenizer,
            omographs,
        })
    }

    /// Допустимые прочтения слова, если оно омограф.
    pub fn variants(&self, word: &str) -> Option<&[String]> {
        self.omographs.get(word).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.omographs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.omographs.is_empty()
    }

    /// Выбирает прочтение, лучше всего подходящее предложению.
    /// `marked` — предложение, где целевое слово обёрнуто в `<w>...</w>`.
    pub fn resolve(&self, marked: &str, variants: &[String]) -> Option<String> {
        let mut best: Option<(f32, &String)> = None;
        for variant in variants {
            let Ok(p) = self.score(marked, variant) else {
                continue;
            };
            if best.as_ref().is_none_or(|(bp, _)| p > *bp) {
                best = Some((p, variant));
            }
        }
        best.map(|(_, v)| v.clone())
    }

    /// Вероятность того, что вариант ударения верен для этого предложения.
    fn score(&self, sentence: &str, variant: &str) -> Result<f32> {
        let encoding = self
            .tokenizer
            .encode((sentence, variant), true)
            .map_err(|e| anyhow::anyhow!("Ошибка токенизации: {e}"))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        if ids.len() > MAX_TOKENS {
            ids.truncate(MAX_TOKENS);
            mask.truncate(MAX_TOKENS);
        }
        let logits = self.model.logits(&ids, &mask)?;
        if logits.len() < 2 {
            anyhow::bail!("Модель омографов вернула {} логитов", logits.len());
        }
        // softmax по двум классам, берём вероятность «гипотеза верна».
        let max = logits[0].max(logits[1]);
        let e0 = (logits[0] - max).exp();
        let e1 = (logits[1] - max).exp();
        Ok(e1 / (e0 + e1))
    }
}

/// Строит предложение с разметкой `<w>слово</w>` вокруг указанного слова
/// и убирает пробелы перед знаками препинания (как делает ruaccent).
pub fn mark_word(src: &[char], start: usize, end: usize, word: &str) -> String {
    let mut out = String::with_capacity(src.len() + word.len() + 12);
    out.extend(src[..start].iter());
    out.push_str(" <w>");
    out.push_str(word);
    out.push_str("</w> ");
    out.extend(src[end..].iter());
    strip_space_before_punct(&out)
}

fn strip_space_before_punct(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, ',' | '.' | '?' | '!' | ':' | ';' | '…') {
            while out.ends_with(' ') {
                out.pop();
            }
        }
        out.push(c);
    }
    out
}

/// Разбирает прочтение вида «дел+а» в слово и номер ударной гласной.
pub fn split_variant(variant: &str) -> (String, Option<usize>) {
    if let Some(pos) = variant.find('+') {
        let before = &variant[..pos];
        let ordinal = before
            .chars()
            .filter(|c| crate::stress::is_ru_vowel(*c))
            .count()
            + 1;
        (variant.replace('+', ""), Some(ordinal))
    } else {
        // Без «+» ударение несёт «ё».
        let ordinal = {
            let mut seen = 0usize;
            let mut found = None;
            for c in variant.chars() {
                if crate::stress::is_ru_vowel(c) {
                    seen += 1;
                    if c == 'ё' {
                        found = Some(seen);
                        break;
                    }
                }
            }
            found
        };
        (variant.to_string(), ordinal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_parsing() {
        assert_eq!(split_variant("дел+а"), ("дела".to_string(), Some(2)));
        assert_eq!(split_variant("з+амок"), ("замок".to_string(), Some(1)));
        assert_eq!(split_variant("начал+а"), ("начала".to_string(), Some(3)));
        assert_eq!(split_variant("вёсла"), ("вёсла".to_string(), Some(1)));
        assert_eq!(split_variant("мука"), ("мука".to_string(), None));
    }

    #[test]
    fn marking_wraps_word() {
        let src: Vec<char> = "замок стоит на горе.".chars().collect();
        let marked = mark_word(&src, 0, 5, "замок");
        assert!(marked.trim_start().starts_with("<w>замок</w>"), "{marked}");
        assert!(!marked.contains(" ."));
    }

    #[test]
    fn punctuation_spacing() {
        assert_eq!(
            strip_space_before_punct("привет , как дела ?"),
            "привет, как дела?"
        );
        assert_eq!(strip_space_before_punct("да ."), "да.");
        assert_eq!(strip_space_before_punct("норм"), "норм");
    }
}
