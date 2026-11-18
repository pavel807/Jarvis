//! Текстовый токенизатор SuperBPE (8192) из метаданных GGUF.
//!
//! В GGUF лежит полный JSON HF Tokenizers: byte-level BPE с предразбиением
//! по цифрам/пунктуации. Своя реализация BPE здесь не нужна — в проекте уже
//! есть крейт `tokenizers` (он же обслуживает STT), а паттерн `Split`
//! содержит lookahead `(?!\S)`, который поддерживается включённым в нём
//! Oniguruma. Поэтому просто отдаём JSON в `tokenizers` и приводим
//! идентификаторы к текстовому пространству модели ZONOS3.
//!
//! Соглашение чекпоинта (`superbpe.py`): id словаря 0..8191 сдвигаются на
//! +4, а строки BOS/EOS — это id 2 и 3; `text_ids(t) = [2] + bpe(t) + [3]`.
//! Ряд `text_vocab` (8196) — паддинг текста, он в токенизаторе не участвует.

use anyhow::{Context, Result, bail};

/// Токенизатор текста ZONOS3-Mini.
pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
    offset: u32,
    bos: u32,
    eos: u32,
    vocab_size: usize,
}

impl TextTokenizer {
    /// Собрать токенизатор из JSON, встроенного в GGUF.
    ///
    /// `vocab_size` — размер словаря без учёта смещения (`text_vocab - offset`).
    pub fn from_json(
        json: &str,
        offset: u32,
        bos: u32,
        eos: u32,
        vocab_size: usize,
    ) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| anyhow::anyhow!("Токенизатор ZONOS3: {e}"))?;
        let actual = inner.get_vocab_size(true);
        if actual != vocab_size {
            bail!(
                "Токенизатор ZONOS3: словарь {actual} токенов, ожидался {vocab_size}"
            );
        }
        Ok(Self {
            inner,
            offset,
            bos,
            eos,
            vocab_size,
        })
    }

    /// Размер словаря токенизатора (без смещения).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Идентификаторы текста без BOS/EOS, уже сдвинутые в пространство модели.
    pub fn payload_ids(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Токенизатор ZONOS3: {e}"))
            .with_context(|| format!("Не удалось токенизировать текст: {text:?}"))?;
        let mut ids = Vec::with_capacity(encoding.len());
        for &id in encoding.get_ids() {
            let shifted = (id as usize).checked_add(self.offset as usize);
            match shifted {
                Some(v) if v < self.vocab_size + self.offset as usize => ids.push(v as u32),
                _ => bail!("Токенизатор ZONOS3 вернул id {id} вне словаря"),
            }
        }
        Ok(ids)
    }

    /// Полная строка текстовых id: `[BOS] + payload + [EOS]`.
    pub fn text_ids(&self, text: &str) -> Result<Vec<u32>> {
        let payload = self.payload_ids(text)?;
        let mut ids = Vec::with_capacity(payload.len() + 2);
        ids.push(self.bos);
        ids.extend_from_slice(&payload);
        ids.push(self.eos);
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JSON токенизатора из реального GGUF-дампа (фрагмент проверяется в
    /// интеграционном тесте ниже; здесь — только поведение на простом BPE).
    #[test]
    fn byte_level_offsets() {
        let json = r#"{
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<|pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {"<|endoftext|>": 0, "<|pad|>": 1, "Ġ": 2, "a": 3, "Ġa": 4},
                "merges": [["Ġ", "a"]]
            }
        }"#;
        let tok = TextTokenizer::from_json(json, 4, 2, 3, 5).unwrap();
        assert_eq!(tok.text_ids(" a").unwrap(), vec![2, 8, 3]);
    }
}
