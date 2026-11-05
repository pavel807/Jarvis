//! STT: NVIDIA FastConformer-Hybrid Large RU (CC-BY-4.0) — собственная
//! реализация на candle (см. [`crate::asr`]).
//!
//! Фасад для движка: тот же контракт, что был у старого GigaAM-бэкенда
//! (`load` + `transcribe`), чтобы интеграция не менялась.

use crate::asr::model::FastConformer;
use anyhow::Result;
use std::path::Path;

pub struct Stt {
    model: FastConformer,
}

impl Stt {
    /// Скачивает модель при необходимости и загружает её (Metal на macOS).
    pub fn load(models_dir: &Path) -> Result<Self> {
        let model = FastConformer::load(models_dir)?;
        Ok(Self { model })
    }

    /// Распознавание готового буфера 16 кГц mono f32 (RNN-T greedy).
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let transcript = self.model.transcribe(samples)?;
        Ok(transcript.text.trim().to_string())
    }

    /// Доступ к полной модели (декодеры, словарь Jarvis, тайминги).
    pub fn model(&self) -> &FastConformer {
        &self.model
    }
}