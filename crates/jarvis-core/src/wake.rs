//! Детектор wake word (RustPotter): создание, горячая подмена модели.

use anyhow::{Context, Result, anyhow, bail};
use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat};
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;

/// Конфигурация детектора по умолчанию для Jarvis.
pub fn detector_config(threshold: f32) -> RustpotterConfig {
    let mut dcfg = RustpotterConfig::default();
    dcfg.fmt.sample_rate = SAMPLE_RATE;
    dcfg.fmt.channels = 1;
    dcfg.fmt.sample_format = SampleFormat::I16;
    dcfg.detector.threshold = threshold;
    dcfg.filters.band_pass.enabled = false;
    dcfg
}

/// Создаёт детектор с одной зарегистрированной моделью wake word.
pub fn create_detector(model_path: &Path, threshold: f32) -> Result<Rustpotter> {
    if !model_path.exists() {
        bail!("Файл модели вакворда {} не найден", model_path.display());
    }

    let dcfg = detector_config(threshold);
    let mut det = Rustpotter::new(&dcfg).map_err(|e| anyhow!(e))?;

    det.add_wakeword_from_file("jarvis", model_path.to_str().context("путь не UTF-8")?)
        .map_err(|e| anyhow!(e))?;

    Ok(det)
}