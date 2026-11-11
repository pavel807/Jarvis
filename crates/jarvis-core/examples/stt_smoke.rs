//! Диагностика STT на этой машине: грузит FastConformer и расшифровывает
//! wav-файл (16 кГц mono или любой, который hound прочитает — приводится
//! к 16 кГц mono f32, как микрофонный поток).
//! Запуск: cargo run -p jarvis-core --example stt_smoke -- resources/audio/ok1.wav
//! Модель ищется в системной папке моделей (JARVIS_MODELS_DIR переопределяет).

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let wav = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "resources/audio/ok1.wav".into());
    let started = std::time::Instant::now();
    let mut stt = jarvis_core::stt::Stt::load(&jarvis_core::paths::models_dir())?;
    println!("✅ FastConformer загружен за {:.1?}", started.elapsed());

    let mut reader = hound::WavReader::open(&wav)
        .with_context(|| format!("не удалось открыть {wav}"))?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / (1 << (spec.bits_per_sample - 1)) as f32)
        .collect();
    let mono: Vec<f32> = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    };
    let audio_16k = jarvis_core::audio::resample_audio(&mono, spec.sample_rate as usize, 16_000);

    let started = std::time::Instant::now();
    let text = stt.transcribe(&audio_16k)?;
    println!(
        "✅ Расшифровка ({:.1} сек звука за {:.1?}): «{}»",
        audio_16k.len() as f32 / 16_000.0,
        started.elapsed(),
        text
    );
    Ok(())
}
