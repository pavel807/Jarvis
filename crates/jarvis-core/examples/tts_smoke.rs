//! Диагностика TTS на этой машине: загружается ли Silero v5 и синтезирует ли
//! звук (RMS > 0 = не тишина). Модель ищется в системной папке моделей,
//! путь можно переопределить: JARVIS_MODELS_DIR=/path/to/models
//! Запуск: cargo run -p jarvis-core --example tts_smoke

use anyhow::Result;

fn main() -> Result<()> {
    let started = std::time::Instant::now();
    let engine = jarvis_core::tts_silero::SileroEngine::new_no_stress("ru_eduard")?;
    println!("✅ Silero загрузился за {:.1?}", started.elapsed());

    let started = std::time::Instant::now();
    let audio = engine.synthesize("Привет! Проверка синтеза речи на этой машине.", 1.0)?;
    let secs = audio.len() as f32 / 22050.0;
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!(
        "✅ Синтез: {:.1} сек звука за {:.1?}, RMS {:.4}{}",
        secs,
        started.elapsed(),
        rms,
        if rms > 0.001 { "" } else { " ⚠️ тишина!" }
    );
    Ok(())
}
