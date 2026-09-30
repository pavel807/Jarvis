//! Дымовой тест TTS: cargo run -p jarvis-core --example tts_smoke -- [голос]
//!
//! Инициализирует движок (с автозагрузкой модели при отсутствии), синтезирует
//! русскую фразу в /tmp и печатает RTF с разбивкой по стадиям.

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let voice = std::env::args().nth(1).unwrap_or_else(|| "ryan".into());

    jarvis_core::tts::set_current_speed(1.0);
    let engine = jarvis_core::tts::TtsEngine::new(&voice)?;

    let text = "Привет! Я Джарвис, твой персональный голосовой ассистент.";
    let t0 = Instant::now();
    let out = std::env::temp_dir().join(format!("jarvis_tts_smoke_{voice}.wav"));
    engine.synthesize_to_file(text, &out)?;
    let elapsed = t0.elapsed().as_secs_f64();

    let dur = jarvis_core::tts::wav_duration_secs(&out).unwrap_or(0.0);
    println!(
        "синтез {:.2}с, аудио {:.2}с, RTF {:.2} → {}",
        elapsed,
        dur,
        elapsed / dur.max(0.001),
        out.display()
    );
    Ok(())
}
