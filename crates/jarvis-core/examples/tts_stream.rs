//! Дымовой тест стримингового TTS: cargo run -p jarvis-core --example tts_stream -- [голос]
//!
//! Печатает TTFA (время до первого чанка) и суммарный RTF для холодного
//! старта (первый forward компилирует Metal-шейдеры) и для повторного.

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let voice = std::env::args().nth(1).unwrap_or_else(|| "ryan".into());

    jarvis_core::tts::set_current_speed(1.0);
    let engine = jarvis_core::tts::TtsEngine::new(&voice)?;

    for (label, phrase) in [
        ("холодный старт", "Привет! Я Джарвис, твой голосовой ассистент."),
        ("повтор", "Готов к работе."),
    ] {
        let started = Instant::now();
        let mut session = engine.stream_session(phrase)?;

        let mut total_samples = 0usize;
        let mut chunks = 0usize;
        let mut ttfa = None;
        while let Some(audio) = session.next_chunk()? {
            if ttfa.is_none() {
                ttfa = Some(started.elapsed());
            }
            total_samples += audio.samples.len();
            chunks += 1;
        }

        let elapsed = started.elapsed().as_secs_f64();
        let audio_secs = total_samples as f64 / 24000.0;
        println!(
            "{label}: TTFA {:.2?}, чанков {chunks}, синтез {elapsed:.2}с, аудио {audio_secs:.2}с, RTF {:.2}",
            ttfa.unwrap_or_default(),
            elapsed / audio_secs.max(0.001)
        );
    }
    Ok(())
}
