use jarvis_core::tts_silero::SileroEngine;
fn main() -> anyhow::Result<()> {
    let engine = SileroEngine::new("ru_eduard")?;
    let t0 = std::time::Instant::now();
    let audio = engine.synthesize("Привет! Как дела?", 1.0)?;
    let dt = t0.elapsed().as_secs_f64();
    let secs = audio.len() as f64 / 48000.0;
    println!("Привет! Как дела? -> {:.2}с аудио за {:.3}с, RTF {:.3}", secs, dt, dt / secs);
    let t0 = std::time::Instant::now();
    let audio = engine.synthesize("Сейчас на улице двадцать один градус и небольшая облачность, ветра почти нет.", 1.0)?;
    let dt = t0.elapsed().as_secs_f64();
    let secs = audio.len() as f64 / 48000.0;
    println!("длинная фраза -> {:.2}с аудио за {:.3}с, RTF {:.3}", secs, dt, dt / secs);
    Ok(())
}
