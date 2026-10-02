//! Демо Silero v5 на candle: фразы пользователя -> resources/silero_final.wav
use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use jarvis_core::tts_silero::SileroEngine;
use std::io::BufWriter;

fn main() -> Result<()> {
    let engine = SileroEngine::new("ru_eduard")?;
    let sr = engine.sample_rate();
    let phrases = [
        "Привет! Я Джарвис, твой голосовой ассистент.",
        "Сейчас на улице двадцать один градус и небольшая облачность.",
        "Открываю ютуб, одну секунду. Включаю YouTube Music.",
        "Запускаю Premiere Pro и After Effects через Zed.",
    ];
    let mut all = Vec::new();
    for text in phrases {
        let t = std::time::Instant::now();
        let samples = engine.synthesize(text, 0.95)?;
        let dt = t.elapsed().as_secs_f64();
        let secs = samples.len() as f64 / sr as f64;
        println!("«{}» -> {:.2}с за {:.3}с (RTF {:.3})", text, secs, dt, dt / secs);
        all.extend(samples);
        all.extend(std::iter::repeat(0.0f32).take(sr as usize / 3));
    }
    let spec = WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: SampleFormat::Int };
    let file = std::fs::File::create("resources/silero_final.wav")?;
    let mut w = WavWriter::new(BufWriter::new(file), spec)?;
    for &s in &all {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    println!("🔊 resources/silero_final.wav");
    Ok(())
}
