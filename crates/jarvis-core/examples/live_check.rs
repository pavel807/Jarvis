//! Воспроизведение ЖИВОГО пути ассистента: текст -> normalize_for_tts ->
//! фонемы -> синтез. Сравни с тем, что слышишь в приложении.
//!
//! Запуск: cargo run --release --example live_check -p jarvis-core

use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use jarvis_core::text::normalize_for_tts;
use jarvis_core::tts_kokoro::KokoroEngine;
use std::io::BufWriter;

fn main() -> Result<()> {
    let engine = KokoroEngine::new("dima")?;
    let sr = engine.sample_rate();

    // Точные фразы из живого сценария.
    let phrases = [
        "Привет! Как дела?",
        "Какие дела?",
        "У меня всё отлично, дела идут хорошо.",
        "Сейчас на улице двадцать один градус.",
    ];

    let mut all = Vec::new();
    for text in phrases {
        // Ровно то, что делает плеер ассистента.
        let normalized = normalize_for_tts(text);
        let phonemes = engine.phonemize_mixed(&normalized);
        println!("ТЕКСТ:    {text}");
        println!("НОРМАЛ.:  {normalized}");
        println!("ФОНЕМЫ:   {phonemes}");
        println!();

        let samples = engine.synthesize(&normalized, 0.95)?;
        all.extend(samples);
        all.extend(std::iter::repeat(0.0f32).take(sr as usize / 3));
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let file = std::fs::File::create("resources/live_check.wav")?;
    let mut writer = WavWriter::new(BufWriter::new(file), spec)?;
    for &s in &all {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    println!("🔊 {:?} — послушай и сравни с приложением", "resources/live_check.wav");

    Ok(())
}
