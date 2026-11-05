//! Диагностика ударений: печатает токены («+» = id 5) и пишет два WAV —
//! со стрессором и без, чтобы сравнить на слух.
//!
//! Запуск: cargo run --release --example stress_audio -p jarvis-core

use jarvis_core::tts_silero::{SAMPLE_RATE, SileroEngine};

const TEXT: &str = "Слушаю вас, Павел. Привет, как дела? У меня всё хорошо. \
Сейчас две тысячи двадцать шестой год. Замок стоит на горе.";

fn write_wav(path: &str, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let with = SileroEngine::new("ru_bogdan")?;
    let without = SileroEngine::new_no_stress("ru_bogdan")?;

    let tokens_with = with.debug_tokens(TEXT)?;
    let tokens_without = without.debug_tokens(TEXT)?;
    println!(
        "знак «+» (id 5): {} шт. с ударениями, {} шт. без",
        tokens_with.iter().filter(|&&t| t == 5).count(),
        tokens_without.iter().filter(|&&t| t == 5).count(),
    );

    let a = with.synthesize(TEXT, 0.95)?;
    let b = without.synthesize(TEXT, 0.95)?;
    let n = a.len().min(b.len());
    let diff: f32 = (0..n).map(|i| (a[i] - b[i]).abs()).sum::<f32>() / n as f32;
    write_wav("build/tts_with_stress.wav", &a)?;
    write_wav("build/tts_without_stress.wav", &b)?;
    println!("средняя |разница| = {diff:.5}");
    println!("→ build/tts_with_stress.wav / build/tts_without_stress.wav");
    Ok(())
}
