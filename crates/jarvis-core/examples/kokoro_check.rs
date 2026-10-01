//! Проверка качества Kokoro-ru: русский, смешанный EN+RU, темп, RTF и
//! WAV для прослушивания.
//!
//! Запуск: cargo run --release --example kokoro_check -p jarvis-core

use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use jarvis_core::tts_kokoro::KokoroEngine;
use std::io::BufWriter;
use std::path::Path;

fn main() -> Result<()> {
    let voice = "dima";
    let engine = KokoroEngine::new(voice)?;
    let sr = engine.sample_rate();

    let phrases = [
        "Привет! Я Джарвис, твой голосовой ассистент.",
        "Сейчас на улице двадцать один градус и небольшая облачность.",
        // Смешанная речь: то, на чём ломался Piper.
        "Открываю ютуб, одну секунду. Включаю YouTube Music.",
        "Запускаю Premiere Pro и After Effects через Zed.",
    ];

    println!("Голос: {voice}, {sr} Гц\n");
    let mut all_out = Vec::new();
    for text in phrases {
        let t = std::time::Instant::now();
        let samples = engine.synthesize(text, 1.0)?;
        let dt = t.elapsed().as_secs_f64();
        let audio = samples.len() as f64 / sr as f64;
        let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = if samples.is_empty() {
            0.0
        } else {
            (sum_sq / samples.len() as f64).sqrt()
        };

        println!("«{text}»");
        println!(
            "  аудио {audio:.2} с ({:.1} симв/с) | синтез {dt:.2} с | RTF {:.3} | RMS {rms:.3} | пик {peak:.2}\n",
            text.chars().count() as f64 / audio,
            dt / audio,
        );
        all_out.extend(samples);
        all_out.extend(std::iter::repeat(0.0f32).take(sr as usize / 3));
    }

    let out_path = Path::new("resources/kokoro_check.wav");
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let file = std::fs::File::create(out_path)?;
    let mut writer = WavWriter::new(BufWriter::new(file), spec)?;
    for &s in &all_out {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    println!("🔊 Готово к прослушиванию: {:?}", out_path);

    // --- Демо нейро-ударителя: знак ˈ в IPA показывает ударную гласную ---
    println!("--- Ударения (IPA) ---");
    for w in [
        "звонит", "торты", "щавель", "договор", "каталог", "красивее", "еще", "дела",
    ] {
        println!("  {w:<12} -> {}", engine.phonemize_mixed(w));
    }

    let _ = engine.synthesize("Звонит Иван, ещё одна красивее.", 1.0)?;
    println!("🔊 Готово к прослушиванию: {:?}", out_path);

    Ok(())
}
