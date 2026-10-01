//! Проверка качества Piper: покрытие фенем против словаря голоса, темп
//! речи и WAV для прослушивания.
//!
//! Запуск: cargo run --release --example piper_check -p jarvis-core

use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use jarvis_core::tts_piper::PiperEngine;
use std::collections::HashMap;
use std::io::BufWriter;
use std::path::Path;

fn main() -> Result<()> {
    let voice = "ruslan";
    let piper = PiperEngine::new(voice)?;
    let sr = piper.sample_rate();

    // Словарь фенем модели: какие IPA-символы она знает.
    let model_json = jarvis_core::paths::tts_models_dir()
        .join(format!("piper-{voice}/model.json"));
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&model_json)?)?;
    let map: HashMap<String, Vec<i64>> = serde_json::from_value(cfg["phoneme_id_map"].clone())?;
    let known = |c: char| map.contains_key(&c.to_string());

    let phrases = [
        "Привет! Я Джарвис, твой голосовой ассистент.",
        "Сейчас на улице двадцать один градус и небольшая облачность.",
        "Открываю ютуб, одну секунду.",
    ];

    println!("Голос: {voice}, {sr} Гц\n");
    let mut all_out = Vec::new();
    for text in phrases {
        let ipa = piper.phonemize(text)?;
        let chars: Vec<char> = ipa.chars().collect();
        let unknown: Vec<char> = chars.iter().copied().filter(|&c| !known(c)).collect();
        let coverage = 100.0 * (chars.len() - unknown.len()) as f64 / chars.len() as f64;

        let t = std::time::Instant::now();
        let samples = piper.synthesize(text, 1.0)?;
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
        println!("  IPA ({}/{} симв.): {}", chars.len() - unknown.len(), chars.len(), ipa);
        println!(
            "  покрытие словаря {coverage:.1}% | неизвестные: {}",
            if unknown.is_empty() {
                "нет".into()
            } else {
                unknown.iter().collect::<String>()
            }
        );
        println!(
            "  аудио {audio:.2} с ({:.1} симв/с) | синтез {dt:.2} с | RTF {:.3} | RMS {rms:.3} | пик {peak:.2}\n",
            text.chars().count() as f64 / audio,
            dt / audio,
        );
        all_out.extend(samples);
        all_out.extend(std::iter::repeat(0.0f32).take(sr as usize / 3));
    }

    let out_path = Path::new("resources/piper_check.wav");
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

    Ok(())
}
