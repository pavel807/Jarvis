//! Проверка качества Piper: покрытие фенем против словаря голоса, темп
//! речи и WAV для прослушивания. Вторая секция — варианты ритма одной
//! фразы (выразительность/темп/склейка слов), чтобы выбрать на слух.
//!
//! Запуск: cargo run --release --example piper_check -p jarvis-core

use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use jarvis_core::tts_piper::{PiperEngine, PiperParams};
use std::collections::HashMap;
use std::io::BufWriter;

fn save_wav(path: &str, samples: &[f32], sr: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::new(BufWriter::new(std::fs::File::create(path)?), spec)?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let voice = "ruslan";
    let piper = PiperEngine::new(voice)?;
    let sr = piper.sample_rate();

    // Словарь фенем модели: какие IPA-символы она знает.
    let model_json =
        jarvis_core::paths::tts_models_dir().join(format!("piper-{voice}/model.json"));
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&model_json)?)?;
    let map: HashMap<String, Vec<i64>> = serde_json::from_value(cfg["phoneme_id_map"].clone())?;
    let known = |c: char| map.contains_key(&c.to_string());

    let phrases = [
        "Привет! Я Джарвис, твой голосовой ассистент.",
        "Сейчас на улице двадцать один градус и небольшая облачность.",
        "Открываю ютуб, одну секунду.",
    ];

    println!("Голос: {voice}, {sr} Гц\n");
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
        println!("  IPA: {ipa}");
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
    }

    // --- Варианты ритма: одинаковая фраза, разные ручки модели ---
    let phrase = "Открываю ютуб, одну секунду.";
    let _ipa = piper.phonemize(phrase)?;
    println!("--- Варианты ритма для «{phrase}» ---");

    let variants: Vec<(&str, &str, PiperParams)> = vec![
        ("v1", "как сейчас (слова склеены)", PiperParams::default()),
        (
            "v2",
            "выразительнее (noise 1.0)",
            PiperParams {
                noise_scale: Some(1.0),
                ..Default::default()
            },
        ),
        (
            "v3",
            "медленнее (темп 1.15)",
            PiperParams {
                length_scale: Some(1.15),
                ..Default::default()
            },
        ),
    ];

    for (id, label, params) in variants {
        let samples = piper.synthesize_with(phrase, 1.0, params)?;
        let audio = samples.len() as f64 / sr as f64;
        let file = format!("resources/piper_{id}.wav");
        save_wav(&file, &samples, sr)?;
        println!("  {id}: {label:<38} {audio:.2} с -> {file}");
    }

    println!("\n🔊 Послушай resources/piper_v1.wav ... piper_v3.wav при желании сравнить ручки.");

    Ok(())
}
