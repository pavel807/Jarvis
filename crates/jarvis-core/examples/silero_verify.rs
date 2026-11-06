//! Верификация candle-реализации против эталона PyTorch (silero_ref).
//! Запуск: cargo run --release --example silero_verify -p jarvis-core

use anyhow::Result;
use jarvis_core::tts_silero::SileroEngine;

fn main() -> Result<()> {
    let engine = SileroEngine::new_no_stress("ru_eduard")?;
    let audio = engine.synthesize("Привет! Как дела?", 1.0)?;
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!("audio: {} сэмплов, RMS {:.4}", audio.len(), rms);

    let stages = engine.debug_stages("Привет! Как дела?", 1.0)?;
    let ref_st: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("models/silero/ref_stages.json")?)?;
    println!("--- стадии ---");
    for (name, data, _shape) in &stages {
        if let Some(rd) = ref_st.get(name).and_then(|v| v.as_array()) {
            if rd.len() == data.len() {
                let md = data
                    .iter()
                    .zip(rd)
                    .map(|(a, b)| (a - b.as_f64().unwrap_or(0.0) as f32).abs())
                    .fold(0.0f32, f32::max);
                println!("  {:<12} maxdiff {:.6}", name, md);
            } else {
                println!("  {:<12} размер {} против {}", name, data.len(), rd.len());
            }
        }
    }
    let ref_json = std::fs::read_to_string("models/silero/ref.json")?;
    let ref_data: serde_json::Value = serde_json::from_str(&ref_json)?;
    let ref_audio: Vec<f32> = ref_data["audio"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("нет audio"))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    let ref_durs: Vec<i32> = ref_data["durs"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("нет durs"))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as i32)
        .collect();
    let mel_json = std::fs::read_to_string("models/silero/ref_mel.json")?;
    let mel_data: serde_json::Value = serde_json::from_str(&mel_json)?;
    let ref_mel: Vec<f32> = mel_data["mel"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("нет mel"))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    let my_durs = engine.debug_durations("Привет! Как дела?", 1.0)?;
    println!("my  durs: {:?}", my_durs);
    let ref_voc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        "models/silero/ref_voc_stages.json",
    )?)?;
    let voc_stages = engine.debug_voc_from(&ref_mel)?;
    println!("--- вокодер ---");
    for (name, data, _shape) in &voc_stages {
        if let Some(rd) = ref_voc.get(name).and_then(|v| v.as_array()) {
            if rd.len() == data.len() {
                let md = data
                    .iter()
                    .zip(rd)
                    .map(|(a, b)| (a - b.as_f64().unwrap_or(0.0) as f32).abs())
                    .fold(0.0f32, f32::max);
                println!("  {:<10} maxdiff {:.6}", name, md);
            } else {
                println!("  {:<10} размер {} против {}", name, data.len(), rd.len());
            }
        }
    }

    let (my_mel, ch, n) = engine.debug_mel("Привет! Как дела?", 1.0)?;
    if my_mel.len() == ref_mel.len() {
        let md = my_mel
            .iter()
            .zip(&ref_mel)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mmean = my_mel
            .iter()
            .zip(&ref_mel)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / my_mel.len() as f32;
        println!("mel [{},{}]: maxdiff {:.5}, mean {:.5}", ch, n, md, mmean);
    } else {
        println!("⚠️ mel размеры: {} против {}", my_mel.len(), ref_mel.len());
    }
    println!("ref durs: {:?}", ref_durs);

    if audio.len() == ref_audio.len() {
        let diffs: Vec<f32> = audio
            .iter()
            .zip(&ref_audio)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let big = diffs.iter().filter(|d| **d > 0.1).count();
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let max = diffs.iter().cloned().fold(0.0f32, f32::max);
        println!(
            "diff: mean {:.5}, max {:.5}, >0.1: {} из {}",
            mean,
            max,
            big,
            diffs.len()
        );
        let json_out = serde_json::json!({ "rust_audio": audio, "ref_audio": ref_audio });
        std::fs::write("models/silero/diff_dump.json", json_out.to_string())?;
        if max < 0.05 {
            println!("✅ VERIFIED: candle совпадает с PyTorch");
        } else {
            println!("✗ расхождение есть");
        }
    } else {
        println!(
            "⚠️ длина не совпадает: {} против {}",
            audio.len(),
            ref_audio.len()
        );
    }
    Ok(())
}