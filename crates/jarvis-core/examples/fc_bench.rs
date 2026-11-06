//! Бенчмарк этапов FastConformer: Metal против CPU на одной и той же фразе.
//! Цель — найти фиксированную стоимость (~700 мс на команду) и решить,
//! где считать энкодер. Без квантизации: везде FP32.
//!
//! Запуск: cargo run --release --example fc_bench -p jarvis-core

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use jarvis_core::asr::model::FastConformer;
use std::time::Instant;

fn bench_encoder(model: &FastConformer, mel: &Tensor, length: usize, label: &str) -> Result<()> {
    // Тёплый прогон (компиляция шейдеров/прогрев кэшей — отдельно).
    let _ = model.encoder.forward(mel, length)?;

    // Повторяем 3 раза, берём лучший — меряем устойчивое состояние.
    let mut best_total = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        let (enc, t_valid) = model.encoder.forward(mel, length)?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        best_total = best_total.min(dt);
        let _ = (&enc, t_valid);
    }
    println!("{label}: энкодер целиком {:.0} мс (лучший из 3)", best_total);
    Ok(())
}

fn bench_encoder_cpu(models_dir: &std::path::Path, mel: &Tensor, length: usize) -> Result<()> {
    // Полная загрузка модели на CPU — честный замер CPU-пути.
    let started = Instant::now();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[models_dir.join("fastconformer/model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )?
    };
    let cfg = jarvis_core::asr::config::load(&models_dir.join("fastconformer/config.json"))?;
    let encoder = jarvis_core::asr::conformer::Encoder::load(&vb, &cfg, &Device::Cpu)?;
    println!(
        "CPU: загрузка энкодера {:.0} мс",
        started.elapsed().as_secs_f64() * 1000.0
    );

    let mel_cpu = mel.to_device(&Device::Cpu)?;
    let _ = encoder.forward(&mel_cpu, length)?; // прогрев
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        let (enc, t_valid) = encoder.forward(&mel_cpu, length)?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        best = best.min(dt);
        let _ = (&enc, t_valid);
    }
    println!("CPU: энкодер целиком {:.0} мс (лучший из 3)", best);
    Ok(())
}

fn main() -> Result<()> {
    let model = FastConformer::load(&jarvis_core::paths::models_dir())?;
    println!(
        "=== fc_bench: устройство модели — {} ===",
        if model.is_metal() { "Metal" } else { "CPU" }
    );

    // Короткая команда (типовой сценарий) и длинная фраза.
    for (fixture, label) in [("cmd1", "команда ~2с"), ("long1", "фраза ~10с")] {
        let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
            "build/fastconformer/fixtures/{fixture}.json"
        ))?)?;
        let audio: Vec<f32> = fx["audio"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        // mel
        let t0 = Instant::now();
        let (mel, length) = model.frontend.compute(&audio, &model.device)?;
        let mel_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("— {label}: mel {:.0} мс (кадров {}, валидных {length})", mel_ms, mel.dim(2)?);

        bench_encoder(&model, &mel, length, &format!("{label} Metal"))?;
        if fixture == "cmd1" {
            bench_encoder_cpu(&jarvis_core::paths::models_dir(), &mel, length)?;
        }
    }

    // Полный путь transcribe на текущем устройстве (тёплый).
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("build/fastconformer/fixtures/cmd1.json")?)?;
    let audio: Vec<f32> = fx["audio"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let _ = model.transcribe(&audio)?; // прогрев
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        let t = model.transcribe(&audio)?;
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        println!("  transcribe: «{}» mel {:.0} enc {:.0} dec {:.0} мс", t.text, t.mel_ms, t.encoder_ms, t.decode_ms);
    }
    println!("полный transcribe (лучший): {:.0} мс", best);
    Ok(())
}