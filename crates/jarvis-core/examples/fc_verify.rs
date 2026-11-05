//! Численная верификация candle-реализации FastConformer против ONNX-эталона.
//!
//! Фикстуры готовит `tools/fastconformer_fixtures.py` (истина —
//! encoder/decoder_joint-model.onnx istupakov + его NumPy-препроцессор):
//! для каждого wav сравниваются поэтапно:
//! 1. mel-фичи (Rust DFT+нормализация vs NumPy);
//! 2. выход энкодера (candle 17 блоков vs ONNX);
//! 3. жадный CTC-декод (ids);
//! 4. жадный RNNT-декод (ids и текст).
//!
//! Запуск: cargo run --release --example fc_verify -p jarvis-core
//! (требует models/fastconformer и build/fastconformer/fixtures).

use anyhow::{Context, Result};
use jarvis_core::asr::model::FastConformer;
use jarvis_core::asr::{ctc, rnnt};
use std::path::Path;

fn maxdiff(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, usize, usize) {
    let mut max = 0f32;
    let (rows, cols) = (a.len().min(b.len()), a.first().map(|r| r.len()).unwrap_or(0));
    let cols = cols.min(b.first().map(|r| r.len()).unwrap_or(0));
    for (ra, rb) in a.iter().zip(b.iter()).take(rows) {
        for (va, vb) in ra.iter().zip(rb.iter()).take(cols) {
            max = max.max((va - vb).abs());
        }
    }
    (max, rows, cols)
}

fn main() -> Result<()> {
    let fixtures_dir = Path::new("build/fastconformer/fixtures");
    let files: Vec<_> = std::fs::read_dir(fixtures_dir)
        .with_context(|| format!("Нет каталога фикстур {fixtures_dir:?} — запусти tools/fastconformer_fixtures.py"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .map(|e| e.path())
        .collect();

    let model = FastConformer::load(&jarvis_core::paths::models_dir())?;
    println!(
        "=== верификация: {} фикстур, устройство: {} ===",
        files.len(),
        if model.is_metal() { "Metal" } else { "CPU" }
    );

    let blank = model.cfg.blank_id;
    let mut failures = 0;

    for fx in files {
        let raw = std::fs::read_to_string(&fx)?;
        let fx: serde_json::Value = serde_json::from_str(&raw)?;
        let name = fx["wav"].as_str().unwrap_or("?").to_string();
        let audio: Vec<f32> = fx["audio"]
            .as_array()
            .expect("audio")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        // 1. Mel-фронтенд.
        let (mel, length) = model.frontend.compute(&audio, &model.device)?;
        let my_mel = mel.squeeze(0)?.to_vec2::<f32>()?;
        let ref_mel: Vec<Vec<f32>> = fx["mel"]
            .as_array()
            .expect("mel")
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect();
        let (mel_diff, rows, cols) = maxdiff(&ref_mel, &my_mel);

        // 2. Энкодер.
        let (enc, t_valid) = model.encoder.forward(&mel, length)?;
        let my_enc = enc.squeeze(0)?.to_vec2::<f32>()?;
        let ref_enc: Vec<Vec<f32>> = fx["enc_out"][0]
            .as_array()
            .expect("enc_out")
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect();
        // Сравниваем только ВАЛИДНЫЕ кадры (0..t_valid) — это контракт
        // модели. Хвостовые кадры за t_valid ONNX маскирует (Where -10000),
        // реализация считает их без маски: на транскрипт они не влияют
        // (декодер идёт по t_valid кадрам).
        let t_ref = fx["enc_valid"].as_u64().unwrap() as usize;
        let (enc_diff, erows, ecols) = maxdiff(
            &ref_enc[..t_ref.min(ref_enc.len())],
            &my_enc[..t_ref.min(my_enc.len())],
        );
        let enc_len_match = t_ref == t_valid
            && ref_enc.len() >= t_ref
            && my_enc.len() >= t_ref;

        // 3. CTC greedy.
        let logits = model.ctc_head.logits(&enc)?;
        let (frames, vout) = (logits.dim(1)?, logits.dim(2)?);
        let flat = logits.reshape((frames * vout,))?.to_vec1::<f32>()?;
        let my_ctc = ctc::greedy(&flat, frames, vout, blank);
        let ref_ctc: Vec<usize> = fx["ctc_ids"]
            .as_array()
            .expect("ctc_ids")
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let ctc_match = my_ctc == ref_ctc;

        // 4. RNNT greedy.
        let my_rnnt = rnnt::greedy(&enc, t_valid, &model.predictor, &model.joint, blank, model.cfg.decoder.max_symbols_per_step)?;
        let ref_rnnt: Vec<usize> = fx["rnnt_ids"]
            .as_array()
            .expect("rnnt_ids")
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let rnnt_match = my_rnnt == ref_rnnt;
        let my_rnnt_text = model.tokenizer.decode(&my_rnnt);
        let ref_rnnt_text = fx["rnnt_text"].as_str().unwrap_or("").to_string();

        let ok = enc_len_match && ctc_match && rnnt_match && mel_diff < 1e-2 && enc_diff < 1e-2;
        if !ok {
            failures += 1;
        }
        println!(
            "{icon} {name}: mel Δ{mel_diff:.2e} ({rows}×{cols}) | enc Δ{enc_diff:.2e} ({erows}×{ecols}, len={t_valid} match={enc_len_match}) | CTC {ctc_match} | RNNT {rnnt_match}",
            icon = if ok { "✅" } else { "❌" }
        );
        if !rnnt_match {
            println!("   RNNT эталон: «{ref_rnnt_text}»");
            println!("   RNNT наш:    «{my_rnnt_text}»");
            let n = my_rnnt.len().min(ref_rnnt.len());
            if let Some(i) = (0..n).find(|&i| my_rnnt[i] != ref_rnnt[i]) {
                println!(
                    "   первое расхождение: кадр-токен #{i}: мой {} ({:?}) vs эталон {} ({:?}), всего мой {} vs эталон {}",
                    my_rnnt[i],
                    model.tokenizer.piece(my_rnnt[i]),
                    ref_rnnt[i],
                    model.tokenizer.piece(ref_rnnt[i]),
                    my_rnnt.len(),
                    ref_rnnt.len()
                );
            }
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} фикстур не прошли верификацию");
    }
    println!("🎉 Все фикстуры сошлись с ONNX-эталоном");
    Ok(())
}
