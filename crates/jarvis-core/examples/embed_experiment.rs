//! Эксперимент: какая схема пулинга даёт высокое косинусное сходство
//! между разными голосами на одном слове? Сравниваем:
//! A) mean-pool сырых кадров (текущая) — скор ~0.04
//! B) L2-нормализация КАЖДОГО кадра → mean-pool → L2
//! C) B + вычитание глобального среднего (CMV-like)
use jarvis_core::stt::Stt;
use jarvis_core::tts_silero::SileroEngine;
use rustpotter::FrameEmbedder;
use std::sync::{Arc, Mutex};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (na, nb) = (a.iter().map(|x| x*x).sum::<f32>().sqrt(), b.iter().map(|x| x*x).sum::<f32>().sqrt());
    a.iter().zip(b).map(|(x, y)| x*y).sum::<f32>() / (na * nb + 1e-9)
}

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let engine = SileroEngine::new_no_stress("ru_eduard")?;
    let window = 16_000 * 12 / 10;
    let mk = |voice: &str, text: &str| -> anyhow::Result<Vec<f32>> {
        let audio = engine.synthesize_with(voice, text, 1.0)?;
        let mono = jarvis_core::audio::resample_audio(&audio, 48_000, 16_000);
        let mut w = vec![0f32; window];
        let start = (window - mono.len().min(window / 2)) / 2;
        for (i, s) in mono.iter().take(window - start).enumerate() { w[start + i] = *s; }
        Ok(w)
    };
    let stt = Arc::new(Mutex::new(Stt::load(&jarvis_core::paths::models_dir())?));
    let mut embedder = jarvis_core::wake::FastConformerEmbedder::new(Arc::clone(&stt));

    let voices = ["ru_eduard", "ru_dmitriy", "ru_karina", "ru_roman", "ru_vika"];
    let mut pos_raw = Vec::new(); let mut pos_fr = Vec::new();
    for v in voices {
        let w = mk(v, "Джарвис")?;
        // A: сырой mean-pool
        let raw = embedder.embed(&w).map_err(|e| anyhow::anyhow!(e))?;
        // B/C: кадры энкодера по отдельности
        let stt_g = stt.lock().unwrap();
        let model = stt_g.model();
        let (mel, length) = model.frontend.compute(&w, &model.device)?;
        let (enc, t_valid) = model.encoder.forward(&mel, length)?;
        let (t, d) = (enc.dim(1)?, enc.dim(2)?);
        let flat = enc.reshape((t, d))?.to_vec2::<f32>()?;
        let t_use = t_valid.min(t).max(1);
        let mut fr = vec![0f32; d];
        for frame in flat.iter().take(t_use) {
            let norm: f32 = frame.iter().map(|x| x*x).sum::<f32>().sqrt();
            for (acc, x) in fr.iter_mut().zip(frame) { *acc += x / (norm + 1e-9); }
        }
        drop(stt_g);
        let n: f32 = fr.iter().map(|x| x*x).sum::<f32>().sqrt();
        for x in fr.iter_mut() { *x /= n; }
        pos_raw.push(raw); pos_fr.push(fr);
    }
    let neg_j = mk("ru_eduard", "сервис")?;
    let neg_r = embedder.embed(&neg_j).map_err(|e| anyhow::anyhow!(e))?;

    // B-вариант негатива тоже кадрами:
    let stt_g = stt.lock().unwrap();
    let model = stt_g.model();
    let (mel, length) = model.frontend.compute(&neg_j, &model.device)?;
    let (enc, t_valid) = model.encoder.forward(&mel, length)?;
    let (t, d) = (enc.dim(1)?, enc.dim(2)?);
    let flat = enc.reshape((t, d))?.to_vec2::<f32>()?;
    let mut fr_neg = vec![0f32; d];
    for frame in flat.iter().take(t_valid.min(t)) {
        let norm: f32 = frame.iter().map(|x| x*x).sum::<f32>().sqrt();
        for (acc, x) in fr_neg.iter_mut().zip(frame) { *acc += x / (norm + 1e-9); }
    }
    drop(stt_g);
    let n: f32 = fr_neg.iter().map(|x| x*x).sum::<f32>().sqrt();
    for x in fr_neg.iter_mut() { *x /= n; }

    println!("=== A: сырой mean-pool (текущая схема) ===");
    for (i, p) in pos_raw.iter().enumerate() {
        let best = pos_raw.iter().enumerate().filter(|(j, _)| *j != i)
            .map(|(_, q)| cosine(p, q)).fold(f32::NEG_INFINITY, f32::max);
        println!("  {} ↔ другие позитивы max cos: {:.3} | к негативу: {:.3}", voices[i], best, cosine(p, &neg_r));
    }
    println!("=== B: L2-кадры → mean-pool → L2 ===");
    for (i, p) in pos_fr.iter().enumerate() {
        let best = pos_fr.iter().enumerate().filter(|(j, _)| *j != i)
            .map(|(_, q)| cosine(p, q)).fold(f32::NEG_INFINITY, f32::max);
        println!("  {} ↔ другие позитивы max cos: {:.3} | к негативу: {:.3}", voices[i], best, cosine(p, &fr_neg));
    }
    part_c()
}

#[allow(dead_code)]
fn part_c() -> anyhow::Result<()> {
    // Вариант C: центрмирование — вычитаем среднее фоновых эмбеддингов.
    let engine = SileroEngine::new_no_stress("ru_eduard")?;
    let window = 16_000 * 12 / 10;
    let mk = |voice: &str, text: &str| -> anyhow::Result<Vec<f32>> {
        let audio = engine.synthesize_with(voice, text, 1.0)?;
        let mono = jarvis_core::audio::resample_audio(&audio, 48_000, 16_000);
        let mut w = vec![0f32; window];
        let start = (window - mono.len().min(window / 2)) / 2;
        for (i, s) in mono.iter().take(window - start).enumerate() { w[start + i] = *s; }
        Ok(w)
    };
    let stt = Arc::new(Mutex::new(Stt::load(&jarvis_core::paths::models_dir())?));
    let embed = |w: &[f32]| -> anyhow::Result<Vec<f32>> {
        let stt_g = stt.lock().unwrap();
        let model = stt_g.model();
        let (mel, length) = model.frontend.compute(w, &model.device)?;
        let (enc, t_valid) = model.encoder.forward(&mel, length)?;
        let (t, d) = (enc.dim(1)?, enc.dim(2)?);
        let flat = enc.reshape((t, d))?.to_vec2::<f32>()?;
        let mut out = vec![0f32; d];
        for frame in flat.iter().take(t_valid.min(t).max(1)) {
            for (acc, x) in out.iter_mut().zip(frame) { *acc += x; }
        }
        let n = t_valid.min(t).max(1) as f32;
        for x in out.iter_mut() { *x /= n; }
        Ok(out)
    };
    // Фоновый набор: разные слова — среднее направление «речи вообще».
    let mut background = Vec::new();
    for text in ["сервис", "запись", "нравится", "привет", "музыка", "таймер"] {
        background.push(embed(&mk("ru_eduard", text)?)?);
    }
    let mut mean = vec![0f32; background[0].len()];
    for b in &background {
        for (acc, x) in mean.iter_mut().zip(b) { *acc += x; }
    }
    for x in mean.iter_mut() { *x /= background.len() as f32; }

    let center = |e: &[f32]| -> Vec<f32> {
        let mut v: Vec<f32> = e.iter().zip(&mean).map(|(x, m)| x - m).collect();
        let n: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
        if n > 1e-9 { for x in v.iter_mut() { *x /= n; } }
        v
    };

    let voices = ["ru_eduard", "ru_dmitriy", "ru_karina", "ru_roman", "ru_vika"];
    let mut pos = Vec::new();
    for v in voices {
        let w = mk(v, "Джарвис")?;
        let e = embed(&w)?;
        pos.push(center(&e));
    }
    let mut neg_words = Vec::new();
    for t in ["сервис", "запись", "нравится"] {
        let w = mk("ru_eduard", t)?;
        let e = embed(&w)?;
        neg_words.push(center(&e));
    }
    println!("=== C: центрмирование по фону речи ===");
    for (i, p) in pos.iter().enumerate() {
        let same = pos.iter().enumerate().filter(|(j, _)| *j != i)
            .map(|(_, q)| cosine(p, q)).fold(f32::NEG_INFINITY, f32::max);
        let worst_neg = neg_words.iter().map(|n| cosine(p, n)).fold(f32::NEG_INFINITY, f32::max);
        println!("  {} ↔ позитивы max cos: {:.3} | к похожим словам max: {:.3}", voices[i], same, worst_neg);
    }
    Ok(())
}
