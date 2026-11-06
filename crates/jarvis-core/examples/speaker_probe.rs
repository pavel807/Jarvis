//! Диагностика фильтра голоса: сходство записей диктанта между собой.
//!
//! Запуск: cargo run -p jarvis-core --example speaker_probe
//!
//! Ожидания: косинус между двумя диктантами ОДНОГО голоса ≥ 0.6; косинус
//! длинной записи с её же 3-секундным куском — тоже высокий. Низкие значения
//! означают, что эмбеддинги speakrs несовместимы между длинами входа — и
//! нужен другой способ сравнения (пул по чанкам, PLDA, минимальная длина).

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
use std::path::Path;

fn read_wav_16k(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut samples: Vec<f32> = Vec::new();
    for frame in reader.samples::<i32>() {
        let raw = frame?;
        let max = 1i64 << (spec.bits_per_sample.saturating_sub(1));
        samples.push(raw as f32 / max as f32);
    }
    if channels > 1 {
        samples = samples
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect();
    }
    if spec.sample_rate != 16_000 {
        samples = jarvis_core::audio::resample_audio(&samples, spec.sample_rate as usize, 16_000);
    }
    Ok(samples)
}

/// Усреднение конечных фреймов эмбеддингов: то же, что в speaker.rs.
fn mean_embedding(result: &speakrs::DiarizationResult) -> Option<Vec<f32>> {
    let mut pool: Vec<f32> = Vec::new();
    let mut dim = 0usize;
    let mut total = 0usize;
    let mut bad = 0usize;
    for chunk in result.embeddings.rows() {
        for frame in chunk.rows() {
            total += 1;
            if dim == 0 {
                dim = frame.len();
            }
            if frame.len() == dim && frame.iter().all(|v| v.is_finite()) {
                pool.extend(frame.iter().copied());
            } else {
                bad += 1;
            }
        }
    }
    println!(
        "  фреймов всего {total}, битых {bad}, годных {}",
        pool.len() / dim.max(1)
    );
    if pool.is_empty() || dim == 0 {
        return None;
    }
    let count = pool.len() / dim;
    let mut mean = vec![0.0f32; dim];
    for chunk in pool.chunks(dim) {
        for (m, v) in mean.iter_mut().zip(chunk) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= count as f32;
    }
    Some(mean)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let clean: Vec<f32> = v.iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect();
    let norm = clean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-9 {
        return clean;
    }
    clean.iter().map(|x| x / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new("train/voice");
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "в train/voice нет записей");
    println!("Записей: {}", files.len());

    #[cfg(target_os = "macos")]
    let modes: &[(&str, ExecutionMode)] = &[
        ("CoreML", ExecutionMode::CoreMl),
        ("CPU", ExecutionMode::Cpu),
    ];
    #[cfg(not(target_os = "macos"))]
    let modes: &[(&str, ExecutionMode)] = &[("CPU", ExecutionMode::Cpu)];

    for (label, mode) in modes {
        println!("\n=== Режим: {label} ===");
        let mut pipeline = match OwnedDiarizationPipeline::from_pretrained(*mode) {
            Ok(p) => p,
            Err(e) => {
                println!("  недоступен: {e}");
                continue;
            }
        };

        // Канонические 3-секундные окна (как в новой оконной верификации).
        const WINDOW: usize = 16_000 * 3;
        let mut window_means: Vec<(String, Vec<f32>)> = Vec::new();
        let mut profile_vectors: Vec<Vec<f32>> = Vec::new();
        for (i, path) in files.iter().enumerate().take(2) {
            let samples = read_wav_16k(path)?;
            let len_secs = samples.len() as f32 / 16_000.0;
            let mid = samples.len() / 2;
            let voiced = &samples[mid.saturating_sub(48_000)..(mid + 48_000).min(samples.len())];
            let mut start = 0usize;
            let mut windows = 0usize;
            while start + WINDOW <= voiced.len() {
                let run = pipeline.run(&voiced[start..start + WINDOW])?;
                if let Some(m) = mean_embedding(&run) {
                    let n = normalize(&m);
                    window_means.push((format!("запис{i}.окно{windows}"), n.clone()));
                    profile_vectors.push(n);
                    windows += 1;
                }
                start += WINDOW;
            }
            println!("{:?}: {len_secs:.1} с → {windows} окон", path.file_name());
        }

        // Профиль = среднее окон; сверяем каждое окно против профиля.
        let dim = profile_vectors[0].len();
        let mut profile = vec![0.0f32; dim];
        for v in &profile_vectors {
            for (p, x) in profile.iter_mut().zip(v) {
                *p += x;
            }
        }
        for p in &mut profile {
            *p /= profile_vectors.len() as f32;
        }
        let profile = normalize(&profile);

        println!("\nОкно × профиль (порог 0.6):");
        for (name, w) in &window_means {
            println!("  {name}: {:.3}", cosine(w, &profile));
        }
    }
    Ok(())
}
