//! Верификация wake word на реальном эталоне: статистика счётов по классам
//! (мин/макс/среднее), false positive / false negative на пороге эталона.
//! Позитивы — записи [jarvis] из train-каталога; негативы — [none].
//! Запуск: JARVIS_HOME=~/.local/share/Jarvis cargo run -p jarvis-core --example wake_verify
//!
//! Счёты считаются на БОЕВОЙ шкале (best_p − weight·best_n) — той же,
//! что и в детекторе. Критерии приёмки: худший позитив выше порога,
//! лучший негатив ниже порога с запасом.

use anyhow::{Context, Result};

fn read_window(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f32> = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / (1 << (spec.bits_per_sample - 1)) as f32)
        .collect();
    let mono: Vec<f32> = if spec.channels == 1 {
        raw
    } else {
        raw.chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    };
    let mono = jarvis_core::audio::resample_audio(&mono, spec.sample_rate as usize, 16_000);
    // Окно 1.2 с, идемпотентно: файл уже окно — берём как есть; длиннее —
    // центр-кроп; короче — паддинг по центру. Повторный паддинг падденного
    // сдвигал фразу и резал хвост (скор валидации падал на 0.2).
    let window = 16_000 * 12 / 10;
    match mono.len().cmp(&window) {
        std::cmp::Ordering::Equal => Ok(mono),
        std::cmp::Ordering::Greater => {
            let extra = mono.len() - window;
            Ok(mono[extra / 2..extra / 2 + window].to_vec())
        }
        std::cmp::Ordering::Less => {
            let mut out = vec![0_f32; window];
            let start = (window - mono.len()) / 2;
            out[start..start + mono.len()].copy_from_slice(&mono);
            Ok(out)
        }
    }
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let stt = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));
    let (train_dir, _) = jarvis_core::wake_train::sample_dirs();
    let reference = rustpotter::EmbeddingReference::load_reference_only(
        &jarvis_core::paths::models_dir().join(jarvis_core::wake::EMBEDDING_REFERENCE),
    )
    .context(
        "эталон .jref не загрузился — пересобери: cargo run -p jarvis-core --example wake_enroll",
    )?;
    let threshold = reference.threshold;

    let mut pos_scores: Vec<(String, f32)> = Vec::new();
    let mut neg_scores: Vec<(String, f32)> = Vec::new();
    for entry in std::fs::read_dir(&train_dir)?.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_none_or(|e| !e.eq_ignore_ascii_case("wav"))
        {
            continue;
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let label = jarvis_core::tts::label_of_sample(&name);
        let window = read_window(&path)?;
        let Some(score) = jarvis_core::wake::score_reference(&stt, &window) else {
            continue;
        };
        match label.as_str() {
            "jarvis" => pos_scores.push((name, score)),
            "none" => neg_scores.push((name, score)),
            _ => {}
        }
    }
    if pos_scores.is_empty() || neg_scores.is_empty() {
        anyhow::bail!(
            "Нужны записи обоих классов в {}: [jarvis]*.wav (позитивы) и [none]*.wav (негативы)",
            train_dir.display()
        );
    }

    let stats = |v: &[(String, f32)]| -> (f32, f32, f32) {
        let vals: Vec<f32> = v.iter().map(|(_, s)| *s).collect();
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (
            vals.iter().cloned().fold(f32::INFINITY, f32::min),
            vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            sorted[sorted.len() / 2],
        )
    };

    let (pmin, pmax, pmean) = stats(&pos_scores);
    let (nmin, nmax, nmean) = stats(&neg_scores);
    let fnr = pos_scores.iter().filter(|(_, s)| *s < threshold).count();
    let fpr = neg_scores.iter().filter(|(_, s)| *s >= threshold).count();

    println!("=== Верификация wake word (порог эталона {threshold:.3}) ===");
    println!(
        "эталон v{}: позитивов {}, негативов {}, вес негативов {}, фон: {}",
        reference.version,
        reference.positives_count,
        reference.negatives_count,
        reference.weight,
        reference.mean_source
    );
    println!(
        "позитивы ({}): min {pmin:.3} / mean {pmean:.3} / max {pmax:.3}",
        pos_scores.len()
    );
    println!(
        "негативы ({}): min {nmin:.3} / mean {nmean:.3} / max {nmax:.3}",
        neg_scores.len()
    );
    println!(
        "FNR: {fnr}/{} (позитив ниже порога), FPR: {fpr}/{} (негатив выше порога)",
        pos_scores.len(),
        neg_scores.len()
    );
    for (name, s) in &pos_scores {
        println!("   [+] {name}: {s:.3}");
    }
    for (name, s) in &neg_scores {
        println!("   [-] {name}: {s:.3}");
    }

    // Гейт: FNR/FNR = 0, минимальный позитив ≥ 0.7 (калиброванное правило
    // допускает снижение при min_pos < 0.85), негатив до порога — запас
    // минимум ×1.5 (штатно фонетически близкие слова дают 0.28-0.38).
    let margin = threshold - nmax;
    let pass = fnr == 0 && fpr == 0 && pmin >= 0.7 && margin >= nmax.abs().max(0.1) * 0.5;
    if pass {
        println!(
            "✅ ВЕРИФИКАЦИЯ ПРОЙДЕНА: позитивы ≥ {pmin:.3}, FNR/FPR = 0, запас негатива до порога {margin:.3}"
        );
    } else {
        println!(
            "❌ ВЕРИФИКАЦИЯ НЕ ПРОЙДЕНА: pmin {pmin:.3}, nmax {nmax:.3}, FNR {fnr}, FPR {fpr}, запас {margin:.3}"
        );
        std::process::exit(1);
    }
    Ok(())
}
