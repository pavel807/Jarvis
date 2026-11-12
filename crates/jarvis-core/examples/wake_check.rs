//! Диагностика wake word на этой машине: какой движок выбран (метрический
//! .jref или DTW .rpw), грузится ли эталон и как распределяются счёты
//! эталонных сэмплов из train-каталога (позитивы [jarvis] против негативов).
//! Запуск: cargo run -p jarvis-core --example wake_check
//! Модели ищутся в системной папке моделей (JARVIS_MODELS_DIR переопределяет).

use anyhow::{Context, Result};

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let stt = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));
    let threshold = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())
        .map(|s| s.wake_threshold)
        .unwrap_or(0.6);
    let ww = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())
        .map(|s| s.wakeword_model)
        .unwrap_or_default();

    let (mode, word, window_ms) =
        jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())
            .map(|s| (s.wake_detector, s.wake_word, s.wake_window_ms))
            .unwrap_or_else(|_| ("ctc".to_string(), "Джарвис".to_string(), 1200));
    jarvis_core::wake::create_engine(&stt, &ww, threshold, &mode, &word, window_ms)?;
    println!("✅ Движок wake word создан");

    // Эталонные записи из train-каталога: позитивы должны давать счёт выше
    // порога эталона, негативы — ниже.
    let (train_dir, _) = jarvis_core::wake_train::sample_dirs();
    if !train_dir.exists() {
        println!(
            "ℹ️ Записей для самопроверки нет ({}) — запиши сэмплы в приложении",
            train_dir.display()
        );
        return Ok(());
    }
    let read_window = |path: &std::path::Path| -> Result<Vec<f32>> {
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
        Ok(jarvis_core::audio::resample_audio(
            &mono,
            spec.sample_rate as usize,
            16_000,
        ))
    };

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
        let mut audio = read_window(&path).with_context(|| name.clone())?;
        // Окно 1.2 с, идемпотентно (см. wake_verify): файл уже окно — как есть.
        let window = 16_000 * 12 / 10;
        audio.resize(window, 0.0);
        match jarvis_core::wake::score_reference(&stt, &audio) {
            Some(s) => println!("   [{label}] {name}: счёт {s:.3}"),
            None => println!("   [{label}] {name}: эталон не загружен"),
        }
    }
    Ok(())
}
