//! Дефолтный дикторонезависимый эталон «Джарвис» (.jref) по архитектуре
//! Few-Shot Metric Learning: позитивы — фраза разными дикторами Silero
//! (разные тембры и темпы), негативы — похожие по звучанию слова и тишина.
//! Готовый .jref выкладывается в релиз jarvis-models и качается движком;
//! после записи своих сэмплов эталон стоит переобучить в приложении.
//! Запуск: cargo run -p jarvis-core --example wake_enroll

use anyhow::{Context, Result};
use rustpotter::{EmbeddingConfig, EmbeddingDetector, FrameEmbedder};

const WINDOW_SAMPLES: usize = 16_000 * 12 / 10; // 1.2 с, как в боевом детекторе

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let engine = jarvis_core::tts_silero::SileroEngine::new_no_stress("ru_eduard")?;

    let write_window = |dir: &std::path::Path, name: &str, audio: &[f32]| -> Result<Vec<f32>> {
        std::fs::create_dir_all(dir)?;
        let mono = jarvis_core::audio::resample_audio(audio, 48_000, 16_000);
        // Пауза вокруг фразы — фраза живёт в середине боевого окна 1.2 с.
        let mut window = vec![0_f32; WINDOW_SAMPLES];
        let start = (WINDOW_SAMPLES - mono.len().min(WINDOW_SAMPLES / 2)) / 2;
        for (i, s) in mono.iter().take(WINDOW_SAMPLES - start).enumerate() {
            window[start + i] = *s;
        }
        let path = dir.join(format!("{name}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec)?;
        for s in &window {
            writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        println!("🎙️ {name}: {}", path.display());
        Ok(window)
    };

    let train = jarvis_core::wake_train::sample_dirs().0;

    // --- Позитивы: «Джарвис» всеми дикторами Silero в трёх темпах ---
    // Чем шире покрытие тембров, тем выше шанс, что живой голос пользователя
    // ложится на ближайший эталон (счёт — максимум по позитивам).
    let mut positives = Vec::new();
    let voices: &[&str] = &[
        "ru_aigul", "ru_albina", "ru_alexandr", "ru_alfia", "ru_bogdan",
        "ru_dmitriy", "ru_eduard", "ru_ekaterina", "ru_gamat", "ru_igor",
        "ru_karina", "ru_kejilgan", "ru_kermen", "ru_marat", "ru_miyau",
        // ru_sibday исключён: не-русское произношение «Джарвис» ломает
        // шаблон (скор 0.51) — это кураторство данных, а не порог.
        "ru_nurgul", "ru_oksana", "ru_onaoy", "ru_ramilia", "ru_roman",
        "ru_safarhuja", "ru_saida", "ru_vika", "ru_zara",
        "ru_zhadyra", "ru_zhazira", "ru_zinaida",
    ];
    let tempos = [0.95_f32, 1.0, 1.05];
    let mut skipped = 0;
    for voice in voices {
        for (k, speed) in tempos.iter().enumerate() {
            let Ok(audio) = engine.synthesize_with(voice, "Джарвис", *speed) else {
                skipped += 1;
                continue;
            };
            let window =
                write_window(&train, &format!("[jarvis]_default_{voice}_{k}"), &audio)?;
            positives.push(window);
        }
    }
    println!("🎙️ Позитивов: {} (пропущено голосов: {skipped})", positives.len());

    // --- Негативы: похожие по звучанию слова, шум, тишина ---
    let mut negatives = Vec::new();
    for (i, text) in [
        "сервис", "запись", "нравится", "привет", "музыка",
        "джаз", "джазовый", "джазист", "жарко", "завтрак",
        "джинсы", "сверстник", "университет", "маркетинг",
    ]
    .into_iter()
    .enumerate()
    {
        let audio = engine.synthesize_with("ru_eduard", text, 1.0)?;
        let window = write_window(&train, &format!("[none]_neg_{i}_{text}"), &audio)?;
        negatives.push(window);
    }
    negatives.push(vec![0_f32; WINDOW_SAMPLES]); // тишина
    let noise: Vec<f32> = (0..WINDOW_SAMPLES)
        .map(|i| 0.03 * ((i as f32 * 0.037).sin() + (i as f32 * 0.011).cos()))
        .collect(); // бытовой гул
    negatives.push(noise);

    // --- Детектор: настоящий энкодер FastConformer ---
    let stt = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));
    let embedder = jarvis_core::wake::FastConformerEmbedder::new(std::sync::Arc::clone(&stt));
    println!("🧠 Размерность эмбеддинга: {}", embedder.dims());

    let mut detector = EmbeddingDetector::new(embedder, EmbeddingConfig::default());

    // Замер цены окна в установившемся режиме — должен укладываться в hop.
    {
        let mut embedder_probe =
            jarvis_core::wake::FastConformerEmbedder::new(std::sync::Arc::clone(&stt));
        for i in 0..2 {
            let started = std::time::Instant::now();
            let _ = embedder_probe.embed(&positives[i % positives.len()]);
            println!("⏱ Эмбеддинг окна 1.2 с ({}): {:.0?}", i, started.elapsed());
        }
        // Раскладка цены: mel vs энкодер vs декодер (через полный transcribe).
        if let Ok(t) = stt.lock().unwrap_or_else(|e| e.into_inner()).model().transcribe(&positives[0]) {
            println!("⏱ transcribe 1.2 с: mel {:.0} мс, encoder {:.0} мс, decode {:.0} мс",
                t.mel_ms, t.encoder_ms, t.decode_ms);
        }
    }

    let reference = detector
        .enroll(&positives, &negatives)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Самопроверка: score_reference — та же семантика, что у боевого
    // детектора (центрирование + max по позитивам).
    let mut score = |w: &[f32]| -> f32 {
        jarvis_core::wake::score_reference(&stt, w).expect("score")
    };
    println!("— счёты позитивов:");
    let mut min_pos = f32::MAX;
    for (i, p) in positives.iter().enumerate() {
        let s = score(p);
        min_pos = min_pos.min(s);
        println!("   {i}: {s:.3}");
    }
    println!("— счёты негативов:");
    let mut max_neg = f32::MIN;
    for (i, n) in negatives.iter().enumerate() {
        let s = score(n);
        max_neg = max_neg.max(s);
        println!("   {i}: {s:.3}");
    }
    println!(
        "✅ Зазор: позитив ≥ {min_pos:.3}, негатив ≤ {max_neg:.3}; порог в эталоне {stored:.4}",
        stored = reference.threshold
    );

    let out = jarvis_core::paths::models_dir().join(jarvis_core::wake::EMBEDDING_REFERENCE);
    detector
        .save_reference(&out)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("сохранение {}", out.display()))?;
    println!("✅ Эталон сохранён: {}", out.display());
    Ok(())
}
