//! Дефолтный дикторонезависимый эталон «Джарвис» (.jref) по архитектуре
//! Few-Shot Metric Learning: позитивы — фраза разными дикторами Silero
//! (разные тембры и темпы), негативы — похожие по звучанию слова и тишина.
//! Готовый .jref выкладывается в релиз jarvis-models и качается движком;
//! после записи своих сэмплов эталон стоит переобучить в приложении.
//! Запуск: cargo run -p jarvis-core --example wake_enroll

use anyhow::{Context, Result};
use rustpotter::{EmbeddingConfig, EmbeddingDetector, FrameEmbedder};

const WINDOW_SAMPLES: usize = 16_000 * 12 / 10; // 1.2 с, как в боевом детекторе

fn read_window(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
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
    let voices: &[&str] = &[
        "ru_aigul",
        "ru_albina",
        "ru_alexandr",
        "ru_alfia",
        "ru_bogdan",
        "ru_dmitriy",
        "ru_eduard",
        "ru_ekaterina",
        "ru_gamat",
        "ru_igor",
        "ru_karina",
        "ru_kejilgan",
        "ru_kermen",
        "ru_marat",
        "ru_miyau",
        // ru_sibday исключён: не-русское произношение «Джарвис» ломает
        // шаблон (скор 0.51) — это кураторство данных, а не порог.
        "ru_nurgul",
        "ru_oksana",
        "ru_onaoy",
        "ru_ramilia",
        "ru_roman",
        "ru_safarhuja",
        "ru_saida",
        "ru_vika",
        "ru_zara",
        "ru_zhadyra",
        "ru_zhazira",
        "ru_zinaida",
    ];
    let tempos = [0.95_f32, 1.0, 1.05];
    const HOLD_OUT: usize = 4; // последних кураторских голосов — верификация
    let mut skipped = 0;
    // voice → его окна (до кураторства).
    let mut per_voice: Vec<(&str, Vec<Vec<f32>>)> = Vec::new();
    for voice in voices {
        let mut windows = Vec::new();
        for (k, speed) in tempos.iter().enumerate() {
            let Ok(audio) = engine.synthesize_with(voice, "Джарвис", *speed) else {
                skipped += 1;
                continue;
            };
            let window = write_window(&train, &format!("[jarvis]_default_{voice}_{k}"), &audio)?;
            windows.push(window);
        }
        if !windows.is_empty() {
            per_voice.push((voice, windows));
        }
    }

    // --- Кураторство: голос-аутсайдер (чьё «Джарвис» не похоже на
    // остальные) портит LOO-зазор — такой голос не в эталон. Похожество
    // голоса = max косинус его среднего (центрированного по словам) к
    // средним остальных голосов.
    let stt_probe = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));
    // Размерность эмбеддинга берём из энкодера: хардкод 512 молча обрезал
    // бы средние через zip при любой другой модели (тихо, без ошибки).
    let dims =
        jarvis_core::wake::FastConformerEmbedder::new(std::sync::Arc::clone(&stt_probe)).dims();
    let voice_means: Vec<Vec<f32>> = per_voice
        .iter()
        .map(|(_, windows)| {
            let mut acc = vec![0_f32; dims];
            for w in windows {
                let mut e = jarvis_core::wake::FastConformerEmbedder::new(std::sync::Arc::clone(
                    &stt_probe,
                ))
                .embed(w)
                .expect("embed");
                rustpotter::l2_normalize(&mut e);
                for (a, x) in acc.iter_mut().zip(&e) {
                    *a += x;
                }
            }
            let n = windows.len() as f32;
            for x in acc.iter_mut() {
                *x /= n;
            }
            rustpotter::l2_normalize(&mut acc);
            acc
        })
        .collect();
    // Среднее направление слов-негативов для центрирования при кураторстве:
    // слова синтезируем здесь же (те же, что пойдут в негативы).
    let mut bg_mean = vec![0_f32; dims];
    let bg_words = [
        "сервис",
        "запись",
        "нравится",
        "привет",
        "музыка",
        "джаз",
        "джазовый",
        "джазист",
        "жарко",
        "завтрак",
        "джинсы",
        "сверстник",
        "университет",
        "маркетинг",
    ];
    let mut bg_count = 0;
    for text in bg_words {
        let Ok(audio) = engine.synthesize_with("ru_eduard", text, 1.0) else {
            continue;
        };
        let mono = jarvis_core::audio::resample_audio(&audio, 48_000, 16_000);
        let mut w = vec![0_f32; 16_000 * 12 / 10];
        let start = (w.len() - mono.len().min(w.len() / 2)) / 2;
        for (i, s) in mono.iter().take(w.len() - start).enumerate() {
            w[start + i] = *s;
        }
        let mut e =
            jarvis_core::wake::FastConformerEmbedder::new(std::sync::Arc::clone(&stt_probe))
                .embed(&w)
                .expect("embed");
        rustpotter::l2_normalize(&mut e);
        for (a, x) in bg_mean.iter_mut().zip(&e) {
            *a += x;
        }
        bg_count += 1;
    }
    if bg_count == 0 {
        anyhow::bail!(
            "ни одного фонового слова не синтезировалось — нечем центрировать эмбеддинги"
        );
    }
    for x in bg_mean.iter_mut() {
        *x /= bg_count as f32;
    }
    rustpotter::l2_normalize(&mut bg_mean);

    let voice_sim = |i: usize| -> f32 {
        let mut best = f32::NEG_INFINITY;
        for (j, m) in voice_means.iter().enumerate() {
            if j == i {
                continue;
            }
            // Центрированные голосовые средние: убираем общий фон слов.
            let mut a: Vec<f32> = voice_means[i]
                .iter()
                .zip(&bg_mean)
                .map(|(x, m)| x - m)
                .collect();
            let mut b: Vec<f32> = m.iter().zip(&bg_mean).map(|(x, m)| x - m).collect();
            rustpotter::l2_normalize(&mut a);
            rustpotter::l2_normalize(&mut b);
            best = best.max(a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>());
        }
        best
    };
    let sim_before: Vec<(String, f32)> = per_voice
        .iter()
        .enumerate()
        .map(|(i, (v, _))| (v.to_string(), voice_sim(i)))
        .collect();
    for (v, sim) in &sim_before {
        println!("🎙️ голос {v}: похожесть на остальные {sim:.3}");
    }
    const MIN_VOICE_SIM: f32 = 0.5;
    let kept: Vec<usize> = (0..per_voice.len())
        .filter(|&i| voice_sim(i) >= MIN_VOICE_SIM)
        .collect();
    let dropped: Vec<&str> = per_voice
        .iter()
        .enumerate()
        .filter(|(i, _)| voice_sim(*i) < MIN_VOICE_SIM)
        .map(|(_, (v, _))| *v)
        .collect();
    if !dropped.is_empty() {
        println!(
            "🗑️ Голоса-аутсайдеры (похожесть < {MIN_VOICE_SIM}): {}",
            dropped.join(", ")
        );
    }
    assert!(
        kept.len() >= 8,
        "после кураторства осталось {} голосов — нужно ≥ 8",
        kept.len()
    );
    let enrolled_count = kept.len().saturating_sub(HOLD_OUT);

    let mut positives = Vec::new();
    for (order, &vi) in kept.iter().enumerate() {
        let windows = &per_voice[vi].1;
        for window in windows {
            // Hold-out голоса идут в файлы для верификации, но НЕ в эталон:
            // их счёты — честная оценка на невиданных голосах.
            if order < enrolled_count {
                positives.push(window.clone());
            }
        }
    }
    println!(
        "🎙️ Позитивов в эталоне: {} (голосов {}, hold-out голосов: {HOLD_OUT}, пропущено: {skipped})",
        positives.len(),
        enrolled_count,
    );

    // --- Негативы: похожие по звучанию слова, шум, тишина ---
    let mut negatives = Vec::new();
    for (i, text) in [
        "сервис",
        "запись",
        "нравится",
        "привет",
        "музыка",
        "джаз",
        "джазовый",
        "джазист",
        "жарко",
        "завтрак",
        "джинсы",
        "сверстник",
        "университет",
        "маркетинг",
    ]
    .into_iter()
    .enumerate()
    {
        let audio = engine.synthesize_with("ru_eduard", text, 1.0)?;
        let window = write_window(&train, &format!("[none]_neg_{i}_{text}"), &audio)?;
        negatives.push(window);
    }
    // Тишина и гул не добавляем: нулевой/низкочастотный эмбеддинг после
    // центрирования даёт фантомные направления (см. embedding.rs guard).

    // --- Детектор: настоящий энкодер FastConformer ---
    let stt = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));
    let embedder = jarvis_core::wake::FastConformerEmbedder::new(std::sync::Arc::clone(&stt));
    println!("🧠 Размерность эмбеддинга: {}", embedder.dims());

    // negative_weight 0.3: при w=1.0 жёсткие негативы подавляют и
    // позитивы (ближайший негатив «сервис» лежит в ~0.4 от любого
    // «Джарвиса» — фонетическое родство), мин-скор позитивов падал до
    // 0.58. 0.3 держит позитивы ≥ 0.78 и негативы ≤ 0.3. Вес замораживается
    // в эталоне — боевой счёт использует тот же.
    let mut detector = EmbeddingDetector::new(
        embedder,
        EmbeddingConfig {
            negative_weight: 0.3,
            ..EmbeddingConfig::default()
        },
    )
    .expect("конфиг детектора");

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
        if let Ok(t) = stt
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .model()
            .transcribe(&positives[0])
        {
            println!(
                "⏱ transcribe 1.2 с: mel {:.0} мс, encoder {:.0} мс, decode {:.0} мс",
                t.mel_ms, t.encoder_ms, t.decode_ms
            );
        }
    }

    let reference = detector
        .enroll(&positives, &negatives)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // Сохраняем ДО самопроверки: score_reference читает эталон с диска.
    let out = jarvis_core::paths::models_dir().join(jarvis_core::wake::EMBEDDING_REFERENCE);
    detector
        .save_reference(&out)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("сохранение {}", out.display()))?;

    // Самопроверка: score_reference — та же семантика, что у боевого
    // детектора (центрирование + max по позитивам).
    let score = |w: &[f32]| -> f32 { jarvis_core::wake::score_reference(&stt, w).expect("score") };
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

    // Калибровка порога по hold-out голосам: само-совпадение записей
    // регистрации завышает min_pos (они внутри эталона), из-за чего порог
    // оказывался выше счёта живого голоса. Считаем невиданные голоса
    // уже сохранённым эталоном — и опускаем порог в середину их зазора.
    let mut holdout_scores: Vec<f32> = Vec::new();
    for entry in std::fs::read_dir(&train)?.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if !name.ends_with(".wav") || !name.contains("[jarvis]") {
            continue;
        }
        // Hold-out — файлы голосов из хвоста кураторского списка.
        let enrolled = kept[..enrolled_count]
            .iter()
            .any(|&vi| name.contains(per_voice[vi].0));
        if enrolled {
            continue;
        }
        if let Some(score) = jarvis_core::wake::score_reference(&stt, &read_window(&path)?) {
            holdout_scores.push(score);
        }
    }
    if let Some(&worst_holdout) = holdout_scores
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
    {
        let mut reference = rustpotter::EmbeddingReference::load_reference_only(&out)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Порог: существенно ниже худшего невиданного TTS-голоса — живой
        // голос в эмбеддинг-пространстве сидит ниже синтетики, и запас
        // должен покрывать этот сдвиг. Негативы (≤ 0.34) остаются далеко
        // снизу; шумовые аугментации позитивов слегка проседают — учтено.
        let calibrated = (worst_holdout - 0.25)
            .max(0.0)
            .max(reference.threshold * 0.3);
        if (calibrated - reference.threshold).abs() > f32::EPSILON {
            reference.set_threshold(calibrated);
            println!(
                "📊 Калибровка по hold-out: {} невиданных голосов, худший счёт {worst_holdout:.3} → порог {calibrated:.3} (был {:.3})",
                holdout_scores.len(),
                reference.threshold
            );
            // Сохраняем ИЗМЕНЁННУЮ копию: detector.save_reference перезаписал
            // бы файл своей устаревшей копией (баг перекалибровки).
            reference
                .save_to_file(&out)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        } else {
            println!(
                "📊 Калибровка не требуется: худший hold-out {worst_holdout:.3} выше порога {:.3}",
                reference.threshold
            );
        }
    }
    println!("✅ Эталон сохранён: {}", out.display());
    Ok(())
}
