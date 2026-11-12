//! Замер: wake word по CTC-постерам против метрического детектора.
//!
//! Печатает разделение классов на одном и том же корпусе, что и
//! `wake_verify`: позитивы — записи [jarvis], негативы — [none].
//! Слово задаётся аргументом: `cargo run -p jarvis-core --example wake_ctc_probe -- Джарвис`
//!
//! Запуск: JARVIS_HOME=~/.local/share/Jarvis cargo run -p jarvis-core --example wake_ctc_probe

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
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
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
    let word = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Джарвис".to_string());
    let stt = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));

    // Переиспользуемые буферы декода: в горячем пути аллокаций нет.
    let mut scratch = jarvis_core::wake_ctc::CtcScratch::default();

    let target = jarvis_core::wake_ctc::encode_word(&stt, &word)
        .with_context(|| format!("не закодировать слово «{word}»"))?;
    let pieces: Vec<String> = {
        let guard = stt.lock().unwrap();
        target
            .iter()
            .map(|&id| {
                guard
                    .model()
                    .tokenizer
                    .piece(id)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| format!("#{id}"))
            })
            .collect()
    };
    println!("🎯 слово «{word}» → {} токенов: {pieces:?}", target.len());

    // Синтетические шумовые негативы: тишина, белый шум, бытовой гул,
    // «не-речь» (шум с формантной модуляцией). Настоящего фона комнаты
    // на машине нет — эти случаи хоть как-то проверяют ложные срабатывания.
    let window_len = 16_000 * 12 / 10;
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 20_000) as f32 / 20_000.0 - 0.5
    };
    let mut noise_cases: Vec<(&str, Vec<f32>)> = Vec::new();
    noise_cases.push(("тишина", vec![0_f32; window_len]));
    noise_cases.push(("белый шум", (0..window_len).map(|_| rnd()).collect()));
    noise_cases.push((
        "бытовой гул",
        (0..window_len)
            .map(|i| 0.03 * ((i as f32 * 0.037).sin() + (i as f32 * 0.011).cos()))
            .collect(),
    ));
    noise_cases.push((
        "шум с формантной модуляцией",
        (0..window_len)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.02 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin())
            })
            .collect(),
    ));

    // Режим --synth: синтезируем САМО слово разными дикторами Silero и
    // меряем recall на нём. Это и есть проверка «любого активационного
    // слова»: эталон — текст, переобучение и синтез эталона не нужны.
    let synth = std::env::args().any(|a| a == "--synth");
    if synth {
        let voices = [
            "ru_aigul",
            "ru_albina",
            "ru_alexandr",
            "ru_dmitriy",
            "ru_eduard",
            "ru_ekaterina",
            "ru_igor",
            "ru_karina",
            "ru_marat",
            "ru_oksana",
        ];
        let engine = jarvis_core::tts_silero::SileroEngine::new_no_stress("ru_eduard")?;
        let mut hits = 0usize;
        let mut total = 0usize;
        let mut worst = f32::INFINITY;
        for voice in voices {
            for speed in [0.95f32, 1.0, 1.05] {
                let Ok(audio) = engine.synthesize_with(voice, &word, speed) else {
                    continue;
                };
                let mono = jarvis_core::audio::resample_audio(&audio, 48_000, 16_000);
                let mut w = vec![0_f32; window_len];
                let start = (window_len - mono.len().min(window_len / 2)) / 2;
                for (i, s) in mono.iter().take(window_len - start).enumerate() {
                    w[start + i] = *s;
                }
                if jarvis_core::wake_ctc::ctc_frames(&stt, &w, &mut scratch).is_none() {
                    continue;
                }
                total += 1;
                let m = jarvis_core::wake_ctc::match_word_text(&stt, &mut scratch, &word);
                match m {
                    Some(m) => {
                        hits += 1;
                        worst = worst.min(m.score);
                    }
                    None => {
                        jarvis_core::wake_ctc::greedy_stream(&scratch.frames, &mut scratch.stream);
                        let pieces: Vec<String> = {
                            let guard = stt.lock().unwrap();
                            scratch
                                .stream
                                .iter()
                                .map(|(id, _, _)| {
                                    guard
                                        .model()
                                        .tokenizer
                                        .piece(*id)
                                        .map(|p| p.to_string())
                                        .unwrap_or_default()
                                })
                                .collect()
                        };
                        println!("   промах: {voice} @ {speed} → CTC «{pieces:?}»");
                    }
                }
            }
        }
        println!("🎚 синтез слова «{word}»: recall {hits}/{total}, худший счёт {worst:.4}");
    }

    let (train_dir, _) = jarvis_core::wake_train::sample_dirs();
    let mut pos: Vec<(String, f32)> = Vec::new();
    let mut neg: Vec<(String, f32)> = Vec::new();
    let mut samples_printed = 0;

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
        if jarvis_core::wake_ctc::ctc_frames(&stt, &window, &mut scratch).is_none() {
            eprintln!("⚠️ {name}: постеры не посчитались");
            continue;
        }
        let m = jarvis_core::wake_ctc::match_word_text(&stt, &mut scratch, &word);
        let score = m.as_ref().map(|m| m.score).unwrap_or(f32::NEG_INFINITY);
        if samples_printed < 3 {
            jarvis_core::wake_ctc::greedy_stream(&scratch.frames, &mut scratch.stream);
            let decoded: String = {
                let guard = stt.lock().unwrap();
                let ids: Vec<usize> = scratch.stream.iter().map(|(id, _, _)| *id).collect();
                guard.model().tokenizer.decode(&ids)
            };
            println!(
                "   {name}: CTC-декод «{decoded}» | совпадение: {:?}",
                m.as_ref().map(|m| (m.score, m.span_frames, m.tokens))
            );
            samples_printed += 1;
        }
        match label.as_str() {
            "jarvis" => pos.push((name, score)),
            "none" => neg.push((name, score)),
            _ => {}
        }
    }

    // Шумовые случаи: считаем отдельно от корпуса.
    for (title, w) in &noise_cases {
        if jarvis_core::wake_ctc::ctc_frames(&stt, w, &mut scratch).is_none() {
            eprintln!("⚠️ {title}: постеры не посчитались");
            continue;
        }
        let m = jarvis_core::wake_ctc::match_word_text(&stt, &mut scratch, &word);
        jarvis_core::wake_ctc::greedy_stream(&scratch.frames, &mut scratch.stream);
        let decoded: String = {
            let guard = stt.lock().unwrap();
            let ids: Vec<usize> = scratch.stream.iter().map(|(id, _, _)| *id).collect();
            guard.model().tokenizer.decode(&ids)
        };
        println!(
            "   шум «{title}»: CTC-декод «{decoded}» | совпадение: {:?}",
            m.as_ref().map(|m| (m.score, m.span_frames, m.tokens))
        );
    }

    let missed: Vec<String> = pos
        .iter()
        .filter(|(_, s)| !s.is_finite())
        .map(|(n, _)| n.clone())
        .collect();
    if !missed.is_empty() {
        println!("   без вхождения слова: {}", missed.join(", "));
    }

    // Режим --stream: сквозная проверка потокового детектора. Корпус
    // подаётся как живой поток (чанками по 160 сэмплов) — проверяются
    // окно, VAD, min_hits и cooldown, а не только сопоставление слова.
    if std::env::args().any(|a| a == "--diag") {
        // Диагностика окон: как модель декодит слово, когда оно накрыто
        // окном целиком, с полями и обрезано.
        let wav_arg = std::env::args()
            .find(|a| a.ends_with(".wav"))
            .unwrap_or_else(|| "[jarvis]_default_ru_eduard_1.wav".to_string());
        let body = read_window(&train_dir.join(&wav_arg))?;
        let mut stream = vec![0_f32; 16_000 / 2];
        stream.extend_from_slice(&body);
        stream.extend(std::iter::repeat(0_f32).take(16_000 * 2));
        for w in [1200usize, 1600, 2000] {
            println!("── окно {w} мс ──");
            let mut det = jarvis_core::wake_ctc::CtcWakeEngine::new(
                std::sync::Arc::clone(&stt),
                &word,
                jarvis_core::wake_ctc::CtcConfig {
                    window_ms: w,
                    hop_ms: 240,
                    min_hits: 1,
                    ..Default::default()
                },
            );
            let mut at = 0usize;
            for chunk in stream.chunks(160) {
                at += chunk.len();
                det.process(chunk);
                if let Some(diag) = det.last_diag() {
                    if diag.encoder_ran {
                        println!(
                            "   t={:5.2} c: декод «{}» счёт {:?}",
                            at as f32 / 16_000.0,
                            diag.decoded,
                            diag.score
                        );
                    }
                }
            }
        }
    }

    // Фон микрофона: тихий шум вместо цифровой тишины.
    let mut floor_seed: u64 = 0x243F_6A88_85A3_08D3;
    let mut floor_noise = |n: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            floor_seed ^= floor_seed << 13;
            floor_seed ^= floor_seed >> 7;
            floor_seed ^= floor_seed << 17;
            out.push((floor_seed % 20_000) as f32 / 20_000.0 - 0.5);
        }
        out
    };

    if std::env::args().any(|a| a == "--sweep") {
        // Свип конфигурации окна: сколько позиций скользящего окна накрывают
        // слово целиком — (окно − слово) / шаг. От этого зависит, наберётся
        // ли min_hits на живом потоке.
        let configs = [
            (1200usize, 160usize, 2usize),
            (1200, 240, 1),
            (1600, 160, 2),
            (1000, 160, 2),
        ];
        for (w, h, hits) in configs {
            let mut det = jarvis_core::wake_ctc::CtcWakeEngine::new(
                std::sync::Arc::clone(&stt),
                &word,
                jarvis_core::wake_ctc::CtcConfig {
                    window_ms: w,
                    hop_ms: h,
                    min_hits: hits,
                    ..Default::default()
                },
            );
            let mut fired = 0usize;
            let mut total = 0usize;
            let mut fp = 0usize;
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
                let body = read_window(&path)?;
                let mut stream = floor_noise(16_000 / 2);
                stream.extend_from_slice(&body);
                stream.extend(floor_noise(16_000 * 3 / 2));
                det.reset();
                let mut n = 0;
                for chunk in stream.chunks(160) {
                    if det.process(chunk).is_some() {
                        n += 1;
                    }
                }
                total += 1;
                if label == "jarvis" {
                    if n > 0 {
                        fired += 1;
                    }
                } else if label == "none" && n > 0 {
                    fp += 1;
                }
            }
            // Шумовые потоки — в счёт ложных срабатываний.
            let mut fp_noise = 0usize;
            for (_, w) in &noise_cases {
                let mut stream = floor_noise(16_000 / 2);
                stream.extend_from_slice(w);
                stream.extend(floor_noise(16_000 * 3 / 2));
                det.reset();
                for chunk in stream.chunks(160) {
                    if det.process(chunk).is_some() {
                        fp_noise += 1;
                        break;
                    }
                }
            }
            println!(
                "   окно {w} мс / шаг {h} мс / min_hits {hits}: recall {fired}/{total}, ложных {fp} (+{fp_noise} на шуме)"
            );
        }
    }

    if std::env::args().any(|a| a == "--stream") {
        let mut det = jarvis_core::wake_ctc::CtcWakeEngine::new(
            std::sync::Arc::clone(&stt),
            &word,
            jarvis_core::wake_ctc::CtcConfig::default(),
        );
        let mut fired = 0usize;
        let mut total = 0usize;
        let mut fp = 0usize;
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
            // Живой поток: тишина перед словом и после — окно должно
            // проскользить по фразе, как в микрофоне.
            let mut stream = vec![0_f32; 16_000 / 2];
            stream.extend_from_slice(&window);
            stream.extend(std::iter::repeat(0_f32).take(16_000 * 3 / 2));
            det.reset();
            let mut hits = 0;
            for chunk in stream.chunks(160) {
                if det.process(chunk).is_some() {
                    hits += 1;
                }
            }
            total += 1;
            if label == "jarvis" {
                if hits > 0 {
                    fired += 1;
                } else {
                    println!("   не сработало: {name}");
                }
            } else if label == "none" && hits > 0 {
                fp += 1;
                println!("   ЛОЖНОЕ срабатывание: {name}");
            }
        }
        println!("🎚 поток: recall {fired}/{total}, ложных срабатываний на негативах: {fp}");
    }

    let stat = |v: &[(String, f32)]| {
        let vals: Vec<f32> = v
            .iter()
            .map(|(_, s)| *s)
            .filter(|s| s.is_finite())
            .collect();
        let hits = vals.len();
        (
            vals.iter().cloned().fold(f32::INFINITY, f32::min),
            vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            if hits > 0 {
                vals.iter().sum::<f32>() / hits as f32
            } else {
                f32::NAN
            },
            hits,
        )
    };
    let (pmin, pmax, pmean, phits) = stat(&pos);
    let (nmin, nmax, nmean, nhits) = stat(&neg);
    println!("\n=== CTC wake word: слово «{word}» ===");
    println!(
        "позитивы ({} всего, {} со словом): min {pmin:.3} / mean {pmean:.3} / max {pmax:.3}",
        pos.len(),
        phits
    );
    println!(
        "негативы ({} всего, {} со словом): min {nmin:.3} / mean {nmean:.3} / max {nmax:.3}",
        neg.len(),
        nhits
    );
    // Разделение: порог посередине между худшим позитивом и лучшим негативом.
    if phits > 0 && nhits > 0 {
        let threshold = (pmin + nmax) / 2.0;
        let fnr = pos.iter().filter(|(_, s)| *s < threshold).count();
        let fpr = neg.iter().filter(|(_, s)| *s >= threshold).count();
        println!(
            "порог {threshold:.3}: FNR {fnr}/{}, FPR {fpr}/{}, запас {:.3}",
            pos.len(),
            neg.len(),
            pmin - nmax
        );
    }
    Ok(())
}
