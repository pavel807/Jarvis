//! Калибровка CTC-детектора wake word на живых записях.
//!
//! Прогоняет **боевой** потоковый детектор (`CtcWakeEngine`) по записанному
//! wav — тот же код, что крутится в движке: окно, VAD, min_hits, cooldown —
//! и разбирает результат по окнам: где фраза декодировалась, какой счёт,
//! где сработало срабатывание.
//!
//! Два режима:
//! - `--silence <файл>` — фон комнаты: показывает, какой максимальный счёт
//!   детектор вообще может набрать в тишине (это и есть потолок ложных
//!   срабатываний) и сколько раз фраза «проявилась» сама;
//! - `--speech <файл>` — произнесения: делит запись на речевые сегменты
//!   (по энергии), считает, сколько сегментов сработало (recall) и какой
//!   счёт был у каждого;
//! - вместе — подбор порога: максимум по тишине + запас, и recall на этом
//!   пороге.
//!
//! Запуск:
//!   wake_calibrate --word Джарвис --silence тишина.wav --speech речь.wav
//!   wake_calibrate --word Компьютер --speech речь.wav

use anyhow::{Context, Result, bail};
use jarvis_core::stt::Stt;
use jarvis_core::wake_ctc::{CtcConfig, CtcWakeEngine};
use std::sync::{Arc, Mutex};

/// Окно анализа: что детектор увидел в этом окне.
struct Win {
    /// Время в аудио, с.
    t: f32,
    /// Признак речи по VAD.
    voiced: bool,
    /// Что услышала модель.
    decoded: String,
    /// Счёт совпадения (None — фраза не декодировалась).
    score: Option<f32>,
    /// Сработало ли срабатывание в этом окне.
    fired: bool,
}

/// Речевой сегмент: пауза между произнесениями разбивает запись.
struct Seg {
    start: f32,
    end: f32,
    best: Option<f32>,
    fired: bool,
    /// Декод в окне с лучшим счётом — для разбора промахов.
    best_decoded: String,
}

fn read_wav(path: &std::path::Path, gain: f32) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("открыть {}", path.display()))?;
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
    let mut out = jarvis_core::audio::resample_audio(&mono, spec.sample_rate as usize, 16_000);
    if (gain - 1.0).abs() > f32::EPSILON {
        for x in out.iter_mut() {
            *x *= gain;
        }
    }
    Ok(out)
}

/// Прогнать потоковый детектор по записи и собрать окна.
fn run_stream(stt: &Arc<Mutex<Stt>>, word: &str, samples: &[f32], cfg: &CtcConfig) -> Vec<Win> {
    let mut eng = CtcWakeEngine::new(Arc::clone(stt), word, cfg.clone());
    let hop = (cfg.hop_ms * 16_000 / 1000).max(1);
    let mut wins = Vec::new();
    for (i, chunk) in samples.chunks(hop).enumerate() {
        let fired = eng.process(chunk).is_some();
        if let Some(d) = eng.last_diag()
            && d.encoder_ran
        {
            wins.push(Win {
                t: (i + 1) as f32 * cfg.hop_ms as f32 / 1000.0,
                voiced: d.voiced,
                decoded: d.decoded,
                score: d.score,
                fired,
            });
        }
    }
    wins
}

/// Разбить окна на речевые сегменты.
///
/// Сегменты строятся только по ГОЛОСНЫМ окнам (пауза > gap сек — разрыв),
/// затем каждое окно прикрепляется к ближайшему сегменту. Неголосные окна
/// нельзя выбрасывать: лучшее окно часто в hangover после конца слова
/// (тогда фраза целиком влезла в окно), и его счёт относится к этому слову.
/// Прикрепление к голосным спанам заодно чинит склейку произнесений:
/// hangover-окна физически заполняют паузу между словами и при простом
/// сравнении «время − конец сегмента» мостят любой разрыв.
fn segments(wins: &[Win], gap: f32) -> Vec<Seg> {
    let mut spans: Vec<(f32, f32)> = Vec::new();
    for w in wins {
        if !w.voiced {
            continue;
        }
        match spans.last_mut() {
            Some(s) if w.t - s.1 <= gap => s.1 = w.t,
            _ => spans.push((w.t, w.t)),
        }
    }
    let mut segs: Vec<Seg> = spans
        .into_iter()
        .map(|(start, end)| Seg {
            start,
            end,
            best: None,
            fired: false,
            best_decoded: String::new(),
        })
        .collect();
    for w in wins {
        let Some((idx, _)) = segs.iter().enumerate().min_by(|a, b| {
            let d = |s: &Seg| {
                if w.t < s.start {
                    s.start - w.t
                } else if w.t > s.end {
                    w.t - s.end
                } else {
                    0.0
                }
            };
            d(a.1).total_cmp(&d(b.1))
        }) else {
            break;
        };
        let s = &mut segs[idx];
        if w.score.is_some_and(|v| s.best.is_none_or(|b| v > b)) {
            s.best = w.score;
            s.best_decoded = w.decoded.clone();
        }
        s.fired |= w.fired;
    }
    segs
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut word = "Джарвис".to_string();
    let mut silence: Option<String> = None;
    let mut speech: Option<String> = None;
    let mut window_ms = 1200usize;
    let mut hop_ms = 160usize;
    let mut min_hits = 2usize;
    let mut vad_rms: Option<f32> = None;
    let mut vad_ratio: Option<f32> = None;
    let mut gain = 1.0f32;
    let mut dump = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--word" => {
                i += 1;
                word = args
                    .get(i)
                    .context("после --word должно идти слово")?
                    .clone();
            }
            "--silence" => {
                i += 1;
                silence = Some(
                    args.get(i)
                        .context("после --silence должен идти файл")?
                        .clone(),
                );
            }
            "--speech" => {
                i += 1;
                speech = Some(
                    args.get(i)
                        .context("после --speech должен идти файл")?
                        .clone(),
                );
            }
            "--window-ms" => {
                i += 1;
                window_ms = args.get(i).context("после --window-ms число")?.parse()?;
            }
            "--hop-ms" => {
                i += 1;
                hop_ms = args.get(i).context("после --hop-ms число")?.parse()?;
            }
            "--min-hits" => {
                i += 1;
                min_hits = args.get(i).context("после --min-hits число")?.parse()?;
            }
            "--vad-rms" => {
                i += 1;
                vad_rms = Some(args.get(i).context("после --vad-rms число")?.parse()?);
            }
            "--vad-ratio" => {
                i += 1;
                vad_ratio = Some(args.get(i).context("после --vad-ratio число")?.parse()?);
            }
            "--gain" => {
                i += 1;
                gain = args.get(i).context("после --gain число")?.parse()?;
            }
            "--dump" => dump = true,
            other => bail!("неизвестный аргумент: {other}"),
        }
        i += 1;
    }
    if silence.is_none() && speech.is_none() {
        bail!("нужен --silence и/или --speech");
    }

    let mut cfg = CtcConfig {
        window_ms,
        hop_ms,
        min_hits,
        ..Default::default()
    };
    if let Some(v) = vad_rms {
        cfg.vad_rms = v;
    }
    if let Some(v) = vad_ratio {
        cfg.vad_ratio = v;
    }
    println!(
        "🎯 слово «{word}»: окно {window_ms} мс, шаг {hop_ms} мс, min_hits {min_hits}, порог {:.2}",
        cfg.threshold
    );
    println!(
        "   VAD: rms > {:.4}, ratio > {:.1}; усиление записи ×{gain:.1}",
        cfg.vad_rms, cfg.vad_ratio
    );

    let stt = Arc::new(Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));

    // Разбор по окнам: что модель слышит в каждом окне.
    if dump {
        for path in [&silence, &speech].into_iter().flatten() {
            let samples = read_wav(std::path::Path::new(path), gain)?;
            let wins = run_stream(&stt, &word, &samples, &cfg);
            println!("\n── ПООКОННО ({path}) ──");
            for w in &wins {
                println!(
                    "   t={:7.2} c  rms-окно  голос {:5}  счёт {:?}  декод «{}»{}",
                    w.t,
                    if w.voiced { "да" } else { "нет" },
                    w.score,
                    w.decoded,
                    if w.fired { "  ← СРАБОТАЛО" } else { "" }
                );
            }
        }
        return Ok(());
    }

    // Потолок ложных срабатываний: максимум счёта в тишине.
    let mut silence_max: Option<f32> = None;
    let mut silence_matches = 0usize;
    let mut silence_fired = 0usize;

    if let Some(path) = &silence {
        let samples = read_wav(std::path::Path::new(path), gain)?;
        let wins = run_stream(&stt, &word, &samples, &cfg);
        let silence_windows = wins.len();
        for w in &wins {
            if let Some(s) = w.score {
                silence_matches += 1;
                silence_max = Some(silence_max.map_or(s, |m: f32| m.max(s)));
            }
            if w.fired {
                silence_fired += 1;
            }
        }
        println!(
            "\n── ТИШИНА ({path}, {:.1} с) ──\n   окон считано: {silence_windows}, фраза декодировалась в {silence_matches}, срабатываний: {silence_fired}",
            samples.len() as f32 / 16_000.0
        );
        match silence_max {
            Some(m) => println!("   максимальный счёт в тишине: {m:.4}"),
            None => println!("   фраза в тишине не декодировалась ни разу (счёт = None)"),
        }
        if silence_fired > 0 {
            println!("   ⚠️ ЛОЖНЫЕ СРАБАТЫВАНИЯ: {silence_fired}");
        }
    }

    // Произнесения: recall по речевым сегментам.
    let mut segs_all: Vec<Seg> = Vec::new();

    if let Some(path) = &speech {
        let samples = read_wav(std::path::Path::new(path), gain)?;
        let wins = run_stream(&stt, &word, &samples, &cfg);
        let speech_windows = wins.len();
        let segs = segments(&wins, 1.0);
        let fired = segs.iter().filter(|s| s.fired).count();
        // «Услышано» — сегмент, где фраза декодировалась с��ётом ≥ порога.
        // Это и есть recall детектора: срабатывание может не произойти из-за
        // cooldown (после срабатывания 2.5 с игнорируется), но слово было
        // услышано.
        let heard = segs
            .iter()
            .filter(|s| s.best.is_some_and(|v| v >= cfg.threshold))
            .count();
        println!(
            "\n── ПРОИЗНЕСЕНИЯ ({path}, {:.1} с) ──\n   окон считано: {speech_windows}, сегментов речи: {}\n   услышано: {heard} (recall {:.0} %), сработало: {fired} (с учётом cooldown)",
            samples.len() as f32 / 16_000.0,
            segs.len(),
            if segs.is_empty() {
                0.0
            } else {
                heard as f32 * 100.0 / segs.len() as f32
            }
        );
        for (n, s) in segs.iter().enumerate() {
            let mark = if s.best.is_some_and(|v| v >= cfg.threshold) {
                "✓"
            } else {
                "✗"
            };
            println!(
                "   {mark} сегмент {:2}: {:.2}–{:.2} с, счёт {:?}, декод «{}»{}",
                n + 1,
                s.start,
                s.end,
                s.best,
                s.best_decoded,
                if s.fired { "  ← СРАБОТАЛО" } else { "" }
            );
        }
        segs_all = segs;
    }

    // Подбор порога.
    if !segs_all.is_empty() {
        let mut best_miss: Option<(f32, String)> = None;
        for s in segs_all
            .iter()
            .filter(|s| !s.best.is_some_and(|v| v >= cfg.threshold))
        {
            if let Some(v) = s.best
                && best_miss.as_ref().is_none_or(|(b, _)| v > *b)
            {
                best_miss = Some((v, s.best_decoded.clone()));
            }
        }
        // Порог, при котором ложных срабатываний в тишине не будет:
        // выше максимального счёта, который детектор набрал в тишине.
        let proposed = silence_max.map(|m| m + 0.05);
        println!("\n── ПОРГОР ──");
        match proposed {
            Some(p) => {
                let detected = segs_all
                    .iter()
                    .filter(|s| s.fired)
                    .count();
                let heard = segs_all
                    .iter()
                    .filter(|s| s.best.is_some_and(|v| v >= p))
                    .count();
                println!(
                    "   предложенный порог {p:.3} (максимум тишины {:.3} + 0.05): услышано {heard}/{}, сработало {detected}/{}",
                    silence_max.unwrap_or(0.0),
                    segs_all.len(),
                    segs_all.len()
                );
            }
            None => println!(
                "   порог по тишине не ограничен (фраза в тишине не декодировалась) — \
                 оставить текущий {:.2}",
                cfg.threshold
            ),
        }
        match best_miss {
            Some((v, d)) => println!(
                "   лучший счёт среди непойманных произнесений: {v:.3} (декод «{d}») — \
                 опустить порог нельзя: ложные срабатывания вырастут"
            ),
            None => println!("   все сегменты пойманы"),
        }
    }
    Ok(())
}
