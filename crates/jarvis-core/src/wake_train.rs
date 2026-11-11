//! Обучение wake word модели (RustPotter): запись сэмплов, тренировка с
//! прогрессом, тест на проверочных записях, применение в движке.
//!
//! Конвенция разметки rustpotter: ярлык берётся из имени файла `[label]_...`,
//! файлы без тега считаются негативами (ярлык `none`).

use crate::audio::{calculate_rms, resample_audio};
use crate::events::{EventTx, JarvisEvent};
use crate::paths;
use crate::tts::list_wav_samples;
use anyhow::{Context, Result, bail};
use rustpotter::{
    Rustpotter, WakewordRefBuildFromFiles, WakewordSave,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

/// Сколько секунд тишины завершают запись сэмпла.
const RECORD_SILENCE_SECS: f32 = 0.7;
const RECORD_RMS_THRESHOLD: f32 = 0.02;
const TARGET_RATE: usize = 16_000;
/// Значение mfcc_size, совпадающее с дефолтами rustpotter.
const MFCC_SIZE: u16 = 30;

/// Каталоги с сэмплами: (обучающие, проверочные).
pub fn sample_dirs() -> (PathBuf, PathBuf) {
    (
        paths::train_dir().join("jarvis"),
        paths::train_dir().join("jarvis_test"),
    )
}

/// Каталог записей диктанта для профиля голоса (`train/voice`).
pub fn voice_samples_dir() -> PathBuf {
    paths::train_dir().join("voice")
}

/// Фраза диктанта для профиля голоса: покрывает все фонемы русской речи
/// (шипящие ш/щ/ж/ч, ц, ф, х, ы, э, ю, я, ё) и разные интонации — вопрос,
/// перечисление, команда, смена темпа. Слово «Джарвис» сознательно не входит:
/// его тональность покрывает обучение wake word, а здесь важен весь остальной
/// диапазон голоса. Чтение занимает ~40 секунд.
pub const VOICE_TRAINING_PHRASE: &str = "\
За окном светит солнце, и щебечут воробьи. Ты слышишь этот шум? \
Запиши в задачник: купить хлеб, молоко и шесть спелых яблок. \
Открой жёлтую папку, найди чужой документ и проверь цифры: восемьсот сорок семь. \
Маленькая Юля едет на юг в июле. \
Что ты видишь на экране? \
Повтори всё чётко и с выражением — громко, тихо, быстро и медленно.";

/// Доля слов эталонной фразы, найденных в расшифровке (0..1).
///
/// Сравнение устойчиво к ошибкам ASR: слово считается найденным при полном
/// совпадении после нормализации (нижний регистр, ё→е, только буквы/цифры).
pub fn phrase_overlap(target: &str, transcript: &str) -> f32 {
    fn words(text: &str) -> Vec<String> {
        text.to_lowercase()
            .replace('ё', "е")
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }
    let target = words(target);
    if target.is_empty() {
        return 0.0;
    }
    let spoken_words = words(transcript);
    let spoken: std::collections::HashSet<&String> = spoken_words.iter().collect();
    let matched = target.iter().filter(|w| spoken.contains(w)).count();
    matched as f32 / target.len() as f32
}

/// Жёсткий предел диктанта: защита от забытой кнопки «Стоп».
pub const VOICE_MAX_SECS: f32 = 180.0;

/// Открытая запись диктанта: пишет микрофон до флага `stop` (или жёсткого
/// капа [`VOICE_MAX_SECS`]). В отличие от [`record_sample`], паузы в речи
/// запись не завершают — пользователь читает текст в своём темпе. Файл
/// сохраняется всегда (проверку прочитанного делает вызывающая сторона).
pub fn record_until_stop(dir: &Path, label: &str, stop: &AtomicBool) -> Result<PathBuf> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::mpsc::sync_channel;

    fs::create_dir_all(dir)?;
    let label_tag = format!("[{label}]");

    let (tx, rx) = sync_channel::<Vec<f32>>(64);

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("Нет устройства ввода")?;
    let sup = device.default_input_config()?;
    let ch = sup.channels() as usize;
    let native_sr = sup.sample_rate() as usize;

    let mut cfg: cpal::StreamConfig = sup.into();
    cfg.buffer_size = cpal::BufferSize::Fixed(1024);

    let stream = match sup.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            cfg,
            move |d: &[f32], _| {
                let _ = tx.send(crate::audio::downmix_f32(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            cfg,
            move |d: &[i16], _| {
                let _ = tx.send(crate::audio::downmix_i16(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            cfg,
            move |d: &[u16], _| {
                let _ = tx.send(crate::audio::downmix_u16(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,
        f => bail!("Неподдерживаемый формат микрофона: {f:?}"),
    };
    stream.play()?;

    println!("🎙️ Диктант: запись до кнопки «Стоп»");
    let max_samples = (TARGET_RATE as f32 * VOICE_MAX_SECS) as usize;
    let mut buffer: Vec<f32> = Vec::with_capacity(max_samples.min(TARGET_RATE * 120));

    while buffer.len() < max_samples {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let chunk = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(c) => c,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!("Поток микрофона закрыт"),
        };
        let chunk_16k = if native_sr != TARGET_RATE {
            resample_audio(&chunk, native_sr, TARGET_RATE)
        } else {
            chunk
        };
        buffer.extend_from_slice(&chunk_16k);
    }

    drop(stream);

    if buffer.len() < (TARGET_RATE as f32 * 0.5) as usize {
        bail!("Слишком короткая запись — нажми «Стоп», когда дочитаешь");
    }

    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("{label_tag}_{}_{}.wav", ts, std::process::id()));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)
        .with_context(|| format!("Не удалось создать {:?}", path))?;
    for &s in &buffer {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;

    println!("💾 Диктант сохранён: {:?}", path);
    Ok(path)
}

/// Записывает сэмпл с микрофона: ждёт начала речи, копит до тишины,
/// сохраняет 16 кГц mono WAV с ярлыком в имени файла.
pub fn record_sample(
    dir: &Path,
    label: &str,
    max_secs: f32,
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::mpsc::sync_channel;

    fs::create_dir_all(dir)?;
    let label_tag = if label == "none" {
        "none".to_string()
    } else {
        format!("[{label}]")
    };

    let (tx, rx) = sync_channel::<Vec<f32>>(64);

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("Нет устройства ввода")?;
    let sup = device.default_input_config()?;
    let ch = sup.channels() as usize;
    let native_sr = sup.sample_rate() as usize;

    let mut cfg: cpal::StreamConfig = sup.into();
    cfg.buffer_size = cpal::BufferSize::Fixed(1024);

    let stream = match sup.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            cfg,
            move |d: &[f32], _| {
                let _ = tx.send(crate::audio::downmix_f32(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            cfg,
            move |d: &[i16], _| {
                let _ = tx.send(crate::audio::downmix_i16(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            cfg,
            move |d: &[u16], _| {
                let _ = tx.send(crate::audio::downmix_u16(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,
        f => bail!("Неподдерживаемый формат микрофона: {f:?}"),
    };
    stream.play()?;

    println!("🎙️ Запись сэмпла: скажи wake word...");
    let max_samples = (TARGET_RATE as f32 * max_secs) as usize;
    let silence_trigger = (TARGET_RATE as f32 * RECORD_SILENCE_SECS) as usize;
    let preroll = (TARGET_RATE as f32 * 0.3) as usize;

    let mut buffer: Vec<f32> = Vec::with_capacity(max_samples);
    let mut history: Vec<f32> = Vec::new();
    let mut started = false;
    let mut silence = 0usize;

    while buffer.len() < max_samples {
        if cancel.load(Ordering::SeqCst) {
            bail!("Запись отменена");
        }

        let chunk = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(c) => c,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!("Поток микрофона закрыт"),
        };

        let chunk_16k = if native_sr != TARGET_RATE {
            resample_audio(&chunk, native_sr, TARGET_RATE)
        } else {
            chunk
        };

        let rms = calculate_rms(&chunk_16k);
        if !started {
            for &s in &chunk_16k {
                if history.len() >= preroll {
                    history.remove(0);
                }
                history.push(s);
            }
            if rms >= RECORD_RMS_THRESHOLD {
                started = true;
                buffer.append(&mut history);
                buffer.extend_from_slice(&chunk_16k);
            }
        } else {
            buffer.extend_from_slice(&chunk_16k);
            if rms < RECORD_RMS_THRESHOLD {
                silence += chunk_16k.len();
                if silence >= silence_trigger {
                    break;
                }
            } else {
                silence = 0;
            }
        }
    }

    drop(stream);

    if !started || buffer.len() < (TARGET_RATE as f32 * 0.4) as usize {
        bail!("Речь не распознана, сэмпл не сохранён");
    }

    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("{label_tag}_{}_{}.wav", ts, std::process::id()));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)
        .with_context(|| format!("Не удалось создать {:?}", path))?;
    for &s in &buffer {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;

    println!("💾 Сэмпл сохранён: {:?}", path);
    Ok(path)
}

/// Строит DTW-модель wake word (reference) по записям пользователя.
///
/// Это не нейросеть: сравниваются MFCC-признаки записей с потоком с микрофона
/// (dynamic time warping). Нужно 3–8 сэмплов фразы, построение занимает секунды.
pub fn build_reference_wakeword(events: &EventTx, cancel: &AtomicBool) -> Result<PathBuf> {
    let (train_dir, _) = sample_dirs();

    // В reference-модель попадают только записи фразы: файлы с меткой
    // [jarvis] в имени. Фоновые шумы (метка none) для DTW не нужны.
    let samples: Vec<String> = list_wav_samples(&train_dir)
        .into_iter()
        .filter(|(p, _)| {
            crate::tts::label_of_sample(&p.file_name().unwrap_or_default().to_string_lossy())
                != "none"
        })
        .map(|(p, _)| p.to_string_lossy().to_string())
        .collect();

    if !(3..=10).contains(&samples.len()) {
        bail!(
            "Нужно от 3 до 10 сэмплов фразы, сейчас: {}. Запишите их на этой же странице.",
            samples.len()
        );
    }

    if cancel.load(Ordering::SeqCst) {
        bail!("Отменено");
    }

    println!(
        "🧩 Построение DTW-модели wake word из {} сэмплов...",
        samples.len()
    );
    let started = std::time::Instant::now();

    let wakeword = <rustpotter::WakewordRef as WakewordRefBuildFromFiles>::new_from_sample_files(
        "jarvis".to_string(),
        None,
        None,
        samples,
        MFCC_SIZE,
    )
    .map_err(|e| anyhow::anyhow!("Не удалось построить модель: {e}"))?;

    let out = paths::models_dir().join("jarvis_model_trained.rpw");
    wakeword
        .save_to_file(&out.to_string_lossy())
        .map_err(|e| anyhow::anyhow!("Не удалось сохранить модель: {e}"))?;

    println!(
        "✅ DTW-модель готова за {:.1?}: {:?}",
        started.elapsed(),
        out
    );
    let _ = events.send(JarvisEvent::Log {
        message: format!("Модель wake word построена ({:.1?})", started.elapsed()),
    });
    Ok(out)
}

/// Возвращает лучший score детектора для одного WAV-файла.
pub fn test_wakeword(rpw: &Path, wav: &Path) -> Result<f32> {
    let samples = read_wav_i16_16k(wav)?;

    // Тот же конфиг, что у боевого детектора (band_pass выключен): с
    // дефолтным конфигом совпадение с reference-моделью не находится.
    let cfg = crate::wake::detector_config(0.0);

    let mut det = Rustpotter::new(&cfg).map_err(|e| anyhow::anyhow!("{e}"))?;
    det.add_wakeword_from_file("jarvis", rpw.to_str().context("путь не UTF-8")?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let frame = det.get_samples_per_frame();
    let mut best = 0f32;
    let mut window: Vec<i16> = samples.iter().copied().take(frame).collect();

    for &s in samples.iter().skip(frame) {
        window.push(s);
        if window.len() > frame {
            window.remove(0);
        }
        if window.len() == frame
            && let Some(d) = det.process_samples(window.clone())
        {
            best = best.max(d.score);
        }
    }

    Ok(best)
}

/// Подменяет активную модель wake word обученной, архивируя старую.
/// `active` — путь активной модели (имя берётся из настроек).
/// Движок подхватит новую модель при следующем переходе в режим wake word
/// (флаг `wakeword_reload` в EngineHandles).
pub fn apply_wakeword(trained: &Path, active: &Path) -> Result<PathBuf> {
    if active.exists() {
        let backup = active.with_extension(format!(
            "rpw.bak.{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ));
        fs::rename(active, &backup).with_context(|| format!("Бэкап {:?}", backup))?;
        println!("🗄️ Прошлая модель сохранена как {:?}", backup);
    }
    fs::copy(trained, active).with_context(|| format!("Копирование в {:?}", active))?;
    Ok(active.to_path_buf())
}

fn read_wav_i16_16k(path: &Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("Чтение {:?}", path))?;
    let spec = reader.spec();
    let sr = spec.sample_rate as usize;

    let f32_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    let mono = if spec.channels > 1 {
        crate::audio::downmix_f32(&f32_samples, spec.channels as usize)
    } else {
        f32_samples
    };

    let resampled = if sr != TARGET_RATE {
        resample_audio(&mono, sr, TARGET_RATE)
    } else {
        mono
    };

    Ok(resampled
        .into_iter()
        .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect())
}
#[cfg(test)]
mod voice_train_tests {
    use super::*;

    #[test]
    fn phrase_overlap_full_reading() {
        // Читал ровно фразу — совпадение 1.0 независимо от пунктуации и регистра.
        assert_eq!(phrase_overlap(VOICE_TRAINING_PHRASE, VOICE_TRAINING_PHRASE), 1.0);
        assert_eq!(
            phrase_overlap("Привет, мир!", "привет   МИР!!!"),
            1.0
        );
    }

    #[test]
    fn phrase_overlap_partial_reading() {
        // Дочитал до середины — примерно половина слов.
        let half = "За окном светит солнце, и щебечут воробьи. Ты слышишь этот шум?";
        let overlap = phrase_overlap(VOICE_TRAINING_PHRASE, half);
        assert!(
            (0.2..=0.3).contains(&overlap),
            "четверть фразы ≈ 25% слов, получено {overlap:.2}"
        );
        assert_eq!(phrase_overlap(VOICE_TRAINING_PHRASE, "полная чушь"), 0.0);
        assert_eq!(phrase_overlap(VOICE_TRAINING_PHRASE, ""), 0.0);
    }

    #[test]
    fn phrase_overlap_tolerates_ash_and_cases() {
        // ё/е и регистр не влияют.
        assert_eq!(phrase_overlap("жёлтый чётко", "ЖЕЛТЫЙ ЧЕТКО"), 1.0);
    }

    #[test]
    fn training_phrase_covers_rare_phonemes() {
        // Ключевые фонемы присутствуют в тексте диктанта.
        let t = VOICE_TRAINING_PHRASE.to_lowercase();
        for (phoneme, sample) in [
            ("щ", "щебечут"),
            ("ж", "жёлтую"),
            ("ч", "чётко"),
            ("ц", "цифры"),
            ("ф", "цифры"),
            ("х", "хлеб"),
            ("ш", "шум"),
            ("ы", "ты"),
            ("э", "экран"),
            ("ю", "юля"),
        ] {
            assert!(t.contains(sample), "фонема «{phoneme}» потеряна: нет слова «{sample}»");
        }
    }
}
