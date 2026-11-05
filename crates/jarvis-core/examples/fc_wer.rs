//! Замер точности (WER/CER/точные совпадения команд) и скорости (RTF,
//! латентность по этапам) FastConformer на тестовых русских записях.
//!
//! Тестовый набор: build/fastconformer/wavs — голосовые команды («Проиграй
//! музыку», «Открой Premiere Pro», …), длинное предложение, быстрая фраза.
//! Эталоны — тексты, которыми записи сгенерированы.
//!
//! Запуск: cargo run --release --example fc_wer -p jarvis-core
//! Декодер задаётся аргументом: rnnt (по умолчанию) | ctc | ctc_beam.

use jarvis_core::asr::model::{Decoder, FastConformer};
use std::time::Instant;

/// (файл, эталонный текст, это короткая команда?)
const CASES: &[(&str, &str, bool)] = &[
    ("cmd1.wav", "Джарвис, проиграй музыку", true),
    ("cmd2.wav", "Открой Premiere Pro", true),
    ("cmd3.wav", "Поставь музыку на паузу", true),
    ("cmd4.wav", "Переключи следующий трек", true),
    ("slow1.wav", "Останови воспроизведение", true),
    (
        "long1.wav",
        "Сегодня на улице двадцать пять градусов тепла, вероятность дождя шестьдесят процентов, а вечером ожидается усиление ветра до пятнадцати метров в секунду.",
        false,
    ),
];

/// Нормализация для WER/CER: lowercase, без пунктуации, одиночные пробелы.
fn normalize_words(text: &str) -> Vec<String> {
    text.chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect()
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ai) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, bj) in b.iter().enumerate() {
            cur.push((prev[j] + 1).min(cur[j] + 1).min(prev[j] + (ai != bj) as usize));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Набор голосов Silero TTS (движок самого Jarvis): один текст, 8 голосов.
/// build/voice_compare_16k/*.wav; эталон — текст генерации.
const TTS_REFERENCE: &str = "Слушаю вас, Павел. Все системы в норме: процессор загружен на тридцать процентов, оперативной памяти свободно двенадцать гигабайт. Открываю премьер про. Похоже, сегодня будет дождь — взять с собой зонт? До связи.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let decoder = match args.next().as_deref() {
        Some("ctc") => Decoder::CtcGreedy,
        Some("ctc_beam") => Decoder::CtcBeam { width: 8 },
        _ => Decoder::Rnnt,
    };
    let set = args.next().unwrap_or_else(|| "say".into());
    let tts_set = set == "tts";
    let model = FastConformer::load(&jarvis_core::paths::models_dir())?;
    // Словарь Jarvis для ctc_beam: триггеры команд + русские формы
    // произношения (как в движке).
    let dict = (decoder == Decoder::CtcBeam { width: 8 })
        .then(|| {
            let cfg = jarvis_core::commands::CommandsConfig::load(&jarvis_core::paths::commands_file()).ok()?;
            let phrases: Vec<String> = cfg
                .commands
                .iter()
                .filter(|c| c.enabled)
                .flat_map(|c| {
                    let mut variants = c.trigger_words.clone();
                    for w in &c.trigger_words {
                        let ru = jarvis_core::tts_normalizer::apply_tech_pronunciation(&w.to_lowercase());
                        if ru != w.to_lowercase() && !variants.contains(&ru) {
                            variants.push(ru);
                        }
                    }
                    variants
                })
                .collect();
            model.jarvis_dictionary(&phrases).ok()
        })
        .flatten();
    println!(
        "=== fc_wer: декодер {:?}, устройство {} ===",
        decoder,
        if model.is_metal() { "Metal" } else { "CPU" }
    );

    let mut total_ref_words = 0usize;
    let mut total_word_errors = 0usize;
    let mut total_ref_chars = 0usize;
    let mut total_char_errors = 0usize;
    let mut command_total = 0usize;
    let mut command_exact = 0usize;
    let mut audio_secs = 0f64;
    let mut proc_secs = 0f64;
    let mut first_chunk_ms = Vec::new();

    if tts_set {
        // Голоса Silero: один эталон на все файлы каталога.
        let mut voices: Vec<String> = std::fs::read_dir("build/voice_compare_16k")?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        voices.sort();
        let mut total_wer = 0usize;
        let mut total_words = 0usize;
        for voice in &voices {
            let wav = format!("build/voice_compare_16k/{voice}");
            let (samples, sr) = crate_read_wav(&wav)?;
            assert_eq!(sr, 16_000, "ожидаю 16 кГц в {voice}");
            let t = model.transcribe_with(&samples, decoder, dict.as_ref())?;
            let got = normalize_words(&t.text);
            let want = normalize_words(TTS_REFERENCE);
            let wer = levenshtein(&got, &want);
            total_wer += wer;
            total_words += want.len();
            println!(
                "{voice}: WER {wer}/{} | «{}»",
                want.len(),
                t.text
            );
        }
        println!("─────");
        println!(
            "Silero TTS (8 голосов): WER {:.1}%  — декодер {:?}, устройство {}",
            100.0 * total_wer as f64 / total_words.max(1) as f64,
            decoder,
            if model.is_metal() { "Metal" } else { "CPU" }
        );
        return Ok(());
    }

    for (file, reference, is_command) in CASES {
        let wav = format!("build/fastconformer/wavs/{file}");
        let (samples, sr) = crate_read_wav(&wav)?;
        assert_eq!(sr, 16_000, "ожидаю 16 кГц в {file}");
        audio_secs += samples.len() as f64 / 16_000.0;

        let started = Instant::now();
        let t = model.transcribe_with(&samples, decoder, dict.as_ref())?;
        let elapsed = started.elapsed();
        proc_secs += elapsed.as_secs_f64();
        first_chunk_ms.push((t.mel_ms, t.encoder_ms, t.decode_ms));

        let got = normalize_words(&t.text);
        let want = normalize_words(reference);
        let wer = levenshtein(&got, &want);
        let cer = levenshtein(
            &got.concat().chars().map(String::from).collect::<Vec<_>>(),
            &want.concat().chars().map(String::from).collect::<Vec<_>>(),
        );
        total_ref_words += want.len();
        total_word_errors += wer;
        total_ref_chars += want.concat().chars().count();
        total_char_errors += cer;
        if *is_command {
            command_total += 1;
            if got == want {
                command_exact += 1;
            }
        }

        println!(
            "{file}: «{text}»\n    эталон: «{reference}»\n    WER {wer}/{n_w} CER {cer}/{n_c} | {:.2?} (RTF {:.2})",
            elapsed,
            elapsed.as_secs_f64() / (samples.len() as f64 / 16_000.0),
            text = t.text,
            n_w = want.len(),
            n_c = want.concat().chars().count(),
        );
    }

    let n = first_chunk_ms.len() as f64;
    let (mel, enc, dec) = first_chunk_ms
        .iter()
        .fold((0.0, 0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2));
    println!("─────");
    println!(
        "Итог: WER {:.1}%  CER {:.1}%  точные команды {command_exact}/{command_total}",
        100.0 * total_word_errors as f64 / total_ref_words.max(1) as f64,
        100.0 * total_char_errors as f64 / total_ref_chars.max(1) as f64,
    );
    println!(
        "Скорость: аудио {audio_secs:.1} с, обработка {proc_secs:.2} с, RTF {:.2} | средние этапы: mel {:.0} мс, encoder {:.0} мс, decode {:.0} мс",
        proc_secs / audio_secs.max(1e-9),
        mel / n,
        enc / n,
        dec / n,
    );
    Ok(())
}

/// Минимальный WAV-ридер: 16 кГц, mono, PCM16.
fn crate_read_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    Ok((samples, spec.sample_rate))
}
