//! Активационная фраза: пользователь выбирает **любую**, Silero её
//! синтезирует, а CTC-детектор проверяет, что модель фразу слышит.
//!
//! Зачем синтез, если эталон — текст? Проверка: ASR-модель может кодить
//! необычную фразу неверно (для «Джарвис» в скользящем окне она давала
//! «Дервись»). Синтез разными дикторами и темпами показывает, сколько
//! вариантов произношения модель понимает буквально, и даёт воспроизводимый
//! эталон произношения для самопроверки.
//!
//! Длина фразы важна для детектора: окно должно накрывать её целиком,
//! поэтому в отчёте — рекомендованный `window_ms`.

use crate::stt::Stt;
use crate::wake_ctc::{self, CtcScratch};
use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Дикторы Silero для проверки — разные тембры.
const VOICES: &[&str] = &[
    "ru_aigul",
    "ru_albina",
    "ru_dmitriy",
    "ru_eduard",
    "ru_ekaterina",
    "ru_karina",
    "ru_oksana",
    "ru_vika",
];

/// Темпы произнесения.
const TEMPOS: &[f32] = &[0.95, 1.0, 1.05];

/// Отчёт по проверке активационной фразы.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhraseReport {
    /// Фраза как задал пользователь.
    pub phrase: String,
    /// Сколько вариантов произнесения синтезировано.
    pub total: usize,
    /// Сколько вариантов модель услышала буквально.
    pub detected: usize,
    /// Худший счёт среди услышанных (средний лог-постер, ближе к 0 — лучше).
    pub worst_score: f32,
    /// Рекомендованная длина окна детектора под эту фразу, мс.
    pub window_ms: usize,
    /// Каталог с синтезированными записями.
    pub samples_dir: String,
}

/// Синтезировать фразу разными дикторами и проверить, слышит ли её модель.
pub fn generate_and_check(stt: &Arc<Mutex<Stt>>, phrase: &str) -> Result<PhraseReport> {
    let phrase = phrase.trim();
    if phrase.is_empty() {
        bail!("Фраза-активация пустая");
    }
    let engine = crate::tts_silero::SileroEngine::new_no_stress("ru_eduard")
        .context("Silero недоступен — фразу нечем синтезировать")?;

    // Путь абсолютный: без JARVIS_HOME train_dir() относительный, и записи
    // легли бы в текущий каталог процесса.
    let dir: PathBuf = std::path::absolute(crate::paths::train_dir().join("phrase"))
        .unwrap_or_else(|_| crate::paths::train_dir().join("phrase"));
    std::fs::create_dir_all(&dir).with_context(|| format!("создать {}", dir.display()))?;

    let mut scratch = CtcScratch::default();
    let mut total = 0usize;
    let mut detected = 0usize;
    let mut worst = f32::INFINITY;
    let mut longest_ms = 0usize;

    for voice in VOICES {
        for (ti, speed) in TEMPOS.iter().enumerate() {
            let Ok(audio) = engine.synthesize_with(voice, phrase, *speed) else {
                continue;
            };
            let mono = crate::audio::resample_audio(&audio, 48_000, 16_000);
            longest_ms = longest_ms.max(mono.len() * 1000 / 16_000);
            // Окно для проверки — фраза целиком с полями по краям.
            let win = (mono.len() + 8_000).max(16_000 * 12 / 10);
            let mut window = vec![0_f32; win];
            let start = (win - mono.len().min(win / 2)) / 2;
            for (i, s) in mono.iter().take(win - start).enumerate() {
                window[start + i] = *s;
            }
            let path = dir.join(format!("phrase_{voice}_{ti}.wav"));
            write_wav(&path, &window)?;
            total += 1;
            if wake_ctc::ctc_frames(stt, &window, &mut scratch).is_none() {
                continue;
            }
            if let Some(m) = wake_ctc::match_word_text(stt, &mut scratch, phrase) {
                detected += 1;
                worst = worst.min(m.score);
            }
        }
    }

    if total == 0 {
        bail!("Silero не синтезировал ни одного варианта фразы");
    }
    // Детектор должен накрывать фразу целиком: окно = фраза + 400 мс запаса.
    let window_ms = (longest_ms + 400).max(1200);

    Ok(PhraseReport {
        phrase: phrase.to_string(),
        total,
        detected,
        worst_score: if worst.is_finite() { worst } else { f32::NAN },
        window_ms,
        samples_dir: dir.to_string_lossy().to_string(),
    })
}

/// Записать моно- wav 16 кГц (i16).
fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
