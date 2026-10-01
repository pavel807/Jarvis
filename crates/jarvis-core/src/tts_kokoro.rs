//! Kokoro-ru TTS (82M StyleTTS2-lite поверх ONNX Runtime): быстрый русский
//! синтез с корректными ударениями и смешанной речью EN+RU.
//!
//! Модель неавторегрессивная: вся фраза за один forward, RTF ~0.1 на CPU.
//! Качество выше Piper ruslan (WER 2.5% против 4.4%), фонемизация —
//! порт `ru_g2p` (espeak ru + позиционная редукция), латинские слова
//! фонемизируются espeak en-us — модель содержит английские IPA-фенемы.
//! Модель ~132 МБ (q8) + стайл-пак голоса, скачивается при первом запуске.
//!
//! Источник: zaakirio/kokoro-ru (OpenRAIL); инференс повторяет их пример.

use crate::kokoro_g2p;
use crate::kokoro_vocab::tokenize;
use crate::paths;
use anyhow::{bail, Context, Result};
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Tensor, Value};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Доступные голоса kokoro-ru (студийные записи с согласием актёров).
pub const KOKORO_VOICES: &[(&str, &str)] = &[
    ("dima", "мужской"),
    ("sveta", "женский, основной"),
    ("masha", "женский"),
];

/// Частота дискретизации выходного аудио.
pub const SAMPLE_RATE: u32 = 24000;

/// Максимальный размер чанка в фенемах: стайл-пак имеет 510 строк.
const MAX_CHUNK_PHONEMES: usize = 480;

/// Пауза между чанками после деления, сэмплов.
const CHUNK_PAUSE_SAMPLES: usize = (SAMPLE_RATE as usize) / 8;

const INPUT_IDS: &str = "input_ids";
const STYLE: &str = "style";
const SPEED: &str = "speed";
const WAVEFORM: &str = "waveform";

/// Имя файла модели по голосу: sveta/masha делят базовый чекпоинт,
/// dima — отдельный.
fn model_file_for(voice: &str) -> &'static str {
    match voice {
        "dima" => "onnx/model_dima.onnx",
        _ => "onnx/model.onnx",
    }
}

pub struct KokoroEngine {
    /// ort-сессия; run требует &mut, поэтому Mutex (инференс однопоточный).
    session: Mutex<Session>,
    /// Строки стайл-пака: [510][256] — выбирается по длине фенемной строки.
    style_rows: Vec<[f32; 256]>,
    espeak_ru: Mutex<espeak_ng::EspeakNg>,
    espeak_en: Mutex<espeak_ng::EspeakNg>,
    voice: String,
}

/// Распаковывает встроенные данные espeak-ng (фонемы + словари ru и en).
fn ensure_espeak_data() -> Result<PathBuf> {
    let data_dir = paths::models_dir().join("espeak-ng-data");
    std::fs::create_dir_all(&data_dir)?;
    if !data_dir.join("phondata").exists() {
        espeak_ng::install_bundled_language(&data_dir, "ru").with_context(|| {
            format!("Не удалось распаковать данные espeak-ng в {:?}", data_dir)
        })?;
    }
    // Словарь английского для смешанной речи (файлы словарей лежат
    // в корне каталога данных без расширения).
    if !data_dir.join("en_dict").exists() && !data_dir.join("en").exists() {
        espeak_ng::install_bundled_language(&data_dir, "en").with_context(|| {
            format!("Не удалось распаковать английский словарь espeak-ng в {:?}", data_dir)
        })?;
    }
    Ok(data_dir)
}

fn init_espeak(data_dir: &Path, voice: &str) -> Result<espeak_ng::EspeakNg> {
    espeak_ng::EspeakNg::with_data_dir(voice, data_dir)
        .map_err(|e| anyhow::anyhow!("Не удалось инициализировать espeak-ng ({voice}): {e}"))
}

impl KokoroEngine {
    /// Загружает голос (скачивает ~132 МБ при первом запуске).
    pub fn new(voice: &str) -> Result<Self> {
        if !KOKORO_VOICES.iter().any(|(v, _)| *v == voice) {
            bail!("Неизвестный голос kokoro: {voice}");
        }

        let dir = ensure_kokoro_model(voice)?;
        let model_path = dir.join(format!(
            "{}.onnx",
            model_file_for(voice)
                .trim_start_matches("onnx/")
                .trim_end_matches(".onnx")
        ));
        let voice_path = dir.join(format!("voices/{voice}.bin"));

        let started = std::time::Instant::now();
        let session = Session::builder()
            .context("Не удалось создать ONNX Session")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Не удалось загрузить модель {:?}", model_path))?;

        let style_rows = load_style_pack(&voice_path)
            .with_context(|| format!("Не удалось загрузить стайл-пак {:?}", voice_path))?;

        let espeak_data = ensure_espeak_data()?;
        let espeak_ru = Mutex::new(init_espeak(&espeak_data, "ru")?);
        let espeak_en = Mutex::new(init_espeak(&espeak_data, "en-us")?);

        println!(
            "✅ Kokoro готов ({voice}, {SAMPLE_RATE} Гц) за {:.1?}",
            started.elapsed()
        );

        Ok(Self {
            session: Mutex::new(session),
            style_rows,
            espeak_ru,
            espeak_en,
            voice: voice.to_string(),
        })
    }

    pub fn voice(&self) -> &str {
        &self.voice
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Синтез фразы; `speed` (0.5..2) — темп.
    pub fn synthesize(&self, text: &str, speed: f32) -> Result<Vec<f32>> {
        let mut audio = Vec::new();
        for (i, chunk) in self.chunk_phonemes(text).iter().enumerate() {
            if i > 0 {
                audio.extend(std::iter::repeat(0.0f32).take(CHUNK_PAUSE_SAMPLES));
            }
            audio.extend(self.synthesize_phonemes(chunk, speed)?);
        }
        Ok(audio)
    }

    /// Текст -> IPA: токены делятся по письменности — кириллица через
    /// espeak ru + правила ru_g2p, латиница через espeak en-us.
    fn phonemize_mixed(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() * 2);
        for token in text.split_whitespace() {
            let ipa = if token.chars().any(|c| ('а'..='я').contains(&c) || c == 'ё' || c.is_uppercase() && ('А'..='Я').contains(&c)) {
                let ru = self
                    .espeak_ru
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .text_to_phonemes(token)
                    .unwrap_or_default();
                kokoro_g2p::normalize_ru(&ru)
            } else {
                self.espeak_en
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .text_to_phonemes(token)
                    .unwrap_or_default()
            };
            if ipa.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&ipa);
        }
        out
    }

    /// Делит по предложениям/словам, фонемизирует каждый кусок отдельно.
    fn chunk_phonemes(&self, text: &str) -> Vec<String> {
        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        let text = text.replace(['[', '{'], "(").replace([']', '}'], ")");
        let sentences: Vec<&str> = text
            .split(['.', '!', '?', ';', '\n'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        for sentence in sentences {
            let sentence = format!("{}.", sentence.trim());
            let lowered = kokoro_g2p::respell_text(&sentence).to_lowercase();
            let phonemes = self.phonemize_mixed(&lowered);
            if phonemes.chars().count() > MAX_CHUNK_PHONEMES {
                // Слишком длинное предложение — добиваем по словам.
                for word in sentence.split_whitespace() {
                    let word_lower = kokoro_g2p::respell_text(word).to_lowercase();
                    let word_phonemes = self.phonemize_mixed(&word_lower);
                    if current.chars().count() + word_phonemes.chars().count() + 1
                        > MAX_CHUNK_PHONEMES
                    {
                        if !current.is_empty() {
                            chunks.push(std::mem::take(&mut current));
                        }
                        current = word_phonemes;
                    } else if current.is_empty() {
                        current = word_phonemes;
                    } else {
                        current.push(' ');
                        current.push_str(&word_phonemes);
                    }
                }
            } else if current.chars().count() + phonemes.chars().count() + 1 > MAX_CHUNK_PHONEMES
            {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                current = phonemes;
            } else if current.is_empty() {
                current = phonemes;
            } else {
                current.push(' ');
                current.push_str(&phonemes);
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    }

    /// Синтез одного фенемного чанка за один forward.
    fn synthesize_phonemes(&self, phonemes: &str, speed: f32) -> Result<Vec<f32>> {
        let tokens = tokenize(phonemes);
        if tokens.is_empty() {
            bail!("Фенемы не дали токенов: \"{}\"", phonemes);
        }

        let mut input_ids: Vec<i64> = Vec::with_capacity(tokens.len() + 2);
        input_ids.push(0);
        input_ids.extend(tokens.iter().copied());
        input_ids.push(0);

        let style_idx =
            (phonemes.chars().count().saturating_sub(1)).min(self.style_rows.len() - 1);
        let style: Vec<f32> = self.style_rows[style_idx].to_vec();

        let tokens_tensor =
            Tensor::from_array(([1usize, input_ids.len()], input_ids)).context("input_ids")?;
        let style_tensor =
            Tensor::from_array(([1usize, style.len()], style)).context("style")?;
        let speed_tensor =
            Tensor::from_array(([1usize], vec![speed.clamp(0.5, 2.0)])).context("speed")?;

        let inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = vec![
            (
                Cow::Borrowed(INPUT_IDS),
                SessionInputValue::Owned(Value::from(tokens_tensor)),
            ),
            (
                Cow::Borrowed(STYLE),
                SessionInputValue::Owned(Value::from(style_tensor)),
            ),
            (
                Cow::Borrowed(SPEED),
                SessionInputValue::Owned(Value::from(speed_tensor)),
            ),
        ];

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session
            .run(SessionInputs::from(inputs))
            .context("Ошибка инференса kokoro")?;

        let (shape, data) = outputs[WAVEFORM]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Нет выхода 'waveform': {e}"))?;

        let total: usize = shape.iter().map(|&d| d as usize).product();
        let mut audio = data.to_vec();
        audio.truncate(total);
        Ok(audio)
    }
}

/// Стайл-пак голоса: сырой float32 [510, 256] (zaakirio/kokoro-ru voices/*.bin).
fn load_style_pack(path: &Path) -> Result<Vec<[f32; 256]>> {
    let bytes = std::fs::read(path).with_context(|| format!("Чтение {:?}", path))?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let rows = floats.len() / 256;
    if rows == 0 || floats.len() % 256 != 0 {
        bail!("Стайл-пак {:?}: неожиданный размер {} f32", path, floats.len());
    }

    Ok(floats
        .chunks_exact(256)
        .map(|c| {
            let mut row = [0f32; 256];
            row.copy_from_slice(c);
            row
        })
        .collect())
}

/// Скачивает модель kokoro-ru (~311 МБ fp32) и стайл-пак голоса
/// в собственную папку models/kokoro-ru/.
fn ensure_kokoro_model(voice: &str) -> Result<PathBuf> {
    let dir = paths::models_dir().join("kokoro-ru");
    let model = dir.join(
        model_file_for(voice)
            .trim_start_matches("onnx/")
            .trim_end_matches(".onnx")
            .to_string()
            + ".onnx",
    );
    let voicepack = dir.join(format!("voices/{voice}.bin"));

    if model.exists() && voicepack.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(dir.join("voices"))?;

    let client = hf_hub::HFClientSync::new().context("Не удалось создать HuggingFace-клиент")?;
    let repo = client.model("zaakirio".to_string(), "kokoro-ru".to_string());

    let fetch = |filename: &str, dest: &Path| -> Result<()> {
        if dest.exists() {
            return Ok(());
        }
        println!("📥 Загрузка kokoro-ru ({filename})...");
        let downloaded = repo
            .download_file()
            .filename(filename.to_string())
            .send()
            .with_context(|| format!("Не удалось скачать {filename}"))?;
        std::fs::copy(&downloaded, dest)
            .with_context(|| format!("Не удалось сохранить {filename}"))?;
        Ok(())
    };

    fetch(model_file_for(voice), &model)?;
    fetch(&format!("voices/{voice}.bin"), &voicepack)?;

    println!("✅ Модель kokoro-ru готова");
    Ok(dir)
}
