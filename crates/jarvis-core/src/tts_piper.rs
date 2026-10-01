//! Piper TTS (VITS на ONNX Runtime): быстрый русский синтез на CPU.
//!
//! Неавторегрессивная модель: вся фраза за один forward, RTF < 0.1 на
//! Apple Silicon — на порядки быстрее Qwen3-TTS на Metal, ценой более
//! простой интонации. Модель ~63 МБ, скачивается при первом запуске.
//!
//! Формат инференса повторяет rhasspy/piper: IPA-фенемы от espeak-ng
//! (чистый Rust-порт, крейт `espeak-ng`, данные встроены) -> id из
//! `phoneme_id_map` конфига с BOS `^`, PAD `_`, EOS `$` -> вход
//! `input`/`input_lengths`/`scales`.

use anyhow::{bail, Context, Result};
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Tensor, Value};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Доступные русские голоса piper (rhasspy/piper-voices, качество medium).
pub const PIPER_VOICES: &[(&str, &str)] = &[
    ("ruslan", "мужской"),
    ("denis", "мужской"),
    ("dmitri", "мужской"),
    ("irina", "женский"),
];

/// Частота дискретизации выходного аудио (у всех medium-голосов 22050).
pub const SAMPLE_RATE: u32 = 22050;

const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

/// Максимальная длина фенемной строки на один forward (как в piper:
/// длинные фразы делятся, между кусками вставляется пауза).
const MAX_CHUNK_PHONEMES: usize = 450;

/// Пауза между кусками после деления, сэмплов.
const CHUNK_PAUSE_SAMPLES: usize = (SAMPLE_RATE as usize) / 10;

#[derive(Deserialize)]
struct AudioCfg {
    sample_rate: u32,
}

#[derive(Deserialize)]
struct InferenceCfg {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

#[derive(Deserialize)]
struct PiperConfig {
    audio: AudioCfg,
    // espeak.voice из конфига не используем: фонемизация всегда "ru".
    inference: InferenceCfg,
    num_speakers: u32,
    #[serde(default)]
    speaker_id_map: HashMap<String, i64>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

pub struct PiperEngine {
    /// ort-сессия; run требует &mut, поэтому Mutex (инференс однопоточный).
    session: Mutex<Session>,
    config: PiperConfig,
    /// Фонемизатор espeak-ng (чистый Rust-порт), голос ru.
    espeak: Mutex<espeak_ng::EspeakNg>,
    voice: String,
}

/// Переопределение параметров инференса (None = значения из конфига
/// модели). noise_scale — вариативность/выразительность, length_scale —
/// темп (больше = медленнее), noise_w — шум формы волны.
#[derive(Clone, Copy, Debug, Default)]
pub struct PiperParams {
    pub noise_scale: Option<f32>,
    pub length_scale: Option<f32>,
    pub noise_w: Option<f32>,
}

impl PiperEngine {
    /// Загружает голос (скачивает ~63 МБ при первом запуске).
    pub fn new(voice: &str) -> Result<Self> {
        if !PIPER_VOICES.iter().any(|(v, _)| *v == voice) {
            bail!("Неизвестный голос piper: {voice}");
        }

        let dir = ensure_piper_model(voice)?;
        let model_path = dir.join("model.onnx");
        let config_path = dir.join("model.json");

        let config: PiperConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("Не удалось прочитать {:?}", config_path))?,
        )
        .context("Ошибка парсинга конфига piper")?;

        let started = std::time::Instant::now();
        let session = Session::builder()
            .context("Не удалось создать ONNX Session")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Не удалось загрузить модель {:?}", model_path))?;

        let espeak = Mutex::new(init_espeak()?);

        println!(
            "✅ Piper готов ({voice}, {} Гц) за {:.1?}",
            config.audio.sample_rate,
            started.elapsed()
        );

        Ok(Self {
            session: Mutex::new(session),
            config,
            espeak,
            voice: voice.to_string(),
        })
    }

    pub fn voice(&self) -> &str {
        &self.voice
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.audio.sample_rate
    }

    /// Синтез фразы. Длинный текст делится по фенемам; `speed` (0.5..2)
    /// меняет темп через length_scale.
    pub fn synthesize(&self, text: &str, speed: f32) -> Result<Vec<f32>> {
        self.synthesize_with(text, speed, PiperParams::default())
    }

    /// То же с переопределением параметров инференса.
    pub fn synthesize_with(
        &self,
        text: &str,
        speed: f32,
        params: PiperParams,
    ) -> Result<Vec<f32>> {
        let phonemes = self.phonemize(text)?;
        if phonemes.trim().is_empty() {
            bail!("Фенемы не дали символов: \"{text}\"");
        }
        self.synthesize_phonemes(&phonemes, speed, params)
    }

    /// Синтез из готовой фенемной строки (публично: подбор ритма речи).
    pub fn synthesize_phonemes(
        &self,
        phonemes: &str,
        speed: f32,
        params: PiperParams,
    ) -> Result<Vec<f32>> {
        let mut audio = Vec::new();
        for (i, chunk) in split_phonemes(phonemes).iter().enumerate() {
            if i > 0 {
                audio.extend(std::iter::repeat(0.0f32).take(CHUNK_PAUSE_SAMPLES));
            }
            audio.extend(self.infer_chunk(chunk, speed, params)?);
        }
        Ok(audio)
    }

    /// Текст -> IPA-фенемы espeak-ng (голос ru, со стресс-метками — так
    /// обучены голоса piper). Публично: используется проверкой качества.
    ///
    /// Пробелы и границы клауз espeak удаляются:ruslan medium произносит
    /// межсловные паузы слишком рвано (выбор пользователя — «склеенные»
    /// слова, вариант v4). Знаки препинания остаются и дают паузы.
    pub fn phonemize(&self, text: &str) -> Result<String> {
        let espeak = self.espeak.lock().unwrap_or_else(|e| e.into_inner());
        let ipa = espeak
            .text_to_phonemes(text)
            .map_err(|e| anyhow::anyhow!("Ошибка фонемизации espeak-ng: {e}"))?;
        Ok(ipa
            .chars()
            .filter(|&c| c != ' ' && c != '\n')
            .collect())
    }

    /// Один forward: фенемы -> ids -> waveform.
    fn infer_chunk(&self, phonemes: &str, speed: f32, params: PiperParams) -> Result<Vec<f32>> {
        let ids = phonemes_to_ids(&self.config.phoneme_id_map, phonemes);
        if ids.len() <= 3 {
            bail!("Фенемы не дали токенов: \"{phonemes}\"");
        }

        let inf = &self.config.inference;
        // Темп: length_scale умножает длительности, поэтому speed -> 1/speed.
        let length_scale = (1.0 / speed.clamp(0.5, 2.0))
            * params.length_scale.unwrap_or(inf.length_scale);
        let scales = vec![
            params.noise_scale.unwrap_or(inf.noise_scale),
            length_scale,
            params.noise_w.unwrap_or(inf.noise_w),
        ];

        let input_t = Tensor::from_array(([1usize, ids.len()], ids.clone()))
            .context("input")?;
        let lengths_t = Tensor::from_array(([1usize], vec![ids.len() as i64]))
            .context("input_lengths")?;
        let scales_t =
            Tensor::from_array(([3usize], scales)).context("scales")?;

        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = vec![
            (
                Cow::Borrowed("input"),
                SessionInputValue::Owned(Value::from(input_t)),
            ),
            (
                Cow::Borrowed("input_lengths"),
                SessionInputValue::Owned(Value::from(lengths_t)),
            ),
            (
                Cow::Borrowed("scales"),
                SessionInputValue::Owned(Value::from(scales_t)),
            ),
        ];

        if self.config.num_speakers > 1 {
            let sid = *self
                .config
                .speaker_id_map
                .get(&self.voice)
                .unwrap_or(&0);
            let sid_t = Tensor::from_array(([1usize], vec![sid as i64])).context("sid")?;
            inputs.push((
                Cow::Borrowed("sid"),
                SessionInputValue::Owned(Value::from(sid_t)),
            ));
        }

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session
            .run(SessionInputs::from(inputs))
            .context("Ошибка инференса piper")?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Нет выхода аудио: {e}"))?;
        let total: usize = shape.iter().map(|&d| d as usize).product();
        let mut audio = data.to_vec();
        audio.truncate(total);
        Ok(audio)
    }
}

/// Инициализирует фонемизатор: распаковывает встроенные данные espeak-ng
/// (фонемы + русский словарь) и создаёт движок с голосом ru.
fn init_espeak() -> Result<espeak_ng::EspeakNg> {
    let data_dir = crate::paths::tts_models_dir().join("espeak-ng-data");
    if !data_dir.join("phondata").exists() {
        std::fs::create_dir_all(&data_dir)?;
        espeak_ng::install_bundled_language(&data_dir, "ru").with_context(|| {
            format!("Не удалось распаковать данные espeak-ng в {:?}", data_dir)
        })?;
    }
    espeak_ng::EspeakNg::with_data_dir("ru", &data_dir)
        .map_err(|e| anyhow::anyhow!("Не удалось инициализировать espeak-ng (ru): {e}"))
}

/// BOS + (id фенемы + PAD) для каждого символа + EOS; неизвестные
/// символы пропускаются (как в piper).
fn phonemes_to_ids(map: &HashMap<char, Vec<i64>>, phonemes: &str) -> Vec<i64> {
    let first = |c: char| map.get(&c).and_then(|v| v.first()).copied();
    let pad = first(PAD).unwrap_or(0);

    let mut ids = Vec::with_capacity(phonemes.len() * 2 + 2);
    ids.push(first(BOS).unwrap_or(0));
    for ch in phonemes.chars() {
        if let Some(id) = first(ch) {
            ids.push(id);
            ids.push(pad);
        }
    }
    ids.push(first(EOS).unwrap_or(2));
    ids
}

/// Делит фенемную строку на куски <= MAX_CHUNK_PHONEMES. Пробелов после
/// склейки слов нет, поэтому режем по знакам паузы (запятая, точка и т.д.),
/// в крайнем случае — жёстко.
fn split_phonemes(phonemes: &str) -> Vec<String> {
    let chars: Vec<char> = phonemes.chars().collect();
    if chars.len() <= MAX_CHUNK_PHONEMES {
        return vec![phonemes.to_string()];
    }

    const LOOKBACK: usize = 80;
    let pause = |c: char| matches!(c, ',' | '.' | '!' | '?' | ':' | ';');

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let limit = (start + MAX_CHUNK_PHONEMES).min(chars.len());
        let cut = if limit == chars.len() {
            limit
        } else {
            (limit.saturating_sub(LOOKBACK)..limit)
                .rev()
                .find(|&i| pause(chars[i]))
                .map(|i| i + 1)
                .unwrap_or(limit)
        };
        chunks.push(chars[start..cut].iter().collect());
        start = cut;
    }
    chunks
}

/// Скачивает голос piper (~63 МБ) в models/piper-<голос>/ при отсутствии.
fn ensure_piper_model(voice: &str) -> Result<PathBuf> {
    let dir = crate::paths::tts_models_dir().join(format!("piper-{voice}"));
    let model = dir.join("model.onnx");
    let config = dir.join("model.json");

    if model.exists() && config.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)?;

    let client = hf_hub::HFClientSync::new().context("Не удалось создать HuggingFace-клиент")?;
    let repo = client.model("rhasspy".to_string(), "piper-voices".to_string());
    let base = format!("ru/ru_RU/{voice}/medium/ru_RU-{voice}-medium");

    let fetch = |filename: &str, dest: &Path| -> Result<()> {
        if dest.exists() {
            return Ok(());
        }
        println!("📥 Загрузка piper-голоса ({filename})...");
        let downloaded = repo
            .download_file()
            .filename(filename.to_string())
            .send()
            .with_context(|| format!("Не удалось скачать {filename}"))?;
        std::fs::copy(&downloaded, dest)
            .with_context(|| format!("Не удалось сохранить {filename}"))?;
        Ok(())
    };

    fetch(&format!("{base}.onnx"), &model)?;
    fetch(&format!("{base}.onnx.json"), &config)?;

    println!("✅ Модель piper готова");
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> HashMap<char, Vec<i64>> {
        HashMap::from([
            ('^', vec![1]),
            ('$', vec![2]),
            ('_', vec![0]),
            ('a', vec![3]),
            ('b', vec![4]),
        ])
    }

    #[test]
    fn ids_bos_pad_eos() {
        let ids = phonemes_to_ids(&map(), "aba");
        assert_eq!(ids, vec![1, 3, 0, 4, 0, 3, 0, 2]);
    }

    #[test]
    fn unknown_chars_are_skipped() {
        let ids = phonemes_to_ids(&map(), "a?,b");
        assert_eq!(ids, vec![1, 3, 0, 4, 0, 2]);
    }

    #[test]
    fn long_phonemes_are_split() {
        let long = "ab ".repeat(400);
        let chunks = split_phonemes(&long);
        assert!(chunks.len() > 1);
        assert!(chunks
            .iter()
            .all(|c| c.chars().count() <= MAX_CHUNK_PHONEMES));
    }
}
