//! Движок kokoro-ru (Kokoro-82M / StyleTTS2-lite) поверх ONNX Runtime.
//!
//! Модель неавторегрессивная: вся фраза синтезируется за один
//! forward — отсюда RTF ~0.1 на CPU. Вход: токены фенем [1, T] + стайл-вектор
//! [1, 256] + скорость; выход: waveform 24 кГц + длительности фенем.

use crate::g2p;
use crate::vocab::tokenize;
use anyhow::{bail, Context, Result};
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Tensor, Value};
use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;

/// Частота дискретизации выходного аудио.
pub const SAMPLE_RATE: u32 = 24000;

/// Максимальный размер чанка в фенемах: стайл-пак имеет 510 строк,
/// индекс = длина фенемной строки - 1.
const MAX_CHUNK_PHONEMES: usize = 480;

const INPUT_IDS: &str = "input_ids";
const STYLE: &str = "style";
const SPEED: &str = "speed";
const WAVEFORM: &str = "waveform";

pub struct KokoroEngine {
    /// ort-сессия; run требует &mut, поэтому Mutex (инференс однопоточный).
    session: Mutex<Session>,
    /// Строки стайл-пака: [510][256] — выбирается по длине фенемной строки.
    style_rows: Vec<[f32; 256]>,
}

impl KokoroEngine {
    pub fn new(model_path: &Path, voice_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("Не удалось создать ONNX Session")?
            .commit_from_file(model_path)
            .with_context(|| format!("Не удалось загрузить модель {:?}", model_path))?;

        let style_rows = load_style_pack(voice_path)
            .with_context(|| format!("Не удалось загрузить стайл-пак {:?}", voice_path))?;

        Ok(Self {
            session: Mutex::new(session),
            style_rows,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Синтез текста: чанкинг по фенемам, пауза ~0.12 с между чанками.
    pub fn synthesize(&self, text: &str, speed: f32) -> Result<Vec<f32>> {
        let mut audio = Vec::new();
        for phonemes in chunk_phonemes(text) {
            if audio.len() > SAMPLE_RATE as usize / 8 {
                audio.extend(std::iter::repeat(0.0f32).take(SAMPLE_RATE as usize / 8));
            }
            audio.extend(self.synthesize_phonemes(&phonemes, speed)?);
        }
        Ok(audio)
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

        let style_idx = (phonemes.chars().count().saturating_sub(1)).min(self.style_rows.len() - 1);
        let style: Vec<f32> = self.style_rows[style_idx].to_vec();

        let tokens_tensor = Tensor::from_array(([1usize, input_ids.len()], input_ids))
            .context("input_ids")?;
        let style_tensor = Tensor::from_array(([1usize, style.len()], style)).context("style")?;
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

/// Текст -> список фенемных строк по <= MAX_CHUNK_PHONEMES символов.
/// Делит по предложениям/словам, фонемизирует каждый кусок отдельно.
fn chunk_phonemes(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    let sentences: Vec<&str> = text
        .split(['.', '!', '?', ';', '\n'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    for sentence in sentences {
        let sentence = format!("{}.", sentence.trim());
        let phonemes = g2p::phonemize(&sentence);
        if phonemes.chars().count() > MAX_CHUNK_PHONEMES {
            // Слишком длинное предложение — добиваем по словам.
            for word in sentence.split_whitespace() {
                let word_phonemes = g2p::phonemize(word);
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
        } else if current.chars().count() + phonemes.chars().count() + 1 > MAX_CHUNK_PHONEMES {
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
        .chunks(256)
        .map(|r| {
            let mut row = [0.0f32; 256];
            row.copy_from_slice(r);
            row
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_respects_limit() {
        let text = "Привет! ".repeat(50);
        let chunks = chunk_phonemes(&text);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= MAX_CHUNK_PHONEMES);
        }
    }
}
