//! Адаптер Silero TTS v5 -> Seed-VC.
//!
//! Схема работы: Silero синтезирует речь (48 кГц) -> модуль превращает её
//! в голос целевого профиля -> результат уходит в аудиопоток как обычно.
//!
//! Важные свойства:
//! * реализация Silero **не меняется** — конвертация стоит отдельным шагом
//!   после синтеза;
//! * при любой ошибке (нет модели, сбой загрузки, пустой результат)
//!   возвращается исходный сигнал — голосовой ассистент не должен замолчать;
//! * настройки (включение, профиль, сдвиг F0, интенсивность) живут в
//!   `settings.toml` и применяются без перезапуска движка.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};

/// Настройки конвертации.
#[derive(Clone, Debug)]
pub struct VoiceConversionSettings {
    /// Включено ли преобразование голоса.
    pub enabled: bool,
    /// Каталог с экспортированной моделью (`model.safetensors` + `config.json`).
    pub model_dir: Option<PathBuf>,
    /// Референсный фрагмент целевого голоса (wav).
    pub reference_path: Option<PathBuf>,
    /// Сдвиг тона в полутонах: +1 — голос выше.
    pub f0_shift: f32,
    /// Сила преобразования: 0 — исходный голос, 1 — полное превращение.
    pub intensity: f32,
    /// Шагов интегрирования flow matching (больше — точнее и медленнее).
    pub steps: usize,
}

impl Default for VoiceConversionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: None,
            reference_path: None,
            f0_shift: 0.0,
            intensity: 1.0,
            steps: 16,
        }
    }
}

/// Загруженный конвертер вместе с референсом целевого голоса.
pub struct Converter {
    inner: jarvis_vc::VoiceConverter,
    reference: Vec<f32>,
    reference_sr: u32,
    sample_rate: u32,
}

impl Converter {
    /// Загружает модель и референсный фрагмент.
    pub fn load(settings: &VoiceConversionSettings) -> Result<Self> {
        let model_dir = settings
            .model_dir
            .clone()
            .unwrap_or_else(default_model_dir);
        let device = device();
        let inner = jarvis_vc::VoiceConverter::load(&model_dir, &device)
            .with_context(|| format!("Не удалось загрузить модель конвертации из {model_dir:?}"))?;

        let reference_path = settings
            .reference_path
            .clone()
            .unwrap_or_else(default_reference_path);
        let (reference, reference_sr) = read_wav_mono(&reference_path)
            .with_context(|| format!("Не удалось прочитать референс {reference_path:?}"))?;

        Ok(Self {
            inner,
            reference,
            reference_sr,
            sample_rate: 48_000,
        })
    }

    /// Преобразует речь Silero. Возвращает PCM на частоте исходника.
    pub fn convert(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let opts = jarvis_vc::ConvertOptions {
            f0_shift: self.f0_shift,
            intensity: self.intensity,
            steps: self.steps,
        };
        self.inner
            .convert(
                samples,
                sample_rate,
                &self.reference,
                self.reference_sr,
                &opts,
            )
            .context("Ошибка конвертации голоса")
    }
}

// ---------------------------------------------------------------------------
// Глобальное состояние: один загруженный конвертер на процесс.
// ---------------------------------------------------------------------------

fn default_settings() -> VoiceConversionSettings {
    VoiceConversionSettings::default()
}

static SETTINGS: RwLock<Option<VoiceConversionSettings>> = RwLock::new(None);
static CONVERTER: RwLock<Option<Arc<Converter>>> = RwLock::new(None);

/// Текущие настройки (копия).
pub fn settings() -> VoiceConversionSettings {
    SETTINGS
        .read()
        .ok()
        .and_then(|s| s.clone())
        .unwrap_or_else(default_settings)
}

/// Каталог модели по умолчанию: `models/voice-convert/`.
pub fn default_model_dir() -> PathBuf {
    crate::paths::models_dir().join("voice-convert")
}

/// Референс целевого голоса по умолчанию: `models/voice-convert/reference.wav`.
pub fn default_reference_path() -> PathBuf {
    crate::paths::models_dir()
        .join("voice-convert")
        .join("reference.wav")
}

/// Применяет настройки: перезагружает модель, если её не было или путь изменился.
pub fn apply_settings(settings: VoiceConversionSettings) {
    let needs_reload = {
        let current = SETTINGS.read().ok().and_then(|s| s.clone());
        match (&current, &settings) {
            (None, _) => true,
            (Some(old), new) => {
                old.enabled != new.enabled
                    || old.model_dir != new.model_dir
                    || old.reference_path != new.reference_path
            }
        }
    };

    if let Ok(mut slot) = SETTINGS.write() {
        *slot = Some(settings.clone());
    }
    if !settings.enabled {
        if let Ok(mut slot) = CONVERTER.write() {
            *slot = None;
        }
        return;
    }
    if needs_reload {
        match Converter::load(&settings) {
            Ok(conv) => {
                if let Ok(mut slot) = CONVERTER.write() {
                    *slot = Some(Arc::new(conv));
                }
            }
            Err(err) => {
                eprintln!("⚠️ Конвертация голоса недоступна: {err:#}");
                if let Ok(mut slot) = CONVERTER.write() {
                    *slot = None;
                }
            }
        }
    }
}

/// Преобразует речь, если конвертация включена и модель загружена.
///
/// При любой ошибке возвращается исходный сигнал: голосовой ассистент
/// не должен замолчать из-за проблем с моделью.
pub fn convert_speech(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return samples.to_vec();
    }
    let guard = match CONVERTER.read() {
        Ok(g) => g,
        Err(_) => return samples.to_vec(),
    };
    let Some(conv) = guard.as_ref() else {
        return samples.to_vec();
    };
    match conv.convert(samples, sample_rate) {
        Ok(out) if !out.is_empty() => out,
        Ok(_) => samples.to_vec(),
        Err(err) => {
            eprintln!("⚠️ Конвертация не удалась, озвучиваю исходный голос: {err:#}");
            samples.to_vec()
        }
    }
}

/// Готов ли конвертер (модель загружена и включён).
pub fn is_ready() -> bool {
    CONVERTER
        .read()
        .map(|g| g.is_some())
        .unwrap_or(false)
}

/// Склеивает два фрагмента с перекрёстным затуханием.
///
/// Используется при потоковой озвучке: конец предыдущей фразы плавно
/// переходит в начало следующей, без щелчка на стыке.
pub fn crossfade(prev: &[f32], next: &[f32], sample_rate: u32, fade_ms: u32) -> Vec<f32> {
    if prev.is_empty() {
        return next.to_vec();
    }
    if next.is_empty() {
        return prev.to_vec();
    }
    let fade = ((sample_rate as u64 * fade_ms as u64) / 1000).max(1) as usize;
    let fade = fade.min(prev.len()).min(next.len());

    let mut out = Vec::with_capacity(prev.len() + next.len() - fade);
    out.extend_from_slice(&prev[..prev.len() - fade]);
    for i in 0..fade {
        let t = i as f32 / fade as f32;
        let a = prev[prev.len() - fade + i];
        let b = next[i];
        out.push(a * (1.0 - t) + b * t);
    }
    out.extend_from_slice(&next[fade..]);
    out
}

// ---------------------------------------------------------------------------
// Вспомогательное
// ---------------------------------------------------------------------------

/// Устройство вычислений: Metal на macOS (если доступен), иначе CPU.
fn device() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    {
        if candle_core::Device::MetalCpuIfAvailable(0).is_ok() {
            return candle_core::Device::MetalCpuIfAvailable(0).unwrap();
        }
    }
    candle_core::Device::Cpu
}

/// Читает моно-WAV любой разрядности -> (samples, sample_rate).
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Не удалось открыть {path:?}"))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let raw: Vec<i32> = reader.samples::<i32>().collect::<Result<Vec<_>, _>>()?;
            let scale = match bits {
                8 => 1.0 / 128.0,
                16 => 1.0 / 32768.0,
                24 => 1.0 / 8_388_608.0,
                _ => 1.0 / 2_147_483_648.0,
            };
            raw.into_iter().map(|v| v as f32 * scale).collect()
        }
    };
    if channels > 1 {
        let mono: Vec<f32> = samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect();
        Ok((mono, spec.sample_rate))
    } else {
        Ok((samples, spec.sample_rate))
    }
}
