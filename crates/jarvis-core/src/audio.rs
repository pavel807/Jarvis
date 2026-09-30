//! Аудио: захват с микрофона, даунмикс, ресемплинг, воспроизведение WAV.

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavReader;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::sync::mpsc::SyncSender;

pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

pub fn downmix_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|c| c.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

pub fn downmix_i16(data: &[i16], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|c| c.iter().map(|&x| x as f32 / 32768.0).sum::<f32>() / channels as f32)
        .collect()
}

pub fn downmix_u16(data: &[u16], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|c| {
            c.iter()
                .map(|&x| (x as f32 - 32768.0) / 32768.0)
                .sum::<f32>()
                / channels as f32
        })
        .collect()
}

/// Линейная интерполяция без антиалиасинга — для речи достаточно.
pub fn resample_audio(samples: &[f32], src_rate: usize, target_rate: usize) -> Vec<f32> {
    if src_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = target_rate as f32 / src_rate as f32;
    let target_length = (samples.len() as f32 * ratio) as usize;
    let last_idx = samples.len() - 1;

    let mut resampled = Vec::with_capacity(target_length);

    for i in 0..target_length {
        let src_idx_float = i as f32 / ratio;
        let idx_low = src_idx_float.floor() as usize;
        let idx_high = (idx_low + 1).min(last_idx);
        let weight = src_idx_float - idx_low as f32;

        let sample = samples[idx_low] * (1.0 - weight) + samples[idx_high] * weight;
        resampled.push(sample);
    }

    resampled
}

/// Открывает микрофон по умолчанию и шлёт моно-f32 чанки в канал.
pub fn spawn_stream(tx: SyncSender<Vec<f32>>) -> Result<(cpal::Stream, usize)> {
    println!("🔍 Поиск аудиоустройств...");

    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .context("❌ Не найдено устройство ввода по умолчанию.")?;

    let sup = dev
        .default_input_config()
        .context("❌ Не удалось получить конфигурацию микрофона.")?;

    let ch = sup.channels() as usize;
    let sr = sup.sample_rate() as usize;

    println!(
        "🎙️ Параметры микрофона: {}ch {}Hz {:?}",
        ch,
        sr,
        sup.sample_format()
    );

    let mut cfg: cpal::StreamConfig = sup.clone().into();
    cfg.buffer_size = cpal::BufferSize::Fixed(4096);

    let s = match sup.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            cfg,
            move |d: &[f32], _| {
                let _ = tx.send(downmix_f32(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,

        cpal::SampleFormat::I16 => dev.build_input_stream(
            cfg,
            move |d: &[i16], _| {
                let _ = tx.send(downmix_i16(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,

        cpal::SampleFormat::U16 => dev.build_input_stream(
            cfg,
            move |d: &[u16], _| {
                let _ = tx.send(downmix_u16(d, ch));
            },
            |e| eprintln!("❌ Ошибка входного потока: {e}"),
            None,
        )?,

        f => bail!("Неподдерживаемый формат микрофона: {f:?}"),
    };

    println!("✅ Аудиопоток микрофона создан");
    Ok((s, sr))
}

/// Блокирующее воспроизведение WAV-файла с проверкой флага остановки.
///
/// Используется разовыми сценариями (тесты, UI-проверки); плеер движка
/// работает через [`OutputPlayer`] — один постоянный стрим.
pub fn play_wav_blocking(path: &str, stop_flag: &AtomicBool) -> Result<()> {
    let player = OutputPlayer::new()?;
    player.play_wav(path, stop_flag)
}

/// Один постоянный выходной поток: WAV декодируется в буфер, колбэк устройства
/// непрерывно его вычитывает. Никаких лагов на пересоздание стрима между фразами.
///
/// Поддерживает потоковую подачу: [`OutputPlayer::enqueue`] дописывает сэмплы
/// в очередь по мере готовности, [`OutputPlayer::wait_drain`] дожидается конца.
pub struct OutputPlayer {
    queue: Arc<Mutex<Vec<f32>>>,
    consumed: Arc<std::sync::atomic::AtomicUsize>,
    /// Сколько interleaved-сэмплов выдвинуто в очередь всего (монотонно).
    pushed: std::sync::atomic::AtomicUsize,
    /// Держим стрим живым: колбэк работает, пока существует структура.
    _stream: cpal::Stream,
    sample_rate: usize,
    channels: usize,
}

impl OutputPlayer {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("Нет устройства вывода звука")?;
        let default_config = device.default_output_config()?;

        let sample_rate = default_config.sample_rate() as usize;
        let channels = default_config.channels() as usize;

        let queue: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let consumed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let cb_queue = Arc::clone(&queue);
        let cb_consumed = Arc::clone(&consumed);

        let config = cpal::StreamConfig {
            channels: default_config.channels(),
            sample_rate: default_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device.build_output_stream(
            config,
            move |data: &mut [f32], _| {
                let mut q = cb_queue.lock().unwrap_or_else(|e| e.into_inner());
                let take = data.len().min(q.len());
                let head: Vec<f32> = q.drain(..take).collect();
                drop(q);
                cb_consumed.fetch_add(take, Ordering::Relaxed);
                data[..take].copy_from_slice(&head);
                for c in data[take..].iter_mut() {
                    *c = 0.0;
                }
            },
            move |err| eprintln!("Output stream error: {err}"),
            None,
        )?;

        stream.play()?;

        Ok(Self {
            queue,
            consumed,
            pushed: std::sync::atomic::AtomicUsize::new(0),
            _stream: stream,
            sample_rate,
            channels,
        })
    }

    /// Частота выходного устройства (сэмплы для [`OutputPlayer::enqueue`]).
    pub fn device_sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Дописывает моно-сэмплы (уже в частоте устройства) в очередь вывода.
    /// Не блокируется: воспроизведение идёт параллельно через колбэк.
    pub fn enqueue(&self, mono: &[f32]) {
        if mono.is_empty() {
            return;
        }
        let out_channels = self.channels;
        let interleaved: Vec<f32> = if out_channels == 1 {
            mono.to_vec()
        } else {
            mono.iter()
                .flat_map(|&s| std::iter::repeat_n(s, out_channels))
                .collect()
        };
        let n = interleaved.len();
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(&interleaved);
        drop(interleaved);
        self.pushed.fetch_add(n, Ordering::SeqCst);
    }

    /// Сбрасывает очередь вывода (используется при остановке речи).
    pub fn clear(&self) {
        let pushed = self.pushed.load(Ordering::SeqCst);
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        // Счётчики должны сойтись, иначе wait_drain решит, что всё уже сыграно.
        self.consumed.store(pushed, Ordering::SeqCst);
    }

    /// Ждёт, пока устройство вычитает всё выдвинутое в очередь, либо не придёт стоп.
    pub fn wait_drain(&self, stop_flag: &AtomicBool) {
        loop {
            if stop_flag.load(Ordering::SeqCst) {
                self.clear();
                return;
            }
            if self.consumed.load(Ordering::Relaxed) >= self.pushed.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Декодирует WAV, даунмиксит, ресемплирует под выходное устройство
    /// и проигрывает целиком.
    pub fn play_wav(&self, path: &str, stop_flag: &AtomicBool) -> Result<()> {
        if !Path::new(path).exists() {
            bail!("Файл звука не найден: {path}");
        }

        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        };

        let mono = downmix_f32(&raw, spec.channels as usize);
        let final_samples = if spec.sample_rate as usize != self.sample_rate {
            resample_audio(&mono, spec.sample_rate as usize, self.sample_rate)
        } else {
            mono
        };

        self.enqueue(&final_samples);
        self.wait_drain(stop_flag);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_length_and_dc() {
        let src = vec![0.5f32; 1600]; // 0.1s @ 16k
        let out = resample_audio(&src, 16_000, 48_000);
        assert_eq!(out.len(), 4800);
        assert!(out.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn resample_same_rate_is_clone() {
        let src = vec![0.1f32, 0.2, 0.3];
        let out = resample_audio(&src, 16_000, 16_000);
        assert_eq!(out, src);
    }

    #[test]
    fn downmix_stereo_averages() {
        let out = downmix_f32(&[0.0, 1.0, 0.5, 0.5], 2);
        assert_eq!(out, vec![0.5, 0.5]);
    }

    #[test]
    fn downmix_i16_normalizes() {
        let out = downmix_i16(&[i16::MAX], 1);
        assert!((out[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rms_zero_and_nonzero() {
        assert_eq!(calculate_rms(&[]), 0.0);
        assert_eq!(calculate_rms(&[0.0; 100]), 0.0);
        assert!(calculate_rms(&[0.5; 100]) > 0.49);
    }
}
