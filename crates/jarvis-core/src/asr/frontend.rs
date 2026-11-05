//! Mel-фронтенд FastConformer: preemph → STFT → power → mel → log →
//! per-feature нормализация.
//!
//! Формулы сняты с эталонного препроцессора onnx-asr (istupakov), с которым
//! проведена побайтовая сверка на тестовых записях (пример `fc_verify`):
//! - preemph 0.97, первый семпл не изменяется;
//! - окно Hann 400 семплов (из весов чекпоинта), нулевое дополнение до 512
//!   по центру (56 с каждой стороны), нулевой паддинг сигнала на 256 семпла
//!   с обеих сторон;
//! - спектр мощности через ДПФ (матричная форма: окно × cos/sin базис —
//!   эквивалент прямого ДПФ, выполняется на GPU одним matmul);
//! - mel-фильтрбанк берётся ИЗ ВЕСОВ чекпоинта (`preprocessor.featurizer.fb`),
//!   поэтому расхождения фильтров исключены;
//! - log(x + 2**-24);
//! - per-feature нормализация: по каждому mel-каналу mean и несмещённая
//!   дисперсия (деление на len-1) по валидным кадрам, y = (x-mean)/(std+1e-5),
//!   кадры за пределами length обнуляются.

use super::config::PreprocessorConfig;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct MelFrontend {
    cfg: PreprocessorConfig,
    /// Окно Hann из весов, дополненное нулями до n_fft: [n_fft].
    window: Vec<f32>,
    /// cos-базис ДПФ: [n_fft/2+1, n_fft] на целевом устройстве.
    cos: Tensor,
    /// sin-базис ДПФ: [n_fft/2+1, n_fft].
    sin: Tensor,
    /// Mel-фильтрбанк из весов: [n_fft/2+1, n_mels].
    fb: Tensor,
}

impl MelFrontend {
    /// window — тензор окна из safetensors [400]; fb — [1, n_mels, n_fft/2+1].
    pub fn load(
        cfg: PreprocessorConfig,
        window: &Tensor,
        fb: &Tensor,
        device: &Device,
    ) -> Result<Self> {
        let n_fft = cfg.n_fft;
        let win_length = cfg.win_length;
        let mut w = vec![0f32; n_fft];
        let win_vec = window.flatten_all()?.to_vec1::<f32>()?;
        let left = (n_fft - win_length) / 2;
        for (i, v) in win_vec.into_iter().take(win_length).enumerate() {
            w[left + i] = v;
        }

        // Базис ДПФ: X[k] = sum_n x[n] * e^{-2πikn/N}; мощность = Xr² + Xi².
        let bins = n_fft / 2 + 1;
        let mut cos = vec![0f32; bins * n_fft];
        let mut sin = vec![0f32; bins * n_fft];
        for k in 0..bins {
            for n in 0..n_fft {
                let angle = -2.0 * std::f64::consts::PI * (k as f64 * n as f64) / n_fft as f64;
                cos[k * n_fft + n] = angle.cos() as f32;
                sin[k * n_fft + n] = angle.sin() as f32;
            }
        }
        let fb = fb.squeeze(0)?.t()?.to_device(device)?; // [1,80,257] -> [257,80]

        Ok(Self {
            cfg,
            window: w,
            // Базис хранится [bin][n]; matmul нужен [n_fft, bins] —
            // транспонируем представление.
            cos: Tensor::from_vec(cos, (bins, n_fft), device)?.t()?,
            sin: Tensor::from_vec(sin, (bins, n_fft), device)?.t()?,
            fb,
        })
    }

    pub fn config(&self) -> &PreprocessorConfig {
        &self.cfg
    }

    /// Распознавание готового буфера 16 кГц mono f32.
    ///
    /// Возвращает (mel [n_mels, frames] на `device`, число валидных кадров).
    /// Кадров всегда floor(N/hop)+1; валидных — floor(N/hop), последний
    /// частичный кадр обнуляется (так же делает эталонный препроцессор).
    pub fn compute(&self, samples: &[f32], device: &Device) -> Result<(Tensor, usize)> {
        let cfg = &self.cfg;
        let n = samples.len();
        let hop = cfg.hop_length;
        let frames = n / hop + 1;
        let length = n / hop;

        // preemph: x[i] -= preemph * x[i-1], первый семпл не трогаем.
        // Паддинг — по n_fft/2 с ОБЕИХ сторон (как np.pad в эталоне), иначе
        // все кадры съезжают на пол-окна.
        let pad = cfg.n_fft / 2;
        let mut padded = vec![0f32; n + 2 * pad];
        if n > 0 {
            padded[pad] = samples[0];
        }
        for i in 1..n {
            padded[pad + i] = samples[i] - cfg.preemph * samples[i - 1];
        }

        // Окно: frame[t] = padded[t*hop .. t*hop + n_fft] * window.
        let mut windowed = vec![0f32; frames * cfg.n_fft];
        for (t, row) in windowed.chunks_mut(cfg.n_fft).enumerate() {
            let base = t * hop;
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = padded[base + j] * self.window[j];
            }
        }
        let x = Tensor::from_vec(windowed, (frames, cfg.n_fft), &Device::Cpu)?.to_device(device)?;

        // Спектр мощности: |X[k]|² = (x·cos)² + (x·sin)², затем mel и log.
        let xr = x.matmul(&self.cos)?; // [frames, bins]
        let xi = x.matmul(&self.sin)?;
        let power = (xr.sqr()? + xi.sqr()?)?; // [frames, bins]
        let mel = power.matmul(&self.fb)?; // [frames, n_mels]
        let log_mel = (mel + cfg.log_guard as f64)?.log()?; // [frames, n_mels]
        let mut values = log_mel.to_dtype(DType::F32)?.to_vec2::<f32>()?; // values[t][b]

        // Кадры за пределами length обнуляются (как в эталоне onnx-asr).
        for frame in values.iter_mut().skip(length) {
            for v in frame.iter_mut() {
                *v = 0.0;
            }
        }
        // per-feature нормализация: статистики по КАЖДОМУ mel-каналу за
        // length валидных кадров (несмещённая дисперсия, деление на len-1).
        // Оси важны: [frames][n_mels], внешний цикл — канал.
        if length > 1 {
            for b in 0..cfg.n_mels {
                let mut sum = 0f64;
                for t in 0..length {
                    sum += values[t][b] as f64;
                }
                let mean = (sum / length as f64) as f32;
                let mut var = 0f64;
                for t in 0..length {
                    let d = values[t][b] as f64 - mean as f64;
                    var += d * d;
                }
                let std = (var / (length - 1) as f64).sqrt() as f32 + 1e-5;
                for t in 0..length {
                    values[t][b] = (values[t][b] - mean) / std;
                }
            }
        } else if length == 1 {
            // Дисперсия не определена: эталон даёт (x-mean)/inf = 0.
            for v in values[0].iter_mut() {
                *v = 0.0;
            }
        }

        // Тензор [n_mels, frames]: данные раскладываются построчно по каналам
        // (flat[b * frames + t]).
        let mut flat = vec![0f32; cfg.n_mels * frames];
        for (t, row) in values.iter().enumerate() {
            for (b, v) in row.iter().enumerate() {
                flat[b * frames + t] = *v;
            }
        }
        // Тензор с batch-осью: [1, n_mels, frames] — вход препро-энкодера.
        let mel = Tensor::from_vec(flat, (cfg.n_mels, frames), device)?.unsqueeze(0)?;
        Ok((mel, length))
    }
}

/// Длина выходных кадров энкодера: floor((len-1)/2)+1 на каждую свёртку
/// stride 2 (последовательность снята с графа ONNX-эталона).
pub fn calc_length(len: usize, subsampling_factor: usize) -> usize {
    let mut l = len;
    let convs = subsampling_factor.trailing_zeros().max(1) as usize;
    for _ in 0..convs {
        if l == 0 {
            return 0;
        }
        l = (l - 1) / 2 + 1;
    }
    l
}
