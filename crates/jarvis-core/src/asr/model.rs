//! Сборка FastConformer целиком: загрузка весов/конфига/токенизатора,
//! выбор декодера (RNNT greedy / CTC greedy / CTC beam search) и точки
//! входа распознавания.

use super::config::{self, FastConformerConfig};
use super::{conformer, ctc, frontend, rnnt, tokenizer};
use anyhow::{Context, Result};
use candle_core::{D, DType, Device};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Выбор декодера.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoder {
    /// RNN-T жадный (по умолчанию: точнее на русской речи).
    Rnnt,
    /// CTC жадный.
    CtcGreedy,
    /// CTC beam search (width задаётся отдельно).
    CtcBeam { width: usize },
}

impl Default for Decoder {
    fn default() -> Self {
        Self::Rnnt
    }
}

/// Результат распознавания: текст и тайминги этапов.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub ids: Vec<usize>,
    pub mel_ms: f64,
    pub encoder_ms: f64,
    pub decode_ms: f64,
}

pub struct FastConformer {
    pub device: Device,
    pub frontend: frontend::MelFrontend,
    pub encoder: conformer::Encoder,
    pub ctc_head: ctc::CtcHead,
    pub predictor: rnnt::Predictor,
    pub joint: rnnt::Joint,
    pub tokenizer: tokenizer::SpTokenizer,
    pub cfg: FastConformerConfig,
    /// Каталог модели (для словаря и отладки).
    pub dir: PathBuf,
}

impl FastConformer {
    /// Загружает модель из `models_dir/fastconformer`.
    ///
    /// Ожидаемые файлы (готовит `tools/convert_fastconformer.py` +
    /// `model_fetch::ensure_fastconformer`):
    /// `model.safetensors`, `config.json`, `tokenizer.vocab`.
    pub fn load(models_dir: &Path) -> Result<Self> {
        // Автозагрузка ассетов (веса ~460 МБ из релиза jarvis-models).
        if let Err(e) = crate::model_fetch::ensure_fastconformer() {
            eprintln!("⚠️ Не удалось скачать FastConformer: {e:#}");
        }
        let dir = models_dir.join("fastconformer");
        let safetensors = dir.join("model.safetensors");
        anyhow::ensure!(
            safetensors.exists(),
            "Файл весов не найден: {:?} — модель ещё не скачана",
            safetensors
        );

        let cfg = config::load(&dir.join("config.json"))?;
        let device = pick_device();

        let start = Instant::now();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors], DType::F32, &device)
                .with_context(|| format!("Не удалось открыть {safetensors:?}"))?
        };

        // Фронтенд: окно и mel-фильтрбанк приходят из весов чекпоинта —
        // расхождения с эталонным препроцессором исключены.
        let window = vb.get(
            (cfg.preprocessor.win_length,),
            "preprocessor.featurizer.window",
        )?;
        let fb = vb.get(
            (1, cfg.preprocessor.n_mels, cfg.preprocessor.n_fft / 2 + 1),
            "preprocessor.featurizer.fb",
        )?;
        let frontend = frontend::MelFrontend::load(cfg.preprocessor, &window, &fb, &device)?;

        let encoder = conformer::Encoder::load(&vb, &cfg, &device)?;
        let ctc_head =
            ctc::CtcHead::load(vb.pp("ctc_decoder").pp("decoder_layers").pp("0"), cfg.encoder.d_model, cfg.decoder.vocab_size + 1)?;
        let predictor = rnnt::Predictor::load(
            vb.pp("decoder").pp("prediction"),
            cfg.decoder.vocab_size + 1,
            cfg.decoder.pred_hidden,
        )?;
        let joint = rnnt::Joint::load(
            vb.pp("joint"),
            cfg.encoder.d_model,
            cfg.decoder.pred_hidden,
            cfg.decoder.joint_hidden,
            cfg.decoder.vocab_size + 1,
        )?;
        let tokenizer =
            tokenizer::SpTokenizer::load(&dir.join("tokenizer.vocab"))?;

        let model = Self {
            device,
            frontend,
            encoder,
            ctc_head,
            predictor,
            joint,
            tokenizer,
            cfg,
            dir,
        };
        println!(
            "✅ FastConformer загружен за {:.2?} ({})",
            start.elapsed(),
            if model.device.is_metal() { "Metal" } else { "CPU" }
        );
        Ok(model)
    }

    /// Полное распознавание буфера 16 кГц mono f32 выбранным декодером.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        self.transcribe_with(samples, Decoder::Rnnt, None)
    }

    pub fn transcribe_with(
        &self,
        samples: &[f32],
        decoder: Decoder,
        dict: Option<&ctc::JarvisDictionary>,
    ) -> Result<Transcript> {
        // 1. Mel-фронтенд (DFT — matmul на устройстве модели).
        let t0 = Instant::now();
        let (mel, length) = self.frontend.compute(samples, &self.device)?;
        let mel_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Энкодер (pre-encode + 17 Conformer-блоков).
        let t1 = Instant::now();
        let (enc, t_valid) = self.encoder.forward(&mel, length)?;
        let encoder_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // 3. Декодер.
        let t2 = Instant::now();
        let ids = match decoder {
            Decoder::Rnnt => rnnt::greedy(
                &enc,
                t_valid,
                &self.predictor,
                &self.joint,
                self.cfg.blank_id,
                self.cfg.decoder.max_symbols_per_step,
            )?,
            Decoder::CtcGreedy => {
                let logits = self.ctc_head.logits(&enc)?; // [1, T, V]
                let (frames, vocab_out) = (logits.dim(1)?, logits.dim(2)?);
                let flat = logits.reshape((frames * vocab_out,))?.to_vec1::<f32>()?;
                ctc::greedy(&flat, frames, vocab_out, self.cfg.blank_id)
            }
            Decoder::CtcBeam { width } => {
                let logits = candle_nn::ops::log_softmax(&self.ctc_head.logits(&enc)?, D::Minus1)?;
                let (frames, vocab_out) = (logits.dim(1)?, logits.dim(2)?);
                let flat = logits.reshape((frames * vocab_out,))?.to_vec1::<f32>()?;
                ctc::beam_search(&flat, frames, vocab_out, self.cfg.blank_id, width, dict)
            }
        };
        let decode_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let text = self.tokenizer.decode(&ids);
        Ok(Transcript {
            text,
            ids,
            mel_ms,
            encoder_ms,
            decode_ms,
        })
    }

    /// Словарь Jarvis для beam search (кодирование фраз токенизатором).
    pub fn jarvis_dictionary(&self, phrases: &[String]) -> Result<ctc::JarvisDictionary> {
        ctc::JarvisDictionary::new(phrases, &self.tokenizer, 0.3, 1.5)
    }

    pub fn is_metal(&self) -> bool {
        self.device.is_metal()
    }
}

/// Устройство ASR. По умолчанию CPU: замер fc_bench (M4 Pro) показал, что
/// энкодер на Metal стоит фиксированные ~720 мс на ЛЮБУЮ фразу (208 кадров —
/// 719 мс, 978 кадров — 729 мс): это чистый overhead командных буферов Metal
/// (~700 мелких операций с commit+wait на каждую), а не вычисления. На CPU
/// тот же энкодер (FP32, без квантизации) — 98 мс: короткая команда
/// распознаётся за ~130 мс вместо ~780. Metal включается переменной
/// окружения JARVIS_ASR_DEVICE=metal (для экспериментов и длинных потоков).
fn pick_device() -> Device {
    if std::env::var("JARVIS_ASR_DEVICE").as_deref() == Ok("metal") {
        #[cfg(target_os = "macos")]
        match Device::new_metal(0) {
            Ok(d) => return d,
            Err(e) => eprintln!("⚠️ Metal недоступен ({e}), ASR будет работать на CPU"),
        }
    }
    Device::Cpu
}