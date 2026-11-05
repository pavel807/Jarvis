//! Конфигурация FastConformer — читается из `config.json` (полный копия
//! `model_config.yaml` из чекпоинта, см. `tools/convert_fastconformer.py`).
//!
//! Приоритет всегда у конфигурации конкретного checkpoint; значения,
//! которых в нём нет (preemph, log guard), берутся из эталонного
//! препроцессора onnx-asr (istupakov), с которым проведена численная
//! верификация.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    /// Число mel-полос (features).
    pub n_mels: usize,
    pub n_fft: usize,
    /// Длина окна в семплах (window_size * sample_rate = 0.025 * 16000 = 400).
    pub win_length: usize,
    /// Шаг окна в семплах (window_stride * sample_rate = 0.01 * 16000 = 160).
    pub hop_length: usize,
    /// Предыскажение сигнала.
    pub preemph: f32,
    /// Слагаемое внутри log(x + guard) — защита от log(0).
    pub log_guard: f32,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 80,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            preemph: 0.97,
            // 2**-24: значение из эталонного препроцессора onnx-asr.
            log_guard: 2f32.powi(-24),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub feat_in: usize,
    /// Коэффициент сжатия pre-encode (8× = три свёртки stride 2).
    pub subsampling_factor: usize,
    /// Расширение FFN (ff_expansion_factor * d_model = 2048).
    pub ff_dim: usize,
    /// Ядро свёрточного модуля (9).
    pub conv_kernel_size: usize,
    /// Максимальная длина относительной позиционной таблицы (2*5000-1 = 9999).
    pub pos_emb_max_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    /// Размер словаря SentencePiece (1024) без blank.
    pub vocab_size: usize,
    /// Скрытый размер prediction network (640).
    pub pred_hidden: usize,
    /// Число рекуррентных слоёв prediction network.
    pub pred_rnn_layers: usize,
    /// Скрытый размер joint network (640).
    pub joint_hidden: usize,
    /// Максимальное число токенов на один шаг времени RNNT.
    pub max_symbols_per_step: usize,
}

/// Полная конфигурация модели. Blank-токен всегда последний: id = vocab_size.
#[derive(Debug, Clone, Copy)]
pub struct FastConformerConfig {
    pub preprocessor: PreprocessorConfig,
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub blank_id: usize,
}

impl Default for FastConformerConfig {
    fn default() -> Self {
        Self {
            preprocessor: PreprocessorConfig::default(),
            encoder: EncoderConfig {
                d_model: 512,
                n_layers: 17,
                n_heads: 8,
                feat_in: 80,
                subsampling_factor: 8,
                ff_dim: 2048,
                conv_kernel_size: 9,
                pos_emb_max_len: 5000,
            },
            decoder: DecoderConfig {
                vocab_size: 1024,
                pred_hidden: 640,
                pred_rnn_layers: 1,
                joint_hidden: 640,
                max_symbols_per_step: 10,
            },
            blank_id: 1024,
        }
    }
}

// Поля config.json, которые реально читаем; остальное игнорируется.
#[derive(Debug, Deserialize)]
struct RawConfig {
    preprocessor: Option<RawPreprocessor>,
    encoder: Option<RawEncoder>,
    #[serde(rename = "decoder")]
    rnnt_decoder: Option<RawRnntDecoder>,
}

#[derive(Debug, Deserialize)]
struct RawPreprocessor {
    sample_rate: Option<usize>,
    features: Option<usize>,
    n_fft: Option<usize>,
    window_size: Option<f64>,
    window_stride: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawEncoder {
    d_model: Option<usize>,
    n_layers: Option<usize>,
    n_heads: Option<usize>,
    feat_in: Option<usize>,
    subsampling_factor: Option<usize>,
    ff_expansion_factor: Option<usize>,
    conv_kernel_size: Option<usize>,
    pos_emb_max_len: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawRnntDecoder {
    #[allow(dead_code)]
    #[serde(rename = "_target_")]
    target: Option<String>,
    vocab_size: Option<usize>,
    prednet: Option<RawPrednet>,
    #[serde(rename = "jointnet")]
    jointnet: Option<RawJointnet>,
}

#[derive(Debug, Deserialize)]
struct RawPrednet {
    pred_hidden: Option<usize>,
    pred_rnn_layers: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawJointnet {
    joint_hidden: Option<usize>,
}

/// Загружает конфигурацию из `config.json`. Отсутствующие поля — дефолты
/// (зафиксированы для stt_ru_fastconformer_hybrid_large_pc).
pub fn load(path: &Path) -> Result<FastConformerConfig> {
    let raw: RawConfig = serde_json::from_str(
        &std::fs::read_to_string(path)
            .with_context(|| format!("Не удалось прочитать {path:?}"))?,
    )
    .with_context(|| format!("Не удалось разобрать {path:?}"))?;

    let mut cfg = FastConformerConfig::default();
    if let Some(p) = &raw.preprocessor {
        let prep = &mut cfg.preprocessor;
        if let Some(v) = p.sample_rate {
            prep.sample_rate = v;
        }
        if let Some(v) = p.features {
            prep.n_mels = v;
        }
        if let Some(v) = p.n_fft {
            prep.n_fft = v;
        }
        if let Some(v) = p.window_size {
            prep.win_length = (v * prep.sample_rate as f64).round() as usize;
        }
        if let Some(v) = p.window_stride {
            prep.hop_length = (v * prep.sample_rate as f64).round() as usize;
        }
    }
    if let Some(e) = &raw.encoder {
        let enc = &mut cfg.encoder;
        if let Some(v) = e.d_model {
            enc.d_model = v;
        }
        if let Some(v) = e.n_layers {
            enc.n_layers = v;
        }
        if let Some(v) = e.n_heads {
            enc.n_heads = v;
        }
        if let Some(v) = e.feat_in {
            enc.feat_in = v;
        }
        if let Some(v) = e.subsampling_factor {
            enc.subsampling_factor = v;
        }
        if let Some(v) = e.ff_expansion_factor {
            enc.ff_dim = v * enc.d_model;
        }
        if let Some(v) = e.conv_kernel_size {
            enc.conv_kernel_size = v;
        }
        if let Some(v) = e.pos_emb_max_len {
            enc.pos_emb_max_len = v;
        }
    }
    if let Some(d) = &raw.rnnt_decoder {
        let dec = &mut cfg.decoder;
        if let Some(v) = d.vocab_size {
            dec.vocab_size = v;
        }
        if let Some(pn) = &d.prednet {
            if let Some(v) = pn.pred_hidden {
                dec.pred_hidden = v;
            }
            if let Some(v) = pn.pred_rnn_layers {
                dec.pred_rnn_layers = v;
            }
        }
        if let Some(jn) = &d.jointnet {
            if let Some(v) = jn.joint_hidden {
                dec.joint_hidden = v;
            }
        }
    }
    cfg.blank_id = cfg.decoder.vocab_size;
    Ok(cfg)
}