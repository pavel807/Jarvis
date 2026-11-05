//! Qwen3.5-0.8B на чистом candle — без FFI и без candle-transformers.
//!
//! Архитектура (`Qwen3_5ForConditionalGeneration`, текстовая часть
//! `qwen3_5_text`) — гибрид в духе Qwen3-Next: 24 слоя, из них 18
//! linear-attention (Gated DeltaNet) и 6 полных (каждый четвёртый,
//! `full_attention_interval: 4`). Референс математики —
//! `transformers/models/qwen3_5/modeling_qwen3_5.py`.
//!
//! Что здесь реализовано:
//! - **Gated DeltaNet** (linear-attention слои): проекции qkv/z/b/a,
//!   причинная depthwise-свёртка (ядро 4) + SiLU, L2-норма q/k,
//!   `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`,
//!   рекуррентность дельта-правила в f32 (`mamba_ssm_dtype: float32`),
//!   gated RMSNorm (`norm` на выходе, gate из `z`), `out_proj`.
//! - **Gated attention** (full-attention слои): GQA 8 Q-голов / 2 KV,
//!   head_dim 256, QK-RMSNorm, частичный RoPE (`partial_rotary_factor`
//!   0.25 → вращаются первые 64 из 256 компонент головы, θ=1e7),
//!   выход умножается на `sigmoid(gate)` из второй половины q_proj.
//! - MLP (SwiGLU) и RMSNorm — во всех слоях; MTP-блок и vision-башня
//!   чекпоинта не загружаются (только текст, не-thinking режим).
//!
//! Рекуррентность дельта-правила на каждый токен (f32, состояние
//! `S: (головы, V, K)`):
//!   1. `S ← exp(g) ⊙ S` (затухание до извлечения);
//!   2. `kv = S·k`;
//!   3. `delta = (v − kv) · beta`;
//!   4. `S ← S + delta ⊗ k`;
//!   5. `o = S·q` (по уже обновлённому состоянию).
//!
//! Prefill тоже идёт по токенам: точный рекуррентный пересчёт вместо
//! чанкового алгоритма из референса. Дешевле по коду и математически
//! идентично; узкое место — запуски GPU-ядер на маленьких тензорах.

use super::ChatTurn;
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Локальный каталог модели внутри `models/`.
pub const LOCAL_DIR_NAME: &str = "qwen3.5-0.8b";

/// Потолок таблиц RoPE: у модели `max_position_embeddings = 262144`,
/// держать таблицу такого размера незачем.
const ROPE_TABLE_MAX_SEQ: usize = 8192;

/// Порог softplus из PyTorch: выше — линейный участок.
const SOFTPLUS_THRESHOLD: f32 = 20.0;

const EPS_L2NORM: f64 = 1e-6;

// =========================================================================
// Конфиг
// =========================================================================

#[derive(Debug, Deserialize)]
struct RawConfig {
    text_config: RawTextConfig,
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    rms_norm_eps: f64,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    layer_types: Vec<String>,
    rope_parameters: RawRopeParameters,
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    rope_theta: Option<f32>,
    partial_rotary_factor: Option<f64>,
}

/// Разобранный `text_config` из `config.json`.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_heads: usize,
    pub linear_head_k_dim: usize,
    pub linear_head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// Доля компонент головы, охваченных RoPE (0.25 → 64 из 256).
    pub partial_rotary_factor: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// Тип каждого слоя в порядке следования: `"linear_attention"`
    /// (Gated DeltaNet) или `"full_attention"`.
    pub layer_types: Vec<LayerType>,
    pub eos_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen35Config {
    pub fn parse(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json)
            .context("config.json: ожидалась секция text_config (qwen3_5)")?;
        let t = raw.text_config;

        if t.linear_num_value_heads != t.linear_num_key_heads {
            bail!(
                "Qwen3.5-0.8B: linear_num_value_heads ({}) != linear_num_key_heads ({}) — \
                 вариант с разными головами не поддержан",
                t.linear_num_value_heads,
                t.linear_num_key_heads
            );
        }
        if t.layer_types.len() != t.num_hidden_layers {
            bail!(
                "layer_types ({}) не совпадает с num_hidden_layers ({})",
                t.layer_types.len(),
                t.num_hidden_layers
            );
        }
        let layer_types = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(LayerType::LinearAttention),
                "full_attention" => Ok(LayerType::FullAttention),
                other => Err(anyhow::anyhow!("неизвестный тип слоя: {other:?}")),
            })
            .collect::<Result<Vec<_>>>()?;

        let eos_ids = t
            .eos_token_id
            .as_ref()
            .map(|v| match v {
                serde_json::Value::Array(items) => items
                    .iter()
                    .filter_map(|x| x.as_u64().map(|x| x as u32))
                    .collect(),
                other => other.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            })
            .unwrap_or_default();

        Ok(Self {
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            vocab_size: t.vocab_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            linear_num_heads: t.linear_num_value_heads,
            linear_head_k_dim: t.linear_key_head_dim,
            linear_head_v_dim: t.linear_value_head_dim,
            conv_kernel_size: t.linear_conv_kernel_dim,
            rms_norm_eps: t.rms_norm_eps,
            rope_theta: t.rope_parameters.rope_theta.unwrap_or(1_000_000.0),
            partial_rotary_factor: t.rope_parameters.partial_rotary_factor.unwrap_or(1.0),
            max_position_embeddings: t.max_position_embeddings.unwrap_or(4096),
            tie_word_embeddings: t.tie_word_embeddings.unwrap_or(true),
            layer_types,
            eos_ids,
        })
    }

    /// Сколько компонент головы вращает RoPE (у 0.8B: 256 × 0.25 = 64).
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// Определяет по `config.json`, что каталог содержит модель семейства
    /// qwen3_5 (гибрид с Gated DeltaNet), а не Llama-подобную.
    pub fn is_qwen35_config(config_path: &Path) -> bool {
        std::fs::read_to_string(config_path)
            .ok()
            .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
            .and_then(|v| {
                v.get("text_config")
                    .and_then(|t| t.get("model_type"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .as_deref()
            == Some("qwen3_5_text")
    }
}

// =========================================================================
// Линейный слой (веса уже в раскладке [in, out])
// =========================================================================

struct Linear {
    w: Tensor,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.matmul(&self.w)
    }
}

// =========================================================================
// Свёртка и дельта-правило (свободные функции — тестируются на CPU)
// =========================================================================

/// Причинная depthwise-свёртка с состоянием. `x: (seq, channels)`,
/// `w: (kernel, channels)`, `state: (kernel-1, channels)` — последние
/// `kernel-1` входных строк (при старте — нули). Возвращает `(seq, channels)`,
/// обновляя `state`. Активация не применяется.
fn causal_conv(x: &Tensor, w: &Tensor, state: &mut Tensor) -> candle_core::Result<Tensor> {
    let (seq, channels) = x.dims2()?;
    let (kw, w_channels) = w.dims2()?;
    debug_assert_eq!(channels, w_channels);
    debug_assert_eq!(state.dims()[0], kw - 1);

    let padded = Tensor::cat(&[state, x], 0)?; // (kw-1+seq, channels)
    let mut acc = padded
        .narrow(0, 0, seq)?
        .broadcast_mul(&w.narrow(0, 0, 1)?)?;
    for k in 1..kw {
        let term = padded
            .narrow(0, k, seq)?
            .broadcast_mul(&w.narrow(0, k, 1)?)?;
        acc = acc.broadcast_add(&term)?;
    }
    *state = padded.narrow(0, seq, kw - 1)?.contiguous()?;
    Ok(acc)
}

/// Softplus с порогом PyTorch: `log(1+exp(x))`, а при `x > 20` — само `x`.
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let threshold = Tensor::new(SOFTPLUS_THRESHOLD, x.device())?.broadcast_as(x.dims())?;
    let linear = x.gt(&threshold)?;
    let smooth = x.exp()?.affine(1.0, 1.0)?.log()?;
    linear.where_cond(x, &smooth)
}

/// L2-норма по последней размерности в духе fla:
/// `x · rsqrt(Σx² + eps)` (сумма, не среднее).
fn l2norm(x: &Tensor) -> candle_core::Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0, EPS_L2NORM)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

/// Рекуррентное дельта-правило (gated delta rule) — точный пересчёт по токенам.
///
/// `q, k: (seq, heads, K)`, `v: (seq, heads, V)`, `g, beta: (seq, heads)`,
/// `state: (heads, V, K)` — изменяется на месте. Все входы в f32; `q`
/// должен быть уже промасштабирован на `K^-0.5` и нормирован.
/// Возвращает `(seq, heads, V)`.
fn gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> candle_core::Result<Tensor> {
    let (seq, heads, k_dim) = q.dims3()?;
    let (_, _, v_dim) = v.dims3()?;
    let g_exp = g.exp()?; // (seq, heads)
    let mut outputs = Vec::with_capacity(seq);

    for t in 0..seq {
        // 1. Затухание состояния до извлечения.
        let decay = g_exp.narrow(0, t, 1)?.reshape((heads, 1, 1))?;
        *state = state.broadcast_mul(&decay)?;

        let kt = k.narrow(0, t, 1)?.reshape((heads, k_dim, 1))?;
        let vt = v.narrow(0, t, 1)?.reshape((heads, v_dim, 1))?;
        let qt = q.narrow(0, t, 1)?.reshape((heads, k_dim, 1))?;
        let bt = beta.narrow(0, t, 1)?.reshape((heads, 1, 1))?;

        // 2. Извлечение: kv_mem = S·k.
        let kv_mem = state.matmul(&kt)?;
        // 3-4. Запись поправки: S += beta·(v − kv_mem) ⊗ k.
        let delta = vt.broadcast_sub(&kv_mem)?.broadcast_mul(&bt)?;
        *state = state.broadcast_add(&delta.matmul(&kt.transpose(1, 2)?)?)?;
        // 5. Выход по уже обновлённому состоянию.
        let o_t = state.matmul(&qt)?; // (heads, V, 1)
        outputs.push(o_t);
    }

    Tensor::stack(&outputs, 0)?.squeeze(D::Minus1)?.contiguous()
}

// =========================================================================
// KV-кэш
// =========================================================================

/// Zero-centered RMSNorm Qwen3.5 (`Qwen3_5RMSNorm`): вес хранится как
/// отклонение от единицы и применяется как `1 + weight`, вся математика в
/// f32, округление до bf16 один раз в конце.
fn rms_norm_zero_centered(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_mul(&var.affine(1.0, eps)?.sqrt()?.recip()?)?;
    let gain = w.to_dtype(DType::F32)?.affine(1.0, 1.0)?; // 1 + weight
    normed.broadcast_mul(&gain)?.to_dtype(x.dtype())
}

/// Кэш одного full-attention слоя: `(kv_heads, total_seq, head_dim)`.
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<()> {
        let k = k.transpose(0, 1)?.contiguous()?; // (seq, nkv, hd) → (nkv, seq, hd)
        let v = v.transpose(0, 1)?.contiguous()?;
        self.k = Some(match &self.k {
            None => k,
            Some(prev) => Tensor::cat(&[prev, &k], 1)?,
        });
        self.v = Some(match &self.v {
            None => v,
            Some(prev) => Tensor::cat(&[prev, &v], 1)?,
        });
        Ok(())
    }

    fn kv(&self) -> candle_core::Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Ok((k.clone(), v.clone())),
            _ => Err(candle_core::Error::msg("KV-кэш пуст")),
        }
    }
}

/// Состояние одного слоя.
pub enum LayerCache {
    Linear {
        /// Свёртка: последние `kernel-1` входов, `(kernel-1, conv_dim)`.
        conv: Tensor,
        /// Состояние дельта-правила, `(heads, V, K)`, f32.
        ssm: Tensor,
    },
    Full(KvCache),
}

/// Полный кэш модели (по слою на каждый).
pub struct Qwen35Cache {
    layers: Vec<LayerCache>,
}

impl Qwen35Cache {
    pub fn reset(&mut self, model: &Qwen35Model) {
        *self = model.new_cache();
    }
}

// =========================================================================
// Слои
// =========================================================================

/// Linear-attention слой (Gated DeltaNet).
struct DeltaNet {
    in_proj_qkv: Linear, // [key*2 + value, hidden] = [6144, 1024]
    in_proj_z: Linear,   // [value, hidden] = [2048, 1024]
    in_proj_b: Linear,   // [heads, hidden] = [16, 1024]
    in_proj_a: Linear,   // [heads, hidden] = [16, 1024]
    /// Свёрточное ядро, `(kernel, conv_dim)`, bf16.
    conv_w: Tensor,
    /// `-exp(A_log)`, `(heads,)`, f32 — уже с минусом.
    neg_exp_a_log: Tensor,
    /// `(heads,)`, f32.
    dt_bias: Tensor,
    /// Вес gated RMSNorm, `(head_v_dim,)`, f32.
    norm_w: Tensor,
    out_proj: Linear,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    rms_norm_eps: f64,
}

impl DeltaNet {
    fn forward(
        &self,
        x: &Tensor,        // (seq, hidden), bf16
        conv: &mut Tensor, // (kernel-1, conv_dim)
        ssm: &mut Tensor,  // (heads, V, K), f32
    ) -> candle_core::Result<Tensor> {
        let (seq, _) = x.dims2()?;
        let key_dim = self.num_heads * self.head_k_dim;
        let value_dim = self.num_heads * self.head_v_dim;
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?; // (seq, value_dim) — в gate, без свёртки
        let b = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;

        // Свёртка только над qkv, затем SiLU.
        let qkv = candle_nn::ops::silu(&causal_conv(&qkv, &self.conv_w, conv)?)?;

        // Раскладка conv-выхода: [q | k | v] по key_dim/value_dim компонент.
        let q = qkv
            .narrow(1, 0, key_dim)?
            .reshape((seq, self.num_heads, self.head_k_dim))?;
        let k = qkv
            .narrow(1, key_dim, key_dim)?
            .reshape((seq, self.num_heads, self.head_k_dim))?;
        let v = qkv.narrow(1, 2 * key_dim, value_dim)?.reshape((
            seq,
            self.num_heads,
            self.head_v_dim,
        ))?;

        // Всё в f32, как в референсе: L2-норма q/k, масштаб на K^-0.5.
        let q = l2norm(&q.to_dtype(DType::F32)?)?;
        let k = l2norm(&k.to_dtype(DType::F32)?)?;
        let q_scale = (self.head_k_dim as f64).sqrt().recip();
        let q = q.affine(q_scale, 0.0)?;
        let v = v.to_dtype(DType::F32)?;

        let beta = candle_nn::ops::sigmoid(&b.to_dtype(DType::F32)?)?;
        // g = -exp(A_log) * softplus(a + dt_bias), (seq, heads), f32.
        let a = a.to_dtype(DType::F32)?;
        let a = a.broadcast_add(&self.dt_bias.reshape((1, self.num_heads))?)?;
        let g = softplus(&a)?.broadcast_mul(&self.neg_exp_a_log.reshape((1, self.num_heads))?)?;

        let o = gated_delta_rule(&q, &k, &v, &g, &beta, ssm)?; // (seq, heads, V), f32

        // Gated RMSNorm: нормируем выход, затем gate из z (silu), всё в f32.
        let o = o.reshape((seq * self.num_heads, self.head_v_dim))?;
        let var = o.sqr()?.mean_keepdim(D::Minus1)?;
        let o = o
            .broadcast_mul(&var.affine(1.0, self.rms_norm_eps)?.sqrt()?.recip()?)?
            // Зеркалим референс: после нормировки — округление до bf16.
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        let z = z.reshape((seq * self.num_heads, self.head_v_dim))?;
        let o = o.broadcast_mul(&self.norm_w.reshape((1, self.head_v_dim))?)?;
        let o = o.broadcast_mul(&candle_nn::ops::silu(&z.to_dtype(DType::F32)?)?)?;
        let o = o.to_dtype(DType::BF16)?.reshape((seq, value_dim))?;

        let o = self.out_proj.forward(&o)?;
        Ok(o)
    }
}

/// Full-attention слой: GQA + QK-norm + частичный RoPE + sigmoid-gate.
struct FullAttention {
    q: Linear, // [heads * head_dim * 2, hidden] — половина под gate
    k: Linear, // [kv_heads * head_dim, hidden]
    v: Linear,
    o: Linear, // [hidden, heads * head_dim]
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f64,
}

impl FullAttention {
    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut KvCache,
    ) -> candle_core::Result<Tensor> {
        let (seq, _hidden) = x.dims2()?;
        let hd = self.head_dim;
        let half_rot = self.rotary_dim / 2;
        // q_proj даёт [значения | gate] по 256 компонент на голову.
        let qkv = self.q.forward(x)?.reshape((seq, self.num_heads, hd * 2))?;
        let gate = candle_nn::ops::sigmoid(&qkv.narrow(2, hd, hd)?)?;
        // narrow по последней размерности даёт неконтiguous тензор —
        // rms_norm в candle умеет только contiguous.
        let q = qkv.narrow(2, 0, hd)?.contiguous()?;
        let k = self.k.forward(x)?.reshape((seq, self.num_kv_heads, hd))?;
        let v = self.v.forward(x)?.reshape((seq, self.num_kv_heads, hd))?;

        // QK-RMSNorm (zero-centered) по head_dim, до RoPE.
        let q = rms_norm_zero_centered(&q, &self.q_norm_w, self.eps)?;
        let k = rms_norm_zero_centered(&k, &self.k_norm_w, self.eps)?;

        // Частичный RoPE: вращаем первые rotary_dim компонент.
        let rope_one = |t: &Tensor| -> candle_core::Result<Tensor> {
            let t_rot = t.narrow(2, 0, self.rotary_dim)?;
            let t_pass = t.narrow(2, self.rotary_dim, hd - self.rotary_dim)?;
            let t1 = t_rot.narrow(2, 0, half_rot)?;
            let t2 = t_rot.narrow(2, half_rot, half_rot)?;
            let cos = cos
                .narrow(0, start_pos, seq)?
                .unsqueeze(1)?
                .to_dtype(t.dtype())?;
            let sin = sin
                .narrow(0, start_pos, seq)?
                .unsqueeze(1)?
                .to_dtype(t.dtype())?;
            let o1 = t1
                .broadcast_mul(&cos)?
                .broadcast_sub(&t2.broadcast_mul(&sin)?)?;
            let o2 = t2
                .broadcast_mul(&cos)?
                .broadcast_add(&t1.broadcast_mul(&sin)?)?;
            Tensor::cat(&[o1, o2, t_pass], D::Minus1)
        };
        let q = rope_one(&q)?;
        let k = rope_one(&k)?;

        cache.append(&k, &v)?;
        let (k, v) = cache.kv()?;
        let total = k.dim(1)?;

        // GQA без копий K/V — группировка q-голов по kv-головам.
        let rep = self.num_heads / self.num_kv_heads;
        let q = q.transpose(0, 1)?.contiguous()?; // (nh, seq, hd)
        let q = if rep > 1 {
            q.reshape((self.num_kv_heads, rep, seq, hd))?
                .permute((0, 2, 1, 3))?
                .reshape((self.num_kv_heads, seq * rep, hd))?
                .contiguous()?
        } else {
            q
        };

        let scale = (hd as f64).sqrt().recip();
        // Скоры сразу в f32: и маска, и softmax в референсе считаются в f32.
        let mut scores = q
            .matmul(&k.transpose(1, 2)?)?
            .affine(scale, 0.0)?
            .to_dtype(DType::F32)?;

        if seq > 1 {
            let mask = causal_mask(seq, total, start_pos, x.device(), DType::F32)?;
            scores = scores
                .reshape((self.num_kv_heads, seq, rep, total))?
                .broadcast_add(&mask.unsqueeze(2)?)?
                .reshape((self.num_kv_heads, seq * rep, total))?;
        }

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(x.dtype())?;
        let out = attn.matmul(&v)?; // (nkv, seq*rep, hd)
        let out = out
            .reshape((self.num_kv_heads, seq, rep, hd))?
            .permute((1, 0, 2, 3))?
            .reshape((seq, self.num_heads * hd))?;

        // Sigmoid-gate на выход внимания.
        let out = out.broadcast_mul(&gate.reshape((seq, self.num_heads * hd))?)?;
        let o = self.o.forward(&out)?;
        Ok(o)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        self.down
            .forward(&candle_nn::ops::silu(&gate)?.broadcast_mul(&up)?)
    }
}

enum LayerKind {
    Linear(DeltaNet),
    Full(FullAttention),
}

struct Block {
    input_norm_w: Tensor,
    post_norm_w: Tensor,
    kind: LayerKind,
    mlp: Mlp,
    eps: f64,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope: &RopeTables,
        cache: &mut LayerCache,
    ) -> candle_core::Result<Tensor> {
        let normed = rms_norm_zero_centered(x, &self.input_norm_w, self.eps)?;
        let attn_out = match (&self.kind, cache) {
            (LayerKind::Linear(dn), LayerCache::Linear { conv, ssm }) => {
                dn.forward(&normed, conv, ssm)?
            }
            (LayerKind::Full(attn), LayerCache::Full(kv)) => {
                attn.forward(&normed, start_pos, &rope.cos, &rope.sin, kv)?
            }
            _ => unreachable!("тип слоя и кэш не согласованы"),
        };
        let x = (x + attn_out)?;

        let normed = rms_norm_zero_centered(&x, &self.post_norm_w, self.eps)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x + mlp_out
    }
}

// =========================================================================
// Каузальная маска (кэшируется для полного префилла)
// =========================================================================

/// Кэш каузальных масок: (seq, total, start_pos) → маска.
type MaskCache = Mutex<HashMap<(usize, usize, usize), Tensor>>;

static MASK_CACHE: OnceLock<MaskCache> = OnceLock::new();

fn causal_mask(
    seq: usize,
    total: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    if start_pos == 0 && seq > 1 {
        let cache = MASK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = guard.get(&(seq, total, start_pos)) {
            return Ok(m.clone());
        }
        let mask = build_causal_mask(seq, total, start_pos, device, dtype)?;
        if guard.len() > 64 {
            guard.clear();
        }
        guard.insert((seq, total, start_pos), mask.clone());
        return Ok(mask);
    }
    build_causal_mask(seq, total, start_pos, device, dtype)
}

fn build_causal_mask(
    seq: usize,
    total: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    let data: Vec<f32> = (0..seq)
        .flat_map(|i| {
            (0..total).map(move |j| {
                if j <= start_pos + i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (seq, total), device)?
        .unsqueeze(0)?
        .to_dtype(dtype)
}

// =========================================================================
// RoPE
// =========================================================================

pub struct RopeTables {
    /// `(max_seq, rotary_dim/2)`, f32.
    cos: Tensor,
    sin: Tensor,
}

fn build_rope_tables(
    max_seq: usize,
    rotary_dim: usize,
    theta: f32,
    device: &Device,
) -> candle_core::Result<RopeTables> {
    let half = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    let inv = Tensor::from_vec(inv_freq, (1, half), device)?;
    let positions: Vec<f32> = (0..max_seq).map(|i| i as f32).collect();
    let pos = Tensor::from_vec(positions, (max_seq, 1), device)?;
    let freqs = pos.broadcast_mul(&inv)?;
    Ok(RopeTables {
        cos: freqs.cos()?,
        sin: freqs.sin()?,
    })
}

// =========================================================================
// Загрузка весов
// =========================================================================

/// Доступ к весам: два VarBuilder'а поверх одного mmap — bf16 для основных
/// тензоров и f32 для гейтов затухания (`A_log`, `dt_bias`) и веса gated
/// RMSNorm, которые в чекпоинте лежат в f32 и понижать их нельзя.
struct Weights {
    bf16: candle_nn::VarBuilder<'static>,
    f32: candle_nn::VarBuilder<'static>,
}

impl Weights {
    fn load(weights: &[std::path::PathBuf], device: &Device) -> Result<Self> {
        // SAFETY: mmap read-only поверх наших собственных файлов модели;
        // тот же паттерн, что в `model.rs` для MiniCPM5 (candle 0.11 помечает
        // mmap-загрузку unsafe из-за memmap2).
        let bf16 =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(weights, DType::BF16, device) }
                .context("Не удалось открыть safetensors-веса (bf16)")?;
        let f32 =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(weights, DType::F32, device) }
                .context("Не удалось открыть safetensors-веса (f32)")?;
        Ok(Self { bf16, f32 })
    }

    fn tensor(&self, name: &str, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let vb = match dtype {
            DType::F32 => &self.f32,
            _ => &self.bf16,
        };
        vb.get(shape, name)
            .with_context(|| format!("Нет тензора {name} или не совпала форма {shape:?}"))
    }

    /// Тензор слоя; f32-тензоры чекпоинта остаются в f32, остальные — bf16.
    fn layer_tensor(&self, layer: usize, suffix: &str, shape: &[usize]) -> Result<Tensor> {
        let is_f32 = suffix.ends_with("A_log")
            || suffix.ends_with("dt_bias")
            || suffix == "linear_attn.norm.weight";
        let dtype = if is_f32 { DType::F32 } else { DType::BF16 };
        self.tensor(
            &format!("model.language_model.layers.{layer}.{suffix}"),
            shape,
            dtype,
        )
    }

    fn linear(&self, name: &str, out: usize, inn: usize) -> Result<Linear> {
        let w = self
            .tensor(name, &[out, inn], DType::BF16)?
            .t()?
            .contiguous()?; // [in, out] для matmul
        Ok(Linear { w })
    }
}

// =========================================================================
// Модель
// =========================================================================

pub struct Qwen35Model {
    embed: Tensor,   // (vocab, hidden), bf16
    lm_head: Tensor, // (hidden, vocab), bf16
    layers: Vec<Block>,
    norm_w: Tensor,
    rope: RopeTables,
    rope_len: usize,
    device: Device,
    pub config: Qwen35Config,
}

impl Qwen35Model {
    /// Загружает модель из каталога (`config.json` + safetensors-веса).
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let started = std::time::Instant::now();
        let config_path = dir.join("config.json");
        let json = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Не удалось прочитать {:?}", config_path))?;
        let config = Qwen35Config::parse(&json)?;

        let mut weights: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with("model.safetensors") && name.ends_with(".safetensors")
            })
            .collect();
        if weights.is_empty() {
            bail!("В {:?} нет safetensors-весов Qwen3.5-0.8B", dir);
        }
        weights.sort();

        let w = Weights::load(&weights, device)?;
        let cfg = &config;
        let hidden = cfg.hidden_size;

        let embed = w.tensor(
            "model.language_model.embed_tokens.weight",
            &[cfg.vocab_size, hidden],
            DType::BF16,
        )?;
        // tie_word_embeddings: lm_head = embedᵀ; транспонируем один раз,
        // чтобы каждый forward не таскал 500-МБ копию.
        if !cfg.tie_word_embeddings {
            bail!("Qwen3.5-0.8B ожидается с tie_word_embeddings=true");
        }
        let lm_head = embed.t()?.contiguous()?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let input_norm_w = w.layer_tensor(i, "input_layernorm.weight", &[hidden])?;
            let post_norm_w = w.layer_tensor(i, "post_attention_layernorm.weight", &[hidden])?;

            let kind = match cfg.layer_types[i] {
                LayerType::LinearAttention => {
                    let key_dim = cfg.linear_num_heads * cfg.linear_head_k_dim;
                    let value_dim = cfg.linear_num_heads * cfg.linear_head_v_dim;
                    let conv_w = w
                        .layer_tensor(
                            i,
                            "linear_attn.conv1d.weight",
                            &[key_dim * 2 + value_dim, 1, cfg.conv_kernel_size],
                        )?
                        .squeeze(1)?
                        .t()?
                        .contiguous()?; // (kernel, conv_dim)
                    let a_log = w.layer_tensor(i, "linear_attn.A_log", &[cfg.linear_num_heads])?;
                    let neg_exp_a_log = a_log.exp()?.affine(-1.0, 0.0)?;
                    let dt_bias =
                        w.layer_tensor(i, "linear_attn.dt_bias", &[cfg.linear_num_heads])?;
                    let norm_w =
                        w.layer_tensor(i, "linear_attn.norm.weight", &[cfg.linear_head_v_dim])?;
                    LayerKind::Linear(DeltaNet {
                        in_proj_qkv: w.linear(
                            &format!(
                                "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"
                            ),
                            key_dim * 2 + value_dim,
                            hidden,
                        )?,
                        in_proj_z: w.linear(
                            &format!(
                                "model.language_model.layers.{i}.linear_attn.in_proj_z.weight"
                            ),
                            value_dim,
                            hidden,
                        )?,
                        in_proj_b: w.linear(
                            &format!(
                                "model.language_model.layers.{i}.linear_attn.in_proj_b.weight"
                            ),
                            cfg.linear_num_heads,
                            hidden,
                        )?,
                        in_proj_a: w.linear(
                            &format!(
                                "model.language_model.layers.{i}.linear_attn.in_proj_a.weight"
                            ),
                            cfg.linear_num_heads,
                            hidden,
                        )?,
                        conv_w,
                        neg_exp_a_log,
                        dt_bias,
                        norm_w,
                        out_proj: w.linear(
                            &format!("model.language_model.layers.{i}.linear_attn.out_proj.weight"),
                            hidden,
                            value_dim,
                        )?,
                        num_heads: cfg.linear_num_heads,
                        head_k_dim: cfg.linear_head_k_dim,
                        head_v_dim: cfg.linear_head_v_dim,
                        rms_norm_eps: cfg.rms_norm_eps,
                    })
                }
                LayerType::FullAttention => {
                    let hd = cfg.head_dim;
                    LayerKind::Full(FullAttention {
                        q: w.linear(
                            &format!("model.language_model.layers.{i}.self_attn.q_proj.weight"),
                            cfg.num_attention_heads * hd * 2,
                            hidden,
                        )?,
                        k: w.linear(
                            &format!("model.language_model.layers.{i}.self_attn.k_proj.weight"),
                            cfg.num_key_value_heads * hd,
                            hidden,
                        )?,
                        v: w.linear(
                            &format!("model.language_model.layers.{i}.self_attn.v_proj.weight"),
                            cfg.num_key_value_heads * hd,
                            hidden,
                        )?,
                        o: w.linear(
                            &format!("model.language_model.layers.{i}.self_attn.o_proj.weight"),
                            hidden,
                            cfg.num_attention_heads * hd,
                        )?,
                        q_norm_w: w.layer_tensor(i, "self_attn.q_norm.weight", &[hd])?,
                        k_norm_w: w.layer_tensor(i, "self_attn.k_norm.weight", &[hd])?,
                        num_heads: cfg.num_attention_heads,
                        num_kv_heads: cfg.num_key_value_heads,
                        head_dim: hd,
                        rotary_dim: cfg.rotary_dim(),
                        eps: cfg.rms_norm_eps,
                    })
                }
            };

            let mlp = Mlp {
                gate: w.linear(
                    &format!("model.language_model.layers.{i}.mlp.gate_proj.weight"),
                    cfg.intermediate_size,
                    hidden,
                )?,
                up: w.linear(
                    &format!("model.language_model.layers.{i}.mlp.up_proj.weight"),
                    cfg.intermediate_size,
                    hidden,
                )?,
                down: w.linear(
                    &format!("model.language_model.layers.{i}.mlp.down_proj.weight"),
                    hidden,
                    cfg.intermediate_size,
                )?,
            };

            layers.push(Block {
                input_norm_w,
                post_norm_w,
                kind,
                mlp,
                eps: cfg.rms_norm_eps,
            });
        }

        let norm_w = w.tensor("model.language_model.norm.weight", &[hidden], DType::BF16)?;

        let rotary_dim = cfg.rotary_dim();
        let rope_len = cfg.max_position_embeddings.min(ROPE_TABLE_MAX_SEQ);
        let rope = build_rope_tables(rope_len, rotary_dim, cfg.rope_theta, device)?;

        println!(
            "✅ Qwen3.5-0.8B загружена за {:.1?} ({} слоёв: {} delta-net + {} attention, vocab={})",
            started.elapsed(),
            cfg.num_hidden_layers,
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            cfg.vocab_size,
        );

        Ok(Self {
            embed,
            lm_head,
            layers,
            norm_w,
            rope,
            rope_len,
            device: device.clone(),
            config,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn rope_len(&self) -> usize {
        self.rope_len
    }

    pub fn new_cache(&self) -> Qwen35Cache {
        let dev = &self.device;
        let conv_dim = self.config.linear_num_heads
            * (2 * self.config.linear_head_k_dim + self.config.linear_head_v_dim);
        let layers = self
            .layers
            .iter()
            .map(|block| match &block.kind {
                LayerKind::Linear(_) => LayerCache::Linear {
                    conv: Tensor::zeros(
                        (self.config.conv_kernel_size - 1, conv_dim),
                        DType::BF16,
                        dev,
                    )
                    .unwrap(),
                    ssm: Tensor::zeros(
                        (
                            self.config.linear_num_heads,
                            self.config.linear_head_v_dim,
                            self.config.linear_head_k_dim,
                        ),
                        DType::F32,
                        dev,
                    )
                    .unwrap(),
                },
                LayerKind::Full(_) => LayerCache::Full(KvCache { k: None, v: None }),
            })
            .collect();
        Qwen35Cache { layers }
    }

    /// Прямой проход: `ids` — токены, `start_pos` — позиция первого токена
    /// в общем контексте. Возвращает логиты `(seq, vocab)`.
    pub fn forward(
        &self,
        ids: &[u32],
        start_pos: usize,
        cache: &mut Qwen35Cache,
    ) -> Result<Tensor> {
        self.forward_with_inspector(ids, start_pos, cache, &mut |_, _| {})
    }

    /// Как [`forward`](Self::forward), но вызывает `inspect(i, &hidden)`
    /// после эмбеддинга (i = USIZE_MAX), каждого слоя (i — номер слоя)
    /// и финальной нормировки (i = USIZE_MAX - 1). Для отладочной сверки.
    pub fn forward_with_inspector(
        &self,
        ids: &[u32],
        start_pos: usize,
        cache: &mut Qwen35Cache,
        inspect: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("Пустой набор токенов для forward");
        }
        if cache.layers.len() != self.layers.len() {
            bail!(
                "Число кэшей ({}) не совпадает с числом слоёв ({})",
                cache.layers.len(),
                self.layers.len()
            );
        }
        let seq = ids.len();
        if start_pos + seq > self.rope_len {
            bail!(
                "Контекст {}+{} токенов превышает потолок RoPE {}",
                start_pos,
                seq,
                self.rope_len
            );
        }

        let ids_t = Tensor::from_slice(ids, (seq,), &self.device)?;
        let mut x = self.embed.index_select(&ids_t, 0)?; // (seq, hidden), bf16
        inspect(usize::MAX, &x);

        for (i, block) in self.layers.iter().enumerate() {
            x = block.forward(&x, start_pos, &self.rope, &mut cache.layers[i])?;
            inspect(i, &x);
        }

        let x = rms_norm_zero_centered(&x, &self.norm_w, self.config.rms_norm_eps)?;
        inspect(usize::MAX - 1, &x);
        Ok(x.matmul(&self.lm_head)?)
    }
}

// =========================================================================
// Шаблон диалога (ChatML, не-thinking режим)
// =========================================================================

/// Собирает промпт по chat-шаблону Qwen3.5: ChatML с принудительно
/// закрытым блоком рассуждений — модель отвечает сразу, без `<think>`.
pub fn build_chat_prompt(system: &str, history: &[ChatTurn], user: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    for (q, a) in history {
        out.push_str(&format!(
            "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        ));
    }
    out.push_str(&format!(
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn parses_real_config() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.5-0.8b/config.json");
        if !path.exists() {
            return; // модель не скачана — проверять нечего
        }
        let json = std::fs::read_to_string(&path).unwrap();
        let cfg = Qwen35Config::parse(&json).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.linear_num_heads, 16);
        assert_eq!(cfg.rotary_dim(), 64);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            18
        );
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            6
        );
        // Каждый четвёртый слой — полный.
        for (i, t) in cfg.layer_types.iter().enumerate() {
            assert_eq!(*t == LayerType::FullAttention, i % 4 == 3);
        }
        assert!(cfg.eos_ids.contains(&248044));
    }

    #[test]
    fn chatml_template_closes_think() {
        let p = build_chat_prompt("Ты ассистент.", &[], "привет");
        assert!(p.starts_with("<|im_start|>system\nТы ассистент.<|im_end|>\n"));
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let history = vec![("вопрос".to_string(), "ответ".to_string())];
        let p = build_chat_prompt("S.", &history, "новый");
        assert!(p.contains(
            "<|im_start|>user\nвопрос<|im_end|>\n<|im_start|>assistant\nответ<|im_end|>\n"
        ));
        assert!(p.find("вопрос").unwrap() < p.find("новый").unwrap());
    }

    #[test]
    fn softplus_matches_torch() {
        let dev = cpu();
        let x = Tensor::from_vec(vec![-2.0f32, 0.0, 1.5, 25.0], (4,), &dev).unwrap();
        let y = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        let expect = |v: f32| if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
        for (got, x) in y.iter().zip([-2.0f32, 0.0, 1.5, 25.0]) {
            assert!((got - expect(x)).abs() < 1e-5);
        }
    }

    #[test]
    fn l2norm_makes_unit_vectors() {
        let dev = cpu();
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev).unwrap();
        let y = l2norm(&x).unwrap().to_vec2::<f32>().unwrap();
        assert!((y[0][0] - 0.6).abs() < 1e-6);
        assert!((y[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn delta_rule_stores_and_retrieves() {
        // Один токен: v записывается в память вдоль k; чтение по q == k
        // возвращает записанный вектор целиком.
        let dev = cpu();
        let q = Tensor::from_vec(vec![1.0f32, 0.0], (1, 1, 2), &dev).unwrap();
        let k = Tensor::from_vec(vec![1.0f32, 0.0], (1, 1, 2), &dev).unwrap();
        let v = Tensor::from_vec(vec![3.0f32, 4.0], (1, 1, 2), &dev).unwrap();
        let g = Tensor::zeros((1, 1), DType::F32, &dev).unwrap();
        let beta = Tensor::ones((1, 1), DType::F32, &dev).unwrap();
        let mut s = Tensor::zeros((1, 2, 2), DType::F32, &dev).unwrap();
        let o = gated_delta_rule(&q, &k, &v, &g, &beta, &mut s)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((o[0][0][0] - 3.0).abs() < 1e-6);
        assert!((o[0][0][1] - 4.0).abs() < 1e-6);
        // Состояние: v записан в колонку k=(1,0).
        let s_v = s.to_vec3::<f32>().unwrap();
        assert!((s_v[0][0][0] - 3.0).abs() < 1e-6);
        assert!((s_v[0][1][0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn delta_rule_decays_read_before_write() {
        // beta = 0 — записи нет, выход это чистое чтение ЗАТУХШЕГО состояния.
        let dev = cpu();
        // S = [[2, 0], [4, 0]] (V×K): вдоль k=(1,0) хранится вектор (2, 4).
        let mut s = Tensor::from_vec(vec![2.0f32, 0.0, 4.0, 0.0], (1, 2, 2), &dev).unwrap();
        let q = Tensor::from_vec(vec![1.0f32, 0.0], (1, 1, 2), &dev).unwrap();
        let k = Tensor::from_vec(vec![1.0f32, 0.0], (1, 1, 2), &dev).unwrap();
        let v = Tensor::from_vec(vec![7.0f32, 7.0], (1, 1, 2), &dev).unwrap();
        // g = ln 0.5: состояние перед чтением уменьшается вдвое.
        let g = Tensor::from_vec(vec![(0.5f32).ln()], (1, 1), &dev).unwrap();
        let beta = Tensor::zeros((1, 1), DType::F32, &dev).unwrap();
        let o = gated_delta_rule(&q, &k, &v, &g, &beta, &mut s)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((o[0][0][0] - 1.0).abs() < 1e-6);
        assert!((o[0][0][1] - 2.0).abs() < 1e-6);
        // Состояние затухло и осталось таким (записи не было).
        let s_v = s.to_vec3::<f32>().unwrap();
        assert!((s_v[0][0][0] - 1.0).abs() < 1e-6);
        assert!((s_v[0][1][0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn conv_streaming_matches_full_pass() {
        let dev = cpu();
        let (seq, ch, kw) = (7usize, 3usize, 4usize);
        let x: Vec<f32> = (0..seq * ch).map(|i| (i % 11) as f32 - 5.0).collect();
        let w: Vec<f32> = (0..kw * ch).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
        let x_t = Tensor::from_vec(x.clone(), (seq, ch), &dev).unwrap();
        let w_t = Tensor::from_vec(w, (kw, ch), &dev).unwrap();

        // Полный проход с нулевым состоянием.
        let mut state = Tensor::zeros((kw - 1, ch), DType::F32, &dev).unwrap();
        let full = causal_conv(&x_t, &w_t, &mut state)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // Потоково, по одному токену.
        let mut state = Tensor::zeros((kw - 1, ch), DType::F32, &dev).unwrap();
        for (t, row) in full.iter().enumerate() {
            let step = causal_conv(&x_t.narrow(0, t, 1).unwrap(), &w_t, &mut state)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for c in 0..ch {
                assert!((step[0][c] - row[c]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn rope_tables_are_partial() {
        let dev = cpu();
        let tables = build_rope_tables(16, 64, 10_000_000.0, &dev).unwrap();
        let cos = tables.cos.to_vec2::<f32>().unwrap();
        assert_eq!(cos.len(), 16);
        assert_eq!(cos[0].len(), 32); // rotary_dim/2 частот
        // Позиция 0: cos = 1 везде.
        assert!(cos[0].iter().all(|c| (c - 1.0).abs() < 1e-6));
        // Нулевая частота — самая быстрая: cos(7 рад).
        assert!((cos[7][0] - 7.0f32.cos()).abs() < 1e-5);
        // Последняя частота почти не меняется: theta^(-62/64) ≈ 2e-7.
        assert!((cos[7][31] - 1.0).abs() < 1e-4);
    }
}
