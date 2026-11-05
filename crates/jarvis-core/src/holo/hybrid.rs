//! Гибридный мультимодальный декодер Qwen3.5 (`Qwen3_5ForConditionalGeneration`)
//! на чистом candle — база Holo-3.1-4B и Fara1.5-4B.
//!
//! Архитектура (`model_type: qwen3_5`, текст `qwen3_5_text`): 32 слоя, из них
//! 24 linear-attention (Gated DeltaNet) и 8 полных (каждый четвёртый). От
//! текстовой реализации `llm/qwen35.rs` (Qwen3.5-0.8B) отличаются три вещи:
//!
//! 1. **Разные головы DeltaNet**: 16 K-голов / 32 V-головы. q и k
//!    расширяются `repeat_interleave(2)` до V-голов; `beta`/`g` и состояние
//!    рекуррентности — по V-головам (эталон: `modeling_qwen3_5.py`,
//!    `in_proj_b/a → num_v_heads`, состояние `[V, k_head_dim, v_head_dim]`).
//! 2. **MRoPE на полных слоях**: трёхосевые позиции (t/h/w), interleaved
//!    секции [11,11,10], частичный ротор — первые `rotary_dim = 64` из 256
//!    компонент головы, частоты по `rotary_dim` (не по head_dim).
//! 3. **Фьюжн изображений**: вход — готовые `inputs_embeds` (визуальные
//!    токены подставляет рантайм), DeepStack отсутствует (`deepstack_visual_indexes: []`).
//!
//! Для спекулятивного декодирования кэш поддерживает снапшот/восстановление:
//! у linear-attention слоёв состояние рекуррентное, простой truncate невозможен.

use super::processor::PreparedPrompt;
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use serde::Deserialize;
use std::path::Path;
use std::sync::atomic::Ordering;
use tokenizers::Tokenizer;

const SOFTPLUS_THRESHOLD: f32 = 20.0;

fn vb_path(dir: &Path) -> &Path {
    dir
}
const EPS_L2NORM: f64 = 1e-6;

// =========================================================================
// Конфиг
// =========================================================================

#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    pub text: HybridTextConfig,
    pub vision: HybridVisionConfig,
}

#[derive(Debug, Clone)]
pub struct HybridTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// Доля компонент головы, охваченных ротором (0.25 → 64 из 256).
    pub partial_rotary_factor: f64,
    pub mrope_sections: [usize; 3],
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// Тип каждого слоя: `"linear_attention"` (Gated DeltaNet) или `"full_attention"`.
    pub layer_types: Vec<HybridLayerType>,
    pub eos_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct HybridVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
}

impl HybridConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(dir.join("config.json"))
            .context("Не удалось прочитать config.json Qwen3.5-VL")?;
        let mut cfg = Self::parse(&json)?;
        // generation_config.json — канонический источник eos: у Holo
        // eos_token_id = [im_end, endoftext], а в text_config указан один.
        if let Ok(gen_cfg) = std::fs::read_to_string(dir.join("generation_config.json"))
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&gen_cfg)
            && let Some(eos) = value.get("eos_token_id")
        {
            let ids: Vec<u32> = match eos {
                serde_json::Value::Array(items) => {
                    items.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect()
                }
                other => other.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            };
            for id in ids {
                if !cfg.text.eos_ids.contains(&id) {
                    cfg.text.eos_ids.push(id);
                }
            }
        }
        Ok(cfg)
    }

    pub fn parse(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Raw {
            model_type: String,
            image_token_id: u32,
            video_token_id: u32,
            vision_start_token_id: u32,
            vision_end_token_id: u32,
            #[serde(default)]
            tie_word_embeddings: Option<bool>,
            text_config: RawText,
            vision_config: RawVision,
        }
        #[derive(Deserialize)]
        struct RawText {
            model_type: String,
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
            rope_parameters: RawRope,
            max_position_embeddings: usize,
            #[serde(default)]
            tie_word_embeddings: Option<bool>,
            layer_types: Vec<String>,
            #[serde(default)]
            eos_token_id: Option<serde_json::Value>,
        }
        #[derive(Deserialize)]
        struct RawRope {
            rope_theta: Option<f32>,
            #[serde(default = "default_partial")]
            partial_rotary_factor: f64,
            mrope_section: Vec<usize>,
        }
        fn default_partial() -> f64 {
            1.0
        }
        #[derive(Deserialize)]
        struct RawVision {
            depth: usize,
            hidden_size: usize,
            intermediate_size: usize,
            num_heads: usize,
            out_hidden_size: usize,
            patch_size: usize,
            spatial_merge_size: usize,
            temporal_patch_size: usize,
            in_channels: usize,
            num_position_embeddings: usize,
            #[serde(default)]
            deepstack_visual_indexes: Vec<usize>,
        }

        let raw: Raw = serde_json::from_str(json)
            .context("config.json не соответствует архитектуре Qwen3_5ForConditionalGeneration")?;
        if raw.model_type != "qwen3_5" {
            bail!("Ожидался model_type=qwen3_5, получен {}", raw.model_type);
        }
        if raw.text_config.model_type != "qwen3_5_text" {
            bail!("Ожидался text_config.model_type=qwen3_5_text, получен {}", raw.text_config.model_type);
        }
        let t = raw.text_config;
        if t.layer_types.len() != t.num_hidden_layers {
            bail!("layer_types ({}) не совпадает с num_hidden_layers ({})", t.layer_types.len(), t.num_hidden_layers);
        }
        let layer_types = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(HybridLayerType::LinearAttention),
                "full_attention" => Ok(HybridLayerType::FullAttention),
                other => Err(anyhow::anyhow!("неизвестный тип слоя: {other:?}")),
            })
            .collect::<Result<Vec<_>>>()?;
        if t.rope_parameters.mrope_section.len() != 3 {
            bail!("mrope_section должен содержать три оси: {:?}", t.rope_parameters.mrope_section);
        }
        let sections = [
            t.rope_parameters.mrope_section[0],
            t.rope_parameters.mrope_section[1],
            t.rope_parameters.mrope_section[2],
        ];
        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        if sections.iter().sum::<usize>() * 2 != rotary_dim {
            bail!(
                "mrope_section {:?} (сумма {}) не покрывает rotary_dim/2 = {}",
                sections,
                sections.iter().sum::<usize>(),
                rotary_dim / 2
            );
        }
        if t.linear_num_value_heads % t.linear_num_key_heads != 0 {
            bail!(
                "linear_num_value_heads ({}) не кратен linear_num_key_heads ({})",
                t.linear_num_value_heads,
                t.linear_num_key_heads
            );
        }
        if raw.vision_config.out_hidden_size != t.hidden_size {
            bail!(
                "vision out_hidden_size {} != text hidden_size {} — фьюжн невозможен",
                raw.vision_config.out_hidden_size,
                t.hidden_size
            );
        }
        let eos_ids = t.eos_token_id.as_ref().map(|v| match v {
            serde_json::Value::Array(items) => {
                items.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect()
            }
            other => other.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        }).unwrap_or_default();

        Ok(Self {
            image_token_id: raw.image_token_id,
            video_token_id: raw.video_token_id,
            vision_start_token_id: raw.vision_start_token_id,
            vision_end_token_id: raw.vision_end_token_id,
            tie_word_embeddings: raw.tie_word_embeddings.or(t.tie_word_embeddings).unwrap_or(true),
            text: HybridTextConfig {
                hidden_size: t.hidden_size,
                intermediate_size: t.intermediate_size,
                vocab_size: t.vocab_size,
                num_hidden_layers: t.num_hidden_layers,
                num_attention_heads: t.num_attention_heads,
                num_key_value_heads: t.num_key_value_heads,
                head_dim: t.head_dim,
                linear_num_key_heads: t.linear_num_key_heads,
                linear_num_value_heads: t.linear_num_value_heads,
                linear_key_head_dim: t.linear_key_head_dim,
                linear_value_head_dim: t.linear_value_head_dim,
                conv_kernel_size: t.linear_conv_kernel_dim,
                rms_norm_eps: t.rms_norm_eps,
                rope_theta: t.rope_parameters.rope_theta.unwrap_or(1_000_000.0),
                partial_rotary_factor: t.rope_parameters.partial_rotary_factor,
                mrope_sections: sections,
                max_position_embeddings: t.max_position_embeddings,
                tie_word_embeddings: t.tie_word_embeddings.unwrap_or(true),
                layer_types,
                eos_ids,
            },
            vision: HybridVisionConfig {
                depth: raw.vision_config.depth,
                hidden_size: raw.vision_config.hidden_size,
                intermediate_size: raw.vision_config.intermediate_size,
                num_heads: raw.vision_config.num_heads,
                out_hidden_size: raw.vision_config.out_hidden_size,
                patch_size: raw.vision_config.patch_size,
                spatial_merge_size: raw.vision_config.spatial_merge_size,
                temporal_patch_size: raw.vision_config.temporal_patch_size,
                in_channels: raw.vision_config.in_channels,
                num_position_embeddings: raw.vision_config.num_position_embeddings,
            },
        })
    }

    pub fn rotary_dim(&self) -> usize {
        (self.text.head_dim as f64 * self.text.partial_rotary_factor) as usize
    }

    ////storage dtype весов (чекпойнты Qwen3.5 — BF16).
    pub fn weight_dtype(&self) -> DType {
        DType::BF16
    }
}

// =========================================================================
// Математика (локальные копии из llm/qwen35.rs + обобщение на V≠K головы)
// =========================================================================

/// Причинная depthwise-свёртка с состоянием. `x: (seq, channels)`,
/// `w: (kernel, channels)`, `state: (kernel-1, channels)`.
fn causal_conv(x: &Tensor, w: &Tensor, state: &mut Tensor) -> candle_core::Result<Tensor> {
    let (seq, channels) = x.dims2()?;
    let (kw, w_channels) = w.dims2()?;
    debug_assert_eq!(channels, w_channels);
    let padded = Tensor::cat(&[state, x], 0)?;
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

fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let threshold = Tensor::new(SOFTPLUS_THRESHOLD, x.device())?.broadcast_as(x.dims())?;
    let linear = x.gt(&threshold)?;
    let smooth = x.exp()?.affine(1.0, 1.0)?.log()?;
    linear.where_cond(x, &smooth)
}

fn l2norm(x: &Tensor) -> candle_core::Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0, EPS_L2NORM)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

/// Рекуррентное дельта-правило по токенам. `q, k: (seq, heads, K)`,
/// `v: (seq, heads, V)`, `g, beta: (seq, heads)`, `state: (heads, V, K)`.
/// Для 32 V-голов q/k уже расширены repeat_interleave до heads.
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
    let g_exp = g.exp()?;
    let mut outputs = Vec::with_capacity(seq);

    for t in 0..seq {
        let decay = g_exp.narrow(0, t, 1)?.reshape((heads, 1, 1))?;
        *state = state.broadcast_mul(&decay)?;

        let kt = k.narrow(0, t, 1)?.reshape((heads, k_dim, 1))?;
        let vt = v.narrow(0, t, 1)?.reshape((heads, v_dim, 1))?;
        let qt = q.narrow(0, t, 1)?.reshape((heads, k_dim, 1))?;
        let bt = beta.narrow(0, t, 1)?.reshape((heads, 1, 1))?;

        let kv_mem = state.matmul(&kt)?;
        let delta = vt.broadcast_sub(&kv_mem)?.broadcast_mul(&bt)?;
        *state = state.broadcast_add(&delta.matmul(&kt.transpose(1, 2)?)?)?;
        let o_t = state.matmul(&qt)?;
        outputs.push(o_t);
    }

    Tensor::stack(&outputs, 0)?.squeeze(D::Minus1)?.contiguous()
}

/// Zero-centered RMSNorm (`1 + weight`), f32 внутри.
fn rms_norm_zero_centered(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_mul(&var.affine(1.0, eps)?.sqrt()?.recip()?)?;
    let gain = w.to_dtype(DType::F32)?.affine(1.0, 1.0)?;
    normed.broadcast_mul(&gain)?.to_dtype(x.dtype())
}

// =========================================================================
// Частичный interleaved MRoPE: препроцессированные таблицы (быстрый путь)
// =========================================================================

/// Ось каждой частоты (`recomposition_frequencies`): фиксированный паттерн
/// по остатку mod 3 внутри секций.
fn axis_of_freq(freq_idx: usize, sections: [usize; 3]) -> usize {
    if freq_idx < sections[1] * 3 && freq_idx % 3 == 1 {
        1
    } else if freq_idx < sections[2] * 3 && freq_idx % 3 == 2 {
        2
    } else {
        0
    }
}

/// Предвычисленные cos/sin-таблицы по трём осям: (max_seq, half) f32 на GPU.
/// Убирает из каждого шага декодирования Rust-цикл с powf/trig и пары
/// аллокаций тензоров на каждом из 8 attention-слоёв — на decode это главный
/// источник мелких запусков GPU и аллокаций.
pub struct MropeTables {
    cos: [Tensor; 3],
    sin: [Tensor; 3],
    /// Вес частоты по осям: (half,) — 1.0 у своей оси, 0 у прочих.
    weights: [Tensor; 3],
    max_seq: usize,
}

impl MropeTables {
    fn build(
        max_seq: usize,
        rotary_dim: usize,
        sections: [usize; 3],
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        let half = rotary_dim / 2;
        if half == 0 || half > 64 {
            bail!("MropeTables: неподдерживаемый rotary_dim {rotary_dim}");
        }
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        let mut cos_tables = Vec::with_capacity(3);
        let mut sin_tables = Vec::with_capacity(3);
        for _axis in 0..3 {
            let mut cos = Vec::with_capacity(max_seq * half);
            let mut sin = Vec::with_capacity(max_seq * half);
            for p in 0..max_seq {
                for inv in &inv_freq {
                    let angle = p as f32 * inv;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
            cos_tables.push(Tensor::from_vec(cos, (max_seq, half), device)?);
            sin_tables.push(Tensor::from_vec(sin, (max_seq, half), device)?);
        }
        let mut weights = [None, None, None];
        for axis in 0..3 {
            let w: Vec<f32> = (0..half)
                .map(|i| if axis_of_freq(i, sections) == axis { 1.0 } else { 0.0 })
                .collect();
            weights[axis] = Some(Tensor::from_vec(w, (half,), device)?);
        }
        Ok(Self {
            cos: cos_tables.try_into().unwrap(),
            sin: sin_tables.try_into().unwrap(),
            weights: [weights[0].take().unwrap(), weights[1].take().unwrap(), weights[2].take().unwrap()],
            max_seq,
        })
    }

    /// cos/sin-строки `(1, 1, seq, rotary_dim)` для позиций. Для текста
    /// (t=h=w=p) побитово совпадает со скалярным вычислением: смешивание —
    /// это выбор строки своей оси с весами 1.0/0.0.
    fn rows(&self, positions: &[[u32; 3]], dtype: DType) -> Result<(Tensor, Tensor)> {
        let seq = positions.len();
        let device = self.cos[0].device();
        let mut pos = [Vec::with_capacity(seq), Vec::with_capacity(seq), Vec::with_capacity(seq)];
        for p in positions {
            for axis in 0..3 {
                if (p[axis] as usize) >= self.max_seq {
                    bail!("MRoPE позиция {} вне таблицы (max {})", p[axis], self.max_seq);
                }
                pos[axis].push(p[axis]);
            }
        }
        let mix = |tables: &[Tensor; 3]| -> candle_core::Result<Tensor> {
            let mut out: Option<Tensor> = None;
            for axis in 0..3 {
                let ids = Tensor::from_vec(pos[axis].clone(), (seq,), device)?;
                let rows = tables[axis].index_select(&ids, 0)?; // (seq, half)
                let weighted = rows.broadcast_mul(&self.weights[axis])?;
                out = Some(match out {
                    None => weighted,
                    Some(acc) => acc.broadcast_add(&weighted)?,
                });
            }
            Ok(out.expect("три оси всегда дают строку"))
        };
        let cos_half = mix(&self.cos)?; // (seq, half) f32
        let sin_half = mix(&self.sin)?;
        let full = |half_row: &Tensor| -> candle_core::Result<Tensor> {
            Ok(Tensor::cat(&[half_row, half_row], D::Minus1)? // (seq, rotary_dim)
                .to_dtype(dtype)?
                .unsqueeze(0)?
                .unsqueeze(0)?) // (1, 1, seq, rotary_dim)
        };
        Ok((full(&cos_half)?, full(&sin_half)?))
    }
}

/// Ротор над первыми `rotary_dim` компонентами головы, хвост проходит насквозь.
fn rotate_partial(
    x: &Tensor, // (1, heads, seq, head_dim)
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
) -> candle_core::Result<Tensor> {
    let half = rotary_dim / 2;
    let rot = x.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?;
    let pass = x.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
    let a = rot.narrow(D::Minus1, 0, half)?;
    let b = rot.narrow(D::Minus1, half, half)?;
    let neg_b = b.neg()?;
    let rotated = Tensor::cat(&[&neg_b, &a], D::Minus1)?;
    let base = rot.broadcast_mul(&cos)?;
    let spin = rotated.broadcast_mul(&sin)?;
    let out = base.broadcast_add(&spin)?;
    Tensor::cat(&[&out, &pass], D::Minus1)
}

/// Скалярный эталон MRoPE — только для тестов быстрого пути (побитовая
/// сверка) и диагностики.
#[cfg(test)]
fn partial_interleaved_mrope_scalar(
    q: &Tensor, // (1, heads, seq, head_dim)
    k: &Tensor,
    positions: &[[u32; 3]],
    rope_theta: f32,
    rotary_dim: usize,
    sections: [usize; 3],
) -> candle_core::Result<(Tensor, Tensor)> {
    let (batch, _heads, seq, head_dim) = q.dims4()?;
    if batch != 1 || seq != positions.len() || head_dim < rotary_dim {
        return Err(candle_core::Error::Msg(format!(
            "partial MRoPE: q={q:?}, позиций {}, rotary_dim={rotary_dim}",
            positions.len()
        )));
    }
    let half = rotary_dim / 2;
    if half == 0 || half > 64 {
        return Err(candle_core::Error::Msg(format!(
            "partial MRoPE: неподдерживаемый rotary_dim {rotary_dim}"
        )));
    }
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    let mut cos_full = Vec::with_capacity(seq * rotary_dim);
    let mut sin_full = Vec::with_capacity(seq * rotary_dim);
    for pos in positions {
        let mut half_cos = [0f32; 64];
        let mut half_sin = [0f32; 64];
        for (freq_idx, inv) in inv_freq.iter().enumerate() {
            let axis = axis_of_freq(freq_idx, sections);
            let angle = pos[axis] as f32 * inv;
            half_cos[freq_idx] = angle.cos();
            half_sin[freq_idx] = angle.sin();
        }
        cos_full.extend_from_slice(&half_cos[..half]);
        cos_full.extend_from_slice(&half_cos[..half]);
        sin_full.extend_from_slice(&half_sin[..half]);
        sin_full.extend_from_slice(&half_sin[..half]);
    }
    let shape = (1, 1, seq, rotary_dim);
    let cos = Tensor::from_vec(cos_full, shape, q.device())?.to_dtype(q.dtype())?;
    let sin = Tensor::from_vec(sin_full, shape, q.device())?.to_dtype(q.dtype())?;
    Ok((rotate_partial(q, &cos, &sin, rotary_dim, head_dim)?, rotate_partial(k, &cos, &sin, rotary_dim, head_dim)?))
}

// =========================================================================
// Слои
// =========================================================================

struct Linear {
    w: Tensor, // [in, out]
}
impl Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.matmul(&self.w)
    }
}

/// Linear-attention слой (Gated DeltaNet) с 32 V-головами на 16 K-голов.
/// Четыре входные проекции (qkv/z/b/a) склеены на этапе загрузки в одну —
/// один matmul вместо четырёх запусков GPU на каждый токен; математика
/// идентична (линейность: разрез по столбцам = отдельные проекции).
struct HybridDeltaNet {
    in_proj: Linear, // [hidden, key*2 + value + v_heads*2]
    /// Ширины частей внутри склейки: qkv, z, b, a.
    qkv_dim: usize,
    value_dim: usize,
    v_heads: usize,
    conv_w: Tensor,      // (kernel, conv_dim), bf16
    neg_exp_a_log: Tensor, // (v_heads,), f32
    dt_bias: Tensor,       // (v_heads,), f32
    norm_w: Tensor,        // (head_v_dim,), f32
    out_proj: Linear,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    rms_norm_eps: f64,
}

impl HybridDeltaNet {
    fn forward(
        &self,
        x: &Tensor,        // (seq, hidden)
        conv: &mut Tensor, // (kernel-1, conv_dim)
        ssm: &mut Tensor,  // (v_heads, V, K), f32
    ) -> candle_core::Result<Tensor> {
        let (seq, _) = x.dims2()?;
        let key_dim = self.num_k_heads * self.head_k_dim;
        let value_dim = self.num_v_heads * self.head_v_dim;
        // Одна склеенная проекция → разрез на qkv | z | b | a.
        let proj = self.in_proj.forward(x)?; // (seq, qkv+value+2*v_heads)
        let qkv = proj.narrow(1, 0, self.qkv_dim)?;
        let z = proj.narrow(1, self.qkv_dim, self.value_dim)?;
        let b = proj.narrow(1, self.qkv_dim + self.value_dim, self.v_heads)?;
        let a = proj.narrow(1, self.qkv_dim + self.value_dim + self.v_heads, self.v_heads)?;

        let qkv = candle_nn::ops::silu(&causal_conv(&qkv.contiguous()?, &self.conv_w, conv)?)?;

        let q = qkv
            .narrow(1, 0, key_dim)?
            .reshape((seq, self.num_k_heads, self.head_k_dim))?;
        let k = qkv
            .narrow(1, key_dim, key_dim)?
            .reshape((seq, self.num_k_heads, self.head_k_dim))?;
        let v = qkv.narrow(1, 2 * key_dim, value_dim)?.reshape((
            seq,
            self.num_v_heads,
            self.head_v_dim,
        ))?;

        // Всё в f32; q/k расширяются до V-голов (repeat_interleave).
        let rep = self.num_v_heads / self.num_k_heads;
        let expand = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.to_dtype(DType::F32)?
                .reshape((seq, self.num_k_heads, 1, self.head_k_dim))?
                .broadcast_as((seq, self.num_k_heads, rep, self.head_k_dim))?
                .reshape((seq, self.num_v_heads, self.head_k_dim))?
                .contiguous()
        };
        let q = l2norm(&expand(&q)?)?;
        let k = l2norm(&expand(&k)?)?;
        let q_scale = (self.head_k_dim as f64).sqrt().recip();
        let q = q.affine(q_scale, 0.0)?;
        let v = v.to_dtype(DType::F32)?;

        let beta = candle_nn::ops::sigmoid(&b.to_dtype(DType::F32)?)?;
        let a = a.to_dtype(DType::F32)?;
        let a = a.broadcast_add(&self.dt_bias.reshape((1, self.num_v_heads))?)?;
        let g = softplus(&a)?.broadcast_mul(&self.neg_exp_a_log.reshape((1, self.num_v_heads))?)?;

        let o = gated_delta_rule(&q, &k, &v, &g, &beta, ssm)?;

        // Gated RMSNorm по V-головам: (seq*V, head_v_dim), gate из z (silu).
        let o = o.reshape((seq * self.num_v_heads, self.head_v_dim))?;
        let var = o.sqr()?.mean_keepdim(D::Minus1)?;
        let o = o
            .broadcast_mul(&var.affine(1.0, self.rms_norm_eps)?.sqrt()?.recip()?)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        let z = z.reshape((seq * self.num_v_heads, self.head_v_dim))?;
        let o = o.broadcast_mul(&self.norm_w.reshape((1, self.head_v_dim))?)?;
        let o = o.broadcast_mul(&candle_nn::ops::silu(&z.to_dtype(DType::F32)?)?)?;
        let o = o.to_dtype(DType::BF16)?.reshape((seq, value_dim))?;

        self.out_proj.forward(&o)
    }
}

/// Полный attention-слой: GQA + QK-norm + частичный MRoPE + sigmoid-gate.
/// Проекции q/k/v склеены на этапе загрузки (один matmul, разрез на месте).
struct HybridAttention {
    qkv: Linear, // [hidden, heads*hd*2 + kv*hd + kv*hd]
    q_dim: usize,   // heads * hd * 2 (значения + gate)
    kv_dim: usize,  // kv_heads * hd
    o: Linear,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    /// Общие таблицы MRoPE (одни на все attention-слои).
    rope_tables: std::sync::Arc<MropeTables>,
    eps: f64,
}

impl HybridAttention {
    fn forward(
        &self,
        x: &Tensor,
        positions: &[[u32; 3]],
        cache: &mut KvCache,
    ) -> candle_core::Result<Tensor> {
        let (seq, _hidden) = x.dims2()?;
        let hd = self.head_dim;
        // Склеенная проекция q|k|v → один matmul, разрез на месте.
        let proj = self.qkv.forward(x)?; // (seq, q_dim + 2*kv_dim)
        let q = proj
            .narrow(1, 0, self.q_dim)?
            .reshape((seq, self.num_heads, hd * 2))?;
        let gate = candle_nn::ops::sigmoid(&q.narrow(2, hd, hd)?)?;
        let q = q.narrow(2, 0, hd)?.contiguous()?;
        let k = proj
            .narrow(1, self.q_dim, self.kv_dim)?
            .reshape((seq, self.num_kv_heads, hd))?;
        let v = proj
            .narrow(1, self.q_dim + self.kv_dim, self.kv_dim)?
            .reshape((seq, self.num_kv_heads, hd))?;

        let q = rms_norm_zero_centered(&q, &self.q_norm_w, self.eps)?;
        let k = rms_norm_zero_centered(&k, &self.k_norm_w, self.eps)?;

        // (seq, heads, hd) → (1, heads, seq, hd) для MRoPE.
        let q = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        // Быстрый путь: cos/sin из предвычисленных таблиц (без powf/trig
        // и аллокаций на каждом шаге декодирования).
        let (cos, sin) = self
            .rope_tables
            .rows(positions, q.dtype())
            .map_err(|e| candle_core::Error::Msg(format!("MRoPE fast: {e:#}")))?;
        let q = rotate_partial(&q, &cos, &sin, self.rotary_dim, hd)?;
        let k = rotate_partial(&k, &cos, &sin, self.rotary_dim, hd)?;
        let q = q.squeeze(0)?; // (heads, seq, hd)
        let k = k.squeeze(0)?;

        // In-place append в предвыделенный буфер: k уже (kv_heads, seq, hd),
        // v разворачивается из (seq, kv_heads, hd). Возвращаются виды на
        // текущее содержимое — без копий истории. slice_set требует
        // непрерывные тензоры — страхуем явно (no-op, если уже непрерывный).
        let (k_all, v_all) = cache.append(&k.contiguous()?, &v.transpose(0, 1)?.contiguous()?)?;
        let total = k_all.dim(1)?;

        let rep = self.num_heads / self.num_kv_heads;
        // q в (heads, seq, hd): head-индекс = g*rep + r (repeat_interleave),
        // поэтому регруппировка идёт напрямую без транспонирования — лишний
        // transpose здесь перемешивал головы с позициями всего префилла.
        let q = if rep > 1 {
            q.reshape((self.num_kv_heads, rep, seq, hd))?
                .permute((0, 2, 1, 3))?
                .reshape((self.num_kv_heads, seq * rep, hd))?
                .contiguous()?
        } else {
            q
        };

        let scale = (hd as f64).sqrt().recip();
        let mut scores = q
            .matmul(&k_all.transpose(1, 2)?)?
            .affine(scale, 0.0)?
            .to_dtype(DType::F32)?;

        if seq > 1 {
            let start = total - seq;
            let mask = causal_mask(seq, total, start, x.device(), DType::F32)?;
            scores = scores
                .reshape((self.num_kv_heads, seq, rep, total))?
                .broadcast_add(&mask.unsqueeze(2)?)?
                .reshape((self.num_kv_heads, seq * rep, total))?;
        }

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(x.dtype())?;
        let out = attn.matmul(&v_all)?;
        let out = out
            .reshape((self.num_kv_heads, seq, rep, hd))?
            .permute((1, 0, 2, 3))?
            .reshape((seq, self.num_heads * hd))?;

        let out = out.broadcast_mul(&gate.reshape((seq, self.num_heads * hd))?)?;
        self.o.forward(&out.contiguous()?)
    }
}

fn causal_mask(
    seq: usize,
    total: usize,
    start: usize,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    let data: Vec<f32> = (0..seq)
        .flat_map(|i| {
            (0..total).map(move |j| if j <= start + i { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Tensor::from_vec(data, (seq, total), device)?.unsqueeze(0)?.to_dtype(dtype)
}

struct Mlp {
    /// gate|up склеены на этапе загрузки — один matmul вместо двух.
    gate_up: Linear,
    intermediate: usize,
    down: Linear,
}
impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gu = self.gate_up.forward(x)?; // (seq, 2*intermediate)
        let gate = gu.narrow(1, 0, self.intermediate)?;
        let up = gu.narrow(1, self.intermediate, self.intermediate)?;
        self.down.forward(&candle_nn::ops::silu(&gate)?.broadcast_mul(&up)?)
    }
}

enum LayerKind {
    Linear(HybridDeltaNet),
    Full(HybridAttention),
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
        positions: &[[u32; 3]],
        cache: &mut HybridLayerCache,
    ) -> candle_core::Result<Tensor> {
        let normed = rms_norm_zero_centered(x, &self.input_norm_w, self.eps)?;
        let attn_out = match (&self.kind, cache) {
            (LayerKind::Linear(dn), HybridLayerCache::Linear { conv, ssm }) => {
                dn.forward(&normed, conv, ssm)?
            }
            (LayerKind::Full(attn), HybridLayerCache::Full(kv)) => attn.forward(&normed, positions, kv)?,
            _ => unreachable!("тип слоя и кэш не согласованы"),
        };
        let x = (x + attn_out)?;
        let normed = rms_norm_zero_centered(&x, &self.post_norm_w, self.eps)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x + mlp_out
    }
}

// =========================================================================
// Кэш и снапшоты
// =========================================================================

#[derive(Clone)]
pub enum HybridLayerCache {
    Linear {
        conv: Tensor, // (kernel-1, conv_dim)
        ssm: Tensor,  // (v_heads, V, K), f32
    },
    Full(KvCache),
}

/// KV-кэш полного слоя: candle-кэш с **предвыделенным буфером** и in-place
/// `slice_set`-append — вместо конкатенации всей истории на каждом шаге
/// (O(n²) трафик памяти заменён на O(n)). Раскладка (kv_heads, seq, hd),
/// ось последовательности = 1.
pub type KvCache = candle_nn::kv_cache::KvCache;

/// Потолок prealloc KV-кэша; дальше candle дорастает автоматически.
const KV_CACHE_MAX_SEQ: usize = 8192;

/// Полный кэш модели + длина контекста.
pub struct HybridCache {
    pub layers: Vec<HybridLayerCache>,
    pub seq_len: usize,
}

// =========================================================================
// Декодер
// =========================================================================

pub struct HybridDecoder {
    embed: Tensor,   // (vocab, hidden)
    lm_head: Tensor, // (hidden, vocab)
    layers: Vec<Block>,
    norm_w: Tensor,
    cfg: HybridConfig,
    device: Device,
}

impl HybridDecoder {
    pub fn config(&self) -> &HybridConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        let ids_t = Tensor::from_slice(ids, (ids.len(),), &self.device)?;
        Ok(self.embed.index_select(&ids_t, 0)?)
    }

    pub fn new_cache(&self) -> HybridCache {
        let t = &self.cfg.text;
        // conv_dim = key_dim*2 + value_dim (как у in_proj_qkv).
        let conv_dim = 2 * t.linear_num_key_heads * t.linear_key_head_dim
            + t.linear_num_value_heads * t.linear_value_head_dim;
        let layers = self
            .layers
            .iter()
            .map(|block| match &block.kind {
                LayerKind::Linear(_) => HybridLayerCache::Linear {
                    conv: Tensor::zeros((t.conv_kernel_size - 1, conv_dim), DType::BF16, &self.device)
                        .unwrap(),
                    ssm: Tensor::zeros(
                        (
                            t.linear_num_value_heads,
                            t.linear_value_head_dim,
                            t.linear_key_head_dim,
                        ),
                        DType::F32,
                        &self.device,
                    )
                    .unwrap(),
                },
                LayerKind::Full(_) => HybridLayerCache::Full(KvCache::new(1, KV_CACHE_MAX_SEQ)),
            })
            .collect();
        HybridCache { layers, seq_len: 0 }
    }

    /// Прямой проход по готовым эмбеддингам. Возвращает финальные скрытые
    /// состояния `(seq, hidden)`; логиты — через `forward`/`forward_last_logits`.
    pub fn forward_hidden(
        &self,
        inputs_embeds: &Tensor,
        positions: &[[u32; 3]],
        cache: &mut HybridCache,
    ) -> Result<Tensor> {
        self.forward_hidden_inspected(inputs_embeds, positions, cache, &mut |_, _| {})
    }

    /// Как [`forward_hidden`](Self::forward_hidden), но вызывает `inspect`
    /// после каждого слоя — для послойной сверки с эталонной реализацией.
    pub fn forward_hidden_inspected(
        &self,
        inputs_embeds: &Tensor,
        positions: &[[u32; 3]],
        cache: &mut HybridCache,
        inspect: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Tensor> {
        let seq = inputs_embeds.dim(0)?;
        if positions.len() != seq {
            bail!("positions ({}) != seq ({seq})", positions.len());
        }
        if cache.layers.len() != self.layers.len() {
            bail!("кэш {} слоёв не совпадает с моделью {}", cache.layers.len(), self.layers.len());
        }
        if seq == 0 {
            bail!("Пустой вход decoder");
        }
        if cache.seq_len + seq > self.cfg.text.max_position_embeddings {
            bail!(
                "Контекст {}+{} превышает max_position_embeddings {}",
                cache.seq_len,
                seq,
                self.cfg.text.max_position_embeddings
            );
        }
        let mut x = inputs_embeds.contiguous()?;
        for (i, block) in self.layers.iter().enumerate() {
            x = block.forward(&x, positions, &mut cache.layers[i])?;
            inspect(i, &x);
        }
        cache.seq_len += seq;
        let out = rms_norm_zero_centered(&x, &self.norm_w, self.cfg.text.rms_norm_eps)?;
        inspect(usize::MAX - 1, &out);
        Ok(out)
    }

    /// Логиты всех позиций `(seq, vocab)` — шаг verification спекулятивного
    /// декодирования.
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        positions: &[[u32; 3]],
        cache: &mut HybridCache,
    ) -> Result<Tensor> {
        let x = self.forward_hidden(inputs_embeds, positions, cache)?;
        Ok(x.matmul(&self.lm_head)?)
    }

    /// Логиты последней позиции `(vocab,)`.
    pub fn forward_last_logits(
        &self,
        inputs_embeds: &Tensor,
        positions: &[[u32; 3]],
        cache: &mut HybridCache,
    ) -> Result<Tensor> {
        let x = self.forward_hidden(inputs_embeds, positions, cache)?;
        let last = x.narrow(0, x.dim(0)? - 1, 1)?;
        Ok(last.matmul(&self.lm_head)?.reshape((self.cfg.text.vocab_size,))?)
    }

    /// Загрузка из каталога чекпойнта (шарды `model*.safetensors`, BF16).
    /// VarBuilder открывается на CPU поверх mmap: транспонирования делаются на
    /// CPU, и на GPU уходит ровно одна копия каждого веса (двойная копия
    /// переполняет Metal-бюджет).
    pub fn load(dir: &Path, cfg: &HybridConfig, device: &Device) -> Result<Self> {
        let weights = super::checkpoint::weight_files(dir)?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&weights, cfg.weight_dtype(), &Device::Cpu)
                .context("Не удалось открыть safetensors-веса Qwen3.5-VL")?
        };
        Self::build(&vb, cfg, dir, device)
    }

    fn build(
        vb: &candle_nn::VarBuilder,
        cfg: &HybridConfig,
        dir: &Path,
        device: &Device,
    ) -> Result<Self> {
        let t = &cfg.text;
        let hidden = t.hidden_size;
        let q_dim = t.num_attention_heads * t.head_dim;
        let kv_dim = t.num_key_value_heads * t.head_dim;
        let key_dim = t.linear_num_key_heads * t.linear_key_head_dim;
        let value_dim = t.linear_num_value_heads * t.linear_value_head_dim;

        // CPU VarBuilder: get() — zero-copy view поверх mmap; на GPU уходит
        // только финальный (транспонированный) вес.
        let up = |w: Tensor, device: &Device| -> Result<Tensor> { Ok(w.to_device(device)?) };
        // Гейты DeltaNet (A_log, dt_bias, norm) в старых чекпойнтах хранятся
        // f32 — читаем их F32-билдером без даунгрейда в bf16; для all-BF16
        // чекпойнтов (Holo) это точный upcast.
        let vb_f32 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &super::checkpoint::weight_files(vb_path(dir))?,
                DType::F32,
                &Device::Cpu,
            )
            .context("Не удалось открыть f32-слой весов Qwen3.5-VL")?
        };

        let embed = up(vb.get((t.vocab_size, hidden), "model.language_model.embed_tokens.weight")?, device)?;
        // lm_head: в чекпойнте Holo он есть, даже при tie_word_embeddings=true;
        // при отсутствии — транспонированные эмбеддинги.
        let lm_head = match vb.get((t.vocab_size, hidden), "lm_head.weight") {
            Ok(w) => up(w.t()?.contiguous()?, device)?,
            Err(_) if cfg.tie_word_embeddings => embed.t()?.contiguous()?,
            Err(e) => return Err(e.into()),
        };

        let lin = |name: String, out: usize, inn: usize| -> Result<Linear> {
            let w = vb.get((out, inn), &name)?.t()?.contiguous()?;
            Ok(Linear { w: up(w, device)? })
        };

        // Таблицы MRoPE — одни на все attention-слои; потолок как у таблиц
        // RoPE остальной системы.
        const MROPE_TABLE_MAX_SEQ: usize = 8192;
        let rope_max_seq = t
            .max_position_embeddings
            .min(MROPE_TABLE_MAX_SEQ);
        let rope_tables = std::sync::Arc::new(MropeTables::build(
            rope_max_seq,
            cfg.rotary_dim(),
            t.mrope_sections,
            t.rope_theta,
            device,
        )?);

        // Слои грузятся параллельно с ОГРАНИЧЕННОЙ очередью: транспонирование
        // на CPU — самый долгий этап, но заливка на GPU из десятков потоков
        // конкурирует за очередь Metal и замедляет всё (90 с против 26 с).
        // 4 воркера делят CPU-работу, не забивая GPU-канал.
        let loaded: std::sync::Mutex<Vec<(usize, Result<Block>)>> = std::sync::Mutex::new(Vec::new());
        let next_index = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| loop {
                    let i = next_index.fetch_add(1, Ordering::Relaxed);
                    if i >= t.num_hidden_layers {
                        break;
                    }
                    let result = load_layer(
                        &vb,
                        &vb_f32,
                        &cfg,
                        &dir,
                        i,
                        &device,
                        std::sync::Arc::clone(&rope_tables),
                    );
                    loaded
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push((i, result));
                });
            }
        });
        let mut slots: Vec<Option<Result<Block>>> = (0..t.num_hidden_layers).map(|_| None).collect();
        for (i, result) in loaded.into_inner().unwrap_or_else(|e| e.into_inner()) {
            slots[i] = Some(result);
        }
        let layers = slots
            .into_iter()
            .map(|slot| slot.expect("слой не загрузился"))
            .collect::<Result<Vec<_>>>()?;

        let norm_w = vb.get(hidden, "model.language_model.norm.weight")?.to_device(device)?;
        Ok(Self { embed, lm_head, layers, norm_w, cfg: cfg.clone(), device: device.clone() })
    }
}

/// Загружает один слой: транспонирование на CPU, склейка проекций
/// (qkv|z|b|a, q|k|v, gate|up — один matmul вместо трёх-четырёх запусков
/// GPU на токен), заливка на GPU.
#[allow(clippy::too_many_arguments)]
fn load_layer(
    vb: &candle_nn::VarBuilder,
    vb_f32: &candle_nn::VarBuilder,
    cfg: &HybridConfig,
    dir: &Path,
    i: usize,
    device: &Device,
    rope_tables: std::sync::Arc<MropeTables>,
) -> Result<Block> {
    let t = &cfg.text;
    let hidden = t.hidden_size;
    let q_dim = t.num_attention_heads * t.head_dim;
    let kv_dim = t.num_key_value_heads * t.head_dim;
    let key_dim = t.linear_num_key_heads * t.linear_key_head_dim;
    let value_dim = t.linear_num_value_heads * t.linear_value_head_dim;
    let inter = t.intermediate_size;

    let lin = |name: String, out: usize, inn: usize| -> Result<Linear> {
        let w = vb.get((out, inn), &name)?.t()?.contiguous()?;
        Ok(Linear { w: w.to_device(device)? })
    };
    // Склейка нескольких [out, hidden] тензоров в один Linear ([hidden, sum]).
    let merged = |parts: &[Tensor]| -> Result<Linear> {
        let cat = Tensor::cat(parts, 0)?;
        Ok(Linear { w: cat.t()?.contiguous()?.to_device(device)? })
    };
    let gate = |name: &str, shape: (usize,)| -> Result<Tensor> {
        let shape = [shape.0];
        match vb_f32.get(&shape, name) {
            Ok(w) => Ok(w.to_device(device)?),
            // all-BF16 чекпойнт: точный upcast из bf16.
            Err(_) => Ok(vb.get(&shape, name)?.to_dtype(DType::F32)?.to_device(device)?),
        }
    };
    let p = |s: &str| format!("model.language_model.layers.{i}.{s}");

    let input_norm_w = vb.get(hidden, &p("input_layernorm.weight"))?.to_device(device)?;
    let post_norm_w = vb.get(hidden, &p("post_attention_layernorm.weight"))?.to_device(device)?;

    let kind = match t.layer_types[i] {
        HybridLayerType::LinearAttention => {
            let conv_w = vb
                .get(
                    (key_dim * 2 + value_dim, 1, t.conv_kernel_size),
                    &p("linear_attn.conv1d.weight"),
                )?
                .squeeze(1)?
                .t()?
                .contiguous()?
                .to_device(device)?; // (kernel, conv_dim)
            let neg_exp_a_log = gate(&p("linear_attn.A_log"), (t.linear_num_value_heads,))?
                .exp()?
                .affine(-1.0, 0.0)?;
            let dt_bias = gate(&p("linear_attn.dt_bias"), (t.linear_num_value_heads,))?;
            let norm_w = gate(&p("linear_attn.norm.weight"), (t.linear_value_head_dim,))?;
            LayerKind::Linear(HybridDeltaNet {
                in_proj: merged(&[
                    vb.get((key_dim * 2 + value_dim, hidden), &p("linear_attn.in_proj_qkv.weight"))?,
                    vb.get((value_dim, hidden), &p("linear_attn.in_proj_z.weight"))?,
                    vb.get((t.linear_num_value_heads, hidden), &p("linear_attn.in_proj_b.weight"))?,
                    vb.get((t.linear_num_value_heads, hidden), &p("linear_attn.in_proj_a.weight"))?,
                ])?,
                qkv_dim: key_dim * 2 + value_dim,
                value_dim,
                v_heads: t.linear_num_value_heads,
                conv_w,
                neg_exp_a_log,
                dt_bias,
                norm_w,
                out_proj: lin(p("linear_attn.out_proj.weight"), hidden, value_dim)?,
                num_k_heads: t.linear_num_key_heads,
                num_v_heads: t.linear_num_value_heads,
                head_k_dim: t.linear_key_head_dim,
                head_v_dim: t.linear_value_head_dim,
                rms_norm_eps: t.rms_norm_eps,
            })
        }
        HybridLayerType::FullAttention => {
            let hd = t.head_dim;
            LayerKind::Full(HybridAttention {
                qkv: merged(&[
                    vb.get((t.num_attention_heads * hd * 2, hidden), &p("self_attn.q_proj.weight"))?,
                    vb.get((kv_dim, hidden), &p("self_attn.k_proj.weight"))?,
                    vb.get((kv_dim, hidden), &p("self_attn.v_proj.weight"))?,
                ])?,
                q_dim: t.num_attention_heads * hd * 2,
                kv_dim,
                o: lin(p("self_attn.o_proj.weight"), hidden, q_dim)?,
                q_norm_w: vb.get(hd, &p("self_attn.q_norm.weight"))?.to_device(device)?,
                k_norm_w: vb.get(hd, &p("self_attn.k_norm.weight"))?.to_device(device)?,
                num_heads: t.num_attention_heads,
                num_kv_heads: t.num_key_value_heads,
                head_dim: hd,
                rotary_dim: cfg.rotary_dim(),
                rope_tables,
                eps: t.rms_norm_eps,
            })
        }
    };

    let mlp = Mlp {
        gate_up: merged(&[
            vb.get((inter, hidden), &p("mlp.gate_proj.weight"))?,
            vb.get((inter, hidden), &p("mlp.up_proj.weight"))?,
        ])?,
        intermediate: inter,
        down: lin(p("mlp.down_proj.weight"), hidden, inter)?,
    };

    Ok(Block { input_norm_w, post_norm_w, kind, mlp, eps: t.rms_norm_eps })
}

// =========================================================================
// Модель: StepModel + фьюжн изображений
// =========================================================================

/// Загруженная гибридная модель: декодер + визионная башня + процессор +
/// токенизатор. Реализует [`super::runtime::StepModel`].
pub struct HybridModel {
    decoder: HybridDecoder,
    vision: super::vision::Qwen3VlVision,
    processor: super::processor::UiVenusProcessor,
    tokenizer: Tokenizer,
    config: HybridConfig,
    device: Device,
    cache: Option<HybridCache>,
    /// Абсолютная MRoPE-позиция следующего токена.
    next_position: usize,
    pending_prompt: Option<PreparedPrompt>,
}

impl HybridModel {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let config = HybridConfig::load(dir)?;
        super::checkpoint::weight_files(dir)?;
        let decoder = HybridDecoder::load(dir, &config, &device)?;
        // Визионная башня qwen3_5_vl: та же геометрия, что у Holo-3.1
        // (depth 24, hidden 1024, out = text hidden, deepstack пуст).
        let vision = super::vision::Qwen3VlVision::load_hybrid(dir, &config, &device)?;
        let processor = super::processor::UiVenusProcessor::load(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить tokenizer.json: {e}"))?;
        // Коммит + сметание пула: загрузка оставляет в пуле пре-транспонс-
        // оригиналы весов (~10 ГБ); без сметания первый forward падает по
        // Metal-памяти (ResidencySet держит всё резидентным).
        device.synchronize().map_err(|e| anyhow::anyhow!("Metal sync после загрузки: {e}"))?;
        Ok(Self {
            decoder,
            vision,
            processor,
            tokenizer,
            config,
            device,
            cache: None,
            next_position: 0,
            pending_prompt: None,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn processor(&self) -> &super::processor::UiVenusProcessor {
        &self.processor
    }

    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Визионный проход: подготовленный промпт → (визуальные токены, []).
    fn encode_image(
        &self,
        prompt: &PreparedPrompt,
    ) -> Result<Tensor> {
        let dtype = self.config.weight_dtype();
        let pixels = prompt.image.pixel_values_tensor(&self.device, dtype)?;
        let (visual, deepstack) = self.vision.forward(&pixels, prompt.image.grid_thw)?;
        if !deepstack.is_empty() {
            bail!("qwen3_5 не использует DeepStack, но башня вернула фичи");
        }
        Ok(visual)
    }

    /// Фьюжн: визуальные токены подставляются в строки эмбеддингов.
    fn fuse_embeds(&self, prompt: &PreparedPrompt) -> Result<(Tensor, Vec<usize>)> {
        let text_emb = self.decoder.embed_tokens(&prompt.input_ids)?;
        let visual = self.encode_image(prompt)?;
        let (start, end) = prompt.image_span;
        let hidden = text_emb.dim(1)?;
        if end - start != visual.dim(0)? {
            bail!(
                "Разъезд чисел: image_span {} vs визуальных токенов {}",
                end - start,
                visual.dim(0)?
            );
        }
        let mut rows = text_emb.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let vrows = visual.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (row, target) in vrows.iter().zip(start..end) {
            if row.len() != hidden {
                bail!("Ширина визуального токена {} != hidden {hidden}", row.len());
            }
            rows[target] = row.clone();
        }
        let merged = Tensor::from_vec(
            rows.into_iter().flatten().collect::<Vec<f32>>(),
            (prompt.input_ids.len(), hidden),
            &self.device,
        )?
        .to_dtype(self.config.weight_dtype())?;
        Ok((merged, (start..end).collect()))
    }

    /// Ставит мультимодальный промпт в очередь на следующий prefill.
    pub fn queue_prompt(&mut self, prompt: PreparedPrompt) {
        self.pending_prompt = Some(prompt);
    }

    /// Полная мультимодальная генерация (скриншот + задача).
    pub fn generate_for_prompt(
        &mut self,
        prompt: &PreparedPrompt,
        params: &super::runtime::SamplingParams,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        sink: impl FnMut(&super::runtime::StreamEvent) -> bool,
    ) -> Result<super::runtime::GenerationOutput> {
        self.pending_prompt = Some(prompt.clone());
        let ids = prompt.input_ids.clone();
        super::runtime::generate(self, &ids, params, cancel, sink)
    }
}

impl super::runtime::StepModel for HybridModel {
    fn begin_sequence(&mut self, prompt_len: usize) {
        self.cache = Some(self.decoder.new_cache());
        self.next_position = prompt_len;
    }

    fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Tensor> {
        let prepared = self.pending_prompt.take();
        let (embeds, positions, next_pos) = if let Some(pp) = prepared.as_ref() {
            if pp.input_ids.as_slice() != prompt_ids {
                bail!("prefill: id промпта не совпадают с подготовленным мультимодальным промптом");
            }
            let (embeds, _image_positions) = self.fuse_embeds(pp)?;
            // MRoPE-позиции подготовлены процессором; следующая текстовая
            // позиция — за максимумом осей.
            let next = pp
                .mrope_position_ids
                .iter()
                .flat_map(|p| p.iter().copied())
                .max()
                .map(|m| m as usize + 1)
                .unwrap_or(pp.input_ids.len());
            (embeds, pp.mrope_position_ids.clone(), next)
        } else {
            let embeds = self.decoder.embed_tokens(prompt_ids)?;
            let positions: Vec<[u32; 3]> =
                (0..prompt_ids.len()).map(|i| [i as u32; 3]).collect();
            (embeds, positions, prompt_ids.len())
        };

        // Prefill чанками: гигантский command buffer на весь промпт (24
        // DeltaNet-слоя × десятки токенов мелких операций) переполняет Metal-
        // буфер (kIOGPUCommandBufferCallbackErrorOutOfMemory). Математика
        // идентична: рекуррентные слои всё равно токен-за-токеном, полным
        // слоям каузальность обеспечивает маска с абсолютным start.
        // 32 — баланс между числом синков и размером буфера.
        const PREFILL_CHUNK: usize = 32;
        let cache = self.cache.as_mut().context("begin_sequence не вызван перед prefill")?;
        let mut logits = None;
        for (chunk_start, positions_chunk) in positions.chunks(PREFILL_CHUNK).enumerate() {
            let seq = positions_chunk.len();
            let embeds_chunk = embeds.narrow(0, chunk_start * PREFILL_CHUNK, seq)?;
            let last = self
                .decoder
                .forward_last_logits(&embeds_chunk, positions_chunk, cache)?;
            logits = Some(last);
            // Коммит command buffer: без него все операции всего префилла
            // копятся в одном буфере, и Metal падает по памяти.
            self.device.synchronize().map_err(|e| anyhow::anyhow!("Metal sync: {e}"))?;
            #[cfg(feature = "metal")]
            if std::env::var("JARVIS_HOLO_DEBUG").is_ok() {
                eprintln!("prefill chunk {chunk_start} ok");
            }
        }
        self.next_position = next_pos;
        Ok(logits.context("prefill: пустой промпт")?)
    }

    fn step(&mut self, token: u32) -> Result<Tensor> {
        let embeds = self.decoder.embed_tokens(&[token])?;
        let positions = [[self.next_position as u32; 3]];
        self.next_position += 1;
        let cache = self.cache.as_mut().context("begin_sequence не вызван перед step")?;
        self.decoder.forward_last_logits(&embeds, &positions, cache)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Декодирование токенов: {e}"))?)
    }

    fn eos_ids(&self) -> Vec<u32> {
        self.config.text.eos_ids.clone()
    }

}

#[cfg(test)]
mod mrope_fast_tests {
    use super::*;

    /// Быстрый путь (таблицы) обязан совпадать со скалярным эталоном
    /// побитово: смешивание — выбор строки своей оси весами 1/0.
    #[test]
    fn mrope_fast_matches_scalar_exactly() {
        let dev = Device::Cpu;
        let (heads, seq, hd, rotary) = (4usize, 9usize, 64usize, 32usize);
        let sections = [6usize, 5usize, 5usize]; // сумма 16 = rotary/2
        let theta = 1_000_000.0f32;
        let tables = MropeTables::build(64, rotary, sections, theta, &dev).unwrap();

        // Текстовые позиции [p, p, p].
        let text: Vec<[u32; 3]> = (0..seq as u32).map(|p| [p, p, p]).collect();
        // Image-подобные: оси расходятся, как у визуальных токенов.
        let image: Vec<[u32; 3]> = (0..seq as u32)
            .map(|i| [i / 3, i / 2, i % 5])
            .collect();

        for positions in [text, image] {
            let q = Tensor::randn(0f32, 1.0, (1, heads, seq, hd), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, heads, seq, hd), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let (q_fast, k_fast) = {
                let (cos, sin) = tables.rows(&positions, q.dtype()).unwrap();
                (
                    rotate_partial(&q, &cos, &sin, rotary, hd).unwrap(),
                    rotate_partial(&k, &cos, &sin, rotary, hd).unwrap(),
                )
            };
            let (q_ref, k_ref) =
                partial_interleaved_mrope_scalar(&q, &k, &positions, theta, rotary, sections).unwrap();
            for (fast, reference) in [(q_fast, q_ref), (k_fast, k_ref)] {
                let diff = (&fast.to_dtype(DType::F32).unwrap()
                    - &reference.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
                assert_eq!(diff, 0.0, "быстрый MRoPE разошёлся со скалярным");
            }
        }
    }

    /// KV-кэш: decode-шаги и prefill-чанки дают тот же результат, что и
    /// конкатенация полной истории (числа совпадают, длина растёт).
    #[test]
    fn kv_cache_inplace_append_matches_history() {
        let dev = Device::Cpu;
        let (kv_heads, hd) = (2usize, 8usize);
        let mut cache = KvCache::new(1, KV_CACHE_MAX_SEQ);
        let mut history_k = Vec::new();
        let mut history_v = Vec::new();
        // Prefill-чанк 4 токена, затем 3 decode-шага.
        for seq in [4usize, 1, 1, 1] {
            let k = Tensor::randn(0f32, 1.0, (kv_heads, seq, hd), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let v = Tensor::randn(0f32, 1.0, (kv_heads, seq, hd), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let (k_all, v_all) = cache.append(&k, &v).unwrap();
            history_k.push(k);
            history_v.push(v);
            let k_expected = Tensor::cat(&history_k.iter().map(|t| t.to_owned()).collect::<Vec<_>>(), 1).unwrap();
            let v_expected = Tensor::cat(&history_v.iter().map(|t| t.to_owned()).collect::<Vec<_>>(), 1).unwrap();
            let kd = (&k_all.to_dtype(DType::F32).unwrap() - &k_expected.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap();
            let vd = (&v_all.to_dtype(DType::F32).unwrap() - &v_expected.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap();
            assert_eq!(kd.to_scalar::<f32>().unwrap(), 0.0);
            assert_eq!(vd.to_scalar::<f32>().unwrap(), 0.0);
        }
        assert_eq!(cache.current_seq_len(), 7);
    }
}
