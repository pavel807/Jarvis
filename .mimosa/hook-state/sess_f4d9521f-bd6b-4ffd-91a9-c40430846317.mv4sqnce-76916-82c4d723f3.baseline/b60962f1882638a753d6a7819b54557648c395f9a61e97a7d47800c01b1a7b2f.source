//! Реализация Llama-совместимой архитектуры (MiniCPM5-1B-ru-v3) на candle.
//!
//! MiniCPM5-1B — это `LlamaForCausalLM`: GQA (16 Q-голов, 2 KV-головы),
//! RoPE с θ=5e6, RMSNorm (ε=1e-6), SwiGLU (SiLU), явный `head_dim=128`.
//!
//! Отличия от Qwen3-1.7B (предыдущая модель):
//! - **нет QK-нормализации** — `q_norm_w`/`k_norm_w` остаются `None`;
//! - `hidden_size = 1536` (было 2048), `num_hidden_layers = 24` (было 28);
//! - `rope_theta = 5e6` (было 1e6);
//! - Chat-шаблон — Llama (`<|start_header_id|>`), а не ChatML.
//!
//! Веса — BF16 safetensors из `models/minicpm5-1b-ru/`. Формат только
//! safetensors: GGUF больше не поддерживается (MiniCPM5-1B-ru-v3
//! распространяется в BF16, GGUF лежит только у автора русской версии и
//! для candle не нужен).

use super::config::ModelConfig;
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, IndexOp, Result as CResult, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const RMS_EPS_FALLBACK: f64 = 1e-6;

/// Потолок таблиц RoPE. `max_position_embeddings` у MiniCPM5 — 131072;
/// держать таблицу такого размера незачем, промпт с ответом заведомо
/// укладываются в 8k токенов.
const ROPE_TABLE_MAX_SEQ: usize = 8192;

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> CResult<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps as f32)
}

// =========================================================================
// Линейный слой
// =========================================================================

/// Матрица линейного слоя без bias.
///
/// В MiniCPM5-1B (как и в Llama) все линейные слои — без bias. Веса в
/// safetensors хранятся как `[out, in]`, для `matmul` нужен `[in, out]`.
struct Linear {
    w: Tensor,
}

impl Linear {
    fn load(vb: &VarBuilder, out: usize, inn: usize, name: &str) -> CResult<Self> {
        let w = vb.get((out, inn), name)?;
        Ok(Self {
            w: w.t()?.contiguous()?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        x.matmul(&self.w)
    }
}

// =========================================================================
// Эмбеддинги
// =========================================================================

struct Embedding {
    t: Tensor,
}

impl Embedding {
    fn forward(&self, ids: &Tensor) -> CResult<Tensor> {
        self.t.index_select(ids, 0)
    }
}

// =========================================================================
// RoPE
// =========================================================================

/// Пре-вычисленные таблицы RoPE: cos/sin всех позиций до `max_seq`.
pub struct RopeTables {
    cos: Tensor,
    sin: Tensor,
}

fn build_rope_tables(
    max_seq: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> CResult<RopeTables> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / theta.powf((2 * i) as f32 / head_dim as f32))
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

/// RoPE (rotate_half) по готовым таблицам: позиции
/// `start_pos..start_pos+seq` применяются к тензору `(seq, heads, head_dim)`.
fn rope_cached(x: &Tensor, tables: &RopeTables, start_pos: usize) -> CResult<Tensor> {
    let (seq, _heads, hd) = x.dims3()?;
    let half = hd / 2;

    let cos = tables
        .cos
        .narrow(0, start_pos, seq)?
        .unsqueeze(1)?
        .to_dtype(x.dtype())?;
    let sin = tables
        .sin
        .narrow(0, start_pos, seq)?
        .unsqueeze(1)?
        .to_dtype(x.dtype())?;

    let x1 = x.i((.., .., 0..half))?;
    let x2 = x.i((.., .., half..))?;

    let out1 = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out2 = x2
        .broadcast_mul(&cos)?
        .broadcast_add(&x1.broadcast_mul(&sin)?)?;

    Tensor::cat(&[out1, out2], D::Minus1)
}

// =========================================================================
// KV-кэш
// =========================================================================

/// KV-кэш одного слоя в раскладке `(kv_heads, total_seq, head_dim)`.
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> CResult<()> {
        let k = k.transpose(0, 1)?.contiguous()?;
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

    fn kv(&self) -> CResult<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Ok((k.clone(), v.clone())),
            _ => Err(candle_core::Error::msg("KV-кэш пуст: append не был вызван")),
        }
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Каузальная маска
// =========================================================================

/// Кеш каузальных масок. При префилле каждый forward строит матрицу
/// `seq × total` — заметная аллокация. Для одного диалога набор `(seq, total)`
/// мал и переиспользуется между запросами.
static MASK_CACHE: OnceLock<Mutex<HashMap<(usize, usize), Tensor>>> = OnceLock::new();

fn causal_mask(
    seq: usize,
    total: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> CResult<Tensor> {
    // Кешируем только полный префилл (`start_pos = 0`): на декоде (`seq=1`)
    // маска тривиальна и строится быстрее, чем ищется в кеше.
    if start_pos == 0 && seq > 1 {
        let cache = MASK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(m) = guard.get(&(seq, total)) {
            return Ok(m.clone());
        }

        let mask = build_causal_mask(seq, total, start_pos, device, dtype)?;
        if guard.len() > 64 {
            guard.clear();
        }
        guard.insert((seq, total), mask.clone());
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
) -> CResult<Tensor> {
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
// Attention
// =========================================================================

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// QK-нормализация в MiniCPM5-1B **отсутствует** — поля `None`.
    /// Оставлены для совместимости с Qwen3-подобными моделями: если
    /// в safetensors есть `q_norm`/`k_norm`, они загрузятся и применятся.
    q_norm_w: Option<Tensor>,
    k_norm_w: Option<Tensor>,
    eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope: &RopeTables,
        cache: &mut KvCache,
    ) -> CResult<Tensor> {
        let (seq, _hidden) = x.dims2()?;
        let hd = self.head_dim;

        let q = self.q.forward(x)?.reshape((seq, self.num_heads, hd))?;
        let k = self.k.forward(x)?.reshape((seq, self.num_kv_heads, hd))?;
        let v = self.v.forward(x)?.reshape((seq, self.num_kv_heads, hd))?;

        // QK-нормализация: у MiniCPM5 весов нет → ветки пропускаются.
        let q = match &self.q_norm_w {
            Some(w) => rms_norm(&q, w, self.eps)?,
            None => q,
        };
        let k = match &self.k_norm_w {
            Some(w) => rms_norm(&k, w, self.eps)?,
            None => k,
        };

        let q = rope_cached(&q, rope, start_pos)?;
        let k = rope_cached(&k, rope, start_pos)?;

        cache.append(&k, &v)?;
        let (k, v) = cache.kv()?;
        let total = k.dim(1)?;

        // GQA без копий K/V: q-головы сворачиваем в батч по kv-головам.
        // K/V остаются `(nkv, total, hd)` как лежат в кэше.
        let rep = self.num_heads / self.num_kv_heads;

        // q: `(seq, nh, hd)` → `(nkv, seq*rep, hd)`
        let q = q.transpose(0, 1)?.contiguous()?;
        let q = if rep > 1 {
            q.reshape((self.num_kv_heads, rep, seq, hd))?
                .permute((0, 2, 1, 3))?
                .reshape((self.num_kv_heads, seq * rep, hd))?
                .contiguous()?
        } else {
            q
        };

        // `(nkv, seq*rep, hd) × (nkv, hd, total)` → `(nkv, seq*rep, total)`
        let mut scores = (q.matmul(&k.transpose(1, 2)?)? * (1.0 / (hd as f64).sqrt()))?;

        // Маска в раскладке групп: `(nkv, seq, rep, total)` +
        // `(1, seq, 1, total)`.
        if seq > 1 {
            let mask = causal_mask(seq, total, start_pos, x.device(), scores.dtype())?;
            scores = scores
                .reshape((self.num_kv_heads, seq, rep, total))?
                .broadcast_add(&mask.unsqueeze(2)?)?
                .reshape((self.num_kv_heads, seq * rep, total))?;
        }

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        // `(nkv, seq*rep, total) × (nkv, total, hd)` → `(nkv, seq*rep, hd)`
        let out = attn.matmul(&v)?;
        // `(nkv, seq*rep, hd)` → `(seq, nh*hd)`
        let out = out
            .reshape((self.num_kv_heads, seq, rep, hd))?
            .permute((1, 0, 2, 3))?
            .reshape((seq, self.num_heads * hd))?;
        self.o.forward(&out)
    }
}

// =========================================================================
// MLP (SwiGLU)
// =========================================================================

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let activated = (candle_nn::ops::silu(&gate)? * up)?;
        self.down.forward(&activated)
    }
}

// =========================================================================
// Трансформерный блок
// =========================================================================

struct Block {
    input_norm_w: Tensor,
    post_norm_w: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope: &RopeTables,
        cache: &mut KvCache,
    ) -> CResult<Tensor> {
        let normed = rms_norm(x, &self.input_norm_w, self.eps)?;
        let attn_out = self.attn.forward(&normed, start_pos, rope, cache)?;
        let x = (x + attn_out)?;

        let normed = rms_norm(&x, &self.post_norm_w, self.eps)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x + mlp_out
    }
}

// =========================================================================
// Модель
// =========================================================================

pub struct LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm_w: Tensor,
    lm_head: Linear,
    device: Device,
    config: ModelConfig,
    rope: RopeTables,
    rope_len: usize,
}

impl LlamaModel {
    /// Загружает safetensors-веса MiniCPM5-1B (BF16).
    pub fn load(
        files: &super::config::ModelFiles,
        config: ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        // Проверяем, что нам дали именно safetensors. GGUF больше не
        // поддерживается: MiniCPM5-1B-ru-v3 распространяется в BF16.
        let all_safetensors = files.weights.iter().all(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        });
        if !all_safetensors {
            bail!(
                "Ожидались safetensors-веса MiniCPM5-1B, получены: {:?}",
                files.weights
            );
        }
        Self::load_safetensors(files, config, device)
    }

    fn load_safetensors(
        files: &super::config::ModelFiles,
        config: ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&files.weights, DType::BF16, device)
                .context("Не удалось открыть safetensors-веса MiniCPM5-1B")?
        };

        let hd = config.head_dim();
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads();
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps.max(RMS_EPS_FALLBACK);
        let rope_theta = config.rope_theta();
        let rope_len = config.max_position_embeddings().min(ROPE_TABLE_MAX_SEQ);
        let rope = build_rope_tables(rope_len, hd, rope_theta, device)
            .context("Не удалось построить таблицы RoPE")?;

        let embed_tokens = Embedding {
            t: vb
                .get((config.vocab_size, hidden), "model.embed_tokens.weight")
                .context("Не найден model.embed_tokens.weight")?,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = move |suffix: &str| format!("model.layers.{i}.{suffix}");

            // QK-norm у MiniCPM5 отсутствует. `.ok()` вернёт `None`,
            // если тензора нет — это ожидаемое поведение, а не ошибка.
            let q_norm_w = vb.get(hd, &p("self_attn.q_norm.weight")).ok();
            let k_norm_w = vb.get(hd, &p("self_attn.k_norm.weight")).ok();

            let attn = Attention {
                q: Linear::load(&vb, nh * hd, hidden, &p("self_attn.q_proj.weight"))
                    .with_context(|| format!("слой {i}: q_proj"))?,
                k: Linear::load(&vb, nkv * hd, hidden, &p("self_attn.k_proj.weight"))
                    .with_context(|| format!("слой {i}: k_proj"))?,
                v: Linear::load(&vb, nkv * hd, hidden, &p("self_attn.v_proj.weight"))
                    .with_context(|| format!("слой {i}: v_proj"))?,
                o: Linear::load(&vb, hidden, nh * hd, &p("self_attn.o_proj.weight"))
                    .with_context(|| format!("слой {i}: o_proj"))?,
                q_norm_w,
                k_norm_w,
                eps,
                num_heads: nh,
                num_kv_heads: nkv,
                head_dim: hd,
            };
            let mlp = Mlp {
                gate: Linear::load(
                    &vb,
                    config.intermediate_size,
                    hidden,
                    &p("mlp.gate_proj.weight"),
                )
                .with_context(|| format!("слой {i}: gate_proj"))?,
                up: Linear::load(
                    &vb,
                    config.intermediate_size,
                    hidden,
                    &p("mlp.up_proj.weight"),
                )
                .with_context(|| format!("слой {i}: up_proj"))?,
                down: Linear::load(
                    &vb,
                    hidden,
                    config.intermediate_size,
                    &p("mlp.down_proj.weight"),
                )
                .with_context(|| format!("слой {i}: down_proj"))?,
            };
            layers.push(Block {
                input_norm_w: vb
                    .get(hidden, &p("input_layernorm.weight"))
                    .with_context(|| format!("слой {i}: input_layernorm"))?,
                post_norm_w: vb
                    .get(hidden, &p("post_attention_layernorm.weight"))
                    .with_context(|| format!("слой {i}: post_attention_layernorm"))?,
                attn,
                mlp,
                eps,
            });
        }

        let norm_w = vb
            .get(hidden, "model.norm.weight")
            .context("Не найден model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings() {
            // Связка с `embed_tokens`: `lm_head.weight` = `embed_tokens.weight`.
            // Транспонируем `[vocab, hidden]` → `[hidden, vocab]`.
            let embed = &embed_tokens.t;
            Linear {
                w: embed.t()?.contiguous()?,
            }
        } else {
            Linear::load(&vb, config.vocab_size, hidden, "lm_head.weight")
                .context("Не найден lm_head.weight (tie_word_embeddings=false)")?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm_w,
            lm_head,
            device: device.clone(),
            config,
            rope,
            rope_len,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn rope_len(&self) -> usize {
        self.rope_len
    }

    /// Прямой проход: `ids` — токены, `start_pos` — позиция первого токена
    /// в общем контексте (для префикса `0`, для декода — длина уже
    /// обработанного промпта).
    pub fn forward(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        caches: &mut [KvCache],
    ) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("Пустой набор токенов для forward");
        }
        if caches.len() != self.layers.len() {
            bail!(
                "Число KV-кэшей ({}) не совпадает с числом слоёв ({})",
                caches.len(),
                self.layers.len()
            );
        }

        let seq = ids.len();
        let ids_t = Tensor::from_slice(ids, (seq,), &self.device)?;
        let mut x = self.embed_tokens.forward(&ids_t)?;

        for (i, block) in self.layers.iter().enumerate() {
            x = block.forward(&x, start_pos, &self.rope, &mut caches[i])?;
        }

        let x = rms_norm(
            &x,
            &self.norm_w,
            self.config.rms_norm_eps.max(RMS_EPS_FALLBACK),
        )?;
        Ok(self.lm_head.forward(&x)?)
    }
}
