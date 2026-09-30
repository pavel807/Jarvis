//! Реализация Llama-совместимой архитектуры (openbmb/MiniCPM5-1B) на candle.
//!
//! Отличие от `candle-transformers::models::llama`: у MiniCPM5 head_dim (128)
//! не равен hidden_size / num_attention_heads (96), поэтому проекции и RoPE
//! считаются по явному head_dim из config.json. RoPE в конвенции HF
//! (rotate_half), как у чекпоинтов transformers.

use super::config::ModelConfig;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Result as CResult, Tensor, D};
use candle_nn::VarBuilder;

/// eps RMSNorm из config.json; хранится отдельно, т.к. общий для всех норм.
const RMS_EPS_FALLBACK: f64 = 1e-6;

/// Потолок таблиц RoPE: промпт + ответ в этом ассистенте заведомо короче
/// (max_position_embeddings у Qwen3 — 131072, такую таблицу держать незачем).
const ROPE_TABLE_MAX_SEQ: usize = 8192;

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> CResult<Tensor> {
    // Fused-ядро candle (один dispatch на Metal вместо ~9 операций):
    // внутри нормирует в f32 и возвращает dtype входа.
    candle_nn::ops::rms_norm(x, weight, eps as f32)
}

/// Матрица линейного слоя без bias, уже транспонированная до (in, out).
struct Linear {
    w: Tensor,
}

impl Linear {
    fn load(vb: &VarBuilder, out: usize, inn: usize, name: &str) -> CResult<Self> {
        let w = vb.get((out, inn), name)?;
        Ok(Self { w: w.t()?.contiguous()? })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        x.matmul(&self.w)
    }
}

/// Пре-вычисленные таблицы RoPE: cos/sin всех позиций до `max_seq`.
/// Строятся один раз при загрузке модели; в upstream candle-llama сделано так же.
pub struct RopeTables {
    cos: Tensor, // (max_seq, head_dim/2)
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

/// RoPE (rotate_half) по готовым таблицам: позиции start_pos..start_pos+seq
/// применяются к тензору формата (seq, heads, head_dim).
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

    let out1 = x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out2 = x2.broadcast_mul(&cos)?.broadcast_add(&x1.broadcast_mul(&sin)?)?;

    Tensor::cat(&[out1, out2], D::Minus1)
}

/// KV-кэш одного слоя в раскладке (kv_heads, total_seq, head_dim).
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
        // k, v приходят в раскладке (seq, kv_heads, hd) -> (kv_heads, seq, hd).
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
            _ => Err(candle_core::Error::msg(
                "KV-кэш пуст: append не был вызван",
            )),
        }
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// QK-нормализация (Qwen3): RMSNorm по head_dim после проекций, до RoPE.
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
        let k = self
            .k
            .forward(x)?
            .reshape((seq, self.num_kv_heads, hd))?;
        let v = self
            .v
            .forward(x)?
            .reshape((seq, self.num_kv_heads, hd))?;

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

        // GQA: размножаем kv-головы до числа q-голов -> (num_heads, total, hd).
        let rep = self.num_heads / self.num_kv_heads;
        let k = k
            .unsqueeze(1)?
            .expand((self.num_kv_heads, rep, total, hd))?
            .reshape((self.num_heads, total, hd))?
            .contiguous()?;
        let v = v
            .unsqueeze(1)?
            .expand((self.num_kv_heads, rep, total, hd))?
            .reshape((self.num_heads, total, hd))?
            .contiguous()?;

        // (seq, heads, hd) -> (heads, seq, hd); scores (heads, seq, total).
        let q = q.transpose(0, 1)?.contiguous()?;
        let mut scores = (q.matmul(&k.transpose(1, 2)?)? * (1.0 / (hd as f64).sqrt()))?;

        if seq > 1 {
            // Каузальная маска только на префилле; на декоде seq == 1.
            let mask = causal_mask(seq, total, start_pos, x.device(), scores.dtype())?;
            scores = scores.broadcast_add(&mask)?;
        }

        // У candle нет Metal-ядра для fused softmax_last_dim — используем
        // универсальную реализацию.
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // (heads, seq, hd)
        let out = out.transpose(0, 1)?.contiguous()?; // (seq, heads, hd)
        let out = out.reshape((seq, self.num_heads * hd))?;
        self.o.forward(&out)
    }
}

fn causal_mask(seq: usize, total: usize, start_pos: usize, device: &Device, dtype: DType) -> CResult<Tensor> {
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

pub struct LlamaModel {
    embed_tokens: Tensor,
    layers: Vec<Block>,
    norm_w: Tensor,
    lm_head: Linear,
    device: Device,
    config: ModelConfig,
    rope: RopeTables,
    /// Число позиций, покрытых таблицами RoPE.
    rope_len: usize,
}

impl LlamaModel {
    pub fn load(files: &super::config::ModelFiles, config: ModelConfig, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &files.weights,
                DType::BF16,
                device,
            )
            .context("Не удалось открыть safetensors-веса LLM")?
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

        let embed_tokens = vb
            .get((config.vocab_size, hidden), "model.embed_tokens.weight")
            .context("Не найден embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = move |suffix: &str| format!("model.layers.{i}.{suffix}");
            // QK-норма есть у Qwen3; у чистой Llama её нет — грузим по наличию.
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
                gate: Linear::load(&vb, config.intermediate_size, hidden, &p("mlp.gate_proj.weight"))
                    .with_context(|| format!("слой {i}: gate_proj"))?,
                up: Linear::load(&vb, config.intermediate_size, hidden, &p("mlp.up_proj.weight"))
                    .with_context(|| format!("слой {i}: up_proj"))?,
                down: Linear::load(&vb, hidden, config.intermediate_size, &p("mlp.down_proj.weight"))
                    .with_context(|| format!("слой {i}: down_proj"))?,
            };
            layers.push(Block {
                input_norm_w: vb.get(hidden, &p("input_layernorm.weight"))?,
                post_norm_w: vb.get(hidden, &p("post_attention_layernorm.weight"))?,
                attn,
                mlp,
                eps,
            });
        }

        let norm_w = vb.get(hidden, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings() {
            Linear { w: embed_tokens.t()?.contiguous()? }
        } else {
            Linear::load(&vb, config.vocab_size, hidden, "lm_head.weight")
                .context("Не найден lm_head (tie_word_embeddings=false)")?
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

    /// Число позиций, покрытых таблицами RoPE: предел суммарной длины
    /// (промпт + генерация), проверяется перед префиллом.
    pub fn rope_len(&self) -> usize {
        self.rope_len
    }

    /// Полный forward: токены + позиция старта -> логиты (seq, vocab).
    pub fn forward(&mut self, ids: &[u32], start_pos: usize, caches: &mut [KvCache]) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("Пустой набор токенов для forward");
        }
        if caches.len() != self.layers.len() {
            bail!("Число KV-кэшей не совпадает с числом слоёв");
        }

        let seq = ids.len();
        let ids_t = Tensor::from_slice(ids, (seq,), &self.device)?;
        let mut x = self.embed_tokens.index_select(&ids_t, 0)?;

        for (i, block) in self.layers.iter().enumerate() {
            x = block.forward(&x, start_pos, &self.rope, &mut caches[i])?;
        }

        let x = rms_norm(&x, &self.norm_w, self.config.rms_norm_eps.max(RMS_EPS_FALLBACK))?;
        Ok(self.lm_head.forward(&x)?)
    }
}
