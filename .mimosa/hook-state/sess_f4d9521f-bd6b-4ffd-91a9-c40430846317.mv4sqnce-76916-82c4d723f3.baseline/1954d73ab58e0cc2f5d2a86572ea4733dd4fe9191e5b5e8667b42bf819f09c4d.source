//! RoFormer (ruaccent `nn_accent`) — расстановка ударений по слову, на candle.
//!
//! Порт ONNX-модели: 4 слоя, hidden 128, 8 голов, rotary position embeddings
//! (RoFormer/Zhuiyi), post-LayerNorm, голова `classifier` по каждому токену
//! с тремя метками [NO, STRESS_PRIMARY, STRESS_SECONDARY].
//!
//! Веса — `models/stress-ru/accent.safetensors` (де-квантованы из ONNX).

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops::softmax,
};
use std::path::Path;

#[derive(Clone, Copy)]
pub struct RoFormerConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_labels: usize,
    pub vocab: usize,
    pub n_token_types: usize,
    pub max_positions: usize,
    pub layer_norm_eps: f64,
}

impl Default for RoFormerConfig {
    fn default() -> Self {
        Self {
            n_layers: 4,
            n_heads: 8,
            hidden: 128,
            intermediate: 256,
            n_labels: 3,
            vocab: 45,
            n_token_types: 2,
            max_positions: 60,
            layer_norm_eps: 1e-12,
        }
    }
}

struct Layer {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    attn_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    out_norm: LayerNorm,
}

pub struct RoFormer {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embeddings_norm: LayerNorm,
    /// Таблица синусоид для rotary: (max_positions, head_dim).
    embed_positions: Tensor,
    layers: Vec<Layer>,
    classifier: Linear,
    cfg: RoFormerConfig,
    device: Device,
}

impl RoFormer {
    pub fn load(path: &Path, cfg: RoFormerConfig, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)
                .with_context(|| format!("Не удалось открыть {path:?}"))?
        };
        Self::load_from(vb, cfg, device)
    }

    pub fn load_from(vb: VarBuilder, cfg: RoFormerConfig, device: &Device) -> Result<Self> {
        let r = vb.pp("roformer");
        let word_embeddings = Embedding::new(
            r.pp("embeddings")
                .pp("word_embeddings")
                .get((cfg.vocab, cfg.hidden), "weight")?,
            cfg.hidden,
        );
        let token_type_embeddings = Embedding::new(
            r.pp("embeddings")
                .pp("token_type_embeddings")
                .get((cfg.n_token_types, cfg.hidden), "weight")?,
            cfg.hidden,
        );
        let embeddings_norm = layer_norm(
            cfg.hidden,
            cfg.layer_norm_eps,
            r.pp("embeddings").pp("LayerNorm"),
        )?;
        let head_dim = cfg.hidden / cfg.n_heads;
        let embed_positions = r
            .pp("encoder")
            .pp("embed_positions")
            .get((cfg.max_positions, head_dim), "weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let l = r.pp("encoder").pp("layer").pp(i.to_string());
            layers.push(Layer {
                query: linear(
                    cfg.hidden,
                    cfg.hidden,
                    l.pp("attention").pp("self").pp("query"),
                )?,
                key: linear(
                    cfg.hidden,
                    cfg.hidden,
                    l.pp("attention").pp("self").pp("key"),
                )?,
                value: linear(
                    cfg.hidden,
                    cfg.hidden,
                    l.pp("attention").pp("self").pp("value"),
                )?,
                dense: linear(
                    cfg.hidden,
                    cfg.hidden,
                    l.pp("attention").pp("output").pp("dense"),
                )?,
                attn_norm: layer_norm(
                    cfg.hidden,
                    cfg.layer_norm_eps,
                    l.pp("attention").pp("output").pp("LayerNorm"),
                )?,
                intermediate: linear(
                    cfg.hidden,
                    cfg.intermediate,
                    l.pp("intermediate").pp("dense"),
                )?,
                output: linear(cfg.intermediate, cfg.hidden, l.pp("output").pp("dense"))?,
                out_norm: layer_norm(
                    cfg.hidden,
                    cfg.layer_norm_eps,
                    l.pp("output").pp("LayerNorm"),
                )?,
            });
        }
        let classifier = linear(cfg.hidden, cfg.n_labels, vb.pp("classifier"))?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            embeddings_norm,
            embed_positions,
            layers,
            classifier,
            cfg,
            device: device.clone(),
        })
    }

    /// Логиты по каждому токену: `seq` строк по `n_labels` значений.
    pub fn logits(
        &self,
        ids: &[u32],
        token_types: &[u32],
        attention_mask: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let seq = ids.len();
        let head_dim = self.cfg.hidden / self.cfg.n_heads;
        let ids_t = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let tt_t = Tensor::new(token_types, &self.device)?.unsqueeze(0)?;
        let mut x = self
            .word_embeddings
            .forward(&ids_t)?
            .broadcast_add(&self.token_type_embeddings.forward(&tt_t)?)?;
        x = self.embeddings_norm.forward(&x)?;

        // sin/cos по позициям: таблица (seq, head_dim) -> (1, 1, seq, head_dim)
        let half = head_dim / 2;
        let table = self.embed_positions.narrow(0, 0, seq)?;
        let sin = table.narrow(D::Minus1, 0, half)?;
        let cos = table.narrow(D::Minus1, half, half)?;
        let sin_pos = Tensor::cat(
            &[&sin.unsqueeze(D::Minus1)?, &sin.unsqueeze(D::Minus1)?],
            D::Minus1,
        )?
        .reshape((1, 1, seq, head_dim))?;
        let cos_pos = Tensor::cat(
            &[&cos.unsqueeze(D::Minus1)?, &cos.unsqueeze(D::Minus1)?],
            D::Minus1,
        )?
        .reshape((1, 1, seq, head_dim))?;

        // Маска: (1 - mask) * -10000, как в HF (аддитивная).
        let mask = Tensor::new(attention_mask, &self.device)?;
        let ones = Tensor::ones(mask.shape(), DType::F32, &self.device)?;
        let additive =
            ((ones - mask.to_dtype(DType::F32)?)? * -10000.0)?.reshape((1, 1, 1, seq))?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        for layer in &self.layers {
            let (b, s, _) = x.dims3()?;
            let shape = (b, s, self.cfg.n_heads, head_dim);
            let q = layer
                .query
                .forward(&x)?
                .reshape(shape)?
                .permute((0, 2, 1, 3))?;
            let k = layer
                .key
                .forward(&x)?
                .reshape(shape)?
                .permute((0, 2, 1, 3))?;
            let v = layer
                .value
                .forward(&x)?
                .reshape(shape)?
                .permute((0, 2, 1, 3))?;
            let q = rotary(&q, &sin_pos, &cos_pos)?;
            let k = rotary(&k, &sin_pos, &cos_pos)?;

            let scores = q
                .matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(scale, 0.0)?;
            let scores = scores.broadcast_add(&additive)?;
            let probs = softmax(&scores, D::Minus1)?;
            let ctx = probs
                .matmul(&v)?
                .permute((0, 2, 1, 3))?
                .reshape((b, s, self.cfg.hidden))?;

            let attn = layer.dense.forward(&ctx)?;
            let h = layer.attn_norm.forward(&(x + attn)?)?;
            let inter = layer.intermediate.forward(&h)?.gelu_erf()?;
            let out = layer.output.forward(&inter)?;
            x = layer.out_norm.forward(&(h + out)?)?;
        }

        let logits = self.classifier.forward(&x)?; // (1, seq, n_labels)
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        Ok(flat.chunks(self.cfg.n_labels).map(|c| c.to_vec()).collect())
    }
}

/// Rotary position embedding (RoFormer): `x * cos + rotate_half(x) * sin`.
fn rotary(x: &Tensor, sin_pos: &Tensor, cos_pos: &Tensor) -> Result<Tensor> {
    let shape = x.dims().to_vec();
    let last = *shape.last().context("пустой тензор")?;
    let mut pair_shape = shape.clone();
    pair_shape.pop();
    pair_shape.push(last / 2);
    pair_shape.push(2);
    let pairs = x.reshape(pair_shape)?;
    let even = pairs.narrow(D::Minus1, 0, 1)?;
    let odd = pairs.narrow(D::Minus1, 1, 1)?;
    let rotated = Tensor::cat(&[&odd.neg()?, &even], D::Minus1)?.reshape(shape)?;
    Ok((x.broadcast_mul(cos_pos)? + rotated.broadcast_mul(sin_pos)?)?)
}
