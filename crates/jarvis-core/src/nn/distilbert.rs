//! DistilBERT-классификатор (разрешение омографов) на candle.
//!
//! Это порт ONNX-модели ruaccent `nn_omograph/tiny2`: 3 слоя, hidden 264,
//! 12 голов, post-LayerNorm (legacy-вариант DistilBERT), голова
//! `pre_classifier -> ReLU -> classifier` по [CLS]-токену.
//!
//! Веса лежат рядом в `distilbert`-раскладке (`models/stress-ru/omograph/
//! model.safetensors`), они же были де-квантованы из ONNX (int8 -> f32),
//! поэтому расхождение с onnxruntime — только ошибка исходного квантования.

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops::softmax,
};
use std::path::Path;

#[derive(Clone, Copy)]
pub struct DistilBertConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_labels: usize,
    pub max_positions: usize,
    pub layer_norm_eps: f64,
}

impl Default for DistilBertConfig {
    fn default() -> Self {
        Self {
            n_layers: 3,
            n_heads: 12,
            hidden: 264,
            intermediate: 792,
            n_labels: 2,
            max_positions: 512,
            layer_norm_eps: 1e-12,
        }
    }
}

struct Layer {
    q_lin: Linear,
    k_lin: Linear,
    v_lin: Linear,
    out_lin: Linear,
    sa_layer_norm: LayerNorm,
    lin1: Linear,
    lin2: Linear,
    output_layer_norm: LayerNorm,
}

pub struct DistilBertClassifier {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    embeddings_norm: LayerNorm,
    layers: Vec<Layer>,
    pre_classifier: Linear,
    classifier: Linear,
    cfg: DistilBertConfig,
    device: Device,
}

impl DistilBertClassifier {
    pub fn load(path: &Path, cfg: DistilBertConfig, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)
                .with_context(|| format!("Не удалось открыть {path:?}"))?
        };
        Self::load_from(vb, cfg, device)
    }

    pub fn load_from(vb: VarBuilder, cfg: DistilBertConfig, device: &Device) -> Result<Self> {
        let d = vb.pp("distilbert");
        let word_embeddings = Embedding::new(
            d.pp("embeddings")
                .pp("word_embeddings")
                .get((30524, cfg.hidden), "weight")?,
            cfg.hidden,
        );
        let position_embeddings = Embedding::new(
            d.pp("embeddings")
                .pp("position_embeddings")
                .get((cfg.max_positions, cfg.hidden), "weight")?,
            cfg.hidden,
        );
        let embeddings_norm = layer_norm(
            cfg.hidden,
            cfg.layer_norm_eps,
            d.pp("embeddings").pp("LayerNorm"),
        )?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let l = d.pp("transformer").pp("layer").pp(i.to_string());
            let att = l.pp("attention");
            layers.push(Layer {
                q_lin: linear(cfg.hidden, cfg.hidden, att.pp("q_lin"))?,
                k_lin: linear(cfg.hidden, cfg.hidden, att.pp("k_lin"))?,
                v_lin: linear(cfg.hidden, cfg.hidden, att.pp("v_lin"))?,
                out_lin: linear(cfg.hidden, cfg.hidden, att.pp("out_lin"))?,
                sa_layer_norm: layer_norm(cfg.hidden, cfg.layer_norm_eps, l.pp("sa_layer_norm"))?,
                lin1: linear(cfg.hidden, cfg.intermediate, l.pp("ffn").pp("lin1"))?,
                lin2: linear(cfg.intermediate, cfg.hidden, l.pp("ffn").pp("lin2"))?,
                output_layer_norm: layer_norm(
                    cfg.hidden,
                    cfg.layer_norm_eps,
                    l.pp("output_layer_norm"),
                )?,
            });
        }
        let pre_classifier = linear(cfg.hidden, cfg.hidden, vb.pp("pre_classifier"))?;
        let classifier = linear(cfg.hidden, cfg.n_labels, vb.pp("classifier"))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            embeddings_norm,
            layers,
            pre_classifier,
            classifier,
            cfg,
            device: device.clone(),
        })
    }

    /// Логиты классификации по [CLS]-токену: [n_labels].
    pub fn logits(&self, ids: &[u32], attention_mask: &[u32]) -> Result<Vec<f32>> {
        let seq = ids.len();
        let ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let positions = Tensor::arange(0u32, seq as u32, &self.device)?.unsqueeze(0)?;

        let mut x = self
            .word_embeddings
            .forward(&ids)?
            .broadcast_add(&self.position_embeddings.forward(&positions)?)?;
        x = self.embeddings_norm.forward(&x)?;

        // Маска: True там, где токен — паддинг (как `Where(mask == 0, -inf, x)` в ONNX).
        let mask = Tensor::new(attention_mask, &self.device)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let masked = mask.eq(0.0)?.broadcast_as((1, 1, 1, seq))?;

        let head_dim = self.cfg.hidden / self.cfg.n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        for layer in &self.layers {
            let (b, s, _) = x.dims3()?;
            let shape = (b, s, self.cfg.n_heads, head_dim);
            let q = layer
                .q_lin
                .forward(&x)?
                .reshape(shape)?
                .permute((0, 2, 1, 3))?
                .affine(scale, 0.0)?;
            let k = layer
                .k_lin
                .forward(&x)?
                .reshape(shape)?
                .permute((0, 2, 1, 3))?;
            let v = layer
                .v_lin
                .forward(&x)?
                .reshape(shape)?
                .permute((0, 2, 1, 3))?;

            let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
            let scores = where_masked(&masked, &scores, f32::MIN as f64)?;
            let probs = softmax(&scores, D::Minus1)?;
            let ctx = probs
                .matmul(&v)?
                .permute((0, 2, 1, 3))?
                .reshape((b, s, self.cfg.hidden))?;

            let attn = layer.out_lin.forward(&ctx)?;
            let x1 = layer.sa_layer_norm.forward(&(x + attn)?)?;
            let ffn = layer.lin2.forward(&layer.lin1.forward(&x1)?.gelu_erf()?)?;
            x = layer.output_layer_norm.forward(&(x1 + ffn)?)?;
        }

        // [CLS] — нулевая позиция, сохраняем батчевую ось для Linear.
        let cls = x.narrow(1, 0, 1)?.squeeze(1)?;
        let pooled = self.pre_classifier.forward(&cls)?.relu()?;
        Ok(self
            .classifier
            .forward(&pooled)?
            .squeeze(0)?
            .to_vec1::<f32>()?)
    }
}

/// Ставит `value` в позиции, где `mask` истинно (broadcast по всем осям).
fn where_masked(mask: &Tensor, x: &Tensor, value: f64) -> Result<Tensor> {
    let mask = mask.broadcast_as(x.shape())?;
    let filled = Tensor::full(value as f32, x.shape(), x.device())?;
    Ok(mask.where_cond(&filled, x)?)
}