//! Conformer-блоки энкодера FastConformer: свёрточный модуль, блок и стек.
//!
//! Порядок операций внутри блока снят с графа ONNX-эталона (post-LN
//! ConformerLayer из NeMo):
//! x += 0.5 · FF1(LN_ff1(x)); x += Attn(LN_att(x)); x += Conv(LN_conv(x));
//! x += 0.5 · FF2(LN_ff2(x)); x = LN_out(x).
//! FFN: linear → SiLU (x·sigmoid(x)) → linear; residual FF — с 0.5,
//! residual attention и conv — без масштаба.
//!
//! Свёрточный модуль: pointwise 1×1 (512→1024) → GLU (первая половина ·
//! sigmoid второй) → depthwise 1D k=9 с симметричным паддингом 4 (полный
//! контекст) → BatchNorm (eval, статистики из чекпоинта) → SiLU →
//! pointwise 1×1. В ONNX-эталоне BatchNorm свёрнут в веса depthwise-свёртки;
//! здесь он применяется явно по тензорам чекпоинта — эквивалентно.

use super::attention::RelPosAttention;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{
    conv::Conv1dConfig, conv1d, layer_norm, Linear, Module, VarBuilder, LayerNorm,
};

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    norm: LayerNorm,
}

impl FeedForward {
    /// vb — уровень блока: `feed_forward1.linear{1,2}.*` и `norm_feed_forward1.*`.
    fn load(vb: &VarBuilder, d_model: usize, ff_dim: usize, prefix: &str, norm_prefix: &str) -> Result<Self> {
        let ff = vb.pp(prefix);
        Ok(Self {
            linear1: candle_nn::linear(d_model, ff_dim, ff.pp("linear1"))?,
            linear2: candle_nn::linear(ff_dim, d_model, ff.pp("linear2"))?,
            norm: layer_norm(d_model, 1e-5, vb.pp(norm_prefix))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.linear1.forward(&h)?.silu()?;
        Ok(self.linear2.forward(&h)?)
    }
}

struct ConvModule {
    pw1: candle_nn::Conv1d,
    dw: candle_nn::Conv1d,
    /// BatchNorm в eval-режиме, свёрнутый в аффинное преобразование
    /// y = x·scale + shift (тождественно ONNX-эталону, где BN свёрнут
    /// в веса dw-свёртки).
    bn_scale: Tensor,
    bn_shift: Tensor,
    pw2: candle_nn::Conv1d,
    norm: LayerNorm,
}

impl ConvModule {
    fn load(vb: &VarBuilder, d_model: usize, kernel: usize) -> Result<Self> {
        let dw_cfg = Conv1dConfig {
            padding: (kernel - 1) / 2,
            stride: 1,
            dilation: 1,
            groups: d_model,
            cudnn_fwd_algo: None,
        };
        let pw_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        // Свёртка BatchNorm в аффинное преобразование (eval-режим):
        // y = (x - rm)/sqrt(rv+eps)·w + b ≡ x·scale + shift.
        let bnv = vb.pp("conv").pp("batch_norm");
        let bn_w = bnv.get(d_model, "weight")?;
        let bn_b = bnv.get(d_model, "bias")?;
        let bn_rm = bnv.get(d_model, "running_mean")?;
        let bn_rv = bnv.get(d_model, "running_var")?;
        let denom = (bn_rv + 1e-5)?.sqrt()?;
        let bn_scale = bn_w.broadcast_div(&denom)?;
        let bn_shift = (bn_b - bn_w.broadcast_mul(&bn_rm.broadcast_div(&denom)?)?)?;

        Ok(Self {
            pw1: conv1d(d_model, 2 * d_model, 1, pw_cfg, vb.pp("conv").pp("pointwise_conv1"))?,
            dw: conv1d(d_model, d_model, kernel, dw_cfg, vb.pp("conv").pp("depthwise_conv"))?,
            bn_scale,
            bn_shift,
            pw2: conv1d(d_model, d_model, 1, pw_cfg, vb.pp("conv").pp("pointwise_conv2"))?,
            norm: layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?,
        })
    }

    fn forward(&self, x: &Tensor, conv_keep: Option<&Tensor>) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        // [B, T, D] -> [B, D, T] (свёртки работают по времени)
        let h = h.transpose(1, 2)?;
        let h = self.pw1.forward(&h)?; // [B, 2D, T]

        // GLU по каналам: gate — вторая половина (семантика torch.nn.GLU).
        let d = h.dim(1)? / 2;
        let a = h.narrow(1, 0, d)?;
        let gate = candle_nn::ops::sigmoid(&h.narrow(1, d, d)?)?;
        let mut h = (a * gate)?; // [B, D, T]

        // Where-маска эталона: GLU-выход обнуляется по времени до dw-свёртки,
        // иначе хвостовой кадр затекает в окно k=9 валидных кадров.
        if let Some(keep) = conv_keep {
            h = h.broadcast_mul(keep)?;
        }

        let h = self.dw.forward(&h)?;
        // Свёрнутый BatchNorm: y = x·scale + shift по каналам.
        let h = dw_raw_like(&h, &self.bn_scale, &self.bn_shift)?;
        let h = h.silu()?;
        let h = self.pw2.forward(&h)?; // [B, D, T]
        Ok(h.transpose(1, 2)?) // [B, T, D]
    }

    /// Трассировка для верификации: (выход GLU, выход dw-свёртки до BN,
    /// выход модуля).
    pub fn forward_trace(
        &self,
        x: &Tensor,
        conv_keep: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h = self.norm.forward(x)?;
        let h = h.transpose(1, 2)?;
        let h = self.pw1.forward(&h)?;
        let d = h.dim(1)? / 2;
        let a = h.narrow(1, 0, d)?;
        let gate = candle_nn::ops::sigmoid(&h.narrow(1, d, d)?)?;
        let h = (a * gate)?;
        let glu = h.clone();
        let mut h_masked = h;
        if let Some(keep) = conv_keep {
            h_masked = h_masked.broadcast_mul(keep)?;
        }
        let dw_raw = self.dw.forward(&h_masked)?;
        let h = dw_raw_like(&dw_raw, &self.bn_scale, &self.bn_shift)?;
        let h = h.silu()?;
        let h = self.pw2.forward(&h)?;
        let out = h.transpose(1, 2)?;
        Ok((glu, dw_raw, out))
    }
}

/// Маска кадров при T > валидной длины (последний частичный кадр mel).
/// Все три компонента сняты с Where-узлов ONNX-эталона:
/// - `attn_add` (Cast_1): аддитивные -10000 в scores у невалидных колонок-ключей
///   и целиком у невалидных строк-запросов;
/// - `attn_row_zero` (Cast_2/Where_1, 0.0): невалидная строка softmax обнуляется
///   целиком, иначе её значения через conv-окно затекают в валидные кадры;
/// - `conv_keep` (Where свёрточного модуля): GLU-выход обнуляется по времени
///   перед depthwise-свёрткой, чтобы нулевые кадры не затекали в окно k=9.
pub struct FrameMask {
    pub attn_add: Tensor,
    pub attn_row_zero: Tensor,
    pub conv_keep: Tensor,
}

pub struct ConformerBlock {
    ff1: FeedForward,
    ff2: FeedForward,
    attention: RelPosAttention,
    conv: ConvModule,
    norm_self_att: LayerNorm,
    norm_out: LayerNorm,
}

impl ConformerBlock {
    pub fn load(vb: &VarBuilder, d_model: usize, ff_dim: usize, n_heads: usize, kernel: usize) -> Result<Self> {
        Ok(Self {
            ff1: FeedForward::load(vb, d_model, ff_dim, "feed_forward1", "norm_feed_forward1")?,
            ff2: FeedForward::load(vb, d_model, ff_dim, "feed_forward2", "norm_feed_forward2")?,
            attention: RelPosAttention::load(vb.pp("self_attn"), d_model, n_heads)?,
            conv: ConvModule::load(vb, d_model, kernel)?,
            norm_self_att: layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?,
            norm_out: layer_norm(d_model, 1e-5, vb.pp("norm_out"))?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        mask: Option<&FrameMask>,
    ) -> Result<Tensor> {
        // x += 0.5 * FF1(LN_ff1(x))
        let x = (x + self.ff1.forward(x)?.affine(0.5, 0.0)?)?;
        // x += Attn(LN_att(x))
        let att_in = self.norm_self_att.forward(&x)?;
        let att_out = self.attention.forward(&att_in, pos_emb, mask)?;
        let x = (&x + att_out)?;
        // x += Conv(LN_conv(x)) — norm_conv применяется внутри ConvModule
        let conv_keep = mask.map(|m| &m.conv_keep);
        let conv_out = self.conv.forward(&x, conv_keep)?;
        let x = (&x + conv_out)?;
        // x += 0.5 * FF2(LN_ff2(x)); затем LN_out
        let ff2_out = self.ff2.forward(&x)?.affine(0.5, 0.0)?;
        let x = (&x + ff2_out)?;
        Ok(self.norm_out.forward(&x)?)
    }

    /// Трассировка блока для стенда fc_debug: именованные стадии в порядке
    /// выполнения. Имена согласованы со снимком ONNX (attn_ref.json).
    pub fn forward_trace(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        mask: Option<&FrameMask>,
    ) -> Result<Vec<(String, Tensor)>> {
        let mut trace = Vec::new();
        let x_ff1 = (x + self.ff1.forward(x)?.affine(0.5, 0.0)?)?;
        trace.push(("ff1_res".to_string(), x_ff1.clone()));

        let att_in = self.norm_self_att.forward(&x_ff1)?;
        let at = self.attention.forward_trace(&att_in, pos_emb, mask)?;
        trace.push(("a.bd_raw".to_string(), at.bd_raw.clone()));
        trace.push(("a.bd_shifted".to_string(), at.bd_shifted.clone()));
        trace.push(("a.ac".to_string(), at.ac.clone()));
        trace.push(("a.scores".to_string(), at.scores.clone()));
        trace.push(("a.attn".to_string(), at.attn.clone()));
        trace.push(("a.attn_v".to_string(), at.attn_v.clone()));
        trace.push(("a.out".to_string(), at.out_proj.clone()));
        trace.push(("q_v".to_string(), at.q_v.clone()));
        trace.push(("p_t".to_string(), at.p_t.clone()));
        let x_att = (&x_ff1 + at.out_proj)?;
        trace.push(("attn_res".to_string(), x_att.clone()));

        let (glu, dw_raw, conv_out) =
            self.conv
                .forward_trace(&x_att, mask.map(|m| &m.conv_keep))?;
        trace.push(("conv_glu".to_string(), glu));
        trace.push(("conv_dw_raw".to_string(), dw_raw));
        trace.push(("conv_out".to_string(), conv_out.clone()));
        let x_conv = (&x_att + conv_out)?;
        trace.push(("conv_res".to_string(), x_conv.clone()));

        let x_ff2 = (&x_conv + self.ff2.forward(&x_conv)?.affine(0.5, 0.0)?)?;
        trace.push(("out".to_string(), self.norm_out.forward(&x_ff2)?));
        Ok(trace)
    }
}

pub struct Encoder {
    subsampling: super::subsampling::Subsampling,
    pub blocks: Vec<ConformerBlock>,
    d_model: usize,
    /// Полная таблица относительных позиций [2*max_len-1, d_model].
    pos_table: Tensor,
    pos_emb_max_len: usize,
    subsampling_factor: usize,
}

impl Encoder {
    pub fn load(
        vb: &VarBuilder,
        cfg: &super::config::FastConformerConfig,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let enc = vb.pp("encoder");
        let pos_table =
            super::attention::build_pos_table(cfg.encoder.pos_emb_max_len, cfg.encoder.d_model, device)?;
        let mut blocks = Vec::with_capacity(cfg.encoder.n_layers);
        for i in 0..cfg.encoder.n_layers {
            blocks.push(ConformerBlock::load(
                &enc.pp("layers").pp(i.to_string()),
                cfg.encoder.d_model,
                cfg.encoder.ff_dim,
                cfg.encoder.n_heads,
                cfg.encoder.conv_kernel_size,
            )?);
        }
        Ok(Self {
            subsampling: super::subsampling::Subsampling::load(enc.clone(), cfg.encoder.d_model)?,
            blocks,
            d_model: cfg.encoder.d_model,
            pos_table,
            pos_emb_max_len: cfg.encoder.pos_emb_max_len,
            subsampling_factor: cfg.encoder.subsampling_factor,
        })
    }

    /// mel: [B, feat_in, T] → выход [B, T'', d_model] вместе с числом валидных
    /// кадров T' (length после сжатия 8×; T'' может быть на кадр больше).
    pub fn forward(&self, mel: &Tensor, length: usize) -> Result<(Tensor, usize)> {
        let x = self.subsampling.forward(mel)?; // [B, T'', d_model]
        // NeMo RelPositionalEncoding (scale_input=true): вход слоёв
        // масштабируется на sqrt(d_model); pos-таблица — без масштаба.
        // В графе ONNX residual первого блока прибавляется к /pos_enc/Mul —
        // это и есть умножение на xscale.
        let x = x.affine((self.d_model as f64).sqrt(), 0.0)?;
        let t = x.dim(1)?;
        let t_valid = super::frontend::calc_length(length, self.subsampling_factor).min(t);
        let pos_emb = super::attention::slice_pos_emb(&self.pos_table, self.pos_emb_max_len, t)?;
        // Маска кадров: только когда за t_valid остались вычисленные кадры.
        let mask = if t > t_valid {
            let idx = Tensor::arange(0u32, t as u32, x.device())?;
            let bound = Tensor::new(t_valid as u32, x.device())?.broadcast_as(idx.shape())?;
            // 1 для валидных кадров, 0 для хвоста.
            let keep = idx.lt(&bound)?.to_dtype(candle_core::DType::F32)?;
            // Аддитивная маска (1 - keep) · -10000; знак критичен: ранее
            // +10000 усиливал хвост вместо маскировки.
            let invalid = keep.affine(-1.0, 1.0)?.affine(-10000.0, 0.0)?; // [t]
            let attn_add = invalid
                .reshape((1, 1, 1, t))?
                .broadcast_add(&invalid.reshape((1, 1, t, 1))?)?;
            Some(FrameMask {
                attn_add,
                attn_row_zero: keep.reshape((1, 1, t, 1))?,
                conv_keep: keep.reshape((1, 1, t))?,
            })
        } else {
            None
        };
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, &pos_emb, mask.as_ref())?;
        }
        Ok((x, t_valid))
    }

    /// Прямой доступ к subsampling — для стендов верификации.
    pub fn subsampling_forward(&self, mel: &Tensor) -> Result<Tensor> {
        self.subsampling.forward(mel)
    }
}


/// Свёрнутый BatchNorm: y = x·scale + shift по каналам (scale/shift [C]).
fn dw_raw_like(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let scaled = x.broadcast_mul(&scale.unsqueeze(0)?.unsqueeze(2)?)?;
    Ok(scaled.broadcast_add(&shift.unsqueeze(0)?.unsqueeze(2)?)?)
}