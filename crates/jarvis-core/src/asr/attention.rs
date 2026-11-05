//! Multi-Head Self-Attention с относительной позиционной кодировкой (rel_pos)
//! — порт espnet-стиля RelPositionMultiHeadAttention из NeMo, снят с графа
//! ONNX-эталона.
//!
//! Схема:
//! - q = W_q·x + pos_bias_u (сдвиг добавлен ДО matmul, по головам);
//! - k = W_k·x, v = W_v·x;
//! - p = W_pos·pos_emb (линейная проекция таблицы относительных позиций,
//!   без bias);
//! - matrix_ac = q_u·kᵀ, matrix_bd = q_v·pᵀ [B, H, T, 2T-1];
//! - relative_shift(matrix_bd) — та же последовательность операций, что в
//!   эталоне (cat нулевой колонки → reshape [B,H,2T,T] → срез → reshape);
//! - scores = (ac + bd) / sqrt(head_dim) → softmax → ·v → W_out.

use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

/// Синусоидальная таблица относительных позиций NeMo RelPositionalEncoding.
///
/// Строки — позиции от +(max_len-1) до 0 до -(max_len-1) (всего 2*max_len-1);
/// столбцы чередуются: sin(pos·div_k), cos(pos·div_k), …, где
/// div_k = exp(-2k·ln(10000)/d_model). Формула сверина с константой,
/// константно-свёрнутой в ONNX-эталоне (макс. расхождение — шум float32).
pub fn build_pos_table(max_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let half = d_model / 2;
    let total = 2 * max_len - 1;
    let mut table = vec![0f32; total * d_model];
    // div_term в f64, как в NeMo (torch.exp в f32 после умножения на позицию).
    let div_term: Vec<f64> = (0..half)
        .map(|k| (-2.0 * (k as f64) * (10000f64).ln() / d_model as f64).exp())
        .collect();
    for (row, value) in table.chunks_mut(d_model).enumerate() {
        // Позиция строки: +(max_len-1-r) для r < max_len, иначе -(r-max_len+1).
        let pos = if row < max_len {
            (max_len - 1 - row) as f64
        } else {
            -((row - max_len + 1) as f64)
        };
        for (k, dv) in div_term.iter().enumerate() {
            let angle = (pos * *dv) as f32;
            value[2 * k] = angle.sin();
            value[2 * k + 1] = angle.cos();
        }
    }
    Ok(Tensor::from_vec(table, (total, d_model), device)?)
}

/// Нарезка таблицы для длины T: строки [max_len-T .. max_len+T-2] =
/// позиции +(T-1)…0…-(T-1), всего 2T-1.
pub fn slice_pos_emb(table: &Tensor, max_len: usize, t: usize) -> Result<Tensor> {
    let start = max_len - t;
    Ok(table.narrow(0, start, 2 * t - 1)?)
}

pub struct RelPosAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    /// W_pos — проекция позиционной таблицы, без bias.
    pos_proj: Linear,
    /// pos_bias_u [n_heads, head_dim] — добавляется к q перед matmul с k.
    pos_bias_u: Tensor,
    /// pos_bias_v [n_heads, head_dim] — добавляется к q перед matmul с p.
    pos_bias_v: Tensor,
    n_heads: usize,
    head_dim: usize,
}

impl RelPosAttention {
    pub fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        let head_dim = d_model / n_heads;
        Ok(Self {
            q_proj: candle_nn::linear(d_model, d_model, vb.pp("linear_q"))?,
            k_proj: candle_nn::linear(d_model, d_model, vb.pp("linear_k"))?,
            v_proj: candle_nn::linear(d_model, d_model, vb.pp("linear_v"))?,
            out_proj: candle_nn::linear(d_model, d_model, vb.pp("linear_out"))?,
            pos_proj: linear_no_bias(d_model, d_model, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((n_heads, head_dim), "pos_bias_u")?,
            pos_bias_v: vb.get((n_heads, head_dim), "pos_bias_v")?,
            n_heads,
            head_dim,
        })
    }

    /// x: [B, T, d_model]; pos_emb: [2T-1, d_model] (нарезка таблицы);
    /// mask — маска кадров при T > валидной длины ([`super::conformer::FrameMask`]);
    /// None, когда все кадры валидны.
    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&super::conformer::FrameMask>) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let h = self.n_heads;
        let dh = self.head_dim;

        let q = self.q_proj.forward(x)?; // [B, T, D]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // q + bias_u: [B, T, H, Dh] -> [B, H, T, Dh]; bias [H, Dh] -> [1, 1, H, Dh]
        // contiguous после транспонирования: Metal-batched matmul
        // отвергает стрided-представления.
        let q = q.reshape((b, t, h, dh))?;
        let bu = self.pos_bias_u.unsqueeze(0)?.unsqueeze(0)?; // [1,1,H,Dh]
        let bv = self.pos_bias_v.unsqueeze(0)?.unsqueeze(0)?;
        let q_u = q.broadcast_add(&bu)?.transpose(1, 2)?.contiguous()?;
        let q_v = q.broadcast_add(&bv)?.transpose(1, 2)?.contiguous()?;
        let k = k
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?; // [B, H, T, Dh]
        let v = v
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;

        // p: [2T-1, D] -> [2T-1, H, Dh] -> [H, 2T-1, Dh] -> [1, H, 2T-1, Dh]
        let p = self.pos_proj.forward(pos_emb)?; // [2T-1, D]
        let p = p
            .reshape((p.dim(0)?, h, dh))?
            .transpose(0, 1)? // [H, 2T-1, Dh]
            .contiguous()?
            .unsqueeze(0)?; // [1, H, 2T-1, Dh]

        // matrix_ac: q_u · kᵀ -> [B, H, T, T]
        let ac = q_u.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        // matrix_bd: q_v · pᵀ -> [B, H, T, 2T-1]
        let bd_raw = q_v.matmul(&p.transpose(D::Minus2, D::Minus1)?)?;

        let bd = relative_shift(&bd_raw)?; // [B, H, T, T]

        let scores = ((ac + bd)?.affine(1.0 / (dh as f64).sqrt(), 0.0))?;
        // Маска кадров, как в ONNX-эталоне: аддитивная -10000 у невалидных
        // колонок-ключей и целиком у невалидных строк-запросов до softmax.
        // Строки после softmax НЕ обнуляются: эталон оставляет невалидную
        // строку uniform (Where_1 гасит только позиции маски, а они и так
        // ~0 после -10000).
        let attn = match mask {
            // Metal не поддерживает where_cond на F32: аддитивная маска
            // тождественна Where до softmax, обнуление строк — умножением.
            Some(m) => {
                let masked = scores.broadcast_add(&m.attn_add)?;
                let a = candle_nn::ops::softmax(&masked, D::Minus1)?;
                a.broadcast_mul(&m.attn_row_zero)?
            }
            None => candle_nn::ops::softmax(&scores, D::Minus1)?,
        };
        let out = attn.matmul(&v)?; // [B, H, T, Dh]
        let out_h = out.transpose(1, 2)?; // [B, T, H, Dh]
        let out = out_h.contiguous()?.reshape((b, t, h * dh))?;
        Ok(self.out_proj.forward(&out)?)
    }

    /// Полная трассировка для стенда верификации (fc_debug).
    pub fn forward_trace(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        mask: Option<&super::conformer::FrameMask>,
    ) -> Result<AttentionTrace> {
        let (b, t, _d) = x.dims3()?;
        let h = self.n_heads;
        let dh = self.head_dim;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, t, h, dh))?;
        let bu = self.pos_bias_u.unsqueeze(0)?.unsqueeze(0)?;
        let bv = self.pos_bias_v.unsqueeze(0)?.unsqueeze(0)?;
        let q_u = q.broadcast_add(&bu)?.transpose(1, 2)?.contiguous()?;
        let q_v = q.broadcast_add(&bv)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, t, h, dh))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, h, dh))?.transpose(1, 2)?.contiguous()?;

        let p = self.pos_proj.forward(pos_emb)?;
        let p = p
            .reshape((p.dim(0)?, h, dh))?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;

        let bd_raw = q_v.matmul(&p.transpose(D::Minus2, D::Minus1)?)?;
        let bd = relative_shift(&bd_raw)?;
        let ac = q_u.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = ac.broadcast_add(&bd)?.affine(1.0 / (dh as f64).sqrt(), 0.0)?;
        let attn = match mask {
            Some(m) => {
                let masked = scores.broadcast_add(&m.attn_add)?;
                let a = candle_nn::ops::softmax(&masked, D::Minus1)?;
                a.broadcast_mul(&m.attn_row_zero)?
            }
            None => candle_nn::ops::softmax(&scores, D::Minus1)?,
        };
        let attn_v = attn.matmul(&v)?;
        let out_h = attn_v.transpose(1, 2)?;
        let out = out_h.contiguous()?.reshape((b, t, h * dh))?;
        let out_proj = self.out_proj.forward(&out)?;
        Ok(AttentionTrace {
            bd_raw,
            bd_shifted: bd,
            ac,
            scores,
            attn,
            attn_v,
            out_proj,
            q_v: q_v.clone(),
            p_t: p.transpose(D::Minus2, D::Minus1)?.contiguous()?,
        })
    }
}

/// Снимок промежуточных стадий attention для верификации.
pub struct AttentionTrace {
    pub bd_raw: Tensor,
    pub bd_shifted: Tensor,
    pub ac: Tensor,
    pub scores: Tensor,
    pub attn: Tensor,
    pub attn_v: Tensor,
    pub out_proj: Tensor,
    /// Входы matmul'а bd — для изоляции расхождений.
    pub q_v: Tensor,
    pub p_t: Tensor,
}

/// Relative shift — точная последовательность операций из графа ONNX-эталона
/// (узлы Pad → Reshape → Slice → Reshape → Slice):
/// 1. Pad: один ноль в НАЧАЛО последней оси [B,H,T,2T-1] → [B,H,T,2T];
/// 2. reshape [B,H,T,2T] → [B,H,2T,T];
/// 3. срез строк 1.. → [B,H,2T-1,T];
/// 4. reshape назад → [B,H,T,2T-1];
/// 5. срез последней оси до T → [B,H,T,T].
/// Семантика результата: bd_shifted[i][j] = bd_raw[i][T-1+(j-i)] —
/// выравнивание таблицы относительных позиций по диагонали.
fn relative_shift(bd: &Tensor) -> Result<Tensor> {
    let (b, h, t, k2) = bd.dims4()?; // k2 = 2T-1
    let zero = Tensor::zeros((b, h, t, 1), bd.dtype(), bd.device())?;
    let padded = Tensor::cat(&[&zero, bd], D::Minus1)?; // [B,H,T,2T]
    let reshaped = padded.reshape((b, h, 2 * t, t))?; // [B,H,2T,T]
    let sliced = reshaped.narrow(2, 1, k2)?; // [B,H,2T-1,T]
    let back = sliced.contiguous()?.reshape((b, h, t, k2))?; // [B,H,T,2T-1]
    Ok(back.narrow(D::Minus1, 0, t)?) // [B,H,T,T]
}
