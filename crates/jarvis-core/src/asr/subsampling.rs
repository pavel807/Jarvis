//! Pre-encode энкодера: глубоко-свёрточное сжатие 8× (dw_striding).
//!
//! Структура снята с графа ONNX-эталона: [B, 80, T] → транспонирование в
//! [B, T, 80] → [B, 1, T, 80] →
//! conv 3×3 stride 2 (1→256) → ReLU →
//! depthwise 3×3 stride 2 (groups 256) → pointwise 1×1 → ReLU →
//! depthwise 3×3 stride 2 → pointwise 1×1 → ReLU →
//! [B, T', 2560] → Linear(2560 → d_model).
//!
//! BatchNorm в pre-encode отсутствует и в чекпоинте, и в эталоне.
//! Длина выхода: floor((len-1)/2)+1 на каждую stride-2 свёртку
//! (см. [`super::frontend::calc_length`]).

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{conv::Conv2dConfig, conv2d, Linear, Module, VarBuilder};

pub struct Subsampling {
    conv0: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    conv3: candle_nn::Conv2d,
    conv5: candle_nn::Conv2d,
    conv6: candle_nn::Conv2d,
    out: Linear,
}

impl Subsampling {
    pub fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        // Веса: encoder.pre_encode.{conv.0,conv.2,conv.3,conv.5,conv.6,out}.
        let channels = 256;
        let stride2 = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let depthwise = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: channels,
            cudnn_fwd_algo: None,
        };
        let pointwise = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let v = vb.pp("pre_encode");
        let conv = v.pp("conv");
        Ok(Self {
            // Вход — ОДИН канал: [B, 1, T, n_mels]; время и частота —
            // пространственные оси. Вес в чекпоинте [256, 1, 3, 3], group 1
            // (по графу ONNX-эталона).
            conv0: conv2d(1, channels, 3, stride2, conv.pp("0"))?,
            conv2: conv2d(channels, channels, 3, depthwise, conv.pp("2"))?,
            conv3: conv2d(channels, channels, 1, pointwise, conv.pp("3"))?,
            conv5: conv2d(channels, channels, 3, depthwise, conv.pp("5"))?,
            conv6: conv2d(channels, channels, 1, pointwise, conv.pp("6"))?,
            out: candle_nn::linear(channels * 10, d_model, v.pp("out"))?,
        })
    }

    /// mel: [B, feat_in, T] (например [1, 80, 101]) → [B, T', d_model].
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // [B, feat_in, T] -> [B, T, feat_in] -> [B, 1, T, feat_in]
        let (b, _feat, _t) = mel.dims3()?;
        let x = mel
            .t()? // [B, T, feat_in]
            .unsqueeze(1)?; // [B, 1, T, feat_in]

        let x = self.conv0.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = self.conv3.forward(&x)?.relu()?;
        let x = self.conv5.forward(&x)?;
        let x = self.conv6.forward(&x)?.relu()?; // [B, 256, T', 10]

        // [B, 256, T', 10] -> [B, T', 2560] -> [B, T', d_model]
        let x = x.transpose(1, 2)?; // [B, T', 256, 10]
        let t_out = x.dim(1)?;
        let x = x.contiguous()?.reshape((b, t_out, 2560))?;
        Ok(self.out.forward(&x)?)
    }
}