//! RNN-T ветка FastConformer: Prediction Network (Embedding + LSTM 640),
//! Joint Network и жадный транзьюсер-декодер.
//!
//! Поток (снят с графа decoder_joint-model.onnx эталона):
//! - prediction: embed(prev_token) → LSTM (один слой, hidden 640); перед
//!   первым кадром предсказатель обрабатывает blank (SOS), как в NeMo;
//! - joint: Linear(enc 512→640) + Linear(pred 640→640) → ReLU →
//!   Linear(640 → vocab+1); проекции считаются по отдельности и складываются,
//!   как в NeMo RNNTJoint;
//! - жадное декодирование: на каждый кадр энкодера — цикл до
//!   max_symbols_per_step: joint(enc[t], state) → argmax; blank — переход к
//!   следующему кадру, иначе токен эмитится и обновляется состояние LSTM.
//!
//! Состояние предсказателя (h, c) живёт между шагами декодирования всей
//! фразы — контекст накопленных токенов учитывается на каждом кадре.

use anyhow::Result;
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_nn::rnn::{lstm, RNN, LSTM, LSTMConfig, LSTMState};

pub struct Predictor {
    embed: Embedding,
    lstm: LSTM,
}

impl Predictor {
    pub fn load(vb: VarBuilder, vocab_out: usize, pred_hidden: usize) -> Result<Self> {
        // Веса: decoder.prediction.embed.weight [vocab+1, pred_hidden],
        //       decoder.prediction.dec_rnn.lstm.{weight_ih_l0,...}.
        let embed = Embedding::new(vb.get((vocab_out, pred_hidden), "embed.weight")?, pred_hidden);
        let lstm = lstm(
            pred_hidden,
            pred_hidden,
            LSTMConfig::default(),
            vb.pp("dec_rnn").pp("lstm"),
        )?;
        Ok(Self { embed, lstm })
    }

    pub fn initial_state(&self) -> Result<LSTMState> {
        Ok(self.lstm.zero_state(1)?)
    }

    /// Один шаг предсказателя: id токена → новое состояние LSTM.
    pub fn step(&self, token: usize, state: &LSTMState) -> Result<LSTMState> {
        let device = state.h.device();
        let ids = Tensor::from_slice(&[token as u32], (1,), device)?;
        let e = self.embed.forward(&ids)?; // [1, pred_hidden]
        Ok(self.lstm.step(&e, state)?)
    }
}

pub struct Joint {
    enc_proj: candle_nn::Linear,  // d_model → joint_hidden
    pred_proj: candle_nn::Linear, // pred_hidden → joint_hidden
    out: candle_nn::Linear,       // joint_hidden → vocab+1
}

impl Joint {
    pub fn load(
        vb: VarBuilder,
        d_model: usize,
        pred_hidden: usize,
        joint_hidden: usize,
        vocab_out: usize,
    ) -> Result<Self> {
        Ok(Self {
            enc_proj: candle_nn::linear(d_model, joint_hidden, vb.pp("enc"))?,
            pred_proj: candle_nn::linear(pred_hidden, joint_hidden, vb.pp("pred"))?,
            out: candle_nn::linear(joint_hidden, vocab_out, vb.pp("joint_net").pp("2"))?,
        })
    }

    /// Логиты одной пары (кадр энкодера [1, d_model], состояние [1, pred_hidden]).
    pub fn logits(&self, enc_frame: &Tensor, state: &LSTMState) -> Result<Tensor> {
        let e = self.enc_proj.forward(enc_frame)?; // [1, joint_hidden]
        let p = self.pred_proj.forward(&state.h)?; // [1, joint_hidden]
        let h = (e + p)?.relu()?;
        Ok(self.out.forward(&h)?)
    }
}

/// Жадное RNNT-декодирование. enc_out: [1, T, d_model] на устройстве модели,
/// t_valid — число валидных кадров. Возвращает id токенов.
///
/// Семантика onnx-asr: каждый шаг встраивает ТЕКУЩИЙ контекст (`prev`) из
/// текущего состояния предсказателя. Состояние фиксируется только при
/// эмиссии токена — в него вошло встраивание старого контекста; при blank
/// откатывается, чтобы тот же токен не вошёл в LSTM дважды через границу
/// кадров (двойное продвижение ломало транскрипт: syllables выпадали).
pub fn greedy(
    enc_out: &Tensor,
    t_valid: usize,
    predictor: &Predictor,
    joint: &Joint,
    blank_id: usize,
    max_symbols_per_step: usize,
) -> Result<Vec<usize>> {
    let mut tokens = Vec::new();
    let mut state = predictor.initial_state()?;
    let mut prev = blank_id;

    for t in 0..t_valid {
        // Кадр энкодера [512] → [1, 512]: индексация даёт contiguous-срез.
        let frame = enc_out.i((0, t, ..))?.unsqueeze(0)?;
        let mut emitted = 0usize;
        loop {
            // Встраиваем текущий контекст из текущего состояния.
            let next_state = predictor.step(prev, &state)?;
            let logits = joint.logits(&frame, &next_state)?; // [1, vocab+1]
            // argmax даёт [1]: сжимаем до ранга 0 для to_scalar.
            let best = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()? as usize;
            if best == blank_id || emitted >= max_symbols_per_step {
                break; // blank: state не двигается — контекст не менялся
            }
            tokens.push(best);
            prev = best;
            state = next_state;
            emitted += 1;
        }
    }
    Ok(tokens)
}