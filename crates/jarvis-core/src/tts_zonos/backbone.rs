//! Языковая часть ZONOS3-Mini на candle: 11 слоёв, GDN + глобальное
//! внимание, латентный MoE и Full AttnRes.
//!
//! Численный оракул — `model/src/zonos3/reference/lm.py` из репозитория
//! Zyphra/ZONOS3-Mini; здесь повторяется его математика один в один:
//!
//! * эмбеддинг: сумма 11 таблиц (10 аудиокниг + текст), затем замена
//!   speaker-строки и трёх condition-строк, затем RMSNorm без веса;
//! * Full AttnRes: банк из 23 источников, softmax по глубине, смешивание
//!   исходных (ненормализованных) источников;
//! * GDN: `in_proj` → depthwise causal conv (k=4) + SiLU → нормализация
//!   q/k → delta-rule в fp32 → per-head RMSNorm → gate silu(z);
//! * глобальное внимание: QK-RMSNorm, interleaved RoPE, GQA 16/4;
//! * MoE: линейный роутер, top-2 из 64, latent 256, SwiGLU-эксперты;
//! * выход: одна матрица на 10 книг, softcap 15·tanh(x/15).
//!
//! Квантованные веса (Q8_0/Q4_0) не декодируются заранее: считает
//! `candle_core::quantized::QMatMul`, поэтому в памяти лежит ровно то,
//! что лежит в файле.

use super::config::ZonosConfig;
use super::gguf::{GgufFile, Mat};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use candle_core::IndexOp;
use candle_nn::ops::softmax;

/// Миксер слоя: GDN либо глобальное внимание.
#[derive(Clone)]
pub enum Mixer {
    Gdn(GdnLayer),
    Attention(AttentionLayer),
}

/// Слой Gated DeltaNet.
#[derive(Clone)]
pub struct GdnLayer {
    pub in_proj: Mat,
    /// Вес depthwise-свёртки: [1536 каналов, kernel 4].
    pub conv: Vec<Vec<f32>>,
    /// Аффинный вес per-head RMSNorm выхода ([128]).
    pub out_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Проекция выхода: 512 → 1024.
    pub wo: Mat,
}

/// Слой глобального внимания.
#[derive(Clone)]
pub struct AttentionLayer {
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub gate: Mat,
    pub temp: Vec<f32>,
    pub wo: Mat,
}

/// Плотный FFN первого слоя.
#[derive(Clone)]
pub struct DenseLayer {
    pub up: Mat,
    pub gate: Mat,
    pub down: Mat,
}

/// Латентный MoE: 64 эксперта, top-2, SwiGLU.
#[derive(Clone)]
pub struct MoeLayer {
    pub router: Mat,
    pub router_norm: Tensor,
    pub router_bias: Vec<f32>,
    pub latent_down: Mat,
    pub latent_up: Mat,
    /// Эксперты gate/up: `[expert][hidden, routed]`.
    pub gate_exps: Vec<Mat>,
    pub up_exps: Vec<Mat>,
    /// Эксперты down: `[expert][routed, hidden]`.
    pub down_exps: Vec<Mat>,
    pub moe_hidden: usize,
    pub routed_dim: usize,
}

/// Тип FFN слоя.
#[derive(Clone)]
pub enum Ffn {
    Dense(DenseLayer),
    Moe(MoeLayer),
}

/// Один слой бэкбона.
#[derive(Clone)]
pub struct Layer {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub mixer: Mixer,
    pub ffn: Ffn,
}

/// Полный языковой бэкбон.
#[derive(Clone)]
pub struct Backbone {
    pub cfg: ZonosConfig,
    pub device: Device,
    /// 10 аудиотаблиц эмбеддинга: [4098, 1024] (f16).
    pub audio_embd: Vec<Tensor>,
    /// Текстовая таблица: [text_rows, 1024] (f16).
    pub text_embd: Tensor,
    /// Голова выхода: [40980, 1024].
    pub output: Mat,
    pub output_norm: Tensor,
    /// Запросы банка AttnRes: [23, 1024].
    pub attnres_q: Vec<Vec<f32>>,
    /// Проекция спикера: 192 → 1024 ([1024, 192]) + bias [1024].
    pub speaker_proj: Tensor,
    pub speaker_bias: Tensor,
    /// Таблицы условий: 3 штуки, у каждой `bucket_counts[i] + 1` строк.
    pub cond_embd: Vec<Tensor>,
    pub layers: Vec<Layer>,
    /// Состояния рекурсии GDN: на слой [4 головы, 128, 128] в f32.
    pub gdn_states: Vec<Vec<f32>>,
    /// История свёртки GDN: на слой [1536, kernel-1].
    pub gdn_histories: Vec<Vec<f32>>,
    /// Кэш K/V глобального внимания, размноженный до 16 query-голов
    /// (GQA 4→16): на слой [позиции · 16 голов · 32].
    pub attn_k_exp: Vec<Vec<f32>>,
    pub attn_v_exp: Vec<Vec<f32>>,
    /// Позиция следующего шага в контексте.
    pub position: usize,
}

impl Backbone {
    /// Загрузить веса из GGUF.
    pub fn load(cfg: ZonosConfig, gguf: &GgufFile, device: &Device) -> Result<Self> {
        let mut audio_embd = Vec::with_capacity(cfg.n_codebooks);
        for k in 0..cfg.n_codebooks {
            let name = format!("audio_embd.{k}.weight");
            audio_embd.push(
                gguf.tensor_f16(&name, device)
                    .with_context(|| format!("Нет {name}"))?,
            );
        }
        let text_embd = gguf.tensor_f16("text_embd.weight", device)?;
        let output = gguf.mat("output.weight", device)?;
        let output_norm = gguf.tensor_f32("output_norm.weight", device)?;
        let attnres_q = gguf.tensor_f32("attnres_q.weight", device)?.to_vec2::<f32>()?;
        let speaker_proj = gguf.tensor_f32("speaker_proj.weight", device)?;
        let speaker_bias = gguf.tensor_f32("speaker_proj.bias", device)?;

        let mut cond_embd = Vec::new();
        for i in 0..cfg.condition_bucket_counts.len() {
            let name = format!("cond_embd.{i}.weight");
            cond_embd.push(gguf.tensor_f16(&name, device)?);
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let attn_norm = gguf.tensor_f32(&format!("blk.{l}.attn_norm.weight"), device)?;
            let ffn_norm = gguf.tensor_f32(&format!("blk.{l}.ffn_norm.weight"), device)?;

            let mixer = if cfg.is_attention_layer(l) {
                Mixer::Attention(AttentionLayer {
                    q: gguf.mat(&format!("blk.{l}.attn_q.weight"), device)?,
                    k: gguf.mat(&format!("blk.{l}.attn_k.weight"), device)?,
                    v: gguf.mat(&format!("blk.{l}.attn_v.weight"), device)?,
                    gate: gguf.mat(&format!("blk.{l}.attn_gate.weight"), device)?,
                    temp: gguf.tensor_f32(&format!("blk.{l}.attn_temp"), device)?.to_vec1::<f32>()?,
                    wo: gguf.mat(&format!("blk.{l}.attn_output.weight"), device)?,
                })
            } else {
                let conv = gguf.tensor_f32(&format!("blk.{l}.gdn_conv.weight"), device)?;
                Mixer::Gdn(GdnLayer {
                    in_proj: gguf.mat(&format!("blk.{l}.gdn_in.weight"), device)?,
                    conv: conv.to_vec2::<f32>()?,
                    out_weight: gguf
                        .tensor_f32(&format!("blk.{l}.gdn_norm.weight"), device)?
                        .to_vec1::<f32>()?,
                    a_log: gguf.tensor_f32(&format!("blk.{l}.gdn_A_log"), device)?.to_vec1::<f32>()?,
                    dt_bias: gguf
                        .tensor_f32(&format!("blk.{l}.gdn_dt_bias"), device)?
                        .to_vec1::<f32>()?,
                    wo: gguf.mat(&format!("blk.{l}.attn_output.weight"), device)?,
                })
            };

            let ffn = if l == 0 {
                Ffn::Dense(DenseLayer {
                    up: gguf.mat("blk.0.ffn_up.weight", device)?,
                    gate: gguf.mat("blk.0.ffn_gate.weight", device)?,
                    down: gguf.mat("blk.0.ffn_down.weight", device)?,
                })
            } else {
                let n_experts = gguf.stack_len(&format!("blk.{l}.ffn_gate_exps.weight"))?;
                let mut gate_exps = Vec::with_capacity(n_experts);
                let mut up_exps = Vec::with_capacity(n_experts);
                let mut down_exps = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    gate_exps.push(gguf.mat_item(&format!("blk.{l}.ffn_gate_exps.weight"), e, device)?);
                    up_exps.push(gguf.mat_item(&format!("blk.{l}.ffn_up_exps.weight"), e, device)?);
                    down_exps.push(gguf.mat_item(&format!("blk.{l}.ffn_down_exps.weight"), e, device)?);
                }
                Ffn::Moe(MoeLayer {
                    router: gguf.mat(&format!("blk.{l}.router.weight"), device)?,
                    router_norm: gguf.tensor_f32(&format!("blk.{l}.router_norm.weight"), device)?,
                    router_bias: gguf
                        .tensor_f32(&format!("blk.{l}.router_bias"), device)?
                        .to_vec1::<f32>()?,
                    latent_down: gguf.mat(&format!("blk.{l}.ffn_latent_down.weight"), device)?,
                    latent_up: gguf.mat(&format!("blk.{l}.ffn_latent_up.weight"), device)?,
                    gate_exps,
                    up_exps,
                    down_exps,
                    moe_hidden: cfg.moe_hidden,
                    routed_dim: cfg.moe_routed_dim,
                })
            };

            layers.push(Layer {
                attn_norm,
                ffn_norm,
                mixer,
                ffn,
            });
        }

        let gdn_states: Vec<Vec<f32>> = (0..cfg.n_layers)
            .map(|_| vec![0.0; cfg.gdn_heads * cfg.gdn_head_dim * cfg.gdn_head_dim])
            .collect();
        let gdn_histories: Vec<Vec<f32>> = (0..cfg.n_layers)
            // История depthwise-свёртки: qkv-каналы (3*inner) × (kernel-1).
            .map(|_| vec![0.0; 3 * cfg.gdn_inner() * (cfg.gdn_kernel - 1)])
            .collect();
        let attn_k_exp: Vec<Vec<f32>> = (0..cfg.n_layers).map(|_| Vec::new()).collect();
        let attn_v_exp: Vec<Vec<f32>> = (0..cfg.n_layers).map(|_| Vec::new()).collect();

        Ok(Self {
            cfg,
            device: device.clone(),
            audio_embd,
            text_embd,
            output,
            output_norm,
            attnres_q,
            speaker_proj,
            speaker_bias,
            cond_embd,
            layers,
            gdn_states,
            gdn_histories,
            attn_k_exp,
            attn_v_exp,
            position: 0,
        })
    }

    /// Сбросить состояние декодирования (новый запрос).
    pub fn reset(&mut self) {
        self.position = 0;
        let cfg = &self.cfg;
        for l in 0..self.layers.len() {
            self.gdn_states[l] = vec![0.0; cfg.gdn_heads * cfg.gdn_head_dim * cfg.gdn_head_dim];
            self.gdn_histories[l] = vec![0.0; 3 * cfg.gdn_inner() * (cfg.gdn_kernel - 1)];
            self.attn_k_exp[l].clear();
            self.attn_v_exp[l].clear();
        }
    }

    /// Прямой проход по промпту/генерации.
    ///
    /// `ids` — строки промпта, в каждой 11 колонок: 10 аудиокниг и текст.
    /// `speaker` — эмбеддинг голоса и номер строки, которую он заменяет;
    /// `cond_rows` — номера строк условий и их бакеты (по порядку таблиц).
    pub fn forward(
        &mut self,
        ids: &[Vec<u32>],
        speaker: Option<(&[f32], usize)>,
        cond_rows: &[(usize, usize)],
    ) -> Result<Tensor> {
        let rows = ids.len();
        if rows == 0 {
            bail!("Пустой промпт");
        }

        let mut x: Option<Tensor> = None;
        for cb in 0..self.cfg.n_codebooks {
            let idx: Vec<u32> = ids.iter().map(|r| r[cb]).collect();
            let idx = Tensor::from_vec(idx, &[rows], &self.device)?;
            let part = self.audio_embd[cb].index_select(&idx, 0)?.to_dtype(candle_core::DType::F32)?;
            x = Some(match x {
                None => part,
                Some(prev) => prev.broadcast_add(&part)?,
            });
        }
        let text_ids: Vec<u32> = ids.iter().map(|r| r[self.cfg.n_codebooks]).collect();
        let text_idx = Tensor::from_vec(text_ids, &[rows], &self.device)?;
        let text_part = self
            .text_embd
            .index_select(&text_idx, 0)?
            .to_dtype(candle_core::DType::F32)?;
        let mut x = x.context("Эмбеддинги не собраны")?.broadcast_add(&text_part)?;

        let mut extras: Vec<Tensor> = Vec::new();
        let mut owners: Vec<usize> = Vec::new();
        if let Some((vec, row)) = speaker {
            let v = Tensor::from_vec(vec.to_vec(), (1, vec.len()), &self.device)?;
            let projected = v.matmul(&self.speaker_proj.t()?)?.broadcast_add(&self.speaker_bias)?;
            extras.push(projected);
            owners.push(row);
        }
        for (i, (row, bucket)) in cond_rows.iter().enumerate() {
            let idx = Tensor::from_vec(vec![*bucket as u32], &[1], &self.device)?;
            // cond_embd хранится в F16: к F32-вычислениям приводим сразу,
            // как audio_embd/text_embd.
            let emb = self.cond_embd[i]
                .index_select(&idx, 0)?
                .to_dtype(candle_core::DType::F32)?;
            extras.push(emb);
            owners.push(*row);
        }
        if !extras.is_empty() {
            // Замена строк до emb_norm (zonos3-graph.cpp): emb = emb*keep +
            // table[sel]. keep обнуляет заменяемые строки, table — нулевой
            // столбец + векторы замен (speaker после speaker_proj, cond_embd);
            // не-заменяемые строки берут нулевой столбец и остаются собой.
            let mut keep = vec![1.0_f32; rows];
            let mut sel = vec![0_u32; rows];
            for (extra_idx, owner) in owners.iter().enumerate() {
                keep[*owner] = 0.0;
                sel[*owner] = extra_idx as u32 + 1;
            }
            let zero = Tensor::zeros((1, self.cfg.dim), candle_core::DType::F32, &self.device)?;
            let table = Tensor::cat(&[&zero, &Tensor::cat(extras.as_slice(), 0)?], 0)?;
            let keep = Tensor::from_vec(keep, (rows, 1), &self.device)?;
            let sel = Tensor::from_vec(sel, &[rows], &self.device)?;
            x = x
                .broadcast_mul(&keep)?
                .add(&table.index_select(&sel, 0)?)?;
        }

        x = rms_norm(&x, self.cfg.norm_eps)?;

        let eps = self.cfg.norm_eps;
        let mut sources: Vec<Tensor> = vec![x.clone()];
        let mut normed: Vec<Tensor> = vec![rms_norm(&x, eps)?];

        for l in 0..self.cfg.n_layers {
            let attn_norm = self.layers[l].attn_norm.clone();
            let ffn_norm = self.layers[l].ffn_norm.clone();
            let is_attn = !self.cfg.uses_gdn(l);

            let mixed = attnres_route(&sources, &normed, &self.attnres_q[2 * l])?;
            let attn_in = rms_norm_weighted(&mixed, &attn_norm, eps)?;
            let attention = if is_attn {
                self.attention_forward(l, &attn_in)?
            } else {
                self.gdn_forward(l, &attn_in)?
            };
            let n = rms_norm(&attention, eps)?;
            sources.push(attention);
            normed.push(n);

            let mixed2 = attnres_route(&sources, &normed, &self.attnres_q[2 * l + 1])?;
            let ffn_in = rms_norm_weighted(&mixed2, &ffn_norm, eps)?;
            let last = match &self.layers[l].ffn {
                Ffn::Dense(d) => dense_forward(d, &ffn_in)?,
                Ffn::Moe(m) => moe_forward(m, &ffn_in)?,
            };
            let n = rms_norm(&last, eps)?;
            sources.push(last);
            normed.push(n);
        }

        let mixed = attnres_route(&sources, &normed, &self.attnres_q[self.cfg.n_sources() - 1])?;
        let normalized = rms_norm_weighted(&mixed, &self.output_norm, self.cfg.norm_eps)?;
        let logits = self.output.forward(&normalized)?;
        let logits = logits.reshape((rows, self.cfg.n_codebooks, self.cfg.audio_vocab))?;
        let cap = self.cfg.logit_softcap;
        let logits = logits
            .affine(1.0 / cap as f64, 0.0)?
            .tanh()?
            .affine(cap as f64, 0.0)?;
        self.position += rows;
        Ok(logits)
    }

    /// Слой Gated DeltaNet.
    fn gdn_forward(&mut self, layer_idx: usize, x: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let rows = x.dim(0)?;
        let heads = cfg.gdn_heads;
        let hd = cfg.gdn_head_dim;
        let inner = cfg.gdn_inner();
        let kernel = cfg.gdn_kernel;
        let channels = 3 * inner;

        let proj = {
            let layer = &self.layers[layer_idx];
            let Mixer::Gdn(g) = &layer.mixer else {
                bail!("Слой {layer_idx} не GDN");
            };
            g.in_proj.forward(x)?.to_vec2::<f32>()?
        };

        // qkv: первые 3*inner колонок; z: следующие inner; aa и b — по 4.
        let mut qkv = vec![0f32; rows * channels];
        let mut z = vec![0f32; rows * inner];
        let mut aa = vec![0f32; rows * heads];
        let mut bb = vec![0f32; rows * heads];
        for r in 0..rows {
            let row = &proj[r];
            qkv[r * channels..(r + 1) * channels].copy_from_slice(&row[..channels]);
            z[r * inner..(r + 1) * inner].copy_from_slice(&row[channels..channels + inner]);
            aa[r * heads..(r + 1) * heads]
                .copy_from_slice(&row[channels + inner..channels + inner + heads]);
            bb[r * heads..(r + 1) * heads]
                .copy_from_slice(&row[channels + inner + heads..channels + inner + 2 * heads]);
        }

        let (conv_w, out_weight, a_log, dt_bias) = {
            let layer = &self.layers[layer_idx];
            let Mixer::Gdn(g) = &layer.mixer else {
                bail!("Слой {layer_idx} не GDN");
            };
            (
                g.conv.clone(),
                g.out_weight.clone(),
                g.a_log.clone(),
                g.dt_bias.clone(),
            )
        };
        let state = &mut self.gdn_states[layer_idx];
        let history = &mut self.gdn_histories[layer_idx];

        // Depthwise causal conv (kernel 4) + SiLU, с историей слоя.
        let seq_len = rows + kernel - 1;
        let mut seq = vec![0f32; channels * seq_len];
        let hist_len = kernel - 1;
        for c in 0..channels {
            seq[c * seq_len..c * seq_len + hist_len]
                .copy_from_slice(&history[c * hist_len..(c + 1) * hist_len]);
            for r in 0..rows {
                seq[c * seq_len + hist_len + r] = qkv[r * channels + c];
            }
        }

        let mut conv = vec![0f32; channels * rows];
        for c in 0..channels {
            for r in 0..rows {
                let mut acc = 0f32;
                for j in 0..kernel {
                    acc += conv_w[c][j] * seq[c * seq_len + r + j];
                }
                conv[c * rows + r] = silu(acc);
            }
        }
        let mut new_history = vec![0f32; channels * hist_len];
        for c in 0..channels {
            new_history[c * hist_len..(c + 1) * hist_len]
                .copy_from_slice(&seq[c * seq_len + seq_len - hist_len..c * seq_len + seq_len]);
        }
        history.copy_from_slice(&new_history);

        // q/k/v в fp32: каналы conv — q [0..I), k [I..2I), v [2I..3I);
        // q/k нормируются (FLA-семантика: /sqrt(Σx²+1e-6)), v — как есть.
        let mut q = vec![0f32; rows * inner];
        let mut k = vec![0f32; rows * inner];
        let mut v = vec![0f32; rows * inner];
        for r in 0..rows {
            for h in 0..heads {
                let off = h * hd;
                let mut qs = 0f32;
                let mut ks = 0f32;
                for d in 0..hd {
                    let qv = conv[(h * hd + d) * rows + r];
                    let kv = conv[(inner + h * hd + d) * rows + r];
                    let vv = conv[(2 * inner + h * hd + d) * rows + r];
                    q[r * inner + off + d] = qv;
                    k[r * inner + off + d] = kv;
                    v[r * inner + off + d] = vv;
                    qs += qv * qv;
                    ks += kv * kv;
                }
                let qn = 1.0 / (qs + 1e-6).sqrt();
                let kn = 1.0 / (ks + 1e-6).sqrt();
                for d in 0..hd {
                    q[r * inner + off + d] *= qn;
                    k[r * inner + off + d] *= kn;
                }
            }
        }

        let mut decay = vec![0f32; rows * heads];
        let mut beta = vec![0f32; rows * heads];
        for r in 0..rows {
            for h in 0..heads {
                let a = aa[r * heads + h] + dt_bias[h];
                decay[r * heads + h] = (-a_log[h].exp() * softplus(a)).exp();
                beta[r * heads + h] = sigmoid(bb[r * heads + h]);
            }
        }

        // Delta-rule рекурсия (fp32): state[head, value, key].
        let scale = 1.0 / (hd as f32).sqrt();
        let mut recurrent = vec![0f32; rows * inner];
        for r in 0..rows {
            for h in 0..heads {
                let st = &mut state[h * hd * hd..(h + 1) * hd * hd];
                let d = decay[r * heads + h];
                let beta_h = beta[r * heads + h];
                let koff = h * hd;
                for j in 0..hd {
                    let mut pred = 0f32;
                    for i in 0..hd {
                        pred += st[j * hd + i] * k[r * inner + koff + i];
                    }
                    let delta = beta_h * (v[r * inner + koff + j] - pred);
                    for i in 0..hd {
                        st[j * hd + i] = st[j * hd + i] * d + delta * k[r * inner + koff + i];
                    }
                }
                for j in 0..hd {
                    let mut acc = 0f32;
                    for i in 0..hd {
                        acc += st[j * hd + i] * q[r * inner + koff + i];
                    }
                    recurrent[r * inner + koff + j] = acc * scale;
                }
            }
        }

        // Per-head RMSNorm (eps 1e-5) + аффинный вес + gate silu(z).
        let mut gated = vec![0f32; rows * inner];
        for r in 0..rows {
            for h in 0..heads {
                let off = h * hd;
                let mut ss = 0f32;
                for d in 0..hd {
                    let val = recurrent[r * inner + off + d];
                    ss += val * val;
                }
                let inv = 1.0 / (ss / hd as f32 + 1e-5).sqrt();
                for d in 0..hd {
                    let normed = recurrent[r * inner + off + d] * inv * out_weight[d];
                    gated[r * inner + off + d] = normed * silu(z[r * inner + off + d]);
                }
            }
        }

        let gated = Tensor::from_vec(gated, &[rows, inner], &self.device)?;
        let layer = &self.layers[layer_idx];
        let Mixer::Gdn(g) = &layer.mixer else {
            bail!("Слой {layer_idx} не GDN");
        };
        g.wo.forward(&gated)
    }

    /// Слой глобального внимания.
    fn attention_forward(&mut self, layer_idx: usize, x: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let rows = x.dim(0)?;
        let nh = cfg.n_heads;
        let nkv = cfg.n_kv_heads;
        let hd = cfg.head_dim;

        let (wq, wk, wv, wgate, temp, wo) = {
            let layer = &self.layers[layer_idx];
            let Mixer::Attention(a) = &layer.mixer else {
                bail!("Слой {layer_idx} не глобальное внимание");
            };
            (
                a.q.clone(),
                a.k.clone(),
                a.v.clone(),
                a.gate.clone(),
                a.temp.clone(),
                a.wo.clone(),
            )
        };

        let gate = candle_nn::ops::sigmoid(&wgate.forward(x)?)?;

        let q = wq.forward(x)?.reshape((rows, nh, hd))?;
        let k = wk.forward(x)?.reshape((rows, nkv, hd))?;
        let v = wv.forward(x)?.reshape((rows, nkv, hd))?;

        // QK-RMSNorm (без веса) + температура головы.
        let q = rms_norm(&q, cfg.qk_norm_eps)?;
        let k = rms_norm(&k, cfg.qk_norm_eps)?;
        // Температура головы: [nh] → [1, nh, 1] для broadcast на [rows, nh, hd].
        let temp = Tensor::from_vec(temp, (1, nh, 1), &self.device)?;
        let q = q.broadcast_mul(&temp)?;

        let pos: Vec<u32> = (self.position..self.position + rows).map(|p| p as u32).collect();
        let pos = Tensor::from_vec(pos, &[rows], &self.device)?;
        let q = rope_interleaved(&q, &pos, cfg.rope_theta)?;
        let k = rope_interleaved(&k, &pos, cfg.rope_theta)?;

        // Кэш K/V: дописываем новые строки сразу в GQA-раскладке
        // (4 KV-головы, размноженные до 16 query-голов), без пересборки банка.
        let rep = nh / nkv;
        let (k_all, v_all) = {
            let kv_new = k.to_vec3::<f32>()?;
            let vv_new = v.to_vec3::<f32>()?;
            let k_cache = &mut self.attn_k_exp[layer_idx];
            let v_cache = &mut self.attn_v_exp[layer_idx];
            for r in 0..rows {
                for h in 0..nh {
                    let kv_head = h / rep;
                    k_cache.extend_from_slice(&kv_new[r][kv_head]);
                    v_cache.extend_from_slice(&vv_new[r][kv_head]);
                }
            }
            let total = k_cache.len() / (nh * hd);
            (
                Tensor::from_vec(k_cache.clone(), &[total, nh, hd], &self.device)?,
                Tensor::from_vec(v_cache.clone(), &[total, nh, hd], &self.device)?,
            )
        };
        let total = k_all.dim(0)?;;

        let qh = q.transpose(0, 1)?.contiguous()?;
        let kh = k_all.transpose(0, 1)?.contiguous()?;
        let vh = v_all.transpose(0, 1)?.contiguous()?;
        let scores = qh.matmul(&kh.transpose(1, 2)?)?; // [nh, rows, total]
        let scores = scores.affine(1.0 / (hd as f32).sqrt() as f64, 0.0)?;
        // Каузальная маска: ключ не может быть «в будущем».
        let mut mask = vec![0f32; rows * total];
        for r in 0..rows {
            let qp = (self.position + r) as f32;
            for kp in 0..total {
                if (kp as f32) > qp {
                    mask[r * total + kp] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, rows, total), &self.device)?;
        let scores = scores.broadcast_add(&mask)?;
        let probs = softmax(&scores, 2)?;
        let ctx = probs.matmul(&vh)?; // [nh, rows, hd]
        let ctx = ctx.transpose(0, 1)?.contiguous()?; // [rows, nh, hd]
        let gate = gate.reshape((rows, nh, 1))?;
        let ctx = ctx.broadcast_mul(&gate)?;
        let ctx = ctx.reshape((rows, nh * hd))?;
        wo.forward(&ctx)
    }
}

/// RMSNorm без веса: `x * rsqrt(mean_row(x²) + eps)` — норма по последней
/// размерности каждой строки (ggml_rms_norm).
pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    Ok(x.broadcast_mul(&rsqrt_last(x, eps)?)?)
}

/// RMSNorm с весом: тот же норм и умножение на вес последней размерности.
pub fn rms_norm_weighted(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let normed = x.broadcast_mul(&rsqrt_last(x, eps)?)?;
    Ok(normed.broadcast_mul(&w.unsqueeze(0)?)?)
}

/// rsqrt(mean(x²) + eps) по последней размерности: candle убирает
/// редуцированную размерность, возвращаем её как 1 — ранг совпадает с x
/// и broadcast работает для любой размерности входа.
fn rsqrt_last(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let n = dims[dims.len() - 1] as f64;
    let mean = x
        .sqr()?
        .sum(candle_core::D::Minus1)?
        .unsqueeze(candle_core::D::Minus1)?
        .affine(1.0 / n, 0.0)?;
    Ok(mean.affine(1.0, eps as f64)?.sqrt()?.recip()?)
}

/// Маршрутизация банка Full AttnRes: softmax по глубине банка, взвешенная
/// сумма источников. `normed` — заранее посчитанные RMS-нормы тех же
/// источников; суммирование идёт без стек-копии банка.
fn attnres_route(sources: &[Tensor], normed: &[Tensor], query: &[f32]) -> Result<Tensor> {
    let dim = sources[0].dim(1)?;
    let q = Tensor::from_vec(query.to_vec(), &[dim], sources[0].device())?;
    let q = q.unsqueeze(0)?;

    let mut scores: Vec<Tensor> = Vec::with_capacity(normed.len());
    for k in normed {
        scores.push(k.broadcast_mul(&q)?.sum_keepdim(1)?);
    }
    let stacked = Tensor::stack(scores.as_slice(), 0)?; // [S, rows, 1]
    let weights = softmax(&stacked, 0)?; // [S, rows, 1]

    let n = sources.len();
    let mut acc: Option<Tensor> = None;
    for i in 0..n {
        let w = weights.i(i)?; // [rows, 1]
        let term = sources[i].broadcast_mul(&w)?;
        acc = Some(match acc {
            None => term,
            Some(prev) => prev.add(&term)?,
        });
    }
    acc.context("пустой банк источников")
}

/// Плотный SwiGLU первого слоя: `up * silu(gate)`.
fn dense_forward(d: &DenseLayer, x: &Tensor) -> Result<Tensor> {
    let up = d.up.forward(x)?;
    let gate = d.gate.forward(x)?;
    let h = up.mul(&candle_nn::ops::silu(&gate)?)?;
    d.down.forward(&h)
}

/// Латентный MoE: линейный роутер, top-2 из 64, SwiGLU-эксперты.
fn moe_forward(m: &MoeLayer, x: &Tensor) -> Result<Tensor> {
    let rows = x.dim(0)?;
    let n_experts = m.router_bias.len();

    let r = rms_norm_weighted(x, &m.router_norm, 1e-5)?;
    let logits = m.router.forward(&r)?; // [rows, 64]
    let logits = logits.to_vec2::<f32>()?;

    // Выбор экспертов: top-k по (логит − balancing bias).
    let mut choices = vec![[0usize; 2]; rows];
    let mut weights = vec![[0f32; 2]; rows];
    for t in 0..rows {
        let mut idx: Vec<usize> = (0..n_experts).collect();
        idx.sort_by(|&a, &b| {
            let sa = logits[t][a] - m.router_bias[a];
            let sb = logits[t][b] - m.router_bias[b];
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        choices[t] = [idx[0], idx[1]];
        // Веса — softmax по сырым логитам выбранных.
        let l0 = logits[t][idx[0]];
        let l1 = logits[t][idx[1]];
        let m0 = l0.max(l1);
        let e0 = (l0 - m0).exp();
        let e1 = (l1 - m0).exp();
        let sum = e0 + e1;
        weights[t] = [e0 / sum, e1 / sum];
    }

    // Латентная проекция — от «сырого» входа, не от нормализованного.
    let latent = m.latent_down.forward(x)?; // [rows, 256]

    let mut out = vec![0f32; rows * m.routed_dim];
    for rank in 0..2 {
        // Группируем токены по эксперту: один matmul на группу.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for t in 0..rows {
            let e = choices[t][rank];
            match groups.iter_mut().find(|(ge, _)| *ge == e) {
                Some((_, list)) => list.push(t),
                None => groups.push((e, vec![t])),
            }
        }
        for (e, tokens) in groups {
            let idx: Vec<u32> = tokens.iter().map(|t| *t as u32).collect();
            let idx = Tensor::from_vec(idx, &[tokens.len()], x.device())?;
            let lat = latent.index_select(&idx, 0)?;
            let gate = m.gate_exps[e].forward(&lat)?;
            let up = m.up_exps[e].forward(&lat)?;
            // SwiGLU: silu на гейте (up * silu(gate)), как в эталоне.
            let hidden = up.mul(&candle_nn::ops::silu(&gate)?)?;
            let expert_out = m.down_exps[e].forward(&hidden)?; // [n, routed]
            let values = expert_out.to_vec2::<f32>()?;
            for (i, t) in tokens.iter().enumerate() {
                let w = weights[*t][rank];
                for r in 0..m.routed_dim {
                    out[t * m.routed_dim + r] += values[i][r] * w;
                }
            }
        }
    }

    let out = Tensor::from_vec(out, &[rows, m.routed_dim], x.device())?;
    m.latent_up.forward(&out)
}

/// Interleaved RoPE: пары соседних элементов поворачиваются на угол
/// `pos * theta^(-2i/head_dim)`.
fn rope_interleaved(x: &Tensor, positions: &Tensor, theta: f32) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let rows = dims[0];
    let heads = dims[1];
    let hd = dims[2];
    let pairs = hd / 2;

    let freqs: Vec<f32> = (0..pairs)
        .map(|i| theta.powf(-2.0 * i as f32 / hd as f32))
        .collect();
    let pos = positions.to_vec1::<u32>()?;
    let mut cos = vec![0f32; rows * pairs];
    let mut sin = vec![0f32; rows * pairs];
    for r in 0..rows {
        for i in 0..pairs {
            let angle = pos[r] as f32 * freqs[i];
            cos[r * pairs + i] = angle.cos();
            sin[r * pairs + i] = angle.sin();
        }
    }

    let data = x.to_vec3::<f32>()?;
    let mut out = vec![0f32; rows * heads * hd];
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            for i in 0..pairs {
                let a = data[r][h][2 * i];
                let b = data[r][h][2 * i + 1];
                let c = cos[r * pairs + i];
                let s = sin[r * pairs + i];
                out[base + 2 * i] = a * c - b * s;
                out[base + 2 * i + 1] = a * s + b * c;
            }
        }
    }
    Ok(Tensor::from_vec(out, &[rows, heads, hd], x.device())?)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}
