//! Спикер-энкодер ZONOS3 (ownmect_s2): MECT-B2 + ASTP, 192-мерный эмбеддинг.
//!
//! Порт ggml-реализации `servers/ggml/src/spk-encoder.cpp` из
//! Zyphra/ZONOS3-Mini (архитектура `mect_spk`) на чистом Rust:
//!
//! 1. фронтенд (хост): 44.1 кГц → 16 кГц точным полифазным ядром из GGUF
//!    (прочие частоты — оконный sinc), Kaldi-fbank 80 мел-полос (окно
//!    Хэмминга 400, сдвиг 160, n_fft 512, предакцент 0.97) и CMN;
//! 2. MECT-B2: 4 стадии блоков (групповые 3×3-свёртки + pointwise + BN +
//!    трансформер на 64 каналах: позиционная свёртка, внимание, MoE-FFN);
//! 3. слияние стадий (`fe.dense_w`) и ASTP-пулинг по времени, BatchNorm и
//!    линейный выход в 192 измерения.
//!
//! Активации `[время, частота, каналы]`, индекс `(t·F + f)·C + c`.

use super::gguf::GgufFile;
use anyhow::{Context, Result, bail};
use rustfft::{FftDirection, FftPlanner};
use std::path::Path;

const ARCH: &str = "mect_spk";
const FEAT: usize = 2560;
const HIDDEN: usize = 64;
const POS_K: usize = 113;
const N_EXPERTS: usize = 4;
const N_HEADS: usize = 4;
const IN_C: usize = 32;
const IN_F: usize = 80;

/// Блок MECT: глубина CNN, выходные каналы, stride (F, T), expansion.
#[derive(Clone, Copy)]
struct BlockCfg {
    cnn: usize,
    out_ch: usize,
    s_f: usize,
    s_t: usize,
    expansion: usize,
}

/// Раскладка MECT_B2 (backbones/mect.py).
fn mect_b2() -> Vec<Vec<BlockCfg>> {
    fn rep(first: BlockCfg, rest: BlockCfg, n: usize) -> Vec<BlockCfg> {
        let mut v = vec![first];
        v.extend(std::iter::repeat_n(rest, n));
        v
    }
    vec![
        vec![
            BlockCfg { cnn: 3, out_ch: 32, s_f: 1, s_t: 1, expansion: 4 },
            BlockCfg { cnn: 3, out_ch: 32, s_f: 1, s_t: 1, expansion: 4 },
        ],
        rep(
            BlockCfg { cnn: 3, out_ch: 64, s_f: 2, s_t: 1, expansion: 2 },
            BlockCfg { cnn: 3, out_ch: 64, s_f: 1, s_t: 2, expansion: 2 },
            3,
        ),
        rep(
            BlockCfg { cnn: 3, out_ch: 128, s_f: 2, s_t: 1, expansion: 2 },
            BlockCfg { cnn: 3, out_ch: 128, s_f: 1, s_t: 2, expansion: 2 },
            4,
        ),
        rep(
            BlockCfg { cnn: 3, out_ch: 256, s_f: 2, s_t: 1, expansion: 1 },
            BlockCfg { cnn: 2, out_ch: 256, s_f: 1, s_t: 2, expansion: 1 },
            2,
        ),
    ]
}

/// Affine-пара (вес [out, in], смещение [out]).
type Aff = (Vec<f32>, Vec<f32>);

/// Один residual-блок CNN.
struct ResBlockW {
    /// Групповая 3×3-свёртка: слоты, каждый [oc × 9 отводов].
    conv1: Vec<Vec<f32>>,
    conv2: Vec<Vec<f32>>,
    pw1: Aff,
    pw2: Aff,
    bn1: Aff,
    bn2: Aff,
    /// Shortcut 1×1 (без смещения) + BN — есть у каждого R=0.
    sc: Option<(Vec<f32>, Aff)>,
}

/// Один блок MECT.
struct BlockW {
    res: Vec<ResBlockW>,
    /// Сведение каналов (expansion > 1): affine (вес+смещение) + BN.
    reduce: Option<(Aff, Aff)>,
    rproj: Aff,
    rproj_ln: Aff,
    pos_dw: Vec<f32>,
    pos_dw_b: Vec<f32>,
    pos_bn: Aff,
    pos_pw: Aff,
    attn_q: Aff,
    attn_k: Aff,
    attn_v: Aff,
    attn_o: Aff,
    ln1: Aff,
    ln2: Aff,
    moe_gate: Vec<f32>,
    moe_experts: Vec<f32>,
    ffn_out: Aff,
    eproj: Aff,
}

/// Спикер-энкодер: веса + фронтенд.
pub struct SpeakerEncoder {
    n_mels: usize,
    frame_length: usize,
    frame_shift: usize,
    n_fft: usize,
    preemph: f32,
    waveform_scale: f32,
    log_floor: f32,
    ln_eps: f32,
    pool_var_floor: f32,
    embedding_length: usize,
    window: Vec<f32>,
    /// Мел-банки: [n_mels × bins] (строка — мел-полоса).
    mel_banks: Vec<f32>,
    /// Полифазное ядро 44100 → 16000: [rate × kernel_size] (строка — фаза).
    rs_kernel: Vec<f32>,
    head_conv: Aff,
    head_bn: Aff,
    blocks: Vec<BlockW>,
    dense_w: Vec<f32>,
    pool_l1: Aff,
    pool_l2: Aff,
    bn: Aff,
    linear: Vec<f32>,
}

fn flat(t: &candle_core::Tensor) -> Result<Vec<f32>> {
    let n = t.elem_count();
    Ok(t.reshape(&[n])?.to_vec1::<f32>()?)
}

impl SpeakerEncoder {
    /// Открыть `spk-encoder.gguf`.
    pub fn open(path: &Path) -> Result<Self> {
        let file = GgufFile::open(path)?;
        let kv = &file.kv;
        let arch = file.kv_str("general.architecture").unwrap_or_default();
        if arch != ARCH {
            bail!("Ожидалась архитектура {ARCH}, получено «{arch}»");
        }
        if file.kv_str(&format!("{ARCH}.backbone")).unwrap_or_default() != "MECT_B2" {
            bail!("Бэкбон спикер-энкодера не MECT_B2");
        }
        let u32v = |key: &str| -> Result<usize> {
            kv.get(&format!("{ARCH}.{key}"))
                .and_then(|v| v.as_i64())
                .map(|v| v as usize)
                .with_context(|| format!("Нет {ARCH}.{key}"))
        };
        let f32v = |key: &str| -> Result<f32> {
            let mut name = format!("{ARCH}.{key}");
            // В GGUF конвертера ключ предакцента называется «preemphasis».
            if key == "preemph" {
                name = format!("{ARCH}.preemphasis");
            }
            kv.get(&name)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .with_context(|| format!("Нет {name}"))
        };
        if u32v("sample_rate")? != 16000
            || u32v("embedding_length")? != 192
            || u32v("n_mels")? != 80
            || u32v("frame_length")? != 400
            || u32v("frame_shift")? != 160
            || u32v("n_fft")? != 512
            || u32v("n_heads")? != N_HEADS
            || u32v("n_experts")? != N_EXPERTS
            || u32v("pos_kernel")? != POS_K
        {
            bail!("Неожиданные параметры MECT-B2 в {path:?}");
        }

        let t32 = |name: &str| -> Result<Vec<f32>> {
            flat(&file
                .tensor_f32(name, &candle_core::Device::Cpu)
                .with_context(|| format!("Нет тензора «{name}»"))?)
        };
        let affine = |name: &str| -> Result<Aff> {
            Ok((t32(&format!("{name}.weight"))?, t32(&format!("{name}.bias"))?))
        };
        let bn = |name: &str| -> Result<Aff> {
            Ok((t32(&format!("{name}.scale"))?, t32(&format!("{name}.shift"))?))
        };

        let head_conv = affine("fe.head.conv")?;
        let head_bn = bn("fe.head.bn")?;

        let mut blocks = Vec::new();
        let stages = mect_b2();
        let mut c_in = IN_C;
        let mut f_in = IN_F;
        for (s, stage) in stages.iter().enumerate() {
            for (k, bc) in stage.iter().enumerate() {
                let p = format!("fe.b{}.{}.", s + 1, k);
                let oc = bc.out_ch * bc.expansion;
                let mut res = Vec::with_capacity(bc.cnn);
                for r in 0..bc.cnn {
                    let q = format!("{p}cnn.{r}.");
                    let ic = if r == 0 { c_in } else { oc };
                    // Группы: R=0 — входные каналы блока, R≥1 — свои входные.
                    let groups = if r == 0 { c_in } else { oc };
                    let a1 = ic / groups;
                    let a2 = oc / groups;
                    let mut conv1 = Vec::with_capacity(a1);
                    for j in 0..a1 {
                        conv1.push(t32(&format!("{q}conv1.w{j}"))?);
                    }
                    let mut conv2 = Vec::with_capacity(a2);
                    for j in 0..a2 {
                        conv2.push(t32(&format!("{q}conv2.w{j}"))?);
                    }
                    // Shortcut: R=0 у стадий 2-4 (tuple-stride) и стадии 1
                    // (входные каналы 32 ≠ выходных 128).
                    let sc = if r == 0 && (s >= 1 || ic != oc) {
                        Some((t32(&format!("{q}sc.weight"))?, bn(&format!("{q}sc.bn"))?))
                    } else {
                        None
                    };
                    res.push(ResBlockW {
                        conv1,
                        conv2,
                        pw1: affine(&format!("{q}pw1"))?,
                        pw2: affine(&format!("{q}pw2"))?,
                        bn1: bn(&format!("{q}bn1"))?,
                        bn2: bn(&format!("{q}bn2"))?,
                        sc,
                    });
                }
                let reduce = if bc.expansion > 1 {
                    Some((affine(&format!("{p}reduce"))?, bn(&format!("{p}reduce.bn"))?))
                } else {
                    None
                };
                blocks.push(BlockW {
                    res,
                    reduce,
                    rproj: affine(&format!("{p}rproj"))?,
                    rproj_ln: affine(&format!("{p}rproj.ln"))?,
                    pos_dw: t32(&format!("{p}pos.dw.weight"))?,
                    pos_dw_b: t32(&format!("{p}pos.dw.bias"))?,
                    pos_bn: bn(&format!("{p}pos.bn"))?,
                    pos_pw: affine(&format!("{p}pos.pw"))?,
                    attn_q: affine(&format!("{p}attn.q"))?,
                    attn_k: affine(&format!("{p}attn.k"))?,
                    attn_v: affine(&format!("{p}attn.v"))?,
                    attn_o: affine(&format!("{p}attn.o"))?,
                    ln1: affine(&format!("{p}ln1"))?,
                    ln2: affine(&format!("{p}ln2"))?,
                    moe_gate: t32(&format!("{p}moe.gate.weight"))?,
                    moe_experts: t32(&format!("{p}moe.experts.weight"))?,
                    ffn_out: affine(&format!("{p}ffn_out"))?,
                    eproj: affine(&format!("{p}eproj"))?,
                });
                let c_out = if bc.expansion > 1 { c_in * bc.s_f } else { oc };
                let f_out = f_in / bc.s_f;
                if c_out * f_out != FEAT {
                    bail!("Геометрия MECT-B2 не сходится: {c_out}×{f_out}");
                }
                c_in = c_out;
                f_in = f_out;
            }
        }

        Ok(Self {
            n_mels: u32v("n_mels")?,
            frame_length: u32v("frame_length")?,
            frame_shift: u32v("frame_shift")?,
            n_fft: u32v("n_fft")?,
            preemph: f32v("preemph")?,
            waveform_scale: f32v("waveform_scale")?,
            log_floor: f32v("log_floor")?,
            ln_eps: f32v("ln_eps")?,
            pool_var_floor: f32v("pool_var_floor")?,
            embedding_length: u32v("embedding_length")?,
            window: t32("frontend.window")?,
            mel_banks: t32("frontend.mel_banks")?,
            rs_kernel: t32("frontend.resample_44100")?,
            head_conv,
            head_bn,
            blocks,
            dense_w: t32("fe.dense_w")?,
            pool_l1: affine("pool.l1")?,
            pool_l2: affine("pool.l2")?,
            bn: bn("bn")?,
            linear: t32("linear.weight")?,
        })
    }

    /// Эмбеддинг голоса по моно-записи (f32 в диапазоне ±1) и её частоте.
    /// 16000 — как есть, 44100 — точное ядро из GGUF, остальное — оконный sinc.
    pub fn embedding(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        Ok(self.embedding_debug(samples, sample_rate, &mut |_, _, _, _| {})?)
    }

    /// Как [`embedding`], но с покадровым дампом стадий (диагностика):
    /// колбэк получает (имя, данные, строки, столбцы).
    pub fn embedding_debug(
        &self,
        samples: &[f32],
        sample_rate: u32,
        dump: &mut dyn FnMut(&str, &[f32], usize, usize),
    ) -> Result<Vec<f32>> {
        let wav16: Vec<f32> = match sample_rate {
            16000 => samples.to_vec(),
            44100 => self.resample_44100_kernel(samples),
            other => sinc_resample(samples, other, 16000),
        };
        let wav16 = &wav16[..wav16.len().min(60 * 16000)];
        if wav16.len() < self.frame_length {
            bail!("Запись короче 25 мс — эмбеддинг не считается");
        }
        let fbank = self.fbank(wav16)?;
        let t = fbank.len() / self.n_mels;
        dump("fbank", &fbank, t, self.n_mels);
        // CMN: по каждому мел-бину среднее по времени.
        let mut x0 = vec![0.0f32; fbank.len()];
        for b in 0..self.n_mels {
            let mut sum = 0.0f64;
            for i in 0..t {
                sum += fbank[i * self.n_mels + b] as f64;
            }
            let m = (sum / t as f64) as f32;
            for i in 0..t {
                x0[i * self.n_mels + b] = fbank[i * self.n_mels + b] - m;
            }
        }
        dump("fbank_cmn", &x0, t, self.n_mels);
        Ok(self.forward_debug(&x0, t, dump))
    }

    /// Полифазное ядро 44100 → 16000 из GGUF ([160 фаз × 475 отводов]).
    fn resample_44100_kernel(&self, x: &[f32]) -> Vec<f32> {
        let orig = 441usize;
        let rate = 160usize;
        let ks = 475usize;
        let width = (ks - orig) / 2;
        let count = (rate as f64 * x.len() as f64 / orig as f64).ceil() as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let phase = i % rate;
            let row = &self.rs_kernel[phase * ks..(phase + 1) * ks];
            // Окно фазы может заходить за края сигнала — в i64, без underflow.
            let start = (i / rate) as i64 * orig as i64 - width as i64;
            let lo = (-start).clamp(0, ks as i64) as usize;
            let hi = (x.len() as i64 - start).clamp(0, ks as i64) as usize;
            let mut sum = 0.0f32;
            for j in lo..hi {
                sum += row[j] * x[(start + j as i64) as usize];
            }
            out.push(sum);
        }
        out
    }

    /// Kaldi-fbank (torchaudio.compliance.kaldi.fbank): [T, 80].
    fn fbank(&self, wav: &[f32]) -> Result<Vec<f32>> {
        let fl = self.frame_length;
        let nf = self.n_fft;
        let bins = nf / 2 + 1;
        let nm = self.n_mels;
        let t_count = 1 + (wav.len() - fl) / self.frame_shift;
        let mut out = vec![0.0f32; t_count * nm];

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft(nf, FftDirection::Forward);
        let mut spec = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0f32); nf];
        let mut x = vec![0.0f32; fl];
        let mut buf = vec![0.0f32; nf];
        let mut power = vec![0.0f32; bins];

        for t in 0..t_count {
            let src = &wav[t * self.frame_shift..t * self.frame_shift + fl];
            // Масштаб int16 и удаление постоянной составляющей кадра.
            let mut mean = 0.0f64;
            for (j, s) in src.iter().enumerate() {
                x[j] = s * self.waveform_scale;
                mean += x[j] as f64;
            }
            let fmean = (mean / fl as f64) as f32;
            for v in x.iter_mut() {
                *v -= fmean;
            }
            // Предакцент + окно: buf[0] — без сдвига.
            buf[..nf].fill(0.0);
            buf[0] = (x[0] - self.preemph * x[0]) * self.window[0];
            for j in (1..fl).rev() {
                buf[j] = (x[j] - self.preemph * x[j - 1]) * self.window[j];
            }
            for (j, v) in buf.iter().enumerate() {
                spec[j] = rustfft::num_complex::Complex::new(*v, 0.0);
            }
            fft.process(&mut spec);
            power[0] = spec[0].re * spec[0].re;
            for k in 1..bins - 1 {
                power[k] = spec[k].re * spec[k].re + spec[k].im * spec[k].im;
            }
            power[bins - 1] = spec[nf / 2].re * spec[nf / 2].re;
            for b in 0..nm {
                let row = &self.mel_banks[b * bins..(b + 1) * bins];
                let mut acc = 0.0f32;
                for (k, w) in row.iter().enumerate() {
                    acc += w * power[k];
                }
                out[t * nm + b] = acc.max(self.log_floor).ln();
            }
        }
        Ok(out)
    }

    /// Прямой проход MECT-B2 + ASTP: x0 [T, 80] → 192.
    fn forward(&self, x0: &[f32], t: usize) -> Vec<f32> {
        self.forward_debug(x0, t, &mut |_, _, _, _| {})
    }

    /// Прямой проход с дампом стадий (диагностика).
    fn forward_debug(&self, x0: &[f32], t: usize, dump: &mut dyn FnMut(&str, &[f32], usize, usize)) -> Vec<f32> {
        // Голова: conv 3×3 (1 → 32) + BN, без ReLU.
        let mut x = conv2d_3x3(x0, t, IN_F, 1, &self.head_conv.0, Some(&self.head_conv.1), 1, 1);
        x = bn_channels(&x, &self.head_bn.0, &self.head_bn.1);
        dump("head", &x, t, x.len() / t);

        let mut f = IN_F;
        let mut c = IN_C;
        let mut feats: Vec<Vec<f32>> = Vec::new();
        let mut bi = 0usize;
        let mut first_block = true;
        for (si, stage) in mect_b2().iter().enumerate() {
            for bc in stage {
                let (nx, _xf) = if first_block {
                    first_block = false;
                    let r = self.block_dbg(&x, t, f, c, *bc, &self.blocks[bi], &mut |name, data, rows, cols| {
                        dump(&format!("b1_{name}"), data, rows, cols);
                    });
                    dump("b1_out", &r.0, t, r.0.len() / t);
                    r
                } else {
                    self.block(&x, t, f, c, *bc, &self.blocks[bi])
                };
                bi += 1;
                c *= bc.s_f;
                f /= bc.s_f;
                x = nx;
            }
            // feat_S — flatten выхода последнего блока стадии.
            feats.push(x.to_vec());
            dump(&format!("feat{si}"), &feats[si], t, FEAT);
        }
        // Слияние стадий: fused[t, k] = Σ_S feat_S[t, k]·dense_w[k + 2560·S].
        let mut fused = vec![0.0f32; t * FEAT];
        for (s, feat) in feats.iter().enumerate() {
            let w = &self.dense_w[s * FEAT..(s + 1) * FEAT];
            for i in 0..t * FEAT {
                fused[i] += feat[i] * w[i % FEAT];
            }
        }

        dump("fused", &fused, t, FEAT);
        // ASTP: u = tanh(l1(x)); att = l2(u); внимание по времени.
        let mut u = mat_rows(&fused, t, &self.pool_l1, true);
        for v in u.iter_mut() {
            *v = v.tanh();
        }
        let att = mat_rows(&u, t, &self.pool_l2, true);
        let mut mean = vec![0.0f32; FEAT];
        let mut stdv = vec![0.0f32; FEAT];
        for k in 0..FEAT {
            let mut mx = f32::NEG_INFINITY;
            for ti in 0..t {
                mx = mx.max(att[ti * FEAT + k]);
            }
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            let mut wsum2 = 0.0f64;
            for ti in 0..t {
                let e = (att[ti * FEAT + k] - mx).exp() as f64;
                let v = fused[ti * FEAT + k] as f64;
                sum += e;
                wsum += e * v;
                wsum2 += e * v * v;
            }
            let m = wsum / sum;
            let var = wsum2 / sum - m * m;
            mean[k] = m as f32;
            stdv[k] = var.max(self.pool_var_floor as f64).sqrt() as f32;
        }
        let mut pooled = vec![0.0f32; 2 * FEAT];
        pooled[..FEAT].copy_from_slice(&mean);
        pooled[FEAT..].copy_from_slice(&stdv);
        for (i, v) in pooled.iter_mut().enumerate() {
            *v = *v * self.bn.0[i] + self.bn.1[i];
        }
        let mut emb = vec![0.0f32; self.embedding_length];
        for (o, ev) in emb.iter_mut().enumerate() {
            let w = &self.linear[o * 2 * FEAT..(o + 1) * 2 * FEAT];
            let mut acc = 0.0f32;
            for (i, wv) in w.iter().enumerate() {
                acc += wv * pooled[i];
            }
            *ev = acc;
        }
        emb
    }

    /// Один блок MECT: CNN-цепочка → трансформер → вычет по flatten входа.
    /// Возвращает (пространственный выход [T, F', C'], flatten [T, 2560]).
    fn block(&self, x: &[f32], t: usize, f: usize, c: usize, bc: BlockCfg, w: &BlockW) -> (Vec<f32>, Vec<f32>) {
        self.block_dbg(x, t, f, c, bc, w, &mut |_, _, _, _| {})
    }

    fn block_dbg(&self, x: &[f32], t: usize, f: usize, c: usize, bc: BlockCfg, w: &BlockW, dump: &mut dyn FnMut(&str, &[f32], usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let mut y = x.to_vec();
        let mut cf = f;
        let mut cc = c;
        let mut ct = t;
        for (r, rb) in w.res.iter().enumerate() {
            let (s_f, s_t) = if r == 0 { (bc.s_f, bc.s_t) } else { (1, 1) };
            let t1 = conv_time(ct, s_t);
            let f1 = conv_time(cf, s_f);
            let oc = bc.out_ch * bc.expansion;
            // conv1 (групповая 3×3) → pw1 → BN1 → ReLU.
            let mut z = grouped_conv3x3(&y, ct, cf, cc, s_t, s_f, t1, f1, &rb.conv1, oc);
            z = pointwise(&z, t1, f1, oc, &rb.pw1.0, &rb.pw1.1);
            z = bn_channels(&z, &rb.bn1.0, &rb.bn1.1);
            relu(&mut z);
            // conv2 (те же группы) → pw2 → BN2 → ReLU.
            z = grouped_conv3x3(&z, t1, f1, oc, 1, 1, t1, f1, &rb.conv2, oc);
            z = pointwise(&z, t1, f1, oc, &rb.pw2.0, &rb.pw2.1);
            z = bn_channels(&z, &rb.bn2.0, &rb.bn2.1);
            relu(&mut z);
            // Shortcut: 1×1 со stride + BN, либо тождество.
            let out = match &rb.sc {
                Some((sc_w, (sc_s, sc_sh))) => {
                    let s = conv2d_1x1(&y, ct, cf, cc, sc_w, t1, f1, s_t, s_f);
                    add(&z, &bn_channels(&s, sc_s, sc_sh))
                }
                None => add(&z, &y),
            };
            y = out;
            ct = t1;
            cf = f1;
            cc = oc;
        }
        // reduce: сведение каналов к c·s_f (expansion > 1).
        let mut c_out = cc;
        if bc.expansion > 1 {
            let ((rw, rb), (rs, rsh)) = w.reduce.as_ref().expect("нет reduce у блока");
            let z = pointwise(&y, ct, cf, cc, rw, rb);
            y = bn_channels(&z, rs, rsh);
            c_out = c * bc.s_f;
        }
        let f_out = cf;
        dump("cnn", &y, ct, y.len() / ct);

        // Flatten [T', 2560] → трансформер на 64 каналах.
        let x2 = y.clone();
        let mut h = mat_rows(&x2, ct, &w.rproj, true);
        layer_norm(&mut h, &w.rproj_ln.0, &w.rproj_ln.1, self.ln_eps);
        dump("rproj", &h, ct, HIDDEN);
        let pos = self.pos_conv(&h, ct, w);
        add_into(&mut h, &pos);
        dump("pos", &h, ct, HIDDEN);
        let attn = self.attention(&h, ct, w);
        add_into(&mut h, &attn);
        layer_norm(&mut h, &w.ln1.0, &w.ln1.1, self.ln_eps);
        dump("ln1", &h, ct, HIDDEN);
        let moe = self.moe_ffn(&h, ct, w);
        add_into(&mut h, &moe);
        layer_norm(&mut h, &w.ln2.0, &w.ln2.1, self.ln_eps);
        dump("ln2", &h, ct, HIDDEN);
        // eproj + x2.
        let mut e = x2;
        let proj = mat_rows(&h, ct, &w.eproj, true);
        dump("eproj_raw", &proj, ct, FEAT);
        add_into(&mut e, &proj);
        // Nearest-upsample по времени (s_t), обрезка до T, вычет flatten входа.
        let mut e_full = if bc.s_t > 1 {
            let mut up = vec![0.0f32; t * FEAT];
            for ti in 0..t {
                let src = ti / bc.s_t;
                up[ti * FEAT..(ti + 1) * FEAT].copy_from_slice(&e[src * FEAT..(src + 1) * FEAT]);
            }
            up
        } else {
            e
        };
        let x_in_flat = flatten_ref(x, t, f, c);
        add_into(&mut e_full, &x_in_flat);
        (unflatten_ref(&e_full, t, f_out, c_out), e_full)
    }

    /// Позиционная свёртка k=113 (симметричный паддинг 56) + BN + GELU + pw.
    fn pos_conv(&self, h: &[f32], t: usize, w: &BlockW) -> Vec<f32> {
        let pad = POS_K / 2;
        let mut out = vec![0.0f32; t * HIDDEN];
        for ch in 0..HIDDEN {
            let taps = &w.pos_dw[ch * POS_K..(ch + 1) * POS_K];
            for ti in 0..t {
                let mut acc = w.pos_dw_b[ch];
                for (k, wv) in taps.iter().enumerate() {
                    let src = ti as i64 + k as i64 - pad as i64;
                    if src >= 0 && (src as usize) < t {
                        acc += wv * h[src as usize * HIDDEN + ch];
                    }
                }
                out[ti * HIDDEN + ch] = acc;
            }
        }
        for i in 0..out.len() {
            out[i] = out[i] * w.pos_bn.0[i % HIDDEN] + w.pos_bn.1[i % HIDDEN];
        }
        for v in out.iter_mut() {
            *v = gelu_erf(*v);
        }
        mat_rows(&out, t, &w.pos_pw, true)
    }

    /// Многоголовое внимание (4 × 16 каналов, масштаб 1/4) + o-проекция.
    fn attention(&self, h: &[f32], t: usize, w: &BlockW) -> Vec<f32> {
        let hd = HIDDEN / N_HEADS;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = mat_rows(h, t, &w.attn_q, true);
        let k = mat_rows(h, t, &w.attn_k, true);
        let v = mat_rows(h, t, &w.attn_v, true);
        let mut o = vec![0.0f32; t * HIDDEN];
        for head in 0..N_HEADS {
            let base = head * hd;
            for tq in 0..t {
                let mut scores = vec![0.0f32; t];
                let mut mx = f32::NEG_INFINITY;
                for tk in 0..t {
                    let mut acc = 0.0f32;
                    for i in 0..hd {
                        acc += q[tq * HIDDEN + base + i] * k[tk * HIDDEN + base + i];
                    }
                    acc *= scale;
                    scores[tk] = acc;
                    mx = mx.max(acc);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    let e = (*s - mx).exp();
                    *s = e;
                    sum += e;
                }
                for (i, ov) in o[tq * HIDDEN + base..tq * HIDDEN + base + hd].iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for tk in 0..t {
                        acc += scores[tk] * v[tk * HIDDEN + base + i];
                    }
                    *ov = acc / sum;
                }
            }
        }
        mat_rows(&o, t, &w.attn_o, true)
    }

    /// MoE-FFN: softmax-гейт по 4 экспертам, NewGELU внутри эксперта,
    /// вторая NewGELU перед ffn_out.
    fn moe_ffn(&self, h: &[f32], t: usize, w: &BlockW) -> Vec<f32> {
        let mut out = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            let hin = &h[ti * HIDDEN..(ti + 1) * HIDDEN];
            let mut gate = [0.0f32; N_EXPERTS];
            for (e, g) in gate.iter_mut().enumerate() {
                let row = &w.moe_gate[e * HIDDEN..(e + 1) * HIDDEN];
                let mut acc = 0.0f32;
                for (i, wv) in row.iter().enumerate() {
                    acc += wv * hin[i];
                }
                *g = acc;
            }
            let mx = gate.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            let mut probs = [0.0f64; N_EXPERTS];
            for (e, g) in gate.iter().enumerate() {
                let p = (*g - mx).exp() as f64;
                probs[e] = p;
                sum += p;
            }
            for p in probs.iter_mut() {
                *p /= sum;
            }
            // Эксперт e — строки [64e, 64e+64) таблицы [256, 64].
            let mut mix = vec![0.0f32; HIDDEN];
            for (e, p) in probs.iter().enumerate() {
                for (o, mv) in mix.iter_mut().enumerate() {
                    let row = &w.moe_experts[(e * HIDDEN + o) * HIDDEN..(e * HIDDEN + o + 1) * HIDDEN];
                    let mut acc = 0.0f32;
                    for (i, wv) in row.iter().enumerate() {
                        acc += wv * hin[i];
                    }
                    *mv += (*p as f32) * new_gelu(acc);
                }
            }
            for (o, ov) in out[ti * HIDDEN..(ti + 1) * HIDDEN].iter_mut().enumerate() {
                let row = &w.ffn_out.0[o * HIDDEN..(o + 1) * HIDDEN];
                let mut acc = w.ffn_out.1[o];
                for (i, wv) in row.iter().enumerate() {
                    acc += wv * new_gelu(mix[i]);
                }
                *ov = acc;
            }
        }
        out
    }
}

/// NewGELU (тanh-форма): 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
fn new_gelu(x: f32) -> f32 {
    // √(2/π).
    const C: f32 = 0.797_884_56;
    let inner = C * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Точный GELU: 0.5·x·(1 + erf(x/√2)) (Абрамовиц-Стегун 7.1.26).
fn gelu_erf(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let u = 1.0 / (1.0 + 0.3275911 * a);
    let poly = ((((1.061405429 * u - 1.453152027) * u + 1.421413741) * u - 0.284496736) * u + 0.254829592) * u;
    let erf = 1.0 - poly * (-z * z).exp();
    0.5 * x * (1.0 + erf.copysign(z))
}

fn relu(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

/// BatchNorm (свёрнутый): scale/shift по каналам (последняя размерность).
fn bn_channels(x: &[f32], scale: &[f32], shift: &[f32]) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, v)| v * scale[i % scale.len()] + shift[i % shift.len()])
        .collect()
}

/// LayerNorm по каналам (последняя размерность) на [T, C].
fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let c = w.len();
    for row in x.chunks_mut(c) {
        let n = c as f64;
        let mut mean = 0.0f64;
        for v in row.iter() {
            mean += *v as f64;
        }
        mean /= n;
        let mut var = 0.0f64;
        for v in row.iter() {
            let d = *v as f64 - mean;
            var += d * d;
        }
        var /= n;
        let inv = 1.0 / ((var as f32) + eps).sqrt();
        let m = mean as f32;
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - m) * inv * w[i] + b[i];
        }
    }
}

/// Матрица [out, in] на строках [T, in] (+ смещение).
fn mat_rows(x: &[f32], t: usize, aff: &Aff, bias: bool) -> Vec<f32> {
    let (w, b) = aff;
    let input = w.len() / b.len().max(1);
    let out_dim = b.len();
    let mut out = vec![0.0f32; t * out_dim];
    for ti in 0..t {
        let xin = &x[ti * input..(ti + 1) * input];
        let orow = &mut out[ti * out_dim..(ti + 1) * out_dim];
        for (o, ov) in orow.iter_mut().enumerate() {
            let row = &w[o * input..(o + 1) * input];
            let mut acc = if bias { b[o] } else { 0.0 };
            for (i, wv) in row.iter().enumerate() {
                acc += wv * xin[i];
            }
            *ov = acc;
        }
    }
    out
}

/// Выходная длина conv с ядром 3, pad 1 и stride s.
fn conv_time(n: usize, s: usize) -> usize {
    (n + 2 - 3) / s + 1
}

/// Точечная свёртка 1×1 со смещением.
fn pointwise(x: &[f32], t: usize, f: usize, ic: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let oc = w.len() / ic;
    let mut out = vec![0.0f32; t * f * oc];
    for ti in 0..t {
        for fi in 0..f {
            let xin = &x[(ti * f + fi) * ic..(ti * f + fi) * ic + ic];
            let orow = &mut out[(ti * f + fi) * oc..(ti * f + fi) * oc + oc];
            for (o, ov) in orow.iter_mut().enumerate() {
                let row = &w[o * ic..(o + 1) * ic];
                let mut acc = b[o];
                for (i, wv) in row.iter().enumerate() {
                    acc += wv * xin[i];
                }
                *ov = acc;
            }
        }
    }
    out
}

/// Свёртка 1×1 без смещения со stride.
#[allow(clippy::too_many_arguments)]
fn conv2d_1x1(
    x: &[f32],
    t: usize,
    f: usize,
    ic: usize,
    w: &[f32],
    t_out: usize,
    f_out: usize,
    s_t: usize,
    s_f: usize,
) -> Vec<f32> {
    let oc = w.len() / ic;
    let mut out = vec![0.0f32; t_out * f_out * oc];
    for ti in 0..t_out {
        for fi in 0..f_out {
            let si = (ti * s_t).min(t - 1);
            let sj = (fi * s_f).min(f - 1);
            let xin = &x[(si * f + sj) * ic..(si * f + sj) * ic + ic];
            let orow = &mut out[(ti * f_out + fi) * oc..(ti * f_out + fi) * oc + oc];
            for (o, ov) in orow.iter_mut().enumerate() {
                let row = &w[o * ic..(o + 1) * ic];
                let mut acc = 0.0f32;
                for (i, wv) in row.iter().enumerate() {
                    acc += wv * xin[i];
                }
                *ov = acc;
            }
        }
    }
    out
}

/// Полная свёртка 3×3 (pad 1, stride) с bias.
#[allow(clippy::too_many_arguments)]
fn conv2d_3x3(
    x: &[f32],
    t: usize,
    f: usize,
    ic: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    s_t: usize,
    s_f: usize,
) -> Vec<f32> {
    let oc = w.len() / (ic * 9);
    let t_out = conv_time(t, s_t);
    let f_out = conv_time(f, s_f);
    let mut out = vec![0.0f32; t_out * f_out * oc];
    for ti in 0..t_out {
        for fi in 0..f_out {
            for o in 0..oc {
                let mut acc = bias.map(|b| b[o]).unwrap_or(0.0);
                for ic2 in 0..ic {
                    let wbase = (o * ic + ic2) * 9;
                    for kh in 0..3 {
                        let si = ti as i64 * s_t as i64 - 1 + kh as i64;
                        if si < 0 || si as usize >= t {
                            continue;
                        }
                        for kw in 0..3 {
                            let sj = fi as i64 * s_f as i64 - 1 + kw as i64;
                            if sj < 0 || sj as usize >= f {
                                continue;
                            }
                            // Оси веса: kH — частота (sj), kW — время (si).
                            acc += w[wbase + kw * 3 + kh] * x[(si as usize * f + sj as usize) * ic + ic2];
                        }
                    }
                }
                out[(ti * f_out + fi) * oc + o] = acc;
            }
        }
    }
    out
}

/// Групповая 3×3-свёртка (паддинг 1, без bias): слот j сворачивает входной
/// канал g·a+j в выходные каналы группы g (b каналов на группу).
#[allow(clippy::too_many_arguments)]
fn grouped_conv3x3(
    x: &[f32],
    t: usize,
    f: usize,
    ic: usize,
    s_t: usize,
    s_f: usize,
    t_out: usize,
    f_out: usize,
    slots: &[Vec<f32>],
    oc: usize,
) -> Vec<f32> {
    let a = slots.len();
    let g = ic / a;
    let b = oc / g;
    let mut out = vec![0.0f32; t_out * f_out * oc];
    for ti in 0..t_out {
        for fi in 0..f_out {
            for j in 0..a {
                let slot = &slots[j];
                for og in 0..g {
                    let src_ch = og * a + j;
                    for ob in 0..b {
                        let o = og * b + ob;
                        let wbase = o * 9;
                        let mut acc = 0.0f32;
                        for kh in 0..3 {
                            let si = ti as i64 * s_t as i64 - 1 + kh as i64;
                            if si < 0 || si as usize >= t {
                                continue;
                            }
                            for kw in 0..3 {
                                let sj = fi as i64 * s_f as i64 - 1 + kw as i64;
                                if sj < 0 || sj as usize >= f {
                                    continue;
                                }
                                // Оси веса: kH — частота (sj), kW — время (si).
                                acc += slot[wbase + kw * 3 + kh]
                                    * x[(si as usize * f + sj as usize) * ic + src_ch];
                            }
                        }
                        out[(ti * f_out + fi) * oc + o] += acc;
                    }
                }
            }
        }
    }
    out
}

/// Flatten [T, F, C] → [T, F·C] (k = f·C + c); раскладка плоского буфера не
/// меняется — функция для явности.
fn flatten_ref(x: &[f32], _t: usize, _f: usize, _c: usize) -> Vec<f32> {
    x.to_vec()
}

/// Обратный flatten: [T, F·C] → [T, F, C]; тоже тождество на плоском буфере.
fn unflatten_ref(x: &[f32], _t: usize, _f: usize, _c: usize) -> Vec<f32> {
    x.to_vec()
}

/// Оконный sinc-ресемплер (Ханн, ~16 отводов на сторону) для записей с
/// произвольной частотой дискретизации. Нормировка на сумму весов.
pub fn sinc_resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let r = to as f64 / from as f64;
    // Срез на 94% полосы min(вход, выход) — как rolloff 0.94 у torchaudio.
    let fc = 0.47 * r.min(1.0);
    let half = (16.0 / (2.0 * fc)).ceil() as i64;
    let out_len = (x.len() as f64 * r).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / r;
        let left = (pos - half as f64).floor() as i64;
        let right = (pos + half as f64).ceil() as i64;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for j in left..=right {
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let d = pos - j as f64;
            if d.abs() >= half as f64 {
                continue;
            }
            let p = std::f64::consts::PI * 2.0 * fc * d;
            let s = if p.abs() < 1e-9 { 1.0 } else { p.sin() / p };
            let w = 0.5 * (1.0 + (std::f64::consts::PI * d / half as f64).cos());
            acc += w * s * x[j as usize] as f64;
            wsum += w * s;
        }
        out.push((acc / wsum.max(1e-12)) as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GELU в нуле — ноль; в единице ≈ 0.8413 (Φ(1)).
    #[test]
    fn gelu_values() {
        assert!(gelu_erf(0.0).abs() < 1e-6);
        assert!((gelu_erf(1.0) - 0.8413447).abs() < 1e-4);
        assert!((gelu_erf(-1.0) + 0.1586553).abs() < 1e-4);
    }

    /// NewGELU близок к точному GELU на малых значениях.
    #[test]
    fn new_gelu_close() {
        for v in [-2.0f32, -0.5, 0.0, 0.5, 2.0] {
            assert!((new_gelu(v) - gelu_erf(v)).abs() < 0.02, "v={v}");
        }
    }

    /// Ресемплер сохраняет постоянный сигнал и длину.
    #[test]
    fn sinc_resample_basics() {
        let x = vec![0.5f32; 4800]; // 0.3 с на 16 кГц
        let out = sinc_resample(&x, 16000, 16000);
        assert_eq!(out.len(), x.len());
        let down = sinc_resample(&x, 24000, 16000);
        assert!((down.len() as i64 - 3200).abs() <= 2, "len={}", down.len());
        for v in &down {
            assert!((*v - 0.5).abs() < 1e-4, "v={v}");
        }
    }

    /// Flatten/UNflatten — тождества на плоском буфере.
    #[test]
    fn flatten_roundtrip() {
        let x: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let f = flatten_ref(&x, 2, 3, 4);
        assert_eq!(f, x);
        assert_eq!(unflatten_ref(&f, 2, 3, 4), x);
    }
}
