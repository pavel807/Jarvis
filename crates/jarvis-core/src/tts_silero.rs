//! Silero v5 TTS (v5_cis_base) — нативная реализация на candle 0.11.
//!
//! Архитектура (восстановлена из TorchScript пакета, docs/silero-v5/):
//! FastPitch (embedding 85 -> 128, энкодер 4 FFT-блока, dur/pitch
//! предикторы по 4 слоя, hourglass-декодер с downsample/upsample ×3,
//! линейный выход 192) + Vocos-вокодер (ConvNeXt ×8 + iSTFT-head 2400).
//!
//! Пайплайн: текст -> нейро-ударение «+» перед гласной -> буквы-токены
//! |текст~ -> длительности (round(exp(d)-1), клампы) -> length regulator
//! -> декодер -> спектрограмма 192 -> вокодер 48 кГц.
//!
//! Веса: models/silero/v5_cis_base.safetensors (317 тензоров).

use crate::stress::RuStress;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::ops::{layer_norm, softmax};
use candle_nn::{Conv1d, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const SAMPLE_RATE: u32 = 48000;

/// Мужские и женские русские голоса v5 (29 шт.).
pub const SILERO_VOICES: &[(&str, &str)] = &[
    ("ru_eduard", "мужской, основной"),
    ("ru_dmitriy", "мужской"),
    ("ru_alexandr", "мужской"),
    ("ru_bogdan", "мужской"),
    ("ru_igor", "мужской"),
    ("ru_marat", "мужской"),
    ("ru_roman", "мужской"),
    ("ru_eugene", "мужской"),
    ("ru_aigul", "женский"),
    ("ru_albina", "женский"),
    ("ru_alfia", "женский"),
    ("ru_alfia2", "женский"),
    ("ru_ekaterina", "женский"),
    ("ru_vika", "женский"),
    ("ru_karina", "женский"),
    ("ru_kermen", "женский"),
    ("ru_miyau", "женский"),
    ("ru_nurgul", "женский"),
    ("ru_oksana", "женский"),
    ("ru_onaoy", "женский"),
    ("ru_ramilia", "женский"),
    ("ru_safarhuja", "женский"),
    ("ru_saida", "женский"),
    ("ru_sibday", "женский"),
    ("ru_zara", "женский"),
    ("ru_zhadyra", "женский"),
    ("ru_zhazira", "женский"),
    ("ru_zinaida", "женский"),
    ("ru_kejilgan", "женский"),
];

/// id символов, которые prepare_text_input выкидывает из текста:
/// '_' (0) и '~' (1) — служебные, '!' (3) и "'" (4) — отбрасывает
/// фильтр symbols[3:] (по позиции в строке символов, не по id).
const DROPPED_IDS: &[i64] = &[0, 1, 3, 4];

#[derive(Deserialize, Clone)]
struct SileroAlphabet {
    symbol_to_id: HashMap<String, i64>,
    speakers: HashMap<String, Vec<i64>>,
}

pub struct SileroEngine {
    model: SileroModel,
    alphabet: HashMap<char, i64>,
    speakers: HashMap<String, Vec<i64>>,
    stress: Option<RuStress>,
    voice: String,
    device: Device,
}

// ---------------------------------------------------------------------------
// Загрузка
// ---------------------------------------------------------------------------

fn ensure_silero_files() -> Result<PathBuf> {
    let dir = crate::paths::models_dir().join("silero");
    let st = dir.join("v5_cis_base.safetensors");
    let alph = dir.join("alphabet.json");
    if st.exists() && alph.exists() {
        return Ok(dir);
    }
    bail!(
        "Файлы Silero v5 не найдены в {:?} (v5_cis_base.safetensors, alphabet.json). \
         Сгенерируйте их: /tmp/silero_env/bin/python — см. docs/silero-v5/",
        dir
    );
}

impl SileroEngine {
    pub fn new(voice: &str) -> Result<Self> {
        Self::new_inner(voice, true)
    }

    /// Без ударителя — для численной верификации против PyTorch
    /// (эталон apply_tts не вставляет «+»).
    pub fn new_no_stress(voice: &str) -> Result<Self> {
        Self::new_inner(voice, false)
    }

    fn new_inner(voice: &str, use_stress: bool) -> Result<Self> {
        if !SILERO_VOICES.iter().any(|(v, _)| *v == voice) {
            bail!("Неизвестный голос silero: {voice}");
        }
        let dir = ensure_silero_files()?;
        let alphabet_cfg: SileroAlphabet = serde_json::from_str(
            &std::fs::read_to_string(dir.join("alphabet.json"))
                .context("Не удалось прочитать alphabet.json")?,
        )
        .context("Ошибка парсинга alphabet.json")?;

        let device = Device::Cpu;
        let started = std::time::Instant::now();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("v5_cis_base.safetensors")],
                DType::F32,
                &device,
            )
            .context("Не удалось открыть веса Silero v5")?
        };
        let model = SileroModel::load(vb, &device)
            .context("Не удалось построить модель Silero v5")?;

        // Нейро-ударитель (та же модель, что и для kokoro).
        let stress = if use_stress {
            match RuStress::load() {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("⚠️ Ударитель не загружен, ударения по умолчанию: {e:#}");
                    None
                }
            }
        } else {
            None
        };

        let alphabet: HashMap<char, i64> = alphabet_cfg
            .symbol_to_id
            .iter()
            .filter_map(|(k, v)| {
                let mut it = k.chars();
                let c = it.next()?;
                it.next().is_none().then(|| (c, *v))
            })
            .collect();

        println!("✅ Silero v5 готов ({voice}, {SAMPLE_RATE} Гц) за {:.1?}", started.elapsed());

        Ok(Self {
            model,
            alphabet,
            speakers: alphabet_cfg.speakers,
            stress,
            voice: voice.to_string(),
            device,
        })
    }

    pub fn voice(&self) -> &str {
        &self.voice
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Мел-спектрограмма (отладка верификации): (данные, каналы, кадры).
    pub fn debug_mel(&self, text: &str, speed: f32) -> Result<(Vec<f32>, usize, usize)> {
        let speaker_ids = self
            .speakers
            .get(&self.voice)
            .cloned()
            .with_context(|| format!("Нет голоса {:?}", self.voice))?;
        let sequence = self.tokenize(text)?;
        self.model.debug_mel(&sequence, &speaker_ids, speed, &self.device)
    }

    /// Стадии вокодера (отладка): mel из файла эталона.
    pub fn debug_voc_from(&self, mel: &[f32]) -> Result<Vec<(String, Vec<f32>, Vec<usize>)>> {
        let mel = Tensor::from_vec(mel.to_vec(), (1, 192, mel.len() / 192), &self.device)?;
        self.model.debug_voc(&mel)
    }

    /// Стадии tacotron (отладка верификации).
    pub fn debug_stages(&self, text: &str, speed: f32)
        -> Result<Vec<(String, Vec<f32>, Vec<usize>)>>
    {
        let speaker_ids = self
            .speakers
            .get(&self.voice)
            .cloned()
            .with_context(|| format!("Нет голоса {:?}", self.voice))?;
        let sequence = self.tokenize(text)?;
        self.model.debug_stages(&sequence, &speaker_ids, speed, &self.device)
    }

    /// Длительности символов (отладка верификации).
    pub fn debug_durations(&self, text: &str, speed: f32) -> Result<Vec<i32>> {
        let speaker_ids = self
            .speakers
            .get(&self.voice)
            .cloned()
            .with_context(|| format!("Нет голоса {:?}", self.voice))?;
        let sequence = self.tokenize(text)?;
        self.model.durations(&sequence, &speaker_ids, speed, &self.device)
            .map(|v| v.iter().map(|d| *d as i32).collect())
    }

    /// Синтез фразы; `speed` (0.5..2) — темп через длительности символов.
    pub fn synthesize(&self, text: &str, speed: f32) -> Result<Vec<f32>> {
        let speaker_ids = self
            .speakers
            .get(&self.voice)
            .cloned()
            .with_context(|| format!("Нет голоса {:?}", self.voice))?;
        let sequence = self.tokenize(text)?;
        if sequence.len() < 4 {
            bail!("Текст не дал токенов: \"{text}\"");
        }
        let audio = self
            .model
            .forward(&sequence, &speaker_ids, speed, &self.device)
            .map_err(|e| {
                eprintln!("SILERO ERR: {e:#}");
                e
            })
            .context("Ошибка инференса Silero v5")?;
        let _ = std::backtrace::Backtrace::force_capture();
        Ok(audio)
    }

    /// Текст -> токены: строчные буквы, ударения «+», фильтр по алфавиту,
    /// обёртка |...~.
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        let lowered = text.to_lowercase();
        let mut chars: Vec<char> = Vec::with_capacity(lowered.len());

        let mut word = String::new();
        for c in lowered.chars() {
            if c.is_alphabetic() {
                word.push(c);
            } else {
                if !word.is_empty() {
                    self.push_word(&word, &mut chars);
                    word.clear();
                }
                if c == '-' || self.alphabet.contains_key(&c) {
                    chars.push(c);
                }
            }
        }
        if !word.is_empty() {
            self.push_word(&word, &mut chars);
        }

        // Фильтр по алфавиту (служебные 0..ALPHABET_SKIP недоступны тексту).
        let mut out: Vec<i64> = Vec::with_capacity(chars.len() + 2);
        out.push(*self.alphabet.get(&'|').context("нет BOS в алфавите")?);
        for c in chars {
            if let Some(id) = self.alphabet.get(&c) {
                if !DROPPED_IDS.contains(id) {
                    out.push(*id);
                }
            }
        }
        out.push(*self.alphabet.get(&'~').context("нет EOS в алфавите")?);
        Ok(out)
    }

    /// Слово с ударением «+» перед ударной гласной (ё восстанавливается).
    fn push_word(&self, word: &str, chars: &mut Vec<char>) {
        let word = self
            .stress
            .as_ref()
            .map(|s| s.yo_restore(word))
            .unwrap_or_else(|| word.to_string());
        let ordinal = self
            .stress
            .as_ref()
            .and_then(|s| s.stressed_vowel_ordinal(&word));
        if let Some(ord) = ordinal {
            let mut vowels = 0usize;
            for c in word.chars() {
                if crate::stress::is_ru_vowel(c) {
                    vowels += 1;
                    if vowels == ord {
                        chars.push('+');
                    }
                }
                chars.push(c);
            }
        } else {
            chars.extend(word.chars());
        }
    }
}

// ---------------------------------------------------------------------------
// Модель
// ---------------------------------------------------------------------------

/// FFT-блок FastPitch: self-attention + conv-ветка, pre-norm.
struct FftBlock {
    attn_in_proj: Linear,
    attn_out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    d_model: usize,
}

impl FftBlock {
    fn load(vb: VarBuilder, d_model: usize, conv_mid: usize, num_heads: usize) -> Result<Self> {
        let attn_in_proj = Linear::new(
            vb.pp("self_attn").get((3 * d_model, d_model), "in_proj_weight")?,
            Some(vb.pp("self_attn").get(3 * d_model, "in_proj_bias")?),
        );
        let attn_out_proj = Linear::new(
            vb.pp("self_attn").get((d_model, d_model), "out_proj.weight")?,
            Some(vb.pp("self_attn").get(d_model, "out_proj.bias")?),
        );
        let cfg9 = candle_nn::Conv1dConfig { padding: 4, ..Default::default() };
        let conv1 = Conv1d::new(
            vb.pp("conv1").get((conv_mid, d_model, 9), "weight")?,
            Some(vb.pp("conv1").get(conv_mid, "bias")?),
            cfg9,
        );
        let cfg1 = candle_nn::Conv1dConfig { padding: 0, ..Default::default() };
        let conv2 = Conv1d::new(
            vb.pp("conv2").get((d_model, conv_mid, 1), "weight")?,
            Some(vb.pp("conv2").get(d_model, "bias")?),
            cfg1,
        );
        Ok(Self {
            attn_in_proj,
            attn_out_proj,
            num_heads,
            head_dim: d_model / num_heads,
            conv1,
            conv2,
            norm1: (
                vb.pp("norm1").get(d_model, "weight")?.contiguous()?,
                vb.pp("norm1").get(d_model, "bias")?.contiguous()?,
            ),
            norm2: (
                vb.pp("norm2").get(d_model, "weight")?.contiguous()?,
                vb.pp("norm2").get(d_model, "bias")?.contiguous()?,
            ),
            d_model,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [1, T, D]
        let qkv = self.attn_in_proj.forward(x)?; // [1, T, 3D]
        let qkv = qkv.squeeze(0)?; // [T, 3D]
        let chunks = qkv.chunk(3, D::Minus1)?; // [T, D] × 3
        let t = chunks[0].dim(0)?;
        let mut attn = None;
        for h in 0..self.num_heads {
            let off = h * self.head_dim;
            let q = chunks[0].i((.., off..off + self.head_dim))?;
            let k = chunks[1].i((.., off..off + self.head_dim))?;
            let v = chunks[2].i((.., off..off + self.head_dim))?;
            let scores = (q.matmul(&k.transpose(0, 1)?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
            let weights = softmax(&scores, D::Minus1)?;
            let a = weights.matmul(&v)?; // [T, head_dim]
            attn = match attn {
                None => Some(a),
                Some(prev) => Some(Tensor::cat(&[&prev, &a], D::Minus1)?),
            };
        }
        let attn = attn.context("attn")?.reshape((t, self.d_model))?.unsqueeze(0)?;
        let attn = self.attn_out_proj.forward(&attn)?;
        let x = (x + attn)?;
        let x = layer_norm(&x.contiguous()?, &self.norm1.0, &self.norm1.1, 1e-5f32)?;

        // Conv-ветка: [1, T, D] -> [1, D, T]
        let c = x.transpose(1, 2)?.contiguous()?;
        let c = self.conv1.forward(&c)?;
        let c = c.relu()?;
        let c = self.conv2.forward(&c)?;
        let c = c.transpose(1, 2)?.contiguous()?;
        let x = (x + c)?;
        Ok(layer_norm(&x.contiguous()?, &self.norm2.0, &self.norm2.1, 1e-5f32)?)
    }
}

/// ForwardTransformer: pos-encoding + N блоков + финальная норма.
struct ForwardTransformer {
    pe: Tensor, // [5000, 1, D]
    scale: Tensor,
    layers: Vec<FftBlock>,
    norm: (Tensor, Tensor),
}

impl ForwardTransformer {
    fn load(vb: VarBuilder, d_model: usize, conv_mid: usize, heads: usize, n_layers: usize) -> Result<Self> {
        let pe = vb.pp("pos_encoder").get((5000, 1, d_model), "pe")?;
        let scale = vb.pp("pos_encoder").get(1, "scale")?;
        let layers = (0..n_layers)
            .map(|i| FftBlock::load(vb.pp("layers").pp(i.to_string()), d_model, conv_mid, heads))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pe,
            scale,
            layers,
            norm: (vb.pp("norm").get(d_model, "weight")?, vb.pp("norm").get(d_model, "bias")?),
        })
    }

    fn pos_enc(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let pe = self.pe.i((..t, 0, ..))?.unsqueeze(0)?; // [1, T, D]
        let scale = self.scale.reshape(())?.to_dtype(x.dtype())?;
        Ok((x + pe.broadcast_mul(&scale))?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.pos_enc(x)?;
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        Ok(layer_norm(&x.contiguous()?, &self.norm.0, &self.norm.1, 1e-5f32)?)
    }
}

/// TransformerSeriesPredictor (dur/pitch): emb + speaker "pre" + transformer + lin.
struct SeriesPredictor {
    embedding: Embedding,
    speaker_embedding: Embedding,
    transformer: ForwardTransformer,
    lin: Linear,
}

impl SeriesPredictor {
    fn load(vb: VarBuilder, heads: usize) -> Result<Self> {
        let embedding = Embedding::new(vb.pp("embedding").get((85, 64), "weight")?, 64);
        let speaker_embedding = Embedding::new(vb.pp("speaker_embedding").get((60, 64), "weight")?, 64);
        let transformer = ForwardTransformer::load(vb.pp("transformer"), 64, 64, heads, 4)?;
        let lin = Linear::new(vb.pp("lin").get((1, 64), "weight")?, Some(vb.pp("lin").get(1, "bias")?));
        Ok(Self { embedding, speaker_embedding, transformer, lin })
    }

    fn forward(&self, sequence: &Tensor, speaker_ids: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(sequence)?; // [1, T, 64]
        let sp = self.speaker_embedding.forward(speaker_ids)?; // [1, 1, 64]
        let x = x.broadcast_add(&sp)?;
        let x = self.transformer.forward(&x)?;
        Ok(self.lin.forward(&x)?) // [1, T, 1]
    }
}

/// HourGlass-декодер FastPitch.
struct HourGlass {
    pe: Tensor,
    scale: Tensor,
    pre: FftBlock,
    shorten: [FftBlock; 2],
    post: FftBlock,
    up_proj: Linear,
    norm: (Tensor, Tensor),
    shorten_factor: usize,
}

impl HourGlass {
    fn pos_enc(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let pe = self.pe.i((..t, 0, ..))?.unsqueeze(0)?;
        let scale = self.scale.reshape(())?.to_dtype(x.dtype())?;
        Ok((x + pe.broadcast_mul(&scale))?)
    }

    fn load(vb: VarBuilder, d_model: usize, heads: usize) -> Result<Self> {
        let pe = vb.pp("pos_encoder").get((5000, 1, d_model), "pe")?;
        let scale = vb.pp("pos_encoder").get(1, "scale")?;
        let pre = FftBlock::load(vb.pp("pre_vanilla_layers").pp("0"), d_model, 512, heads)?;
        let shorten = [
            FftBlock::load(vb.pp("shorten_layers").pp("0"), d_model, 512, heads)?,
            FftBlock::load(vb.pp("shorten_layers").pp("1"), d_model, 512, heads)?,
        ];
        let post = FftBlock::load(vb.pp("post_vanilla_layers").pp("0"), d_model, 512, heads)?;
        let up_proj = Linear::new(vb.pp("upsample").pp("proj").get((3 * d_model, d_model), "weight")?, Some(vb.pp("upsample").pp("proj").get(3 * d_model, "bias")?));
        Ok(Self {
            pe,
            scale,
            pre,
            shorten,
            post,
            up_proj,
            norm: (vb.pp("norm").get(d_model, "weight")?, vb.pp("norm").get(d_model, "bias")?),
            shorten_factor: 3,
        })
    }

    fn pad_to_multiple(_x: &Tensor, _multiple: usize, _v: f64) -> Result<Tensor> {
        // Заменён инлайн-пэддингом в forward (см. ниже).
        bail!("не используется")
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let len_orig = x.dim(1)?;
        let mut x = self.pos_enc(x)?;
        x = self.pre.forward(&x)?;
        // Пэддинг до кратности 3 (нули) — residual до даунсэмпла.
        let t = x.dim(1)?;
        let rem = t % self.shorten_factor;
        let padded = if rem == 0 { x.clone() } else { x.pad_with_zeros(1, 0, self.shorten_factor - rem)? };
        let residual = padded.clone();
        // AvgPool downsample: [1, T, D] -> усреднение окон по 3.
        let x = downsample_avg(&padded, self.shorten_factor)?;
        // Оба shorten-слоя вызываются с одним входом, используется только .1
        let _ = self.shorten[0].forward(&x)?;
        let x = self.shorten[1].forward(&x)?;
        // LinearUpsample: proj 128 -> 384, view [1, 3T, 128]
        let x = self.up_proj.forward(&x)?; // [1, T', 384]
        let b = x.dim(0)?;
        let t2 = x.dim(1)?;
        let d = x.dim(2)? / self.shorten_factor;
        let x = x.reshape((b, t2 * self.shorten_factor, d))?;
        let x = (x + residual)?;
        let x = x.i((.., ..len_orig, ..))?;
        let x = self.post.forward(&x)?;
        Ok(layer_norm(&x.contiguous()?, &self.norm.0, &self.norm.1, 1e-5f32)?)
    }
}

/// Среднее по окнам (AvgPool1d kernel=stride=factor).
fn downsample_avg(x: &Tensor, factor: usize) -> Result<Tensor> {
    let b = x.dim(0)?;
    let t = x.dim(1)?;
    let d = x.dim(2)?;
    let t_out = t / factor;
    let x = x.reshape((b, t_out, factor, d))?;
    Ok(x.mean(D::Minus2)?)
}

/// ConvNeXt-блок Vocos.
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: (Tensor, Tensor),
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let dwconv = Conv1d::new(
            vb.pp("dwconv").get((dim, 1, 7), "weight")?,
            Some(vb.pp("dwconv").get(dim, "bias")?),
            candle_nn::Conv1dConfig { padding: 3, groups: dim, ..Default::default() },
        );
        let pwconv1 = Linear::new(vb.pp("pwconv1").get((1536, dim), "weight")?, Some(vb.pp("pwconv1").get(1536, "bias")?));
        let pwconv2 = Linear::new(vb.pp("pwconv2").get((dim, 1536), "weight")?, Some(vb.pp("pwconv2").get(dim, "bias")?));
        Ok(Self {
            dwconv,
            norm: (vb.pp("norm").get(dim, "weight")?, vb.pp("norm").get(dim, "bias")?),
            pwconv1,
            pwconv2,
            gamma: vb.get(dim, "gamma")?, 
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.dwconv.forward(x)?; // [1, 512, T]
        let x = x.transpose(1, 2)?.contiguous()?; // [1, T, 512]
        let x = layer_norm(&x.contiguous()?, &self.norm.0, &self.norm.1, 1e-6f32)?;
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu_erf()?;
        let x = self.pwconv2.forward(&x)?;
        let x = x.broadcast_mul(&self.gamma.unsqueeze(0)?.unsqueeze(0)?)?;
        let x = x.transpose(1, 2)?.contiguous()?; // [1, 512, T]
        Ok((residual + x)?)
    }
}

pub struct SileroModel {
    // dur/pitch предикторы
    dur: SeriesPredictor,
    pitch: SeriesPredictor,
    // tacotron
    embedding: Embedding,
    speaker_embedding: Embedding,
    encoder: ForwardTransformer,
    pitch_proj: Conv1d,
    decoder: HourGlass,
    lin: Linear,
    // vocoder
    v_embed: Conv1d,
    v_norm: (Tensor, Tensor),
    convnext: Vec<ConvNeXtBlock>,
    v_final_norm: (Tensor, Tensor),
    head_out: Linear,
    window: Tensor,
    device: Device,
}

impl SileroModel {
    fn load(vb: VarBuilder, device: &Device) -> Result<Self> {
        // Число голов attention: подбирается верификацией (SILERO_HEADS).
        let heads: usize = std::env::var("SILERO_HEADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let dur = SeriesPredictor::load(vb.pp("dur_predictor").pp("dur_pred"), heads)?;
        let pitch = SeriesPredictor::load(vb.pp("pitch_predictor").pp("pitch_pred"), heads)?;
        let embedding = Embedding::new(vb.pp("tacotron").pp("embedding").get((85, 128), "weight")?, 128);
        let speaker_embedding = Embedding::new(vb.pp("tacotron").pp("speaker_embedding").get((60, 128), "weight")?, 128);
        let encoder = ForwardTransformer::load(vb.pp("tacotron").pp("encoder"), 128, 512, heads, 4)?;
        let pitch_proj = Conv1d::new(
            vb.pp("tacotron").pp("pitch_proj").get((128, 1, 3), "weight")?,
            Some(vb.pp("tacotron").pp("pitch_proj").get(128, "bias")?),
            candle_nn::Conv1dConfig { padding: 1, ..Default::default() },
        );
        let decoder = HourGlass::load(vb.pp("tacotron").pp("decoder"), 128, heads)?;
        let lin = Linear::new(vb.pp("tacotron").pp("lin").get((192, 128), "weight")?, Some(vb.pp("tacotron").pp("lin").get(192, "bias")?));
        let v_embed = Conv1d::new(
            vb.pp("vocoder").pp("backbone").pp("embed").get((512, 192, 7), "weight")?,
            Some(vb.pp("vocoder").pp("backbone").pp("embed").get(512, "bias")?),
            candle_nn::Conv1dConfig { padding: 3, ..Default::default() },
        );
        let v_norm = (
            vb.pp("vocoder").pp("backbone").pp("norm").get(512, "weight")?,
            vb.pp("vocoder").pp("backbone").pp("norm").get(512, "bias")?,
        );
        let mut convnext = Vec::new();
        for i in 0..8 {
            convnext.push(ConvNeXtBlock::load(vb.pp("vocoder").pp("backbone").pp("convnext").pp(i.to_string()), 512)?);
        }
        let v_final_norm = (
            vb.pp("vocoder").pp("backbone").pp("final_layer_norm").get(512, "weight")?,
            vb.pp("vocoder").pp("backbone").pp("final_layer_norm").get(512, "bias")?,
        );
        let head_out = Linear::new(vb.pp("vocoder").pp("head").pp("out").get((2402, 512), "weight")?, Some(vb.pp("vocoder").pp("head").pp("out").get(2402, "bias")?));
        let window = vb.pp("vocoder").pp("head").pp("istft").get(2400, "window")?;
        Ok(Self {
            dur,
            pitch,
            embedding,
            speaker_embedding,
            encoder,
            pitch_proj,
            decoder,
            lin,
            v_embed,
            v_norm,
            convnext,
            v_final_norm,
            head_out,
            window,
            device: device.clone(),
        })
    }

    fn dur_predict(&self, sequence: &Tensor, speaker_ids: &Tensor) -> Result<Tensor> {
        let x = self.dur.forward(sequence, speaker_ids)?; // [1, T, 1]
        // dur = round(exp(log_dur) - 1), негативы -> 0
        let mut dur = x.exp()?.broadcast_sub(&Tensor::new(1.0f32, x.device())?)?;
        dur = dur.clamp(0.0f32, f32::MAX)?.round()?;
        // BOS <= 5 (первый кламп, до деления на rate)
        dur = candle_core::Tensor::cat(&[&dur.i((.., ..1))?.clamp(0.0f32, 5.0f32)?, &dur.i((.., 1..))?], 1)?;
        // JitDurPredictor делает squeeze(-1): [1, T, 1] -> [1, T]
        Ok(dur.squeeze(2)?)
    }

    /// Длительности символов: предиктор + темп + клампы хвоста.
    fn durations(&self, sequence: &[i64], speaker_ids: &[i64], speed: f32, device: &Device) -> Result<Vec<f32>> {
        let seq = Tensor::new(sequence, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let sp = Tensor::new(speaker_ids, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let dur = self.dur_predict(&seq, &sp)?; // [1, T]
        // Темп: rate > 1 = быстрее
        let t_dur = dur.dim(1)?;
        let rate_val = 1.0f32 / speed.clamp(0.5, 2.0);
        let rate = Tensor::from_vec(vec![rate_val; t_dur], (1, t_dur), device)?;
        let dur = dur.broadcast_div(&rate)?.round()?;
        // Клампы хвоста: BOS<=5, EOS<=7, терминальная пунктуация =13, предыдущий <=13
        let mut durs = dur.flatten_all()?.to_vec1::<f32>()?;
        if let Some(first) = durs.first_mut() {
            *first = first.min(5.0);
        }
        if let Some(last) = durs.last_mut() {
            *last = last.min(7.0);
        }
        let n = durs.len();
        if n >= 2 {
            durs[n - 2] = 13.0;
        }
        if n >= 3 {
            durs[n - 3] = durs[n - 3].min(13.0);
        }
        Ok(durs)
    }

    /// Мел-спектрограмма (отладка верификации): [192, N].
    fn debug_mel(&self, sequence: &[i64], speaker_ids: &[i64], speed: f32, device: &Device) -> Result<(Vec<f32>, usize, usize)> {
        let seq = Tensor::new(sequence, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let sp = Tensor::new(speaker_ids, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let durs_v = self.durations(sequence, speaker_ids, speed, device)?;
        let dur_t = Tensor::new(&durs_v[..], device)?.unsqueeze(0)?;

        let pitch = self.pitch.forward(&seq, &sp)?; // [1, T, 1]
        let pitch = pitch.transpose(1, 2)?; // [1, 1, T]
        let mut pitch_v = pitch.flatten_all()?.to_vec1::<f32>()?;
        {
            let mut cum = 0usize;
            let mut cums: Vec<(usize, usize)> = Vec::new();
            for (i, &id) in sequence.iter().enumerate() {
                let d = durs_v.get(i).copied().unwrap_or(0.0).round() as usize;
                cums.push((cum, cum + d));
                cum += d;
            }
            for (i, &id) in sequence.iter().enumerate() {
                if [3i64, 6, 7, 8, 9, 10, 11, 82, 83].contains(&id) {
                    let (a, b) = cums[i];
                    let a = a.min(pitch_v.len());
                    let b = b.min(pitch_v.len());
                    if a < b {
                        for v in &mut pitch_v[a..b] {
                            *v = 0.0;
                        }
                    }
                }
            }
        }
        let pitch_len = pitch_v.len();
        let pitch = Tensor::from_vec(pitch_v, (1, 1, pitch_len), device)?;

        // Порядок как в JitMultiForward: энкодер -> спикер -> питч.
        let x = self.embedding.forward(&seq)?; // [1, T, 128]
        let mut x = self.encoder.forward(&x)?;
        let sp_emb = self.speaker_embedding.forward(&sp)?; // [1, 1, 128]
        x = x.broadcast_add(&sp_emb)?;
        let pitch_c = pitch.contiguous()?;
        let pp = self.pitch_proj.forward(&pitch_c)?; // [1, 128, T]
        let pp = pp.transpose(1, 2)?; // [1, T, 128]
        x = (x + pp)?;
        let x = length_regulate(&x, &dur_t)?;
        let x = self.decoder.forward(&x)?;
        let mel = self.lin.forward(&x)?; // [1, N, 192]
        let mel = mel.transpose(1, 2)?.contiguous()?; // [1, 192, N]
        let n = mel.dim(2)?;
        Ok((mel.flatten_all()?.to_vec1::<f32>()?, 192, n))
    }


    /// Все стадии tacotron (отладка верификации).
    fn debug_stages(&self, sequence: &[i64], speaker_ids: &[i64], speed: f32, device: &Device)
        -> Result<Vec<(String, Vec<f32>, Vec<usize>)>>
    {
        let seq = Tensor::new(sequence, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let sp = Tensor::new(speaker_ids, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let durs_v = self.durations(sequence, speaker_ids, speed, device)?;
        let dur_t = Tensor::new(&durs_v[..], device)?.unsqueeze(0)?;
        let pitch = self.pitch.forward(&seq, &sp)?.transpose(1, 2)?;
        let pitch_c = pitch.contiguous()?;

        let emb = self.embedding.forward(&seq)?;
        let enc = self.encoder.forward(&emb)?;
        let sp_emb = self.speaker_embedding.forward(&sp)?;
        let pre_enc = emb.broadcast_add(&sp_emb)?; // спикер добавляется ПОСЛЕ энкодера
        let pitch_c2 = pitch_c.contiguous()?;
        let pp = self.pitch_proj.forward(&pitch_c2)?.transpose(1, 2)?.contiguous()?;
        let post_pitch = (&enc.broadcast_add(&sp_emb)? + &pp)?;
        let expanded = length_regulate(&post_pitch, &dur_t)?;
        let dec = self.decoder.forward(&expanded)?;
        let mel = self.lin.forward(&dec)?.transpose(1, 2)?.contiguous()?;

        let dump = |t: &Tensor| -> Result<(Vec<f32>, Vec<usize>)> {
            let shape = t.dims().to_vec();
            Ok((t.flatten_all()?.to_vec1::<f32>()?, shape))
        };
        Ok(vec![
            { let (d, sh) = dump(&pre_enc)?; ("pre_enc".into(), d, sh) },
            { let (d, sh) = dump(&enc)?; ("enc".into(), d, sh) },
            { let (d, sh) = dump(&pitch)?; ("pitch".into(), d, sh) },
            { let (d, sh) = dump(&post_pitch)?; ("post_pitch".into(), d, sh) },
            { let (d, sh) = dump(&expanded)?; ("expanded".into(), d, sh) },
            { let (d, sh) = dump(&dec)?; ("dec".into(), d, sh) },
            { let (d, sh) = dump(&mel)?; ("mel".into(), d, sh) },
        ])
    }


    /// Стадии вокодера (отладка верификации).
    fn debug_voc(&self, mel: &Tensor) -> Result<Vec<(String, Vec<f32>, Vec<usize>)>> {
        let mut stages: Vec<(String, Tensor)> = Vec::new();
        let mut x = self.v_embed.forward(mel)?; // [1, 512, N]
        stages.push(("v_embed".into(), x.clone()));
        let x = x.transpose(1, 2)?.contiguous()?; // [1, N, 512]
        let x = layer_norm(&x, &self.v_norm.0, &self.v_norm.1, 1e-5f32)?;
        stages.push(("v_norm".into(), x.clone()));
        let mut x = x.transpose(1, 2)?.contiguous()?; // [1, 512, N]
        for (i, b) in self.convnext.iter().enumerate() {
            x = b.forward(&x)?;
            stages.push((format!("cn{i}"), x.clone()));
        }
        let x = x.transpose(1, 2)?.contiguous()?; // [1, N, 512]
        let x = layer_norm(&x, &self.v_final_norm.0, &self.v_final_norm.1, 1e-5f32)?;
        stages.push(("v_final".into(), x.clone()));
        let x = self.head_out.forward(&x)?; // [1, N, 2402]
        stages.push(("head_out".into(), x.clone()));
        let mut out = Vec::new();
        for (name, t) in stages {
            let shape = t.dims().to_vec();
            out.push((name, t.flatten_all()?.to_vec1::<f32>()?, shape));
        }
        Ok(out)
    }

    fn forward(&self, sequence: &[i64], speaker_ids: &[i64], speed: f32, device: &Device) -> Result<Vec<f32>> {
        let seq = Tensor::new(sequence, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;
        let sp = Tensor::new(speaker_ids, device)?.unsqueeze(0)?.to_dtype(DType::I64)?;

        // Длительности (общий путь с debug_durations)
        let durs_v = self.durations(sequence, speaker_ids, speed, device)?;
        let dur_t = Tensor::new(&durs_v[..], device)?.unsqueeze(0)?;

        // Pitch
        let pitch = self.pitch.forward(&seq, &sp)?; // [1, T, 1]
        let pitch = pitch.transpose(1, 2)?; // [1, 1, T]
        // update_pitch_coef с coefs=1.0 — no-op; пунктуационное зануление
        // в torch — slice-присваивание pitch_hat[a:b]=0 с КАДРОВЫМИ индексами
        // cumsum(dur_hat) по токен-тензору (для коротких фраз — тихий но-оп).
        // Воспроизводим дословно.
        let mut pitch_v = pitch.flatten_all()?.to_vec1::<f32>()?;
        {
            let mut cum = 0usize;
            let mut cums: Vec<(usize, usize)> = Vec::new();
            for (i, &id) in sequence.iter().enumerate() {
                let d = durs_v.get(i).copied().unwrap_or(0.0).round() as usize;
                cums.push((cum, cum + d));
                cum += d;
            }
            for (i, &id) in sequence.iter().enumerate() {
                if [3i64, 6, 7, 8, 9, 10, 11, 82, 83].contains(&id) {
                    let (a, b) = cums[i];
                    let a = a.min(pitch_v.len());
                    let b = b.min(pitch_v.len());
                    if a < b {
                        for v in &mut pitch_v[a..b] {
                            *v = 0.0;
                        }
                    }
                }
            }
        }
        let pitch_len = pitch_v.len();
        let pitch = Tensor::from_vec(pitch_v, (1, 1, pitch_len), device)?;

        // Tacotron — порядок как в JitMultiForward: энкодер получает
        // чистые эмбеддинги, спикер и питч добавляются ПОСЛЕ энкодера.
        let x = self.embedding.forward(&seq)?; // [1, T, 128]
        let mut x = self.encoder.forward(&x)?;
        let sp_emb = self.speaker_embedding.forward(&sp)?; // [1, 1, 128]
        x = x.broadcast_add(&sp_emb)?;
        let pitch_c = pitch.contiguous()?;
        let pp = self.pitch_proj.forward(&pitch_c)?; // [1, 128, T]
        let pp = pp.transpose(1, 2)?; // [1, T, 128]
        x = (x + pp)?;
        // Length regulator
        let x = length_regulate(&x, &dur_t)?;
        // Декодер
        let x = self.decoder.forward(&x)?;
        let mel = self.lin.forward(&x)?; // [1, N, 192]
        let mel = mel.transpose(1, 2)?.contiguous()?; // [1, 192, N]

        // Вокодер
        let x = self.v_embed.forward(&mel)?; // [1, 512, N]
        let x = x.transpose(1, 2)?.contiguous()?; // [1, N, 512]
        let x = layer_norm(&x.contiguous()?, &self.v_norm.0, &self.v_norm.1, 1e-5f32)?;
        let mut x = x.transpose(1, 2)?;
        for b in &self.convnext {
            x = b.forward(&x)?;
        }
        let x = x.transpose(1, 2)?;
        let x = layer_norm(&x.contiguous()?, &self.v_final_norm.0, &self.v_final_norm.1, 1e-5f32)?;
        let x = self.head_out.forward(&x)?; // [1, N, 2402]
        let x = x.transpose(1, 2)?; // [1, 2402, N]
        let parts = x.chunk(2, 1)?; // [1,1201,N] mag, phase
        let mag = parts[0].exp()?.clamp(0.0f32, 100.0f32)?.contiguous()?;
        let phase = parts[1].contiguous()?;
        let audio = istft_same(&mag, &phase, &self.window, 600, 2400)?;
        Ok(audio)
    }
}

/// LengthRegulator: повтор каждого кадра dur[t] раз.
fn length_regulate(x: &Tensor, dur: &Tensor) -> Result<Tensor> {
    let durs = dur.flatten_all()?.to_vec1::<f32>()?;
    if x.dim(1)? != durs.len() {
        bail!("length_regulate: x={}, dur={}", x.dim(1)?, durs.len());
    }
    let mut chunks = Vec::new();
    for (i, d) in durs.iter().enumerate() {
        let n = d.round().max(0.0) as usize;
        if n == 0 {
            continue;
        }
        chunks.push(x.i((.., i..i + 1, ..))?.contiguous()?.repeat((1, n, 1))?);
    }
    if chunks.is_empty() {
        bail!("Пустая длительность");
    }
    Ok(Tensor::cat(&chunks, 1)?)
}

/// iSTFT с режимом "same": irfft -> окно -> overlap-add -> нормировка огибающей.
fn istft_same(mag: &Tensor, phase: &Tensor, window: &Tensor, hop: usize, n_fft: usize) -> Result<Vec<f32>> {
    let device = mag.device();
    let frames = mag.dim(2)?;
    let freqs = n_fft / 2 + 1; // 1201

    // Комплексный спектр: S = mag · e^{i·phase}
    // re = mag·cos(phase), im = mag·sin(phase) — [1, 1201, T]
    let phase_c = phase.contiguous()?;
    let re = mag.broadcast_mul(&phase_c.cos()?)?.contiguous()?;
    let im = mag.broadcast_mul(&phase_c.sin()?)?.contiguous()?;

    // Таблицы COS/SIN [1201, 2400]
    let mut cos_v = Vec::with_capacity(freqs * n_fft);
    let mut sin_v = Vec::with_capacity(freqs * n_fft);
    for k in 0..freqs {
        for tau in 0..n_fft {
            let w = 2.0 * std::f64::consts::PI * (k as f64) * (tau as f64) / (n_fft as f64);
            cos_v.push(w.cos() as f32);
            // Найкивест (k = n/2) и DC: мнимая часть игнорируется torch.istft
            let sin_val = if k == 0 || k == freqs - 1 { 0.0 } else { w.sin() as f32 };
            sin_v.push(sin_val);
        }
    }
    let cos = Tensor::from_vec(cos_v, (freqs, n_fft), device)?;
    let sin = Tensor::from_vec(sin_v, (freqs, n_fft), device)?;
    // Весовые коэффициенты: k=0 и k=n/2 — без удвоения
    let mut w_v = vec![2.0f32; freqs];
    w_v[0] = 1.0;
    w_v[freqs - 1] = 1.0;
    let w = Tensor::from_vec(w_v, (freqs, 1), device)?;

    let re_w = re.broadcast_mul(&w.unsqueeze(0)?)?.contiguous()?; // [1, 1201, T]
    let im_w = im.broadcast_mul(&w.unsqueeze(0)?)?.contiguous()?;

    // time[τ] = Σ_k (re_w·cos − im_w·sin) / n : матричные произведения по кадрам
    // re [1,1201,T] @ COS [1201,2400] -> [1, T, 2400] для каждого батча
    let mut cos = cos.contiguous()?; // [freqs, n_fft]
    let mut sin = sin.contiguous()?; // [freqs, n_fft]
    let re_t = re_w.transpose(1, 2)?.contiguous()?.reshape((frames, freqs))?; // [T, freqs]
    let im_t = im_w.transpose(1, 2)?.contiguous()?.reshape((frames, freqs))?; // [T, freqs]
    let cos_part = re_t.matmul(&cos)?; // [T, 2400]
    let sin_part = im_t.matmul(&sin)?; // [T, 2400]
    let time = (cos_part - sin_part)?; // [T, 2400]
    let n_f = Tensor::new(n_fft as f32, device)?;
    let time = time.broadcast_div(&n_f)?;

    // Окно и overlap-add
    let win_v = window.to_vec1::<f32>()?;
    let pad = (n_fft - hop) / 2;
    let out_len = (frames - 1) * hop + n_fft;
    let mut y = vec![0.0f32; out_len];
    let mut env = vec![0.0f32; out_len];
    let time_v = time.to_vec2::<f32>()?; // [T][2400]
    for (t, frame) in time_v.iter().enumerate() {
        let base = t * hop;
        for i in 0..n_fft {
            let pos = base + i;
            if pos < out_len {
                y[pos] += frame[i] * win_v[i];
                env[pos] += win_v[i] * win_v[i];
            }
        }
    }
    if std::env::var_os("SILERO_ISTFT_DEBUG").is_some() {
        let mag_c = mag.i((0, ..8, 0))?.to_vec1::<f32>()?;
        let phase_c = phase.i((0, ..8, 0))?.to_vec1::<f32>()?;
        eprintln!("RUST mag[f=0..7, t=0]: {:?}", &mag_c.iter().map(|v| (v*10000.0).round()/10000.0).collect::<Vec<_>>());
        eprintln!("RUST phase[f=0..7, t=0]: {:?}", &phase_c.iter().map(|v| (v*10000.0).round()/10000.0).collect::<Vec<_>>());
        let re_t0 = re_t.i((0, ..8))?.to_vec1::<f32>()?;
        eprintln!("RUST re_t[0][0..8]: {:?}", &re_t0.iter().map(|v| (v*10000.0).round()/10000.0).collect::<Vec<_>>());
        let cos00 = cos.i((0, ..4))?.to_vec1::<f32>()?;
        let cos11 = cos.i((1, ..4))?.to_vec1::<f32>()?;
        eprintln!("RUST COS[0][0..4]: {:?} COS[1][0..4]: {:?}", &cos00.iter().map(|v| (v*10000.0).round()/10000.0).collect::<Vec<_>>(), &cos11.iter().map(|v| (v*10000.0).round()/10000.0).collect::<Vec<_>>());
        let t0: Vec<f32> = time_v[0].iter().take(8).map(|v| (v * 10000.0).round() / 10000.0).collect();
        eprintln!("ISTFT time[0][0..8]: {:?}", t0);
        eprintln!("ISTFT y[900..908]: {:?}", &y[900..908].iter().map(|v| (v * 10000.0).round() / 10000.0).collect::<Vec<_>>());
        eprintln!("ISTFT env[900..908]: {:?}", &env[900..908].iter().map(|v| (v * 10000.0).round() / 10000.0).collect::<Vec<_>>());
        eprintln!("ISTFT win[900..908]: {:?}", &win_v[900..908].iter().map(|v| (v * 10000.0).round() / 10000.0).collect::<Vec<_>>());
    }
    // Обрезка pad..-pad и нормировка
    let start = pad;
    let end = out_len - pad;
    let mut audio = Vec::with_capacity(end - start);
    for p in start..end {
        let v = if env[p] > 1e-12 { y[p] / env[p] } else { 0.0 };
        audio.push(v);
    }
    Ok(audio)
}
