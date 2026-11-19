//! Движок синтеза ZONOS3: текст → коды → (декодер) → звук.
//!
//! Генерация устроена как в эталоне: промпт (строка голоса, три строки
//! условий, текст, кадры тишины) прогоняется через бэкбон одним проходом,
//! затем идёт авторегрессия по кадрам.
//!
//! Delay-паттерн: строка потока `t` несёт коды `code[t][0]` … `code[t-9][9]` —
//! то есть книга `c` кадра `f` лежит в строке `f + c`. Поэтому шаг
//! генерации выдаёт сразу десять кодов (по одному на книгу), и каждый из
//! них принадлежит своему кадру. Кадр считается готовым, когда пришла его
//! десятая книга (строка `f + 9`).
//!
//! Cold start: пока «треугольник» задержки не заполнен, книги с номером
//! больше номера строки заполняются паддингом (`audio_pad_id`) — так же,
//! как в ggml-сервере и web-движке.

use super::backbone::Backbone;
use super::config::ZonosConfig;
use super::gguf::GgufFile;
use super::prompt::{PromptSpec, build_prompt};
use super::sampler::{Sampler, Sampling};
use super::tokenizer::TextTokenizer;
use anyhow::{Context, Result, bail};
use candle_core::{Device, IndexOp, Tensor};

/// Декодер кодов в звук (ZAC-декодер).
pub trait AudioDecoder: Send + Sync {
    /// Коды кадров (де-sheared) → сэмплы на 44.1 кГц.
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>>;
    /// Сэмплов на кадр (hop): для отбрасывания префилл-тишины.
    fn samples_per_frame(&self) -> usize;
}

/// Результат синтеза: коды кадров и, если есть декодер, звук.
pub struct Synthesis {
    /// Кадры (де-sheared): коды по книгам.
    pub frames: Vec<Vec<u32>>,
    /// Сэмплы 44.1 кГц (пусто, если декодера нет).
    pub audio: Vec<f32>,
    /// Частота дискретизации звука.
    pub sample_rate: u32,
}

/// Частота дискретизации на выходе ZONOS3.
pub const SAMPLE_RATE: u32 = 44100;

/// Устройство вычислений для TTS. По умолчанию CPU: Metal быстрее, но на
/// этом бэкбоне заметно портит генерацию (ASR теряет слова), поэтому он
/// включается только явно — `JARVIS_TTS_METAL=1`.
pub fn device() -> Device {
    #[cfg(target_os = "macos")]
    if std::env::var_os("JARVIS_TTS_METAL").is_some() {
        return Device::new_metal(0).unwrap_or(Device::Cpu);
    }
    Device::Cpu
}

/// Устройство ZAC-декодера. Декодер превращает уже выбранные коды в звук и
/// не влияет на текст, поэтому по умолчанию Metal: на нём декодер вдвое
/// быстрее CPU. `JARVIS_TTS_DECODER_CPU=1` возвращает CPU.
pub fn decoder_device() -> Device {
    #[cfg(target_os = "macos")]
    if std::env::var_os("JARVIS_TTS_DECODER_CPU").is_none() {
        return Device::new_metal(0).unwrap_or(Device::Cpu);
    }
    Device::Cpu
}

/// Движок синтеза.
pub struct ZonosEngine {
    pub cfg: ZonosConfig,
    pub tokenizer: TextTokenizer,
    pub backbone: Backbone,
    pub device: Device,
}

impl ZonosEngine {
    /// Загрузить движок из GGUF-бэкбона.
    pub fn load(backbone: &GgufFile, device: Device) -> Result<Self> {
        let cfg = ZonosConfig::from_gguf(backbone)?;
        let tokenizer = TextTokenizer::from_json(
            &cfg.tokenizer_json,
            cfg.text_offset,
            cfg.bos_id,
            cfg.eos_id,
            cfg.text_vocab.saturating_sub(cfg.text_offset as usize),
        )?;
        let backbone = Backbone::load(cfg.clone(), backbone, &device)?;
        Ok(Self {
            cfg,
            tokenizer,
            backbone,
            device,
        })
    }

    /// Синтез: текст → коды кадров (и звук, если передан декодер).
    ///
    /// `speaker` — 192-мерный эмбеддинг голоса (из спикер-энкодера);
    /// `buckets` — бакеты условий: громкость, f_max, темп речи;
    /// `cfg_scale` — офлайн-усилитель клонирования: guided = uncond +
    /// scale·(cond − uncond). В бою не используется (всегда 1.0): даёт
    /// +0.04–0.065 схожести голоса, но ~2× дороже и портит текст.
    pub fn synthesize(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
        cfg_scale: f32,
        decoder: Option<&dyn AudioDecoder>,
    ) -> Result<Synthesis> {
        let cfg = self.cfg.clone();
        let ncb = cfg.n_codebooks;
        // CFG применим только к клонированию (как в эталоне --cfg-scale).
        let cfg_scale = if speaker.is_some() { cfg_scale } else { 1.0 };

        let text_ids = self.tokenizer.text_ids(text)?;
        if text_ids.is_empty() {
            bail!("Пустой текст для синтеза");
        }

        let silence_prefill = cfg.silence_prefill_frames.min(super::prompt::SILENCE_PREFILL_MAX);
        let spec = PromptSpec {
            text_ids: &text_ids,
            speaker,
            buckets,
            silence_prefill,
            reference_codes: None,
        };
        let prompt = build_prompt(&cfg, &spec)?;
        let prefill_rows = prompt.ids.len();
        // Строка, с которой начинается аудиопоток (первый кадр тишины).
        let silence_start = prefill_rows - silence_prefill;

        // Twin для CFG: тот же текст/тишина, но без спикера и с неизвестными
        // условиями — «neutral»-ветка, от которой отталкивается усиление.
        let mut twin = None;
        if cfg_scale != 1.0 {
            let twin_spec = PromptSpec {
                text_ids: &text_ids,
                speaker: None,
                buckets: [
                    cfg.condition_bucket_counts[0],
                    cfg.condition_bucket_counts[1],
                    cfg.condition_bucket_counts[2],
                ],
                silence_prefill,
                reference_codes: None,
            };
            let twin_prompt = build_prompt(&cfg, &twin_spec)?;
            let mut twin_backbone = self.backbone.clone();
            twin_backbone.reset();
            let tl = twin_backbone.forward(&twin_prompt.ids, None, &twin_prompt.cond_rows)?;
            twin = Some((
                twin_backbone,
                tl.i((twin_prompt.ids.len() - 1) as usize)?.to_vec2::<f32>()?,
            ));
        }

        self.backbone.reset();
        let logits = self.backbone.forward(
            &prompt.ids,
            speaker.map(|s| (s, prompt.speaker_row.unwrap_or(0))),
            &prompt.cond_rows,
        )?;
        let mut last_logits = logits.i((prefill_rows - 1) as usize)?.to_vec2::<f32>()?;

        let mut sampler = Sampler::new(
            sampling.clone(),
            ncb,
            cfg.codebook_size,
            cfg.eoa_id,
        );
        // Dev-хак: JARVIS_ZONOS_MIN_FRAMES задаёт минимальное число кадров речи.
        if let Ok(v) = std::env::var("JARVIS_ZONOS_MIN_FRAMES") {
            if let Ok(n) = v.parse() {
                sampler.min_frames = n;
            }
        }
        // Поток кадров: сначала кадры тишины, потом сгенерированные строки.
        let mut stream_rows: Vec<Vec<u32>> = Vec::new();
        for k in 0..silence_prefill {
            stream_rows.push(prompt.ids[silence_start + k][..ncb].to_vec());
        }

        let mut eos_frame: Option<usize> = None;
        let mut countdown = 0usize;
        let max_frames = 2000usize;
        let gen_started = std::time::Instant::now();

        for step in 0..max_frames {
            let stream_row = silence_prefill + step;
            let mut row = vec![cfg.audio_pad_id; ncb];
            let mut eoa_at: Option<usize> = None;
            // CFG-микс: guided = uncond + scale·(cond − uncond), по каждой книге.
            let step_logits: Vec<Vec<f32>> = match &twin {
                None => last_logits.clone(),
                Some((_, twin_logits)) => {
                    let mut g = Vec::with_capacity(ncb);
                    for cb in 0..ncb {
                        let c = &last_logits[cb];
                        let u = &twin_logits[cb];
                        g.push(
                            u.iter()
                                .zip(c)
                                .map(|(&un, &co)| un + cfg_scale * (co - un))
                                .collect(),
                        );
                    }
                    g
                }
            };
            for cb in 0..ncb {
                // Cold start: книги, чья «задержка» ещё не началась, — паддинг.
                if cb > stream_row {
                    row[cb] = cfg.audio_pad_id;
                    continue;
                }
                let token = sampler.sample(&step_logits[cb], cb);
                row[cb] = token;
                if token == cfg.eoa_id && eoa_at.is_none() {
                    eoa_at = Some(cb);
                }
            }
            sampler.accept(&row);
            stream_rows.push(row.clone());

            // Конец речи: EOA задаёт выровненный кадр и запускает обратный отсчёт.
            if let Some(cb) = eoa_at {
                let aligned = stream_row.saturating_sub(cb);
                if eos_frame.is_none() {
                    eos_frame = Some(aligned);
                    countdown = ncb + 1;
                }
            }
            if countdown > 0 {
                countdown -= 1;
                if countdown == 0 {
                    break;
                }
            }

            let mut input_row = row.clone();
            input_row.push(cfg.text_pad_id);
            let logits = self.backbone.forward(std::slice::from_ref(&input_row), None, &[])?;
            last_logits = logits.i(0)?.to_vec2::<f32>()?;
            if let Some((twin_bb, twin_logits)) = twin.as_mut() {
                let tl = twin_bb.forward(std::slice::from_ref(&input_row), None, &[])?;
                *twin_logits = tl.i(0)?.to_vec2::<f32>()?;
            }
        }

        let rows = stream_rows.len();
        let frame_count = rows.saturating_sub(ncb - 1);
        let limit = eos_frame.map(|f| f.min(frame_count)).unwrap_or(frame_count);
        let gen_secs = gen_started.elapsed().as_secs_f32();
        let mut frames: Vec<Vec<u32>> = Vec::with_capacity(limit);
        for f in 0..limit {
            let mut frame = vec![cfg.audio_pad_id; ncb];
            for cb in 0..ncb {
                let row_idx = f + cb;
                if row_idx < rows {
                    frame[cb] = stream_rows[row_idx][cb];
                }
            }
            frames.push(frame);
        }

        let (audio, sample_rate) = match decoder {
            Some(dec) => {
                let dec_started = std::time::Instant::now();
                let mut samples = dec
                    .decode(&frames)
                    .context("ZAC-декодер не справился")?;
                let dec_secs = dec_started.elapsed().as_secs_f32();
                if std::env::var_os("JARVIS_VOICE_DEBUG").is_some() {
                    eprintln!(
                        "⏱ zonos: генерация {gen_secs:.1}s, декодер {dec_secs:.1}s, кадров {limit}"
                    );
                }
                let drop = silence_prefill * dec.samples_per_frame();
                let drop = drop.min(samples.len());
                samples.drain(..drop);
                studio_polish(&mut samples, SAMPLE_RATE);
                (samples, SAMPLE_RATE)
            }
            None => (Vec::new(), SAMPLE_RATE),
        };

        Ok(Synthesis {
            frames,
            audio,
            sample_rate,
        })
    }

    /// Только коды — удобно для тестов и отладки.
    pub fn synthesize_codes(
        &mut self,
        text: &str,
        speaker: Option<&[f32]>,
        buckets: [usize; 3],
        sampling: &Sampling,
    ) -> Result<Vec<Vec<u32>>> {
        Ok(self
            .synthesize(text, speaker, buckets, sampling, 1.0, None)?
            .frames)
    }
}

/// Студийная доводка озвучки: обрезка тишины по краям, анти-щелчок (короткие
/// фейды) и выравнивание громкости до −16 дБFS по активной части — как в
/// чистой студийной записи. Пик ограничен 0.94, усиление — не более ×3.
fn studio_polish(samples: &mut Vec<f32>, rate: u32) {
    if samples.len() < rate as usize / 4 {
        return;
    }
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 1e-3 {
        return;
    }
    // 1. Трим тишины по краям (порог −45 дБ от пика, поля 80 мс).
    let floor = peak * 10f32.powf(-45.0 / 20.0);
    let pad = (rate as usize * 80 / 1000).min(samples.len() / 2);
    let first = samples
        .iter()
        .position(|v| v.abs() > floor)
        .unwrap_or(0)
        .saturating_sub(pad);
    let last = samples
        .iter()
        .rposition(|v| v.abs() > floor)
        .map(|i| (i + pad).min(samples.len()))
        .unwrap_or(samples.len());
    samples.drain(last..);
    samples.drain(..first);

    // 2. Ровная громкость: −16 дБFS по активной части (выше порога).
    let floor_active = peak * 10f32.powf(-40.0 / 20.0);
    let active: Vec<&f32> = samples.iter().filter(|v| v.abs() > floor_active).collect();
    let rms = if active.len() > rate as usize / 10 {
        (active.iter().map(|v| **v).map(|v| v * v).sum::<f32>() / active.len() as f32).sqrt()
    } else {
        (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt()
    };
    if rms > 1e-5 {
        // Цель −16 дБFS; усиление ограничено 3× и пиком 0.94.
        let target = 10f32.powf(-16.0 / 20.0);
        let gain = (target / rms).min(3.0).min(0.94 / peak);
        if (gain - 1.0).abs() > 0.05 {
            for v in samples.iter_mut() {
                *v *= gain;
            }
        }
    }

    // 3. Анти-щелчок: фейды 8 мс по краям.
    let f = (rate as usize * 8 / 1000).min(samples.len() / 2);
    for (i, v) in samples[..f].iter_mut().enumerate() {
        *v *= i as f32 / f as f32;
    }
    let n = samples.len();
    for (i, v) in samples[n - f..].iter_mut().enumerate() {
        *v *= 1.0 - i as f32 / f as f32;
    }
}

/// Бакеты условий по умолчанию: громкость, f_max, темп — «неизвестно».
pub fn default_buckets(cfg: &ZonosConfig) -> [usize; 3] {
    [
        cfg.unknown_bucket(0),
        cfg.unknown_bucket(1),
        cfg.unknown_bucket(2),
    ]
}

/// Логиты последней строки промпта — для отладки.
pub fn last_logits(engine: &mut ZonosEngine, rows: usize) -> Result<Vec<Vec<f32>>> {
    let logits: Tensor = engine
        .backbone
        .forward(&vec![vec![engine.cfg.audio_pad_id; engine.cfg.n_codebooks + 1]; rows], None, &[])?;
    Ok(logits.to_vec2::<f32>()?)
}
