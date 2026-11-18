//! Конфигурация ZONOS3-Mini, собранная из метаданных GGUF.
//!
//! Всё нужное лежит в самом файле модели: `zonos3.*` ключи (архитектура,
//! размеры, бакеты условий, silence-frame) и встроенный токенизатор.

use super::gguf::GgufFile;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Конфигурация языковой части ZONOS3-Mini.
#[derive(Debug, Clone)]
pub struct ZonosConfig {
    pub n_layers: usize,
    pub dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seqlen: usize,
    pub logit_softcap: f32,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub audio_vocab: usize,
    pub eoa_id: u32,
    pub audio_pad_id: u32,
    pub text_vocab: usize,
    /// Дополнительные текстовые строки сверх словаря и паддинга (phoneme/start/pause).
    pub text_extra_tokens: usize,
    pub gdn_heads: usize,
    pub gdn_head_dim: usize,
    pub gdn_kernel: usize,
    /// Период GDN: каждый `gdn_period`-й слой (нумерация с 1) — глобальное
    /// внимание. Паттерн `gdn3_global1` → 3 GDN + 1 global → период 4.
    pub gdn_period: usize,
    pub n_experts: usize,
    pub moe_topk: usize,
    pub moe_hidden: usize,
    pub moe_routed_dim: usize,
    pub ffn_hidden: usize,
    pub speaker_dim: usize,
    /// Число бакетов у каждой таблицы условий (громкость, f_max, темп).
    pub condition_bucket_counts: Vec<usize>,
    /// Границы бакетов: по каждой таблице n-1 границ, `bisect_right` по ним.
    pub condition_edges: Vec<Vec<f32>>,
    pub condition_names: Vec<String>,
    pub condition_units: Vec<String>,
    /// Кадр тишины чекпоинта: десять кодов, которыми начинается промпт.
    pub silence_frame: Vec<u32>,
    /// Смещение идентификаторов текстовых токенов (payload = id + offset).
    pub text_offset: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    /// Текстовый токен «паддинг» (строка эмбеддингов `text_vocab`).
    pub text_pad_id: u32,
    /// Текстовый токенизатор (JSON из метаданных GGUF).
    pub tokenizer_json: String,
    /// JSON-описание схемы текстового токенизатора (kind/path/offset/bos/eos).
    pub text_tokenizer_spec: String,
    /// Образец тишины для префилла (3 кадра по умолчанию).
    pub silence_prefill_frames: usize,
}

impl ZonosConfig {
    /// Собрать конфигурацию из файла GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .kv_str("general.architecture")
            .unwrap_or_default()
            .to_string();
        if arch != "zonos3" {
            bail!("Ожидалась архитектура zonos3, получено «{arch}»");
        }

        let n_layers = file
            .kv_u64("zonos3.block_count")
            .context("zonos3.block_count")? as usize;
        let dim = file
            .kv_u64("zonos3.embedding_length")
            .context("zonos3.embedding_length")? as usize;
        let head_dim = file.kv_u64("zonos3.head_dim").context("zonos3.head_dim")? as usize;
        let n_heads = file
            .kv_u64("zonos3.attention.head_count")
            .context("zonos3.attention.head_count")? as usize;
        let n_kv_heads = file
            .kv_u64("zonos3.attention.head_count_kv")
            .context("zonos3.attention.head_count_kv")? as usize;
        let n_codebooks = file
            .kv_u64("zonos3.n_codebooks")
            .context("zonos3.n_codebooks")? as usize;
        let codebook_size = file
            .kv_u64("zonos3.codebook_size")
            .context("zonos3.codebook_size")? as usize;
        let audio_vocab = file
            .kv_u64("zonos3.audio_vocab")
            .context("zonos3.audio_vocab")? as usize;
        let text_vocab = file
            .kv_u64("zonos3.text_vocab")
            .context("zonos3.text_vocab")? as usize;

        // Паттерн gdnN_globalM: N GDN-слоёв, затем M глобального внимания.
        let pattern = file
            .kv_str("zonos3.attention.pattern")
            .unwrap_or("gdn3_global1");
        let gdn_period = parse_gdn_period(pattern)?;

        let condition_bucket_counts: Vec<usize> = file
            .kv_i64_array("zonos3.condition.bucket_counts")
            .unwrap_or_else(|| vec![10, 10, 30])
            .into_iter()
            .map(|v| v.max(0) as usize)
            .collect();

        let condition_names = file
            .kv_str_array("zonos3.condition.names")
            .unwrap_or_else(|| {
                vec![
                    "loudness".to_string(),
                    "fmax".to_string(),
                    "speaking_rate".to_string(),
                ]
            });
        if condition_names.len() != condition_bucket_counts.len() {
            bail!(
                "Условий {} но таблиц бакетов {}",
                condition_names.len(),
                condition_bucket_counts.len()
            );
        }
        let condition_units = file.kv_str_array("zonos3.condition.units").unwrap_or_default();

        let mut condition_edges = Vec::new();
        for name in &condition_names {
            let key = format!("zonos3.condition.edges.{name}");
            let edges = file
                .kv_f32_array(&key)
                .with_context(|| format!("Нет границ бакетов «{name}»"))?;
            condition_edges.push(edges);
        }

        let silence_frame = file
            .kv_i64_array("zonos3.silence_frame")
            .unwrap_or_default()
            .into_iter()
            .map(|v| v.max(0) as u32)
            .collect::<Vec<_>>();
        if silence_frame.len() != n_codebooks {
            bail!(
                "silence_frame: {} кодов вместо {n_codebooks}",
                silence_frame.len()
            );
        }

        let tokenizer_json = file
            .kv_str("zonos3.tokenizer.json")
            .context("В GGUF нет токенизатора")?
            .to_string();

        let text_tokenizer_spec = file
            .kv_str("zonos3.text_tokenizer_spec")
            .unwrap_or("{}")
            .to_string();

        let text_offset = file.kv_u64("zonos3.tokenizer.offset").unwrap_or(4) as u32;
        let bos_id = file.kv_u64("zonos3.tokenizer.bos").unwrap_or(2) as u32;
        let eos_id = file.kv_u64("zonos3.tokenizer.eos").unwrap_or(3) as u32;

        // Дополнительные текстовые строки: phonemes/start/pauses из спеки.
        let text_extra_tokens = extra_text_rows(&text_tokenizer_spec)?;

        let ffn_hidden = file
            .kv_u64("zonos3.feed_forward_length")
            .unwrap_or(2048) as usize;
        let n_experts = file.kv_u64("zonos3.expert_count").unwrap_or(64) as usize;
        let moe_topk = file
            .kv_i64_array("zonos3.expert_used_count_per_layer")
            .and_then(|v| v.iter().skip(1).copied().max())
            .unwrap_or(2)
            .max(1) as usize;
        let speaker_dim = file
            .kv_u64("zonos3.speaker.embedding_length")
            .unwrap_or(192) as usize;

        // Ширины MoE и GDN восстанавливаем из форм тензоров (в метаданных их нет).
        let moe_hidden = moe_hidden_from_gguf(file)?;
        let moe_routed_dim = latent_dim_from_gguf(file, n_layers)?;

        Ok(Self {
            n_layers,
            dim,
            head_dim,
            n_heads,
            n_kv_heads,
            norm_eps: file
                .kv_f32("zonos3.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            qk_norm_eps: file
                .kv_f32("zonos3.attention.qk_norm_epsilon")
                .unwrap_or(1e-6),
            rope_theta: file.kv_f32("zonos3.rope.freq_base").unwrap_or(10000.0),
            max_seqlen: file.kv_u64("zonos3.context_length").unwrap_or(32768) as usize,
            logit_softcap: file.kv_f32("zonos3.logit_softcap").unwrap_or(15.0),
            n_codebooks,
            codebook_size,
            audio_vocab,
            eoa_id: file.kv_u64("zonos3.eoa_id").unwrap_or(4096) as u32,
            audio_pad_id: file.kv_u64("zonos3.audio_pad_id").unwrap_or(4097) as u32,
            text_vocab,
            text_extra_tokens,
            gdn_heads: file.kv_u64("zonos3.gdn.head_count").unwrap_or(4) as usize,
            gdn_head_dim: file.kv_u64("zonos3.gdn.head_dim").unwrap_or(128) as usize,
            gdn_kernel: file.kv_u64("zonos3.gdn.conv_kernel").unwrap_or(4) as usize,
            gdn_period,
            n_experts,
            moe_topk,
            moe_hidden,
            moe_routed_dim,
            ffn_hidden,
            speaker_dim,
            condition_bucket_counts,
            condition_edges,
            condition_names,
            condition_units,
            silence_frame,
            text_offset,
            bos_id,
            eos_id,
            text_pad_id: text_vocab as u32,
            tokenizer_json,
            text_tokenizer_spec,
            silence_prefill_frames: 3,
        })
    }

    /// Слой — глобальное внимание (нумерация с нуля).
    pub fn is_attention_layer(&self, layer: usize) -> bool {
        (layer + 1) % self.gdn_period == 0
    }

    pub fn uses_gdn(&self, layer: usize) -> bool {
        !self.is_attention_layer(layer)
    }

    /// Внутренняя ширина GDN (голов × размер головы).
    pub fn gdn_inner(&self) -> usize {
        self.gdn_heads * self.gdn_head_dim
    }

    /// Число источников в банке Full AttnRes: эмбеддинг + два на слой.
    pub fn n_sources(&self) -> usize {
        2 * self.n_layers + 1
    }

    /// Строк в текстовой таблице эмбеддингов.
    pub fn text_rows(&self) -> usize {
        self.text_vocab + 1 + self.text_extra_tokens
    }

    /// Бакет значения условия: `bisect_right` по границам, `n` — «неизвестно».
    pub fn bucket_for(&self, table: usize, value: f32) -> usize {
        let edges = &self.condition_edges[table];
        let mut bucket = 0;
        for edge in edges {
            if value > *edge {
                bucket += 1;
            } else {
                break;
            }
        }
        bucket
    }

    /// Бакет «неизвестно» для таблицы.
    pub fn unknown_bucket(&self, table: usize) -> usize {
        self.condition_bucket_counts[table]
    }
}

/// Ширина эксперта из `ffn_gate_exps.weight` (ne = [routed, hidden, experts]).
fn moe_hidden_from_gguf(file: &GgufFile) -> Result<usize> {
    for layer in 1.. {
        let name = format!("blk.{layer}.ffn_gate_exps.weight");
        if let Ok(info) = file.info(&name) {
            if info.dims.len() != 3 {
                bail!("{name}: ожидалась стопка матриц, dims={:?}", info.dims);
            }
            return Ok(info.dims[1]);
        }
        if layer > 1024 {
            break;
        }
    }
    bail!("В GGUF нет экспертов MoE")
}

/// Латентная ширина MoE из `ffn_latent_down.weight` (ne = [dim, routed]).
fn latent_dim_from_gguf(file: &GgufFile, n_layers: usize) -> Result<usize> {
    for layer in 1..n_layers {
        let name = format!("blk.{layer}.ffn_latent_down.weight");
        if let Ok(info) = file.info(&name) {
            return Ok(info.dims[1]);
        }
    }
    bail!("В GGUF нет латентных проекций MoE")
}

/// Паттерн `gdn3_global1` → период 4 (три GDN-слоя, затем глобальное внимание).
fn parse_gdn_period(pattern: &str) -> Result<usize> {
    let rest = pattern
        .strip_prefix("gdn")
        .context("Неизвестный паттерн слоёв")?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let gdn: usize = digits.parse().context("Паттерн gdnN")?;
    let tail = &rest[digits.len()..];
    if !tail.starts_with("_global") {
        bail!("Неизвестный паттерн слоёв «{pattern}»");
    }
    Ok(gdn + 1)
}

/// Дополнительные текстовые строки из спеки токенизатора.
fn extra_text_rows(spec: &str) -> Result<usize> {
    let value: serde_json::Value =
        serde_json::from_str(spec).unwrap_or(serde_json::Value::Null);
    let mut rows = 0usize;
    if let Some(phonemes) = value.get("phonemes") {
        if let (Some(open), Some(close)) = (
            phonemes.get("open_id").and_then(|v| v.as_i64()),
            phonemes.get("close_id").and_then(|v| v.as_i64()),
        ) {
            rows += ((close - open + 1).max(0)) as usize;
        }
    }
    if value.get("start").is_some() {
        rows += 1;
    }
    if let Some(pauses) = value.get("pauses") {
        if let Some(ids) = pauses.get("ids").and_then(|v| v.as_array()) {
            rows += ids.len();
        }
    }
    Ok(rows)
}

/// Куски `params.json` — для сверки с метаданными GGUF.
#[derive(Debug, Deserialize)]
pub struct ParamsFile {
    pub model: ParamsModel,
}

#[derive(Debug, Deserialize)]
pub struct ParamsModel {
    pub n_layers: usize,
    pub dim: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub moe_n_experts: usize,
    pub moe_router_topk: usize,
    pub attention_pattern: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_period_from_pattern() {
        assert_eq!(parse_gdn_period("gdn3_global1").unwrap(), 4);
        assert!(parse_gdn_period("weird").is_err());
    }

    #[test]
    fn bucket_boundaries() {
        // Громкость: 10 бакетов, границы по params.json.
        let edges = vec![
            -29.778f32, -26.947, -25.047, -23.3, -21.743, -20.386, -19.186, -17.851, -15.967,
        ];
        let bucket = |v: f32| {
            let mut b = 0;
            for e in &edges {
                if v > *e {
                    b += 1;
                } else {
                    break;
                }
            };
            b
        };
        assert_eq!(bucket(-40.0), 0);
        assert_eq!(bucket(-29.0), 1);
        assert_eq!(bucket(0.0), 9);
    }
}
