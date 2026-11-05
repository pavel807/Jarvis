use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{path::{Path, PathBuf}, process::Command};

/// Файлы официального чекпойнта. Шардированные веса: имена берём из
/// `model.safetensors.index.json`, но фиксируем их и здесь, чтобы статус
/// «скачано» не зависел от чтения индекса.
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
/// Минимальный размер одного шарда весов: меньше — загрузка неполная.
const MIN_SHARD_BYTES: u64 = 3_500_000_000;

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: String,
    architectures: Vec<String>,
    text_config: TextConfig,
    vision_config: VisionConfig,
}
#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    head_dim: usize,
    /// Holo-3.1 зовёт поле `rope_scaling`, Qwen3.5 (Holo) — `rope_parameters`.
    #[serde(alias = "rope_scaling", alias = "rope_parameters")]
    rope: RopeScaling,
}
#[derive(Debug, Deserialize)]
struct RopeScaling {
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    /// Доля головы, охваченная ротором (Qwen3.5: 0.25 — MRoPE покрывает
    /// не всю голову, поэтому секции сравниваются с rotary_dim/2).
    #[serde(default = "default_partial")]
    partial_rotary_factor: f64,
}
fn default_partial() -> f64 {
    1.0
}
#[derive(Debug, Deserialize)]
struct VisionConfig { depth: usize, patch_size: usize, spatial_merge_size: usize }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStatus {
    pub directory: PathBuf,
    pub downloaded: bool,
    pub weights_bytes: u64,
    pub missing_files: Vec<String>,
    /// Причина, по которой конфиг не принят (для честного статуса в UI).
    pub config_error: Option<String>,
}

pub fn model_dir() -> PathBuf {
    crate::paths::models_dir().join("holo-3.1-4b")
}

pub fn model_status() -> ModelStatus {
    let dir = model_dir();
    let missing_files = FILES.iter().filter(|name| !dir.join(name).is_file())
        .map(|s| (*s).to_string()).collect::<Vec<_>>();
    let weights_bytes = std::fs::read_dir(&dir)
        .map(|entries| entries.filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum())
        .unwrap_or(0);
    let shards_ok = FILES.iter().filter(|f| f.ends_with(".safetensors"))
        .all(|f| std::fs::metadata(dir.join(f)).map(|m| m.len() >= MIN_SHARD_BYTES).unwrap_or(false));
    let config_error = validate_config(&dir.join("config.json")).err().map(|e| format!("{e:#}"));
    let downloaded = missing_files.is_empty() && shards_ok
        && weights_bytes >= 2 * MIN_SHARD_BYTES
        && config_error.is_none();
    ModelStatus { directory: dir, downloaded, weights_bytes, missing_files, config_error }
}

/// Download the exact official checkpoint. curl's resume support keeps a partial
/// multi-gigabyte file on disk and resumes it on the next call.
pub fn ensure_model(progress: &mut dyn FnMut(u8)) -> Result<PathBuf> {
    let dir = model_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("Не удалось создать {}", dir.display()))?;
    let base = format!("https://huggingface.co/{}/resolve/main", super::REPOSITORY);
    let total = FILES.len() as u64;
    for (index, file) in FILES.iter().enumerate() {
        let dest = dir.join(file);
        let complete = dest.is_file() && (file.ends_with(".safetensors")
            && std::fs::metadata(&dest)?.len() >= MIN_SHARD_BYTES
            || !file.ends_with(".safetensors") && std::fs::metadata(&dest)?.len() > 0);
        if complete {
            progress((((index + 1) as u64 * 100) / total) as u8);
            continue;
        }
        let part = dir.join(format!("{file}.part"));
        let url = format!("{base}/{file}");
        let status = Command::new(curl_bin())
            .args(["--fail", "--location", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30", "--continue-at", "-"])
            .arg("--output").arg(&part).arg(&url)
            .status().with_context(|| "Не удалось запустить curl для загрузки Holo-3.1-4B")?;
        if !status.success() { bail!("Загрузка {file} завершилась с кодом {status}"); }
        if file.ends_with(".safetensors") && std::fs::metadata(&part)?.len() < MIN_SHARD_BYTES {
            bail!("Файл {file} слишком мал; загрузка неполная или HF вернул ошибку");
        }
        std::fs::rename(&part, &dest).with_context(|| format!("Не удалось завершить файл {file}"))?;
        progress((((index + 1) as u64 * 100) / total) as u8);
    }
    validate_config(&dir.join("config.json"))?;
    let status = model_status();
    if !status.downloaded { bail!("Holo-3.1-4B скачана не полностью: отсутствуют {:?}", status.missing_files); }
    Ok(dir)
}

pub fn remove_model() -> Result<()> {
    let dir = model_dir();
    if dir.exists() { std::fs::remove_dir_all(&dir).with_context(|| format!("Не удалось удалить {}", dir.display()))?; }
    Ok(())
}

fn validate_config(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("Не найден {}", path.display()))?;
    let cfg: ModelConfig = serde_json::from_slice(&bytes).context("Некорректный config.json Holo-3.1-4B")?;
    if cfg.model_type != "qwen3_5" { bail!("Ожидался model_type=qwen3_5 (Holo/Qwen3.5), получен {}", cfg.model_type); }
    if !cfg.architectures.iter().any(|a| a == super::EXPECTED_ARCHITECTURE) { bail!("Неожиданная архитектура Holo-3.1-4B: {:?}", cfg.architectures); }
    // MRoPE: interleaved, три оси, покрыть head_dim/2.
    if !cfg.text_config.rope.mrope_interleaved {
        bail!("Ожидался interleaved MRoPE (mrope_interleaved=true)");
    }
    if cfg.text_config.rope.mrope_section.len() != 3 {
        bail!("mrope_section должен содержать три оси: {:?}", cfg.text_config.rope.mrope_section);
    }
    let sections_sum: usize = cfg.text_config.rope.mrope_section.iter().sum();
    let rotary_half = (cfg.text_config.head_dim as f64 * cfg.text_config.rope.partial_rotary_factor / 2.0) as usize;
    if sections_sum != rotary_half {
        bail!("mrope_section не покрывает rotary_dim/2: {} из {}", sections_sum, rotary_half);
    }
    Ok(())
}

fn curl_bin() -> String {
    std::env::var("JARVIS_CURL").unwrap_or_else(|_| "curl".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_config_shape_is_family_valid() {
        let json = r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],"text_config":{"hidden_size":2560,"num_hidden_layers":32,"vocab_size":248320,"head_dim":256,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10]}},"vision_config":{"depth":24,"patch_size":16,"spatial_merge_size":2}}"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.model_type, "qwen3_5");
        assert_eq!(cfg.vision_config.patch_size, 16);
        assert!(cfg.text_config.rope.mrope_interleaved);
        assert_eq!(cfg.text_config.rope.mrope_section, [11, 11, 10]);
    }
}
