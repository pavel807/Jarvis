//! Парсинг конфига модели и загрузка файлов с Hugging Face.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// Идентификатор репозитория LLM на Hugging Face.
pub const HF_MODEL_ID: &str = "Qwen/Qwen3-1.7B";
/// Имена файлов весов в репозитории (шарды).
pub const HF_WEIGHTS_FILES: &[&str] = &[
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
/// Локальный каталог модели внутри `models/`.
pub const LOCAL_DIR_NAME: &str = "qwen3-1.7b";

/// Параметры генерации (дефолты из рекомендаций Qwen3).
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub max_tokens: usize,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_window: 64,
            max_tokens: 256,
        }
    }
}

static GENERATION_PARAMS: std::sync::OnceLock<std::sync::RwLock<GenerationParams>> =
    std::sync::OnceLock::new();

/// Текущие параметры генерации (обновляются из settings.toml).
pub fn generation_params() -> &'static std::sync::RwLock<GenerationParams> {
    GENERATION_PARAMS.get_or_init(|| std::sync::RwLock::new(GenerationParams::default()))
}

/// Обновляет параметры генерации LLM.
pub fn update_generation_params(params: GenerationParams) {
    let mut p = generation_params().write().unwrap_or_else(|e| e.into_inner());
    *p = params;
}

/// Разобранный config.json модели (архитектура Llama-семейства).
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// llama (MiniCPM5) | qwen3 (Qwen3: QK-norm, без bos в шаблоне).
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    /// Явный head_dim из конфига, иначе классический hidden_size / num_heads.
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f32>,
    pub max_position_embeddings: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    /// bos/eos в HF бывают числом или массивом — принимаем оба вида.
    pub bos_token_id: Option<serde_json::Value>,
    pub eos_token_id: Option<serde_json::Value>,
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("Не удалось прочитать config.json: {:?}", path))?;
        serde_json::from_str(&raw).with_context(|| format!("Ошибка парсинга {:?}", path))
    }

    /// Использовать ли bos-токен в начале промпта (Qwen3 — нет).
    pub fn use_bos_in_template(&self) -> bool {
        matches!(self.model_type.as_deref(), None | Some("llama"))
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
            .unwrap_or(self.num_attention_heads)
    }

    /// Явный head_dim из конфига, иначе классический hidden_size / num_heads.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or(10_000.0)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(4096)
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_token_id.as_ref().and_then(value_to_u32)
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos_token_id
            .as_ref()
            .map(|v| match v {
                serde_json::Value::Array(items) => items.iter().filter_map(value_to_u32).collect(),
                other => value_to_u32(other).into_iter().collect(),
            })
            .unwrap_or_default()
    }
}

fn value_to_u32(v: &serde_json::Value) -> Option<u32> {
    v.as_u64().map(|x| x as u32).or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// Файлы модели в локальном кэше `models/<LOCAL_DIR_NAME>`.
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// Загружает модель в локальный кэш, докачивая недостающие файлы с HF.
pub fn ensure_model_downloaded(models_dir: &Path) -> Result<ModelFiles> {
    let dir = models_dir.join(LOCAL_DIR_NAME);
    fs::create_dir_all(&dir)?;

    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    let weights: Vec<PathBuf> = HF_WEIGHTS_FILES
        .iter()
        .map(|f| dir.join(f.replace('/', "_")))
        .collect();

    let ready = config.exists()
        && tokenizer.exists()
        && weights.iter().all(|p| p.exists());

    if ready {
        return Ok(ModelFiles {
            config,
            tokenizer,
            weights,
        });
    }

    println!("📥 Загрузка {HF_MODEL_ID} в {:?}...", dir);

    let client = hf_hub::HFClientSync::new()?;
    let (org, repo) = HF_MODEL_ID.split_once('/').unwrap_or((HF_MODEL_ID, ""));
    let api = client.model(org.to_string(), repo.to_string());

    for (remote, local) in [
        ("config.json", &config),
        ("tokenizer.json", &tokenizer),
    ] {
        if local.exists() {
            continue;
        }
        let downloaded = api
            .download_file()
            .filename(remote)
            .send()
            .with_context(|| format!("Не удалось скачать {remote}"))?;
        fs::copy(&downloaded, local)
            .with_context(|| format!("Не удалось скопировать {remote} в {:?}", local))?;
        println!("✅ {remote}");
    }

    for (remote, local) in HF_WEIGHTS_FILES.iter().zip(weights.iter()) {
        if local.exists() {
            continue;
        }
        let downloaded = api
            .download_file()
            .filename((*remote).to_string())
            .send()
            .with_context(|| format!("Не удалось скачать {remote}"))?;
        fs::copy(&downloaded, local)
            .with_context(|| format!("Не удалось скопировать {remote} в {:?}", local))?;
        println!("✅ {remote}");
    }

    println!("✅ {HF_MODEL_ID} готов");
    Ok(ModelFiles {
        config,
        tokenizer,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_eos(v: serde_json::Value) -> ModelConfig {
        let json = format!(
            r#"{{"hidden_size": 1536, "intermediate_size": 4608, "vocab_size": 130560,
                "num_hidden_layers": 24, "num_attention_heads": 16, "head_dim": 128,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 5000000,
                "max_position_embeddings": 131072, "tie_word_embeddings": false,
                "bos_token_id": 0, "eos_token_id": {}}}"#,
            v
        );
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn parses_mini_cpm5_layout() {
        let cfg = config_with_eos(serde_json::json!([1, 130073]));
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_key_value_heads(), 2);
        assert_eq!(cfg.rope_theta(), 5_000_000.0);
        assert!(!cfg.tie_word_embeddings());
        assert_eq!(cfg.eos_ids(), vec![1, 130073]);
        assert_eq!(cfg.bos_id(), Some(0));
    }

    #[test]
    fn eos_as_single_value() {
        let cfg = config_with_eos(serde_json::json!(1));
        assert_eq!(cfg.eos_ids(), vec![1]);
    }

    #[test]
    fn head_dim_fallback_is_hidden_over_heads() {
        let mut cfg = config_with_eos(serde_json::json!(1));
        cfg.head_dim = None;
        assert_eq!(cfg.head_dim(), 96);
    }
}
