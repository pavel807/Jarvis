//! Парсинг конфига модели и доступ к локальным файлам MiniCPM5-1B-ru-v3.
//!
//! Модель лежит в `models/minicpm5-1b-ru/` (config.json, tokenizer.json,
//! model.safetensors — снапшот HF-репозитория). Рантайм **не делает сетевых
//! запросов**: если файлов нет — ошибка с подсказкой, где их взять.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// Идентификатор модели на Hugging Face — подсказка, где взять файлы,
/// если модель не скачана.
pub const HF_MODEL_ID: &str = "vahpetr/MiniCPM5-1B-ru-v3";
/// Локальный каталог модели внутри `models/`.
pub const LOCAL_DIR_NAME: &str = "minicpm5-1b-ru";

/// Параметры генерации.
///
/// `repeat_penalty = 1.1` и `repeat_window = 64` — **обязательное**
/// требование автора MiniCPM5 против зацикливания. Меньшие значения
/// (или `1.0`) приводят к повторам на коротких ответах.
///
/// `max_tokens = 0` — **без искусственного лимита**: генерация идёт до
/// естественного конца (EOS), закрытого tool-JSON, сработавшего фильтра
/// зацикливания, отмены или конца контекста модели. Обрыв по счётчику
/// посередине давал ложное «не могу» на длинный ответ или команду.
/// Ненулевое значение — страховочный кап по желанию пользователя
/// (`llm_max_tokens` в settings.toml).
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
            temperature: 0.5,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_window: 64,
            max_tokens: 0, // 0 — без искусственного лимита, до конца контекста
        }
    }
}

static GENERATION_PARAMS: std::sync::OnceLock<std::sync::RwLock<GenerationParams>> =
    std::sync::OnceLock::new();

/// Каталог модели LLM внутри `models/` (обновляется из `settings.toml`
/// до старта загрузчика).
static MODEL_NAME: std::sync::OnceLock<std::sync::RwLock<String>> = std::sync::OnceLock::new();

/// Бэкенд LLM: "candle" (реализации на candle) или "uzu" (движок Mirai,
/// только macOS). Обновляется из `settings.toml` до старта загрузчика.
static BACKEND: std::sync::OnceLock<std::sync::RwLock<String>> = std::sync::OnceLock::new();

fn default_backend() -> String {
    if cfg!(target_os = "macos") {
        "uzu".to_string()
    } else {
        "candle".to_string()
    }
}

/// Активный бэкенд LLM: по умолчанию uzu на macOS, candle на остальных ОС.
pub fn backend() -> String {
    BACKEND
        .get_or_init(|| std::sync::RwLock::new(default_backend()))
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Задать бэкенд (вызывается из `Settings::apply`).
pub fn set_backend(name: &str) {
    let value = match name {
        "uzu" | "candle" => name.to_string(),
        other => {
            eprintln!(
                "⚠️ Неизвестный llm_backend «{other}» — использую {default}",
                default = default_backend()
            );
            default_backend()
        }
    };
    let mut b = BACKEND
        .get_or_init(|| std::sync::RwLock::new(default_backend()))
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *b = value;
}

/// Каталог модели LLM: по умолчанию Qwen3.5-0.8B.
pub fn model_name() -> String {
    MODEL_NAME
        .get_or_init(|| std::sync::RwLock::new("qwen3.5-0.8b".to_string()))
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Задать каталог модели (вызывается из `Settings::apply`).
pub fn set_model_name(name: &str) {
    let mut n = MODEL_NAME
        .get_or_init(|| std::sync::RwLock::new("qwen3.5-0.8b".to_string()))
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *n = name.to_string();
}

/// Текущие параметры генерации (обновляются из `settings.toml`).
pub fn generation_params() -> &'static std::sync::RwLock<GenerationParams> {
    GENERATION_PARAMS.get_or_init(|| std::sync::RwLock::new(GenerationParams::default()))
}

/// Обновляет параметры генерации LLM.
pub fn update_generation_params(params: GenerationParams) {
    let mut p = generation_params()
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *p = params;
}

/// Разобранный `config.json` модели (Llama-архитектура).
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// `llama` для MiniCPM5. Может быть `null` в некоторых чекпоинтах.
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    /// Явный `head_dim` из конфига; иначе `hidden_size / num_attention_heads`.
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
        let cfg: Self =
            serde_json::from_str(&raw).with_context(|| format!("Ошибка парсинга {:?}", path))?;
        // Защита от случайной подмены модели: qwen3loop требует кастомного
        // forward, которого в model.rs нет.
        if cfg.model_type.as_deref() == Some("qwen3loop") {
            anyhow::bail!(
                "Модель qwen3loop требует кастомного forward — используйте \
                 unrolled GGUF или другую модель"
            );
        }
        Ok(cfg)
    }

    /// Llama-семейство ставит `<|begin_of_text|>` в начале промпта.
    pub fn use_bos_in_template(&self) -> bool {
        matches!(self.model_type.as_deref(), None | Some("llama"))
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// `head_dim` из конфига, иначе классический `hidden_size / num_heads`.
    /// У MiniCPM5 всегда 128, но оставляем fallback для совместимости.
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

    /// По умолчанию `true`: Llama-модели почти всегда связывают
    /// `lm_head` с `embed_tokens`. Если в конфиге явно `false` — тогда
    /// ищем отдельный `lm_head.weight` в safetensors.
    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(true)
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
    v.as_u64()
        .map(|x| x as u32)
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// Файлы модели в локальном кэше.
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// Проверяет, что файлы MiniCPM5-1B уже лежат в `models/minicpm5-1b-ru/`.
///
/// Сетевых запросов здесь **нет**: файлы либо на месте, либо ошибка с
/// подсказкой. Это осознанно — раньше падение HF-клиента выглядело как
/// «LLM недоступна» без внятной причины.
pub fn ensure_model_downloaded(models_dir: &Path) -> Result<ModelFiles> {
    let dir = models_dir.join(LOCAL_DIR_NAME);

    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    let weight = dir.join("model.safetensors");

    let missing: Vec<&str> = [
        ("config.json", &config),
        ("tokenizer.json", &tokenizer),
        ("model.safetensors", &weight),
    ]
    .iter()
    .filter(|(_, p)| !p.exists())
    .map(|(n, _)| *n)
    .collect();

    if !missing.is_empty() {
        anyhow::bail!(
            "Модель не найдена в {:?} (нет: {}). \
             Скачай снапшот HF {HF_MODEL_ID}: config.json, tokenizer.json, model.safetensors",
            dir,
            missing.join(", ")
        );
    }

    Ok(ModelFiles {
        config,
        tokenizer,
        weights: vec![weight],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minicpm5_layout() {
        let json = r#"{
            "hidden_size": 1536,
            "intermediate_size": 4096,
            "vocab_size": 130560,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "head_dim": 128,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "max_position_embeddings": 131072,
            "bos_token_id": 0,
            "eos_token_id": [1, 130073],
            "model_type": "llama"
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_key_value_heads(), 2);
        assert_eq!(cfg.rope_theta(), 5_000_000.0);
        // tie_word_embeddings не указан — по умолчанию true (Llama).
        assert!(cfg.tie_word_embeddings());
        assert_eq!(cfg.eos_ids(), vec![1, 130073]);
        assert_eq!(cfg.bos_id(), Some(0));
        assert!(cfg.use_bos_in_template());
    }

    #[test]
    fn head_dim_fallback_is_hidden_over_heads() {
        let json = r#"{
            "hidden_size": 1536, "intermediate_size": 4096, "vocab_size": 130560,
            "num_hidden_layers": 24, "num_attention_heads": 16,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 96); // 1536 / 16
    }

    #[test]
    fn explicit_untie_is_respected() {
        let json = r#"{
            "hidden_size": 1536, "intermediate_size": 4096, "vocab_size": 130560,
            "num_hidden_layers": 24, "num_attention_heads": 16,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.tie_word_embeddings());
    }

    #[test]
    fn generation_defaults_are_cycle_safe() {
        let p = GenerationParams::default();
        assert!(p.repeat_penalty >= 1.1, "мало penalty — будут циклы");
        assert!(p.repeat_window >= 64);
        // Без искусственного лимита: длинные ответы и команды досказываются
        // до конца, стоп — EOS и фильтр зацикливания, а не обрыв по счёту.
        assert_eq!(p.max_tokens, 0);
    }
}
