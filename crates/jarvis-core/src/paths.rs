//! Централизованные пути к данным Jarvis.
//!
//! По умолчанию всё лежит в текущем каталоге запуска; переменная окружения
//! `JARVIS_HOME` переопределяет корень (нужно, когда процесс запущен не из
//! корня репозитория, например из Tauri-приложения).

use std::env;
use std::path::PathBuf;

pub fn base_dir() -> PathBuf {
    env::var_os("JARVIS_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn models_dir() -> PathBuf {
    base_dir().join("models")
}

pub fn resources_dir() -> PathBuf {
    base_dir().join("resources")
}

pub fn train_dir() -> PathBuf {
    base_dir().join("train")
}

pub fn commands_file() -> PathBuf {
    base_dir().join("commands.toml")
}

pub fn settings_file() -> PathBuf {
    base_dir().join("settings.toml")
}

pub fn tts_models_dir() -> PathBuf {
    models_dir().join("Qwen3-TTS")
}

pub fn tts_cache_dir() -> PathBuf {
    resources_dir().join("tts_cache")
}

pub fn wakeword_model() -> PathBuf {
    models_dir().join("jarvis_model_2.rpw")
}

/// Путь из конфигов (относительные пути отсчитываются от корня данных).
pub fn resolve_data_path(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        base_dir().join(p)
    }
}
