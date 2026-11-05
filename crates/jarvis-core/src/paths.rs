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

/// Задачник: обычный markdown с чекбоксами, его можно править руками.
pub fn tasks_file() -> PathBuf {
    base_dir().join("tasks.md")
}

pub fn tts_cache_dir() -> PathBuf {
    resources_dir().join("tts_cache")
}

/// Активная модель wake word: имя файла задаётся настройкой
/// `wakeword_model` в settings.toml (см. `Settings::default`).
pub fn wakeword_model_file(name: &str) -> PathBuf {
    models_dir().join(name)
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