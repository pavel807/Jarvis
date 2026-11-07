//! Централизованные пути к данным Jarvis.
//!
//! Два корня:
//! - **Данные** (`base_dir`): settings.toml, команды, задачник, образцы
//!   голоса. По умолчанию — каталог запуска; `JARVIS_HOME` переопределяет
//!   (нужно, когда процесс запущен не из корня репозитория).
//! - **Модели** (`models_dir`): системная папка программы по правилам ОС —
//!   macOS `~/Library/Application Support/Jarvis`, Linux
//!   `~/.local/share/Jarvis` (XDG), Windows `%APPDATA%\Jarvis`. Одна папка
//!   на всю систему, все модели в одном месте; `JARVIS_MODELS_DIR`
//!   переопределяет целиком.

use std::env;
use std::path::PathBuf;

pub fn base_dir() -> PathBuf {
    env::var_os("JARVIS_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Домашний каталог пользователя (для системных папок программы).
fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Системная папка данных программы — своя для каждой ОС.
pub fn app_data_dir() -> PathBuf {
    if let Some(dir) = env::var_os("JARVIS_DATA_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(target_os = "macos")]
    {
        home_dir()
            .map(|h| h.join("Library/Application Support/Jarvis"))
            .unwrap_or_else(|| base_dir())
    }
    #[cfg(target_os = "linux")]
    {
        let data_home = env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|h| h.join(".local/share")))
            .unwrap_or_else(|| base_dir());
        data_home.join("Jarvis")
    }
    #[cfg(target_os = "windows")]
    {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|h| h.join("AppData/Roaming")))
            .map(|d| d.join("Jarvis"))
            .unwrap_or_else(|| base_dir())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        base_dir()
    }
}

/// Каталог моделей: системная папка программы, единая на всю систему.
pub fn models_dir() -> PathBuf {
    env::var_os("JARVIS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| app_data_dir().join("models"))
}

/// Перенести модели из старого места (`<JARVIS_HOME>/models`) в системную
/// папку программы. Вызывается при старте движка; после переноса второй раз
/// не срабатывает (старого каталога больше нет). Перекрёстное устройство —
/// переименование не пройдёт, тогда копируем.
pub fn migrate_models() {
    let old = base_dir().join("models");
    let new = models_dir();
    if !old.is_dir() || old == new {
        return;
    }
    if new.exists() && std::fs::read_dir(&new).map(|mut d| d.next().is_some()).unwrap_or(false) {
        // Новая папка уже занята: ничего не переносим, старую не трогаем.
        return;
    }
    if let Some(parent) = new.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::rename(&old, &new) {
        Ok(()) => println!(
            "📦 Модели перенесены: {} → {}",
            old.display(),
            new.display()
        ),
        Err(_) => match copy_dir_recursive(&old, &new) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&old);
                println!(
                    "📦 Модели скопированы: {} → {}",
                    old.display(),
                    new.display()
                );
            }
            Err(e) => eprintln!(
                "⚠️ Не удалось перенести модели из {} в {}: {e:#} — использую старую папку",
                old.display(),
                new.display()
            ),
        },
    }
}

/// Рекурсивное копирование каталога (для переноса между томами).
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

pub fn resources_dir() -> PathBuf {
    base_dir().join("resources")
}

/// Каталог Python-плагинов автоматизации: системная папка программы.
pub fn plugins_dir() -> PathBuf {
    app_data_dir().join("plugins")
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