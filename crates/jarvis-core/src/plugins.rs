//! Python-плагины автоматизации: manifest, скан, запуск через uv.
//!
//! Плагин — один `.py`-файл в `plugins/`. В шапке файла — блок манифеста
//! (TOML, как PEP 723) и зависимости для uv:
//!
//! ```text
//! # /// jarvis
//! # name = "quotes"
//! # title = "Цитаты"
//! # [[command]]
//! # name = "quote"
//! # description = "Случайная цитата программиста"
//! # ///
//! # /// script
//! # dependencies = ["requests"]
//! # ///
//! ```
//!
//! **Строгий контракт запуска**: Джарвис запускает файл через
//! `uv run --no-project <файл>` (uv сам ставит Python и зависимости из
//! блока `/// script`), подаёт на stdin одну JSON-строку
//! `{"command": "...", "args": {...}}` и читает ответ: последняя строка
//! stdout, разобранная как JSON `{"ok": bool, "speech": str, "detail": str?}`.
//! Ненулевой код возврата или невалидный JSON — ошибка плагина.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Таймаут выполнения плагина: первый запуск может ставить зависимости.
pub const PLUGIN_TIMEOUT_SECS: u64 = 120;

/// Одна команда плагина.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PluginCommand {
    pub name: String,
    pub description: String,
}

/// Разобранный плагин.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Plugin {
    /// Имя файла без расширения (идентификатор для запуска).
    pub id: String,
    /// Путь к файлу.
    pub path: PathBuf,
    /// Человекочитаемое название из манифеста.
    pub title: String,
    pub commands: Vec<PluginCommand>,
}

/// Результат выполнения команды плагина.
#[derive(Debug, Clone, PartialEq)]
pub struct PluginOutcome {
    pub ok: bool,
    /// Что произнести вслух.
    pub speech: String,
    /// Технические подробности для модели.
    pub detail: String,
}

/// Разобрать блок манифеста `# /// jarvis ... # ///` из текста плагина.
/// Возвращает (name, title, commands).
pub fn parse_manifest(source: &str) -> Option<(String, String, Vec<PluginCommand>)> {
    let mut block = String::new();
    let mut in_block = false;
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("# /// jarvis") {
            in_block = true;
            block.clear();
            continue;
        }
        if in_block {
            if trimmed.starts_with("# ///") {
                break;
            }
            // Снимаем префикс комментария: "# content" → "content".
            let content = trimmed.strip_prefix('#').unwrap_or(trimmed);
            block.push_str(content.trim_start_matches(' '));
            block.push('\n');
        }
    }
    if block.is_empty() {
        return None;
    }
    let manifest: toml::Value = toml::from_str(&block).ok()?;
    let name = manifest.get("name")?.as_str()?.to_string();
    let title = manifest
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or(&name)
        .to_string();
    let commands = manifest
        .get("command")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    Some(PluginCommand {
                        name: item.get("name")?.as_str()?.to_string(),
                        description: item
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some((name, title, commands))
}

/// Просканировать каталог плагинов: каждый .py с валидным манифестом.
pub fn scan() -> Vec<Plugin> {
    let dir = crate::paths::plugins_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut plugins: Vec<Plugin> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "py"))
        .filter_map(|path| {
            let source = std::fs::read_to_string(&path).ok()?;
            let (name, title, commands) = parse_manifest(&source)?;
            if commands.is_empty() {
                return None;
            }
            let id = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            Some(Plugin { id, path, title, commands })
        })
        .collect();
    plugins.sort_by(|a, b| a.id.cmp(&b.id));
    plugins
}

/// Блок «ПЛАГИНЫ» для промпта действий: имена команд и описания.
/// Пусто — блок не нужен.
pub fn prompt_block() -> String {
    if !plugins_enabled_now() {
        return String::new();
    }
    let plugins = scan();
    if plugins.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nПЛАГИНЫ (run_plugin {\"plugin\":\"id\",\"command\":\"имя\"}):\n");
    for plugin in &plugins {
        for command in &plugin.commands {
            out.push_str(&format!(
                "  run_plugin {{\"plugin\":\"{}\",\"command\":\"{}\"}} — {}\n",
                plugin.id, command.name, command.description
            ));
        }
    }
    out.push_str("  run_plugin {\"plugin\":\"id\",\"command\":\"имя\",\"args\":{\"ключ\":\"значение\"}} — с аргументами\n");
    out
}

/// Выполнить команду плагина через uv: stdin → JSON, последняя строка
/// stdout → JSON-ответ. Строгий контракт, см. модульную документацию.
pub fn run_command(plugin: &Plugin, command: &str, args: &Value) -> Result<PluginOutcome> {
    if !plugin.commands.iter().any(|c| c.name == command) {
        return Err(anyhow!(
            "в плагине «{}» нет команды «{command}»",
            plugin.title
        ));
    }
    let uv = uv_path().ok_or_else(|| {
        anyhow!("uv не установлен — включи автоматизацию в настройках")
    })?;
    let request = serde_json::json!({ "command": command, "args": args });

    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let mut child = Command::new(&uv)
        .args(["run", "--no-project", "--quiet"])
        .arg(&plugin.path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("не удалось запустить uv")?;

    child
        .stdin
        .take()
        .context("stdin плагина недоступен")?
        .write_all(request.to_string().as_bytes())?;

    let started = std::time::Instant::now();
    let output = loop {
        match child.try_wait().context("ошибка ожидания плагина")? {
            Some(status) => {
                // Процесс завершён сам: пайпы уже заполнены, читаем.
                break (status, ());
            }
            None => {
                if started.elapsed() > Duration::from_secs(PLUGIN_TIMEOUT_SECS) {
                    let _ = child.kill();
                    // kill требует &mut: wait после него дочитает статус.
                    let status = child.wait().context("статус плагина недоступен")?;
                    break (status, ());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    };
    let (status, ()) = output;

    // Пайпы: процесс уже завершён или убит — читаем напрямую.
    let mut stdout_bytes = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        use std::io::Read as _;
        let _ = pipe.read_to_end(&mut stdout_bytes);
    }
    let mut stderr_bytes = Vec::new();
    if let Some(mut pipe) = child.stderr.take() {
        use std::io::Read as _;
        let _ = pipe.read_to_end(&mut stderr_bytes);
    }
    let stdout = stdout_bytes;
    let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
    if !status.success() {
        let tail = stderr.lines().last().unwrap_or("").to_string();
        return Err(anyhow!(
            "плагин завершился с ошибкой ({}): {tail}",
            status.code().unwrap_or(-1)
        ));
    }
    let stdout = String::from_utf8_lossy(&stdout);
    // Последняя строка stdout, которая разбирается как JSON-ответ.
    let reply = stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<PluginReply>(line.trim()).ok())
        .ok_or_else(|| anyhow!("плагин не вернул JSON-ответ"))?;
    if !reply.ok {
        return Err(anyhow!(reply.speech));
    }
    Ok(PluginOutcome {
        ok: true,
        speech: reply.speech,
        detail: reply.detail.unwrap_or_default(),
    })
}

#[derive(Deserialize)]
struct PluginReply {
    ok: bool,
    speech: String,
    detail: Option<String>,
}

/// Плагины включены? Читается напрямую из settings.toml (hot-reload его
/// перечитывает), чтобы движку не нужно было таскать настройки в tools.
pub fn plugins_enabled_now() -> bool {
    crate::settings::Settings::load(&crate::paths::settings_file())
        .map(|s| s.plugins_enabled)
        .unwrap_or(false)
}

// --- Установка uv ---

/// Путь к uv: сначала наш каталог bin, затем PATH.
pub fn uv_path() -> Option<PathBuf> {
    let bundled = crate::paths::app_data_dir().join("bin").join(uv_binary_name());
    if bundled.is_file() {
        return Some(bundled);
    }
    // uv из PATH (пользователь ставил сам).
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(std::path::MAIN_SEPARATOR_STR) {
            let candidate = Path::new(dir).join(uv_binary_name());
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn uv_binary_name() -> &'static str {
    if cfg!(windows) { "uv.exe" } else { "uv" }
}

/// Ассет uv для текущей платформы (GitHub releases astral-sh/uv).
fn uv_asset() -> Result<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok("uv-aarch64-apple-darwin.tar.gz")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Ok("uv-x86_64-apple-darwin.tar.gz")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok("uv-x86_64-unknown-linux-gnu.tar.gz")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Ok("uv-aarch64-unknown-linux-gnu.tar.gz")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok("uv-x86_64-pc-windows-msvc.zip")
    } else {
        Err(anyhow!("платформа не поддерживается для авто-установки uv"))
    }
}

/// Установить uv в системную папку программы: скачать релиз astral-sh/uv,
/// распаковать, положить бинарник в `app_data/bin/`. Прогресс — 0..100.
pub fn install_uv(progress: &mut dyn FnMut(u8, String)) -> Result<PathBuf> {
    let asset = uv_asset()?;
    let url = format!("https://github.com/astral-sh/uv/releases/latest/download/{asset}");
    let bin_dir = crate::paths::app_data_dir().join("bin");
    std::fs::create_dir_all(&bin_dir)?;
    let tmp = std::env::temp_dir().join(format!("jarvis-{asset}"));

    progress(5, format!("Скачиваю uv: {asset}"));
    // model_fetch::download качает через curl с прогрессом по размеру файла.
    crate::model_fetch::download_public(&url, &tmp, 0, &mut |pct| {
        progress(5 + pct * 8 / 10, format!("Скачиваю uv: {pct}%"));
    })?;
    progress(85, "Распаковываю…".into());

    let extract_dir = std::env::temp_dir().join(format!("jarvis-uv-extract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&extract_dir);
    std::fs::create_dir_all(&extract_dir)?;
    if asset.ends_with(".zip") {
        // Windows: PowerShell умеет распаковывать zip без внешних утилит.
        let status = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command"])
            .arg(format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}'",
                tmp.display(),
                extract_dir.display()
            ))
            .status()
            .context("не удалось запустить PowerShell")?;
        anyhow::ensure!(status.success(), "распаковка zip не удалась");
    } else {
        let status = std::process::Command::new("tar")
            .arg("-xzf")
            .arg(&tmp)
            .arg("-C")
            .arg(&extract_dir)
            .status()
            .context("не удалось запустить tar")?;
        anyhow::ensure!(status.success(), "распаковка tar.gz не удалась");
    }

    // Бинарник лежит внутри каталога-релиза: ищем uv(.exe) рекурсивно.
    let binary = find_file(&extract_dir, uv_binary_name())
        .ok_or_else(|| anyhow!("в архиве uv не найден бинарник"))?;
    let dest = bin_dir.join(uv_binary_name());
    std::fs::copy(&binary, &dest).context("не удалось положить uv в bin")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755));
    }
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_dir_all(&extract_dir);
    progress(100, format!("Готово: {}", dest.display()));
    Ok(dest)
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
        if path.is_dir()
            && let Some(found) = find_file(&path, name)
        {
            return Some(found);
        }
    }
    None
}

/// Пример плагина: кладётся при первом включении автоматизации, чтобы
/// контракт был виден на живом файле (без внешних зависимостей).
pub fn ensure_sample_plugin() -> Result<PathBuf> {
    let dir = crate::paths::plugins_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("quotes.py");
    if path.exists() {
        return Ok(path);
    }
    std::fs::write(&path, SAMPLE_PLUGIN)?;
    Ok(path)
}

const SAMPLE_PLUGIN: &str = r#"# /// jarvis
# name = "quotes"
# title = "Цитаты программистов"
# [[command]]
# name = "quote"
# description = "Случайная цитата программиста"
# ///
# """Контракт: stdin — JSON {"command": ..., "args": {...}}; stdout —
# последняя строка JSON {"ok": true, "speech": "...", "detail": "..."}."""

import json
import sys
import random

QUOTES = [
    "Простое — надёжнее сложного. Прочность программ в их простоте.",
    "Сначала заставь работать, потом красиво, и только потом быстро.",
    "Хорошая архитектура — это отложенные решения, а не принятые заранее.",
]


def main() -> None:
    request = json.loads(sys.stdin.readline() or "{}")
    command = request.get("command", "")
    if command != "quote":
        reply = {"ok": False, "speech": f"Неизвестная команда: {command}"}
    else:
        topic = (request.get("args") or {}).get("topic")
        quote = random.choice(QUOTES)
        if topic:
            quote += f" (Кстати, про {topic} это тоже верно.)"
        reply = {"ok": True, "speech": quote}
    print(json.dumps(reply, ensure_ascii=False))


if __name__ == "__main__":
    main()
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"# /// jarvis
# name = "quotes"
# title = "Цитаты программистов"
# [[command]]
# name = "quote"
# description = "Случайная цитата программиста"
# [[command]]
# name = "add"
# description = "Складывает два числа"
# ///
import sys
"#;

    #[test]
    fn manifest_parses_name_title_commands() {
        let (name, title, commands) = parse_manifest(SAMPLE).expect("манифест должен разбираться");
        assert_eq!(name, "quotes");
        assert_eq!(title, "Цитаты программистов");
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].name, "quote");
        assert_eq!(commands[0].description, "Случайная цитата программиста");
        assert_eq!(commands[1].name, "add");
    }

    #[test]
    fn manifest_without_block_is_none() {
        assert!(parse_manifest("print('нет манифеста')").is_none());
    }

    #[test]
    fn prompt_block_lists_commands_or_empty() {
        // Пустой каталог плагинов — пустой блок (не ломает промпт).
        let block = prompt_block();
        // В тестовом окружении plugins_dir обычно пуст; блок либо пуст,
        // либо содержит строки run_plugin — оба варианта валидны.
        assert!(block.is_empty() || block.contains("run_plugin"));
    }

    #[test]
    fn sample_plugin_has_valid_manifest() {
        let (name, title, commands) =
            parse_manifest(SAMPLE_PLUGIN).expect("пример плагина должен разбираться");
        assert_eq!(name, "quotes");
        assert!(!title.is_empty());
        assert!(commands.iter().any(|c| c.name == "quote"));
    }

    #[test]
    fn uv_asset_matches_platform() {
        // На поддерживаемых платформах ассет определяется без паники.
        let result = uv_asset();
        assert!(result.is_ok() || cfg!(target_arch = "x86"));
    }
}
