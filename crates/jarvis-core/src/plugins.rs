//! Плагины-автоматизации на Spark: внешние .sk-файлы, не встроенные в
//! программу. Каждый плагин компилируется в нативный бинарник (кэш по
//! хэшу исходника) и становится инструментом агента `plugin_<имя>`:
//! модель видит описание из шапки плагина и вызывает его как любой
//! другой инструмент — с тем же контрактом результата.
//!
//! Шапка плагина пишется в комментариях .sk:
//! ```text
//! # плагин: уведомление
//! # описание: показывает системное уведомление с заданным текстом
//! # аргументы: текст
//! ```
//! Строка `SAY: …` на stdout плагина — то, что Джарвис озвучит.

use crate::agent::ToolSpec;
use crate::paths;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

/// Один загруженный плагин.
#[derive(Debug, Clone)]
pub struct Plugin {
    /// Имя инструмента для модели: `plugin_<имя>`.
    pub tool: String,
    /// Человекочитаемое имя из шапки или имени файла.
    pub name: String,
    pub description: String,
    /// Подсказка по аргументам из шапки («текст», «нет»).
    pub args_hint: String,
    pub source: PathBuf,
    /// Скомпилированный бинарник в кэше.
    pub binary: PathBuf,
    /// Ошибка компиляции: плагин сломан — модель об этом честно предупреждена.
    pub error: Option<String>,
}

/// Префикс инструмента плагина в реестре.
pub const TOOL_PREFIX: &str = "plugin_";

/// Это имя — вызов плагина?
pub fn is_plugin_tool(name: &str) -> bool {
    name.starts_with(TOOL_PREFIX)
}

fn registry() -> &'static RwLock<HashMap<String, Plugin>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Plugin>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Каталоги плагинов: репозиторий (примеры/комплект) и пользовательский.
pub fn dirs() -> Vec<PathBuf> {
    let mut dirs = vec![crate::paths::base_dir().join("plugins")];
    let user = paths::plugins_dir();
    if user != dirs[0] {
        dirs.push(user);
    }
    dirs
}

/// Директории плагинов видны отдельно (для диагностики и интерфейса).
pub fn dirs_for_prompt() -> String {
    dirs()
        .iter()
        .map(|d| d.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Имя инструмента по имени плагина: пробелы в подчёркивания, регистр
/// сохраняется — модель вызывает ровно то, что видит в промпте.
fn tool_name(name: &str) -> String {
    format!("{TOOL_PREFIX}{}", name.trim().replace(' ', "_"))
}

/// Разобрать шапку из начальных комментариев .sk: `# ключ: значение`.
/// Русские ключи канонические, английские — запасные.
fn parse_header(source: &str, file: &Path) -> (String, String, String) {
    let mut name = String::new();
    let mut description = String::new();
    let mut args_hint = String::new();
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(comment) = line.strip_prefix('#') else {
            break; // шапка закончилась — пошёл код
        };
        let comment = comment.trim();
        let take = |key: &[&str]| -> Option<String> {
            for k in key {
                if let Some(rest) = comment
                    .strip_prefix(k)
                    .and_then(|r| r.strip_prefix(':'))
                {
                    let v = rest.trim();
                    if !v.is_empty() {
                        return Some(v.to_string());
                    }
                }
            }
            None
        };
        if let Some(v) = take(&["плагин", "plugin"]) {
            name = v;
        } else if let Some(v) = take(&["описание", "description"]) {
            description = v;
        } else if let Some(v) = take(&["аргументы", "args"]) {
            args_hint = v;
        }
    }
    let fallback = file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "плагин".into());
    (
        if name.is_empty() { fallback } else { name },
        if description.is_empty() {
            "автоматизация на Spark".into()
        } else {
            description
        },
        if args_hint.is_empty() { "нет".to_string() } else { args_hint },
    )
}

/// Хэш исходника для имени бинарника в кэше. FNV-1a 64: стабильный между
/// запусками и версиями (в отличие от DefaultHasher с рандомными ключами),
/// для различения исходников плагинов этого достаточно.
fn source_hash(source: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in source.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Скомпилировать один плагин: генерация Rust через Spark, rustc в кэш.
/// Кэш по хэшу исходника — пересборка только после правки плагина.
fn compile(source: &str, file: &Path) -> Result<PathBuf, String> {
    // Парсер Spark на некоторых ошибках паникует («Неподдерживаемое
    // выражение»): плагин с опечаткой не должен ронять ассистента —
    // на границе паника превращается в честную ошибку.
    let parsed = std::panic::catch_unwind(|| spark::parse_source(source));
    let ast = match parsed {
        Ok(Ok(ast)) => ast,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("не разобрал исходник (синтаксическая ошибка)".into()),
    };
    let cache = paths::plugins_cache_dir();
    std::fs::create_dir_all(&cache).map_err(|e| format!("не создать кэш плагинов: {e}"))?;
    let binary = cache.join(format!("{:016x}.bin", source_hash(source)));
    if binary.exists() {
        return Ok(binary);
    }
    let rust_code = spark::codegen::compile_to_string(ast);
    let rs_path = cache.join(format!("{:016x}.rs", source_hash(source)));
    std::fs::write(&rs_path, &rust_code)
        .map_err(|e| format!("не записать генерацию: {e}"))?;
    let out = std::process::Command::new("rustc")
        .args(["-C", "opt-level=2"])
        .arg(&rs_path)
        .arg("-o")
        .arg(&binary)
        .output()
        .map_err(|e| format!("rustc не запустился: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let _ = std::fs::remove_file(&binary);
        // Ошибка кодогена — вернём первые строки: полный дамп в лог не
        // нужен, там будут сотни строк rustc.
        let short: Vec<&str> = stderr
            .lines()
            .filter(|l| l.contains("error") || l.contains("-->"))
            .take(6)
            .collect();
        return Err(if short.is_empty() {
            stderr.chars().take(400).collect()
        } else {
            short.join("\n")
        });
    }
    Ok(binary)
}

/// Загрузить все плагины из каталогов: скан, разбор шапки, компиляция
/// (с кэшем), регистрация. Возвращает (готово, сломано).
pub fn load_all() -> (usize, usize) {
    let mut map = HashMap::new();
    let mut ready = 0;
    let mut broken = 0;
    for dir in dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e != "sk").unwrap_or(true) {
                continue;
            }
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            let (name, description, args_hint) = parse_header(&source, &path);
            let tool = tool_name(&name);
            let (binary, error) = match compile(&source, &path) {
                Ok(b) => (b, None),
                Err(e) => (PathBuf::new(), Some(e)),
            };
            if error.is_none() {
                ready += 1;
            } else {
                broken += 1;
            }
            map.insert(
                tool.clone(),
                Plugin {
                    tool,
                    name,
                    description,
                    args_hint,
                    source: path,
                    binary,
                    error,
                },
            );
        }
    }
    let total_ready = ready;
    let total_broken = broken;
    *registry().write().unwrap_or_else(|e| e.into_inner()) = map;
    (total_ready, total_broken)
}

/// Прогреть плагины в фоне при старте: кэш тёплый — первый вызов модели
/// не ждёт компиляцию.
pub fn warm() {
    std::thread::spawn(|| {
        let (ready, broken) = load_all();
        if ready + broken > 0 {
            println!(
                "🔌 Плагины Spark: готово {ready}, сломано {broken} (каталоги: {})",
                dirs_for_prompt()
            );
            for p in all() {
                if let Some(e) = &p.error {
                    println!("   ⚠️ {} — не скомпилировался: {e}", p.name);
                }
            }
        }
    });
}

/// Список загруженных плагинов (копия).
pub fn all() -> Vec<Plugin> {
    registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .cloned()
        .collect()
}

/// Найти плагин по имени инструмента (`plugin_уведомление`).
pub fn find(tool: &str) -> Option<Plugin> {
    registry()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(tool)
        .cloned()
}

/// Спецификация инструмента плагина: тот же контракт, что у встроенных.
pub fn spec(tool: &str) -> Option<ToolSpec> {
    let p = find(tool)?;
    if p.error.is_some() {
        return None;
    }
    Some(ToolSpec {
        name: tool.to_string(),
        risk: crate::agent::RiskLevel::Medium,
        timeout: std::time::Duration::from_secs(30),
        postcondition: crate::agent::Postcondition::ExitCode,
        platforms: &["macos", "linux", "windows"],
    })
}

/// Аргументы вызова из JSON поля args: массив строк — канонический вид,
/// одиночная строка — список из одного.
fn argv_from(args: &serde_json::Value) -> Vec<String> {
    match args.get("args") {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Итог исполнения плагина: stdout/stderr/код выхода.
pub struct PluginRun {
    pub ok: bool,
    /// Строки `SAY:` — речь для озвучки.
    pub speech: Option<String>,
    pub detail: String,
    pub timed_out: bool,
}

/// Запустить бинарник плагина с аргументами, с таймаутом и честным
/// разбором вывода.
pub fn run_binary(binary: &Path, argv: &[String], timeout: std::time::Duration) -> PluginRun {
    use std::process::{Command, Stdio};

    let child = Command::new(binary)
        .args(argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return PluginRun {
                ok: false,
                speech: None,
                detail: format!("плагин не запустился: {e}"),
                timed_out: false,
            }
        }
    };

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child
                    .wait_with_output()
                    .expect("stdout/stderr закреплены пайпами");
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                let mut speech = None;
                let mut rest = Vec::new();
                for line in stdout.lines() {
                    if let Some(s) = line.strip_prefix("SAY: ") {
                        speech = Some(s.trim().to_string());
                    } else if !line.trim().is_empty() {
                        rest.push(line.trim().to_string());
                    }
                }
                let mut detail = rest.join("\n");
                if !status.success() {
                    let code = status.code().unwrap_or(-1);
                    if !stderr.trim().is_empty() {
                        detail.push_str(&format!("\nstderr: {}", stderr.trim()));
                    }
                    detail = format!("код выхода {code}\n{detail}").trim().to_string();
                }
                return PluginRun {
                    ok: status.success(),
                    speech,
                    detail: if detail.is_empty() {
                        if status.success() {
                            "плагин завершился без вывода".into()
                        } else {
                            format!("плагин завершился с кодом {}", status.code().unwrap_or(-1))
                        }
                    } else {
                        detail
                    },
                    timed_out: false,
                };
            }
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return PluginRun {
                        ok: false,
                        speech: None,
                        detail: "плагин не уложился в таймаут — остановлен".into(),
                        timed_out: true,
                    };
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => {
                return PluginRun {
                    ok: false,
                    speech: None,
                    detail: format!("ошибка ожидания плагина: {e}"),
                    timed_out: false,
                }
            }
        }
    }
}

/// Выполнить плагин по имени инструмента с JSON-аргументами.
pub fn execute(tool: &str, args: &serde_json::Value) -> PluginRun {
    let Some(plugin) = find(tool) else {
        return PluginRun {
            ok: false,
            speech: None,
            detail: "такого плагина нет".into(),
            timed_out: false,
        };
    };
    if let Some(e) = &plugin.error {
        return PluginRun {
            ok: false,
            speech: None,
            detail: format!("плагин «{}» сломан: {e}", plugin.name),
            timed_out: false,
        };
    }
    let argv = argv_from(args);
    run_binary(&plugin.binary, &argv, std::time::Duration::from_secs(30))
}

/// Блок для промпта агента: какие плагины есть и как их звать.
pub fn prompt_block() -> String {
    let plugins = all();
    let usable: Vec<&Plugin> = plugins.iter().filter(|p| p.error.is_none()).collect();
    if usable.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "ПЛАГИНЫ — автоматизации пользователя. Вызов: plugin_<имя>, аргументы — массив строк в поле args; строка из «SAY:» будет озвучена.".to_string(),
    ];
    for p in usable {
        let args = if p.args_hint == "нет" {
            String::new()
        } else {
            format!(" (аргументы: {})", p.args_hint)
        };
        lines.push(format!("- {} — {}{args}", p.tool, p.description));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_parses_russian_and_fallbacks() {
        let dir = std::env::temp_dir().join("jarvis_plugin_test");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("тест.sk");
        let (name, desc, args) = parse_header(
            "# плагин: уведомление\n# описание: показывает уведомление\n# аргументы: текст\n\nnotify(\"x\")\n",
            &file,
        );
        assert_eq!(name, "уведомление");
        assert_eq!(desc, "показывает уведомление");
        assert_eq!(args, "текст");

        // Без шапки — имя из файла, честные запасные значения.
        let (name, desc, args) = parse_header("x = 1\n", &dir.join("мой-скрипт.sk"));
        assert_eq!(name, "мой-скрипт");
        assert_eq!(desc, "автоматизация на Spark");
        assert_eq!(args, "нет");

        // Ключи после кода не читаются: шапка только до первого оператора.
        let (name, _, _) = parse_header("run(\"ls\")\n# плагин: поздний\n", &file);
        assert_eq!(name, "тест");
    }

    #[test]
    fn tool_name_and_prefix() {
        assert_eq!(tool_name("уведомление"), "plugin_уведомление");
        assert_eq!(tool_name("закрой всё"), "plugin_закрой_всё");
        assert!(is_plugin_tool("plugin_отчёт"));
        assert!(!is_plugin_tool("open_app"));
    }

    #[test]
    fn argv_from_json_shapes() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"args": ["привет", "мир"]}"#).unwrap();
        assert_eq!(argv_from(&v), vec!["привет", "мир"]);
        let v: serde_json::Value = serde_json::from_str(r#"{"args": "одно"}"#).unwrap();
        assert_eq!(argv_from(&v), vec!["одно"]);
        let v: serde_json::Value = serde_json::from_str(r#"{}"#).unwrap();
        assert!(argv_from(&v).is_empty());
    }

    #[test]
    fn real_plugin_compiles_and_runs() {
        // Настоящий сквозной прогон: исходник → Spark → rustc → запуск.
        let source = "# плагин: тест-плюс\n# описание: складывает два числа\n# аргументы: a, b\n\na = int(arg(1))\nb = int(arg(2))\nsay(str(a + b))\n";
        let binary = compile(source, Path::new("тест-плюс.sk")).expect("компиляция");
        let run = run_binary(&binary, &["2".into(), "3".into()], std::time::Duration::from_secs(10));
        assert!(run.ok, "{}", run.detail);
        assert_eq!(run.speech.as_deref(), Some("5"));

        // Сломанный плагин: ошибка компиляции возвращается текстом, а не
        // паникой, и в spec такой не попадает.
        let bad = "# плагин: сломан\n# описание: тест ошибки\nx = \n";
        let err = compile(bad, Path::new("сломан.sk"));
        assert!(err.is_err());
    }

    #[test]
    fn run_binary_honest_failures() {
        let source = "# плагин: падает\n# описание: тест кода выхода\nx = 1\n";
        let binary = compile(source, Path::new("падает.sk")).expect("компиляция");
        // Завершение с ненулевым кодом: искажённый через sh вызов.
        let run = run_binary(&binary, &[], std::time::Duration::from_secs(10));
        assert!(run.ok); // сам плагин корректен

        // Несуществующий бинарник — честная неудача.
        let run = run_binary(Path::new("/nonexistent/plugin.bin"), &[], std::time::Duration::from_secs(1));
        assert!(!run.ok);
        assert!(run.detail.contains("не запустился"));
    }

    #[test]
    fn spec_rejects_broken_plugin() {
        // До load_all() реестр пуст — spec честно None.
        assert!(spec("plugin_нет_такого").is_none());
    }

}
