//! Rust-часть установщика Jarvis: проверка системы, загрузка готового бинарника
//! из GitHub Releases, установка и удаление. Вся работа с процессами и файлами
//! живёт здесь; `App.tsx` зовёт эти методы как промисы. Ответы — JSON-строки
//! `{"ok":bool,"log":string}` (плюс свои поля у `paths` и `preflight`).

use nsh_core::prelude::*;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Экспорт в TSX: `import { Installer } from "jarvis-installer/native"`.
#[native_module("jarvis-installer/native")]
fn native(registry: &mut Registry) {
    registry.module::<Installer>();
}

// ── Помощники ──────────────────────────────────────────────────────────────

/// Строка как JSON-строка (с кавычками и экранированием).
fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jok(log: &str) -> String {
    format!("{{\"ok\":true,\"log\":{}}}", jstr(log))
}

fn jerr(log: &str) -> String {
    format!("{{\"ok\":false,\"log\":{}}}", jstr(log))
}

/// Успех с путём: куда легли скачанные бинарники.
fn jok_dir(log: &str, dir: &Path) -> String {
    format!(
        "{{\"ok\":true,\"log\":{},\"dir\":{}}}",
        jstr(log),
        jstr(&dir.display().to_string())
    )
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn prefix() -> PathBuf {
    std::env::var_os("JARVIS_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".local"))
}

fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".local/share"))
        .join("Jarvis")
}

/// Корень репозитория: JARVIS_REPO, иначе родитель `installer/`, иначе cwd.
fn repo_dir() -> PathBuf {
    if let Some(r) = std::env::var_os("JARVIS_REPO") {
        return PathBuf::from(r);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.file_name().is_some_and(|n| n == "installer") {
        cwd.parent().map(Path::to_path_buf).unwrap_or(cwd)
    } else {
        cwd
    }
}

fn have(bin: &str) -> bool {
    Command::new("sh")
        .args(["-c", &format!("command -v {bin} >/dev/null 2>&1")])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn pkgconfig(name: &str) -> bool {
    Command::new("pkg-config")
        .arg("--exists")
        .arg(name)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Запуск команды с захватом вывода: (успех, stdout+stderr).
fn run(bin: &str, args: &[String], cwd: Option<&Path>) -> (bool, String) {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    match cmd.output() {
        Ok(o) => {
            let mut log = String::from_utf8_lossy(&o.stdout).into_owned();
            log.push_str(&String::from_utf8_lossy(&o.stderr));
            (o.status.success(), log)
        }
        Err(e) => (false, format!("{bin}: {e}")),
    }
}

/// Последние `n` строк журнала: загрузка и распаковка печатают много.
fn tail(log: &str, n: usize) -> String {
    let lines: Vec<&str> = log.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

fn install_bin(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::copy(src, dst)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dst, fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

/// Имя архива с готовой сборкой для этой платформы (как в release.yml).
fn asset_name() -> &'static str {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "jarvis-macos-arm64.tar.gz"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "jarvis-linux-x64.tar.gz"
    } else {
        ""
    }
}

/// Платформа человеческим языком.
fn platform_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else {
        "Windows"
    }
}

// ── Модуль для TSX ─────────────────────────────────────────────────────────

/// Установщик Jarvis: все методы асинхронные, возвращают JSON-строку.
#[module]
pub struct Installer;

#[module]
impl Installer {
    /// Папки установки (JSON: repo, bin, cli, data, desktop).
    pub async fn paths() -> String {
        let repo = repo_dir();
        let p = prefix();
        let d = data_dir();
        format!(
            "{{\"repo\":{},\"bin\":{},\"cli\":{},\"data\":{},\"desktop\":{}}}",
            jstr(&repo.display().to_string()),
            jstr(&p.join("bin/jarvis").display().to_string()),
            jstr(&p.join("bin/jarvis-cli").display().to_string()),
            jstr(&d.display().to_string()),
            jstr(&p.join("share/applications/jarvis.desktop").display().to_string()),
        )
    }

    /// Проверка системы (JSON: platform, asset, curl, linux, deps[{name,ok}], missing[]).
    pub async fn preflight() -> String {
        let linux = cfg!(target_os = "linux");
        let mut deps: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        // Готовому бинарнику на Linux всё равно нужны рантайм-библиотеки Tauri.
        if linux {
            for (name, ok) in [
                ("webkit2gtk-4.1", pkgconfig("webkit2gtk-4.1")),
                ("libpulse", pkgconfig("libpulse")),
            ] {
                deps.push(format!("{{\"name\":{},\"ok\":{}}}", jstr(name), ok));
                if !ok {
                    missing.push(name.to_string());
                }
            }
        }
        let miss = missing
            .iter()
            .map(|m| jstr(m))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"platform\":{},\"asset\":{},\"curl\":{},\"bun\":{},\"cargo\":{},\"linux\":{},\"deps\":[{}],\"missing\":[{}]}}",
            jstr(platform_name()),
            jstr(asset_name()),
            have("curl"),
            have("bun"),
            have("cargo"),
            linux,
            deps.join(","),
            miss
        )
    }

    /// Ставит недостающие системные библиотеки: pkexec pacman, иначе sudo.
    pub async fn install_packages(packages: String) -> String {
        if !have("pacman") {
            return jerr("pacman не найден — поставь пакеты вручную и запусти установщик снова");
        }
        let mut full: Vec<String> = vec!["pacman".to_owned(), "-S".to_owned(), "--needed".to_owned()];
        full.extend(packages.split_whitespace().map(str::to_owned));
        // GUI-процесс не видит терминал для пароля sudo — просим через polkit.
        let (ok, log) = if have("pkexec") {
            run("pkexec", &full, None)
        } else {
            run("sudo", &full, None)
        };
        if ok {
            jok("библиотеки установлены")
        } else {
            jerr(&format!(
                "не удалось поставить пакеты ({}), поставь вручную:\n  pacman -S --needed {}\n{}",
                if have("pkexec") { "pkexec" } else { "sudo" },
                packages,
                tail(&log, 20)
            ))
        }
    }

    /// Скачивает готовый архив из GitHub Releases и распаковывает во временный каталог.
    pub async fn download() -> String {
        let asset = asset_name();
        if asset.is_empty() {
            return jerr(&format!(
                "для {} нет готовой сборки — собери из исходников: bun tauri build в app/",
                platform_name()
            ));
        }
        if !have("curl") {
            return jerr("curl не найден — поставь его и запусти установщик снова");
        }
        let dir = std::env::temp_dir().join("jarvis-installer");
        let _ = fs::remove_dir_all(&dir);
        if let Err(e) = fs::create_dir_all(&dir) {
            return jerr(&format!("{}: {e}", dir.display()));
        }
        let url = format!("https://github.com/pavel807/Jarvis/releases/latest/download/{asset}");
        let archive = dir.join(asset);
        let (ok, log) = run(
            "curl",
            &[
                "-fL".to_owned(),
                "--retry".to_owned(),
                "2".to_owned(),
                "-o".to_owned(),
                archive.display().to_string(),
                url.clone(),
            ],
            None,
        );
        if !ok {
            return jerr(&format!("загрузка не удалась ({}):\n{}", url, tail(&log, 10)));
        }
        let (ok, log) = run(
            "tar",
            &[
                "-xzf".to_owned(),
                archive.display().to_string(),
                "-C".to_owned(),
                dir.display().to_string(),
            ],
            None,
        );
        if !ok {
            return jerr(&format!("распаковка не удалась:\n{}", tail(&log, 10)));
        }
        let app = fs::read_dir(&dir).is_ok_and(|entries| {
            entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().starts_with("jarvis-app"))
        });
        if !app {
            return jerr("в архиве нет jarvis-app — релиз собран не по схеме release.yml");
        }
        jok_dir("скачано и распаковано", &dir)
    }

    /// Резервный путь: сборка релиза из исходников (bun install + bun tauri build).
    pub async fn build() -> String {
        let repo = repo_dir();
        if !have("bun") || !have("cargo") {
            return jerr("нет bun или cargo — для сборки из исходников нужны оба");
        }
        let app = repo.join("app");
        if !app.join("package.json").is_file() {
            return jerr(&format!(
                "не нашёл app/ в {} — запускай установщик из checkout репозитория",
                repo.display()
            ));
        }
        let (ok, log) = run(
            "bun",
            &["install".to_owned(), "--frozen-lockfile".to_owned()],
            Some(&app),
        );
        if !ok {
            return jerr(&format!("bun install не удался:\n{}", tail(&log, 30)));
        }
        let (ok, log) = run("bun", &["tauri".to_owned(), "build".to_owned()], Some(&app));
        if ok {
            jok(&format!(
                "сборка готова: target/release/jarvis-app\n{}",
                tail(&log, 15)
            ))
        } else {
            jerr(&format!("сборка не удалась:\n{}", tail(&log, 40)))
        }
    }

    /// Бинарь в PATH, каталоги данных, плагины .sk, ярлык в меню.
    /// `src` — каталог со скачанными бинарниками; пусто — target/release из репозитория.
    pub async fn install_files(src: String) -> String {
        let repo = repo_dir();
        let find = |prefix: &str| -> Option<PathBuf> {
            if src.is_empty() {
                let p = repo.join("target/release").join(prefix);
                return p.is_file().then_some(p);
            }
            fs::read_dir(&src).ok()?.flatten().find_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.starts_with(prefix).then(|| e.path())
            })
        };
        let Some(app_bin) = find("jarvis-app") else {
            return jerr(if src.is_empty() {
                "нет собранного бинаря target/release/jarvis-app — сначала сборка или загрузка"
            } else {
                "в скачанном архиве нет jarvis-app"
            });
        };
        let mut done: Vec<String> = Vec::new();

        let bin_dir = prefix().join("bin");
        if let Err(e) = fs::create_dir_all(&bin_dir) {
            return jerr(&format!("{}: {e}", bin_dir.display()));
        }
        let dst = bin_dir.join("jarvis");
        if let Err(e) = install_bin(&app_bin, &dst) {
            return jerr(&format!("{}: {e}", dst.display()));
        }
        done.push(dst.display().to_string());

        if let Some(cli_bin) = find("jarvis-cli") {
            let dst = bin_dir.join("jarvis-cli");
            if let Err(e) = install_bin(&cli_bin, &dst) {
                return jerr(&format!("{}: {e}", dst.display()));
            }
            done.push(dst.display().to_string());
        }

        let data = data_dir();
        for d in [data.clone(), data.join("plugins"), data.join("resources")] {
            if let Err(e) = fs::create_dir_all(&d) {
                return jerr(&format!("{}: {e}", d.display()));
            }
            done.push(format!("{}/", d.display()));
        }
        // Плагины не перезаписываем: там могут быть правки пользователя.
        if let Ok(entries) = fs::read_dir(repo.join("plugins")) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "sk") {
                    let dst = data.join("plugins").join(e.file_name());
                    if !dst.exists() {
                        let _ = fs::copy(&p, &dst);
                        done.push(dst.display().to_string());
                    }
                }
            }
        }

        if cfg!(target_os = "linux") {
            let share = prefix().join("share/applications");
            if let Err(e) = fs::create_dir_all(&share) {
                return jerr(&format!("{}: {e}", share.display()));
            }
            let desktop = share.join("jarvis.desktop");
            let icon = repo.join("app/src-tauri/icons/128x128.png");
            let body = format!(
                "[Desktop Entry]\nType=Application\nName=Jarvis\nGenericName=Голосовой ассистент\nComment=Слушает активационную фразу и выполняет команды\nExec={}\nIcon={}\nTerminal=false\nCategories=Utility;AudioVideo;\nStartupNotify=true\n",
                bin_dir.join("jarvis").display(),
                icon.display(),
            );
            if let Err(e) = fs::write(&desktop, body) {
                return jerr(&format!("{}: {e}", desktop.display()));
            }
            done.push(desktop.display().to_string());
            let _ = Command::new("update-desktop-database").arg(&share).output();
        }

        jok(&format!("установлено:\n  {}", done.join("\n  ")))
    }

    /// Удаление бинаря и ярлыка; `purge` сносит данные, модели и плагины.
    pub async fn uninstall(purge: bool) -> String {
        let p = prefix();
        let data = data_dir();
        let mut removed: Vec<String> = Vec::new();
        for path in [
            p.join("bin/jarvis"),
            p.join("bin/jarvis-cli"),
            p.join("share/applications/jarvis.desktop"),
        ] {
            if fs::remove_file(&path).is_ok() {
                removed.push(path.display().to_string());
            }
        }
        if purge && data.is_dir() {
            if let Err(e) = fs::remove_dir_all(&data) {
                return jerr(&format!("{}: {e}", data.display()));
            }
            removed.push(data.display().to_string());
        }
        let note = if purge {
            "данные удалены целиком".to_owned()
        } else {
            format!(
                "данные сохранены ({}) — модели, плагины, настройки",
                data.display()
            )
        };
        jok(&format!(
            "удалено:\n  {}\n{}",
            if removed.is_empty() {
                "(нечего удалять — Jarvis не установлен)".to_owned()
            } else {
                removed.join("\n  ")
            },
            note
        ))
    }
}
