//! Linux: реализация платформенного слоя поверх привычных утилит окружения.
//!
//! Принцип тот же, что на macOS: успех подтверждается делом (существование
//! файла скриншота, изменение состояния playerctl, появление процесса после
//! запуска), а невозможность честно объясняется — чего не хватает и что
//! установить (`wmctrl`, `playerctl`, `brightnessctl`…).
//!
//! Окна: wmctrl/xdotool живут в мире X11. В сессии Wayland ими видны только
//! X11-окна (XWayland); для этого в сообщении об отсутствии окна есть
//! отдельная подсказка, чтобы «не нашёл» не выглядело поломкой.

use crate::platform::{
    COMMON_ALIASES, command_exists, fuzzy_app_match, lookup_alias, process_matches,
};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// =========================================================================
// Словарь русских названий приложений (Linux)
// =========================================================================

const APP_ALIASES: &[(&str, &str)] = &[
    ("браузер", "Firefox"),
    ("терминал", "Terminal"),
    ("файлы", "Files"),
    ("калькулятор", "Calculator"),
    ("редактор", "Text Editor"),
    ("монитор", "System Monitor"),
    ("музыка", "Rhythmbox"),
];

pub fn alias(word: &str) -> Option<&'static str> {
    lookup_alias(word, &[APP_ALIASES, COMMON_ALIASES])
}

// =========================================================================
// Оболочка запуска утилит
// =========================================================================

fn run(bin: &str, args: &[&str]) -> Result<String, String> {
    match Command::new(bin).args(args).output() {
        Ok(out) if out.status.success() => {
            Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let line = err.lines().next().unwrap_or("").to_string();
            Err(if line.is_empty() {
                format!(
                    "{bin} завершился с кодом {}",
                    out.status.code().unwrap_or(-1)
                )
            } else {
                line
            })
        }
        Err(e) => Err(e.to_string()),
    }
}

fn run_ok(bin: &str, args: &[&str]) -> bool {
    run(bin, args).is_ok()
}

fn in_wayland_session() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
}

// =========================================================================
// Приложения: .desktop-файлы и процессы
// =========================================================================

/// Запущено ли приложение с таким именем: точное совпадение имени процесса,
/// затем вхождение в полную строку запуска (`pgrep -f`).
pub fn app_running(name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() {
        return false;
    }
    let pgrep = |args: &[&str]| {
        Command::new("pgrep")
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    pgrep(&["-x", name]) || pgrep(&["-f", name])
}

/// Запись .desktop: как приложение называется и чем запускается.
#[derive(Clone, Debug)]
struct DesktopApp {
    /// Идентификатор для gtk-launch (имя файла без .desktop).
    id: String,
    /// Полный путь к файлу — запасной путь запуска через gio.
    path: String,
    /// Отображаемое имя (Name[ru] или Name).
    name: String,
    /// Имя исполняемого файла из Exec (@@@ — не разобрали).
    exec_bin: String,
}

/// Каталоги .desktop: системные, пользовательские и экспорт Flatpak/Snap.
fn desktop_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = vec![
        "/usr/share/applications".into(),
        "/usr/local/share/applications".into(),
        "/var/lib/flatpak/exports/share/applications".into(),
        "/var/lib/snapd/desktop/applications".into(),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        let home = std::path::PathBuf::from(home);
        dirs.push(home.join(".local/share/applications"));
        dirs.push(home.join(".local/share/flatpak/exports/share/applications"));
    }
    dirs
}

/// Разбор .desktop-файла: Name/Name[ru], Exec (берём первое слово до %-кода),
/// отбрасываем Hidden и NoDisplay — это служебные записи, не приложения.
fn parse_desktop(path: &Path, id: &str) -> Option<DesktopApp> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut in_entry = false;
    let mut name: Option<String> = None;
    let mut name_ru: Option<String> = None;
    let mut exec: Option<String> = None;
    let mut hidden = false;
    let mut no_display = false;
    let mut is_application = false;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_entry = line == "[Desktop Entry]";
            continue;
        }
        if !in_entry {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "Name" => name = Some(value.to_string()),
            "Name[ru]" => name_ru = Some(value.to_string()),
            "Exec" => exec = Some(value.to_string()),
            "Type" => is_application = value == "Application",
            "Hidden" => hidden = value.eq_ignore_ascii_case("true"),
            "NoDisplay" => no_display = value.eq_ignore_ascii_case("true"),
            _ => {}
        }
    }

    if hidden || no_display || !is_application {
        return None;
    }
    let name = name_ru.or(name)?;
    // Служебные записи без собственного окна (foot --server) в каталог
    // не попадают: модель выбирает из списка и «Foot Server» для неё —
    // правдоподобный терминал, но окна он не создаёт (реальный случай).
    let exec_low = exec.as_deref().unwrap_or_default().to_lowercase();
    let name_low = name.to_lowercase();
    let bin_low = exec_low.split_whitespace().next().unwrap_or_default();
    if exec_low.contains("--server")
        || exec_low.contains("--daemon")
        || bin_low.ends_with("-server")
        || bin_low.ends_with("-daemon")
        || name_low
            .split_whitespace()
            .any(|w| w == "server" || w == "daemon" || w == "сервер")
    {
        return None;
    }
    let exec_bin = exec.as_deref().and_then(exec_binary).unwrap_or_default();
    Some(DesktopApp {
        id: id.to_string(),
        path: path.to_string_lossy().to_string(),
        name,
        exec_bin,
    })
}

/// Первое исполняемое слово Exec. Пропускает `env` и присваивания
/// `VAR=значение` (так собраны .desktop у многих GNOME-приложений),
/// снимает кавычки.
fn exec_binary(exec: &str) -> Option<String> {
    let mut words = exec.split_whitespace().map(|w| w.trim_matches('"'));
    let first = words.next()?;
    if first == "env" {
        for w in words {
            if !w.contains('=') && !w.starts_with('%') {
                return Some(w.to_string());
            }
        }
        return None;
    }
    Some(first.to_string())
}

/// Установленные приложения из .desktop-каталогов. Кэш на минуту: список
/// читается при каждом распознавании имени, а меняется редко.
fn desktop_apps() -> Vec<DesktopApp> {
    static CACHE: OnceLock<Mutex<Option<(Instant, Vec<DesktopApp>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, apps)) = guard.as_ref()
            && at.elapsed() < Duration::from_secs(60)
        {
            return apps.clone();
        }
    }

    let mut apps = Vec::new();
    for dir in desktop_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file = entry.file_name();
            let fname = file.to_string_lossy();
            let Some(id) = fname.strip_suffix(".desktop") else {
                continue;
            };
            if let Some(app) = parse_desktop(&entry.path(), id) {
                apps.push(app);
            }
        }
    }
    *cache.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), apps.clone()));
    apps
}

/// Найти приложение по русской/приблизительной просьбе: точное имя, потом
/// сопоставление с транслитерацией, потом словарь, потом нечётко по смыслу.
fn find_desktop_app(want: &str) -> Option<DesktopApp> {
    let apps = desktop_apps();
    // Терминал — категория, а не имя: «открой терминал» резолвится по
    // установленным эмуляторам. Без этого модель в каталоге видит Foot,
    // не связывает его с «терминалом» и выбирает правдоподобный мусор
    // (реальный случай: вместо терминала открывался Docker).
    if crate::platform::is_terminal_request(want) {
        let names: Vec<String> = apps.iter().map(|a| a.name.clone()).collect();
        if let Some(name) = crate::platform::terminal_app_name(&names) {
            if let Some(app) = apps.iter().find(|a| a.name == name) {
                return Some(app.clone());
            }
        }
    }
    let exact = |pred: &dyn Fn(&DesktopApp) -> bool| apps.iter().find(|a| pred(a)).cloned();
    for variant in crate::platform::name_variants(want) {
        let hit = exact(&|a| a.name.eq_ignore_ascii_case(&variant))
            .or_else(|| exact(&|a| process_matches(&a.name, &variant)))
            .or_else(|| {
                alias(&variant).and_then(|target| {
                    exact(&|a| process_matches(&a.name, target))
                        .or_else(|| exact(&|a| process_matches(&a.exec_bin, target)))
                })
            })
            .or_else(|| {
                // Совпадение по имени бинарника: «код» → Exec=code.
                exact(&|a| process_matches(&a.exec_bin, &variant))
            })
            .or_else(|| {
                let names: Vec<String> = apps.iter().map(|a| a.name.clone()).collect();
                fuzzy_app_match(&variant, &names)
                    .and_then(|hit| apps.iter().find(|a| a.name == hit).cloned())
            });
        if hit.is_some() {
            return hit;
        }
    }
    None
}

/// Текущий браузер по умолчанию (desktop id), если xdg-settings отвечает.
fn default_browser_desktop() -> Option<String> {
    run("xdg-settings", &["get", "default-web-browser"])
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Открывает приложение: gtk-launch по desktop id, gio launch по файлу,
/// затем честная попытка запуска по имени. `Ok` — открылось (процесс
/// появился), `Err` — что именно не вышло.
pub fn app_open(name: &str) -> Result<String, String> {
    let want = name.trim();
    if want.is_empty() {
        return Err("Пустое имя приложения".to_string());
    }

    // «Открой браузер» — системный браузер по умолчанию, а не Firefox вслепую.
    if crate::text::compact_name(want) == "браузер" {
        if let Some(desktop) = default_browser_desktop() {
            let display = desktop.trim_end_matches(".desktop").to_string();
            if command_exists("gtk-launch") && run_ok("gtk-launch", &[&desktop]) {
                return Ok(format!("Открыл {display}"));
            }
            return Err(format!(
                "Не удалось запустить браузер по умолчанию ({desktop}): нет gtk-launch"
            ));
        }
    }

    // Разрешаем имя в .desktop-запись — по ней и запускаем.
    let entry = find_desktop_app(want);
    let display = entry
        .as_ref()
        .map(|a| a.name.clone())
        .unwrap_or_else(|| want.to_string());
    let probe = entry
        .as_ref()
        .map(|a| a.exec_bin.clone())
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| want.to_string());
    // pgrep ищет по имени процесса (comm), а в Exec бывает полный путь:
    // /usr/bin/firefox → firefox, иначе точное совпадение не находится.
    let probe_name = Path::new(&probe)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| probe.clone());

    // Приложение уже запущено — это не повод отказывать: просьба «открой»
    // в этом случае означает НОВОЕ ОКНО. Запускаем ещё раз и подтверждаем
    // счётом окон: появилось — честное «открыл новое окно», нет — честное
    // «уже открыт, новое окно не появилось».
    let was_running = proc_alive_exact(&probe_name) || app_running(&display);
    let windows_before = if was_running {
        window_count_for(&display, &probe_name)
    } else {
        0
    };

    // Запуск НЕ через run()/output(): запущенное приложение наследует
    // pipe stdout и держит его открытым — Command::output() ждал бы EOF
    // вечно. Null-потоки + spawn: «запущено» проверяет pgrep следом.
    let launched = match &entry {
        Some(app) if command_exists("gtk-launch") => Command::new("gtk-launch")
            .arg(&app.id)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok(),
        Some(app) if command_exists("gio") => Command::new("gio")
            .args(["launch", &app.path])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok(),
        _ => false,
    } || command_exists(&probe)
        && Command::new(&probe)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok();

    if !launched {
        return match entry {
            Some(app) => Err(format!(
                "Не удалось открыть {}: gtk-launch и gio не сработали",
                app.name
            )),
            None => Err(format!(
                "Не удалось открыть {want}: приложение не найдено среди установленных"
            )),
        };
    }

    // Статус запуска — ещё не гарантия: ждём появления процесса до 3 сек.
    for _ in 0..6 {
        if proc_alive_exact(&probe_name) {
            if !was_running {
                return Ok(format!("Открыл {display}"));
            }
            // Уже запущено: ждём новое окно до 3 сек, итог — по факту.
            for _ in 0..6 {
                std::thread::sleep(Duration::from_millis(500));
                if window_count_for(&display, &probe_name) > windows_before {
                    return Ok(format!("Открыл новое окно {display}"));
                }
            }
            return Ok(format!(
                "{display} уже был открыт — новое окно не появилось"
            ));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "Не удалось открыть {display}: приложение не запустилось"
    ))
}

/// Сколько открытых окон принадлежит приложению (по имени и процессу).
/// Наблюдение идёт через Hyprland IPC на Wayland и wmctrl на X11.
fn window_count_for(display: &str, probe: &str) -> usize {
    let Some(windows) = wmctrl_windows() else {
        return 0;
    };
    windows
        .iter()
        .filter(|w| {
            !w.title.trim().is_empty()
                && (process_matches(&w.class, display)
                    || process_matches(&w.class, probe)
                    || class_words(&w.class)
                        .iter()
                        .any(|c| process_matches(c, display) || process_matches(c, probe)))
        })
        .count()
}

/// Настоящее установленное приложение по русской/приблизительной просьбе.
/// `None` — такого приложения нет: подменять его файлом или сайтом нельзя.
/// Каталог установленных приложений для контекста агента: модель выбирает
/// из реального списка, а не выдумывает имена («Finder» на Linux).
/// Записи базы приложений: (имя, бинарь из Exec, id .desktop). Это и есть
/// «база данных приложений»: сканируется при старте, отдаётся модели в
/// промпт и пишется в JSON — быстро и без повторного обхода каталогов.
pub fn app_db_entries() -> Vec<(String, String, String)> {
    desktop_apps()
        .into_iter()
        .map(|a| (a.name, a.exec_bin.clone(), a.id))
        .collect()
}

pub fn app_catalog() -> Vec<String> {
    let mut names: Vec<String> = desktop_apps()
        .into_iter()
        .map(|a| a.name)
        .collect();
    names.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
    names.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    names.truncate(40);
    names
}

pub fn app_lookup(name: &str) -> Option<String> {
    let want = name.trim();
    if want.is_empty() {
        return None;
    }
    find_desktop_app(want).map(|app| app.name)
}

/// Имя приложения по русской/приблизительной просьбе (для быстрых путей).
/// Ничего не нашлось — возвращаем вход как есть.
pub fn app_resolve(name: &str) -> String {
    let want = name.trim();
    if want.is_empty() {
        return want.to_string();
    }
    match app_lookup(want) {
        Some(app) => app,
        None => match alias(want) {
            Some(target) => target.to_string(),
            None => want.to_string(),
        },
    }
}

/// Просит приложение закрыться: SIGTERM — вежливо, приложения успевают
/// сохранить состояние. `false` — процесс не найден.
pub fn app_quit(name: &str) -> bool {
    let requested = name.trim();
    if requested.is_empty() {
        return false;
    }
    // Для pkill нужно имя процесса (telegram-desktop), а не русское
    // отображаемое имя или «телеграм» из просьбы.
    let entry = find_desktop_app(requested);
    let display = entry
        .as_ref()
        .map(|a| a.name.clone())
        .unwrap_or_else(|| requested.to_string());
    let exec_path = entry
        .as_ref()
        .map(|a| a.exec_bin.trim().to_string())
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| display.clone());
    let probe = Path::new(&exec_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| exec_path.clone());
    if !app_running(&display) && !proc_alive_exact(&probe) {
        return false;
    }
    println!("🛑 Закрываю: {display}");
    // SIGTERM — вежливый сигнал. Комм процесса обрезается до 15 символов
    // (telegram-desktop → telegram-deskt), поэтому пробуем оба имени, а
    // для -f — полный путь из .desktop: -f по короткому слову цепляет
    // собственные процессы ассистента (bash с именем в аргументах).
    let _ = run_ok("pkill", &["-x", &probe])
        || run_ok("pkill", &["-x", &comm15(&probe)])
        || run_ok("pkill", &["-f", &exec_path])
        || run_ok("pkill", &["-f", &probe]);
    // Успех — не код возврата pkill, а исчезнувший процесс, проверенный
    // ТОЧНЫМ именем (comm), а не широким -f: последний матчит cmdline
    // самой цепочки ассистента и вечно «видит живой» процесс.
    for _ in 0..6 {
        if !proc_alive_exact(&probe) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    // Не ушли сами — SIGKILL, это и есть доведение до результата.
    let _ = run_ok("pkill", &["-9", "-x", &probe])
        || run_ok("pkill", &["-9", "-x", &comm15(&probe)])
        || run_ok("pkill", &["-9", "-f", &exec_path])
        || run_ok("pkill", &["-9", "-f", &probe]);
    for _ in 0..4 {
        if !proc_alive_exact(&probe) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    println!("⚠️ {display}: процесс не завершился после SIGTERM и SIGKILL");
    false
}

/// Жив ли процесс с точным именем: comm обрезается ядром до 15 символов,
/// поэтому проверяем и полное имя, и обрезанное.
fn proc_alive_exact(probe: &str) -> bool {
    let comm = comm15(probe);
    run_ok("pgrep", &["-x", probe]) || (comm != probe && run_ok("pgrep", &["-x", &comm]))
}

fn comm15(name: &str) -> String {
    name.chars().take(15).collect()
}

// =========================================================================
// Окна: wmctrl и xdotool
// =========================================================================

/// Окно из `wmctrl -lx`: id, WM_CLASS («instance.Class»), заголовок.
#[derive(Clone, Debug, PartialEq)]
struct LinuxWindow {
    id: String,
    class: String,
    title: String,
}

/// Разбор вывода `wmctrl -lx`:
/// `0x05600007  1 google-chrome.Google-chrome  host Заголовок окна`
fn parse_wmctrl_lx(out: &str) -> Vec<LinuxWindow> {
    out.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let id = parts.next()?.to_string();
            if !id.starts_with("0x") {
                return None;
            }
            let _desktop = parts.next()?;
            let class = parts.next()?.to_string();
            let _host = parts.next()?;
            let title = parts.collect::<Vec<_>>().join(" ");
            Some(LinuxWindow { id, class, title })
        })
        .collect()
}

/// Класс окна в человеческие слова: «google-chrome.Google-chrome» →
/// [google chrome, Google chrome] — по ним и матчим просьбу.
fn class_words(class: &str) -> Vec<String> {
    class
        .split(['.', '-', '_'])
        .map(|w| w.trim().to_string())
        .filter(|w| w.len() >= 2)
        .collect()
}

fn window_matches(window: &LinuxWindow, want: &str) -> bool {
    let words = class_words(&window.class);
    words.iter().any(|w| process_matches(w, want))
        || process_matches(&window.title, want)
        || alias(want).is_some_and(|target| {
            words.iter().any(|w| process_matches(w, target))
                || process_matches(&window.title, target)
        })
}

// =========================================================================
// Hyprland (Wayland): IPC `hyprctl` — единственный источник окон в
// чистой Wayland-сессии, wmctrl/xdotool туда не смотрят.
// =========================================================================

/// Управление окном на Hyprland через hl.dsp.* диспетчеры. Возвращает
/// Some(ответ) для обработанных действий, None — пусть идёт старый путь.
fn hyprland_window_control(action: &str, app: Option<&str>) -> Option<String> {
    let action = action.trim().to_lowercase();
    // У movetodesktop номер стола в конце строки действия — оставляем его:
    // выбор окна идёт по глаголу, номер нужен самому действию.
    let verb = action.split_whitespace().next().unwrap_or("");
    match verb {
        "close" | "minimize" | "maximize" | "fullscreen" | "restore" | "focus"
        | "movetodesktop" | "half_left" | "half_right" => {}
        _ => return None,
    }
    let is_self = app.is_some() && crate::platform::is_self_window_target(app.unwrap_or(""));
    let target = match app {
        Some(wanted)
            if is_self
                || ["активное", "активную", "активный", "это", "текущее", "active"]
                    .contains(&wanted.trim().to_lowercase().as_str()) =>
        {
            // Своё окно ищем по классу, «активное» — по активному.
            if is_self {
                wmctrl_windows()?.into_iter().find(|w| {
                    class_words(&w.class)
                        .iter()
                        .any(|c| c.to_lowercase().contains("jarvis"))
                })
            } else {
                let out = run("hyprctl", &["-j", "activewindow"]).ok()?;
                parse_hypr_single(&out)
            }
        }
        Some(wanted) => find_window(wanted),
        None => wmctrl_windows()?.into_iter().find(|w| {
            class_words(&w.class)
                .iter()
                .any(|c| c.to_lowercase().contains("jarvis"))
        }),
    };
    let Some(window) = target else {
        let name = app.unwrap_or("окно");
        if is_self {
            return Some("Не нашёл своё окно — я в фоне. Скажи, чьё окно трогать".to_string());
        }
        return Some(format!("Приложение {name} не найдено среди открытых"));
    };
    Some(hyprland_window_action(&window, &action))
}

/// Действие над уже найденным окном Hyprland. Вынесено отдельно: выбор
/// конкретного окна приходит из интерфейса уже с заголовком, а не с именем
/// приложения.
fn hyprland_window_action(window: &LinuxWindow, action: &str) -> String {
    let action = action.trim().to_lowercase();
    let addr = window.id.clone();
    let title = if window.title.trim().is_empty() {
        window.class.clone()
    } else {
        window.title.clone()
    };
    let sel = format!("address:{addr}");

    // Матч по глаголу: у movetodesktop номер стола в конце строки действия.
    let verb = action.split_whitespace().next().unwrap_or("");
    match verb {
        "close" => {
            let ok = run_ok(
                "hyprctl",
                &["dispatch", &format!("hl.dsp.window.close(\"{sel}\")")],
            );
            // Верификация: окно обязано исчезнуть из списка. Приложения
            // выходят не мгновенно — окно верификации до 4 секунд.
            for _ in 0..10 {
                if !hypr_window_exists(&addr) {
                    return format!("Закрыл окно {title}");
                }
                std::thread::sleep(Duration::from_millis(400));
            }
            let _ = ok;
            format!("Окно {title} не закрылось — приложение отказалось")
        }
        "minimize" => {
            // В Hyprland нет настоящего минимизирования: окно уезжает в
            // служебный воркспейс «hidden» — из интерфейса это и есть свернуть.
            let ok = run_ok(
                "hyprctl",
                &["dispatch",
                  &format!("hl.dsp.window.move({{ workspace = \"special:hidden\", window = \"{sel}\" }})")],
            );
            for _ in 0..6 {
                match hypr_window_state(&addr) {
                    Some((true, ws, _)) if ws.starts_with("special") => {
                        return format!("Свернул окно {title}");
                    }
                    _ => std::thread::sleep(Duration::from_millis(400)),
                }
            }
            let _ = ok;
            format!("Окно {title} не свернулось")
        }
        "maximize" | "fullscreen" => {
            let mode = if action == "fullscreen" { "fullscreen" } else { "maximized" };
            let before = hypr_window_state(&addr)
                .map(|(_, _, fs)| fs)
                .unwrap_or(0);
            run_ok(
                "hyprctl",
                &["dispatch",
                  &format!("hl.dsp.window.fullscreen({{ window = \"{sel}\", mode = \"{mode}\", action = \"toggle\" }})")],
            );
            let after = hypr_window_state(&addr).map(|(_, _, fs)| fs).unwrap_or(0);
            if before == 0 && after != 0 {
                format!(
                    "{} окно {title}",
                    if mode == "fullscreen" { "Развернул на весь экран" } else { "Развёрнул до максимума" }
                )
            } else if before != 0 && after == 0 {
                format!("Вернул окно {title} к обычному виду")
            } else {
                format!("Окно {title}: состояние экрана не изменилось")
            }
        }
        "restore" => {
            let (exists, ws, fs) = hypr_window_state(&addr)
                .unwrap_or((false, String::new(), 0));
            if !exists {
                return format!("Окно {title} уже закрыто");
            }
            if ws.starts_with("special") {
                // Вернуть из свёрнутого: на текущий воркспейс.
                let current = run("hyprctl", &["-j", "activeworkspace"])
                    .ok()
                    .and_then(|out| serde_json::from_str::<serde_json::Value>(&out).ok())
                    .and_then(|v| v.get("name")?.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "1".to_string());
                run_ok(
                    "hyprctl",
                    &["dispatch",
                      &format!("hl.dsp.window.move({{ workspace = \"{current}\", window = \"{sel}\" }})")],
                );
                return match hypr_window_state(&addr) {
                    Some((true, ws2, _)) if !ws2.starts_with("special") => {
                        format!("Вернул окно {title}")
                    }
                    _ => format!("Окно {title} не вернулось"),
                };
            }
            if fs != 0 {
                // unset у Hyprland 0.56 не снимает разворот — снимает toggle,
                // в том же режиме, в котором окно было развёрнуто.
                let mode = if fs == 2 { "fullscreen" } else { "maximized" };
                run_ok(
                    "hyprctl",
                    &["dispatch",
                      &format!("hl.dsp.window.fullscreen({{ window = \"{sel}\", mode = \"{mode}\", action = \"toggle\" }})")],
                );
                // Hyprland применяет диспетчер асинхронно: читаем с ретраем.
                for _ in 0..6 {
                    match hypr_window_state(&addr) {
                        Some((true, _, fs2)) if fs2 == 0 => {
                            return format!("Вернул окно {title} к обычному виду")
                        }
                        _ => std::thread::sleep(Duration::from_millis(300)),
                    }
                }
                return format!("Окно {title} не вернулось к обычному виду");
            }
            format!("Окно {title} уже в обычном виде")
        }
        "focus" => {
            let ok = run_ok(
                "hyprctl",
                &["dispatch", &format!("hl.dsp.focus({{ window = \"{sel}\" }})")],
            );
            let _ = run_ok(
                "hyprctl",
                &["dispatch",
                  &format!("hl.dsp.window.alter_zorder({{ mode = \"top\", window = \"{sel}\" }})")],
            );
            let active = run("hyprctl", &["-j", "activewindow"])
                .ok()
                .and_then(|out| parse_hypr_single(&out))
                .map(|w| w.id);
            if active.as_deref() == Some(addr.as_str()) {
                format!("Переключился на {title}")
            } else {
                let _ = ok;
                format!("Окно {title} не вышло на передний план")
            }
        }
        "movetodesktop" => {
            // Номер стола приходит в строке действия: «movetodesktop 3».
            match action.split_whitespace().nth(1).and_then(desktop_number) {
                Some(target) => {
                    run_ok(
                        "hyprctl",
                        &["dispatch",
                          &format!("hl.dsp.window.move({{ workspace = \"{target}\", window = \"{sel}\" }})")],
                    );
                    // Верификация: окно обязано оказаться на столе N.
                    for _ in 0..6 {
                        if let Some((true, ws, _)) = hypr_window_state(&addr)
                            && ws == target.to_string()
                        {
                            return format!("Перенёс окно {title} на стол {target}");
                        }
                        std::thread::sleep(Duration::from_millis(300));
                    }
                    format!("Окно {title} не доехало до стола {target}")
                }
                None => "Не понял, на какой стол перенести — назови номер".to_string(),
            }
        }
        "half_left" | "half_right" => hypr_window_half(&addr, &title, action == "half_left"),
        other => format!("Не могу выполнить действие с окном: {other}"),
    }
}

/// Номер стола из аргумента: «3», «стол 3», «3:рабочий» — берём первый
/// чисто цифровой токен.
fn desktop_number(arg: &str) -> Option<u32> {
    arg.split_whitespace()
        .find(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
        .and_then(|t| t.parse().ok())
}

/// Разбор строки действия «глагол номер»: номер приезжает на платформу
/// внутри действия («desktop 3»).
fn split_action_number(action: &str) -> (&str, Option<&str>) {
    match action.trim().split_once(' ') {
        Some((verb, rest)) => (verb, Some(rest.trim())),
        None => (action.trim(), None),
    }
}

/// «Открой стол N»: hl.dsp.focus сам создаёт стол, если его ещё нет.
fn hypr_desktop_switch(number: Option<&str>) -> String {
    let Some(target) = number.and_then(desktop_number) else {
        return "Не понял, на какой стол перейти — назови номер".to_string();
    };
    run_ok(
        "hyprctl",
        &["dispatch", &format!("hl.dsp.focus({{ workspace = {target} }})")],
    );
    for _ in 0..5 {
        let active = run("hyprctl", &["-j", "activeworkspace"])
            .ok()
            .and_then(|out| serde_json::from_str::<serde_json::Value>(&out).ok())
            .and_then(|v| v.get("id")?.as_i64());
        if active == Some(target as i64) {
            return format!("Переключился на стол {target}");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    format!("Не переключился на стол {target}")
}

/// «Сверни все окна»: каждое видимое окно уезжает в special:hidden —
/// тот же приём, что и одиночное «сверни», но по всем окнам разом.
fn hypr_minimize_all() -> String {
    let v = match run("hyprctl", &["-j", "clients"])
        .ok()
        .and_then(|out| serde_json::from_str::<serde_json::Value>(&out).ok())
        .and_then(|v| v.as_array().map(|a| a.clone()))
    {
        Some(v) => v,
        None => return "Не смог прочитать список окон".to_string(),
    };
    let visible: Vec<String> = v
        .iter()
        .filter_map(|w| {
            let id = w.get("workspace")?.get("id")?.as_i64()?;
            if id < 1 {
                return None;
            }
            w.get("address")?.as_str().map(|a| a.to_string())
        })
        .collect();
    if visible.is_empty() {
        return "Все окна уже свернуты — рабочий стол чист".to_string();
    }
    let count = visible.len();
    for addr in &visible {
        run_ok(
            "hyprctl",
            &["dispatch",
              &format!("hl.dsp.window.move({{ workspace = \"special:hidden\", window = \"address:{addr}\" }})")],
        );
    }
    for _ in 0..5 {
        if hypr_visible_windows() == 0 {
            return format!("Свернул все окна ({count}) — рабочий стол чист");
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    "Не все окна свернулись".to_string()
}

/// Сколько окон сейчас на обычных столах (специальные — отрицательные id).
fn hypr_visible_windows() -> usize {
    run("hyprctl", &["-j", "clients"])
        .ok()
        .and_then(|out| serde_json::from_str::<serde_json::Value>(&out).ok())
        .and_then(|v| v.as_array().map(|a| a.clone()))
        .map(|list| {
            list.iter()
                .filter(|w| {
                    w.get("workspace")
                        .and_then(|ws| ws.get("id"))
                        .and_then(|id| id.as_i64())
                        .unwrap_or(0)
                        >= 1
                })
                .count()
        })
        .unwrap_or(0)
}

/// Геометрия окна из IPC: (позиция, размер) в пикселях.
fn hypr_window_geometry(addr: &str) -> Option<((i64, i64), (i64, i64))> {
    let out = run("hyprctl", &["-j", "clients"]).ok()?;
    let v: serde_json::Value = serde_json::from_str(&out).ok()?;
    for w in v.as_array()? {
        if w.get("address")?.as_str()? == addr {
            let nums = |key: &str| -> Option<Vec<i64>> {
                w.get(key)
                    .and_then(|a| a.as_array())
                    .map(|a| a.iter().filter_map(|n| n.as_i64()).collect())
            };
            let at = nums("at")?;
            let size = nums("size")?;
            if at.len() == 2 && size.len() == 2 {
                return Some(((at[0], at[1]), (size[0], size[1])));
            }
            return None;
        }
    }
    None
}

/// Плавает ли окно (float — переключатель, поэтому спрашиваем до вызова).
fn hypr_window_is_floating(addr: &str) -> bool {
    let out = match run("hyprctl", &["-j", "clients"]) {
        Ok(o) => o,
        Err(_) => return false,
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&out) else {
        return false;
    };
    let empty = Vec::new();
    for w in v.as_array().unwrap_or(&empty) {
        if w.get("address").and_then(|a| a.as_str()) == Some(addr) {
            return w.get("floating").and_then(|f| f.as_bool()).unwrap_or(false);
        }
    }
    false
}

/// Половина экрана: окно становится плавающим с точной геометрией.
/// Порядок проверен живьём (Hyprland 0.56, Lua-диспетчеры): resize задаёт
/// размер и центрирует окно, поэтому после него — move в угол; float —
/// toggle, плаваем только тайловое окно.
fn hypr_window_half(addr: &str, title: &str, left: bool) -> String {
    let sel = format!("address:{addr}");
    let (mon_w, mon_h, top) = hypr_monitor_usable();
    let w = (mon_w / 2).max(200);
    let h = mon_h.saturating_sub(top).max(200);
    let x = if left { 0 } else { mon_w / 2 };
    if !hypr_window_is_floating(addr) {
        run_ok(
            "hyprctl",
            &["dispatch", &format!("hl.dsp.window.float({{ window = \"{sel}\" }})")],
        );
    }
    run_ok(
        "hyprctl",
        &["dispatch",
          &format!("hl.dsp.window.resize({{ x = {w}, y = {h}, window = \"{sel}\" }})")],
    );
    run_ok(
        "hyprctl",
        &["dispatch",
          &format!("hl.dsp.window.move({{ x = {x}, y = {top}, window = \"{sel}\" }})")],
    );
    for _ in 0..5 {
        if let Some(((ax, ay), (sw, sh))) = hypr_window_geometry(addr) {
            let ok_pos = (ax - x as i64).abs() <= 4 && (ay - top as i64).abs() <= 4;
            let ok_size = (sw - w as i64).abs() <= 4 && (sh - h as i64).abs() <= 4;
            if ok_pos && ok_size {
                let side = if left { "левую" } else { "правую" };
                return format!("Разместил окно {title} на {side} половину экрана");
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    format!("Окно {title} не встало на половину экрана")
}

/// Размеры сфокусированного монитора и верхний резерв (панель).
/// Порядок в reserved — [left, top, right, bottom]: проверено живьём,
/// верхняя панель Omarchy резервирует 26 px во втором элементе.
fn hypr_monitor_usable() -> (u32, u32, u32) {
    let fallback = (1920, 1080, 0);
    let Some(out) = run("hyprctl", &["-j", "monitors"]).ok() else {
        return fallback;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&out) else {
        return fallback;
    };
    let empty = Vec::new();
    for m in v.as_array().unwrap_or(&empty) {
        let focused = m.get("focused").and_then(|f| f.as_bool()).unwrap_or(false);
        if !focused {
            continue;
        }
        let width = m.get("width").and_then(|n| n.as_u64()).unwrap_or(1920) as u32;
        let height = m.get("height").and_then(|n| n.as_u64()).unwrap_or(1080) as u32;
        let top = m
            .get("reserved")
            .and_then(|r| r.as_array())
            .and_then(|r| r.get(1))
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32;
        // Страховка от кривых значений: резервы не съедают полэкрана.
        let top = top.min(height / 3);
        return (width, height, top);
    }
    fallback
}

/// Состояние окна из IPC: (существует, имя воркспейса, fullscreen 0/1/2).
fn hypr_window_state(addr: &str) -> Option<(bool, String, i64)> {
    let out = run("hyprctl", &["-j", "clients"]).ok()?;
    let v: serde_json::Value = serde_json::from_str(&out).ok()?;
    for w in v.as_array()? {
        if w.get("address")?.as_str()? == addr {
            return Some((
                true,
                w.get("workspace")
                    .and_then(|ws| ws.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
                w.get("fullscreen").and_then(|f| f.as_i64()).unwrap_or(0),
            ));
        }
    }
    Some((false, String::new(), 0))
}

fn hypr_window_exists(addr: &str) -> bool {
    matches!(hypr_window_state(addr), Some((true, _, _)))
}

/// Одно окно из `hyprctl -j activewindow`.
fn parse_hypr_single(out: &str) -> Option<LinuxWindow> {
    let v: serde_json::Value = serde_json::from_str(out).ok()?;
    let class = v.get("class")?.as_str()?.to_string();
    let title = v.get("title")?.as_str()?.to_string();
    let id = v.get("address").and_then(|a| a.as_str()).unwrap_or("").to_string();
    Some(LinuxWindow { id, class, title })
}

/// Разбор `hyprctl clients -j` (JSON): класс и заголовок видимых окон.
fn parse_hypr_clients(out: &str) -> Vec<LinuxWindow> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(out) else {
        return Vec::new();
    };
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter(|w| {
                    // Скрытые и служебные окна наблюдению не нужны.
                    w.get("hidden").and_then(|h| h.as_bool()) != Some(true)
                        && w.get("mapped").and_then(|m| m.as_bool()) != Some(false)
                })
                .filter_map(|w| {
                    let class = w.get("class")?.as_str()?.to_string();
                    let title = w.get("title")?.as_str()?.to_string();
                    let id = w.get("address").and_then(|a| a.as_str()).unwrap_or("").to_string();
                    Some(LinuxWindow { id, class, title })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn wayland_session_env() -> bool {
    in_wayland_session()
}

fn hyprctl_windows() -> Option<Vec<LinuxWindow>> {
    let out = run("hyprctl", &["clients", "-j"]).ok()?;
    // Валидный JSON — массив или объект; мусор (ошибка IPC) — нет.
    (out.starts_with('[') || out.starts_with('{')).then(|| parse_hypr_clients(&out))
}

fn hyprctl_active() -> Option<LinuxWindow> {
    let out = run("hyprctl", &["activewindow", "-j"]).ok()?;
    if !out.starts_with('{') || out.trim().is_empty() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&out).ok()?;
    let class = v.get("class")?.as_str()?.to_string();
    let title = v.get("title")?.as_str()?.to_string();
    let id = v.get("address").and_then(|a| a.as_str()).unwrap_or("").to_string();
    Some(LinuxWindow { id, class, title })
}

fn wmctrl_windows() -> Option<Vec<LinuxWindow>> {
    // Wayland: сначала Hyprland IPC, wmctrl там не видит ничего.
    if wayland_session_env() {
        if let Some(windows) = hyprctl_windows() {
            return Some(windows);
        }
    }
    run("wmctrl", &["-lx"])
        .ok()
        .map(|out| parse_wmctrl_lx(&out))
}

fn xdotool_search(flag: &str, pattern: &str) -> Option<Vec<String>> {
    run("xdotool", &["search", "--onlyvisible", flag, pattern])
        .ok()
        .map(|out| {
            out.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
}

/// Активное окно: id и заголовок. На Hyprland — IPC; xdotool точнее на X11;
/// xprop — запасной путь.
pub fn active_window() -> Option<(String, String)> {
    if wayland_session_env() {
        if let Some(w) = hyprctl_active() {
            return Some((w.class, w.title));
        }
    }
    if command_exists("xdotool") {
        // Без `?`: неудача xdotool не должна закрывать путь к xprop.
        if let Ok(id) = run("xdotool", &["getactivewindow"]) {
            if !id.is_empty() {
                let title = run("xdotool", &["getwindowname", &id]).unwrap_or_default();
                return Some((id, title));
            }
        }
    }
    if command_exists("xprop") {
        let root = run("xprop", &["-root", "_NET_ACTIVE_WINDOW"]).ok()?;
        let id = root.split_whitespace().last()?.to_string();
        if id == "0x0" {
            return None;
        }
        let name = run("xprop", &["-id", &id, "_NET_WM_NAME"]).unwrap_or_default();
        let title = name
            .split_once("= ")
            .map(|(_, t)| t.trim_matches('"').to_string())
            .unwrap_or_default();
        return Some((id, title));
    }
    None
}

fn wayland_limitation() -> &'static str {
    if in_wayland_session() {
        " (в сессии Wayland видны только X11-окна; для полного управления окнами нужна X11-сессия)"
    } else {
        ""
    }
}

fn missing_tools_message() -> String {
    if in_wayland_session() {
        "Не могу управлять окнами в сессии Wayland: нужен wmctrl (X11) или xdotool — \
         sudo apt install wmctrl xdotool; для полного управления войди в X11-сессию"
            .to_string()
    } else {
        "Не могу управлять окнами: нужен wmctrl или xdotool — sudo apt install wmctrl xdotool"
            .to_string()
    }
}

/// Поиск по окнам через wmctrl: точное совпадение класса, потом заголовка,
/// потом словарь алиасов, потом нечётко по смыслу.
fn find_window_wmctrl(want: &str) -> Option<LinuxWindow> {
    let windows = wmctrl_windows()?;
    let find = |pred: &dyn Fn(&LinuxWindow) -> bool| windows.iter().find(|w| pred(w)).cloned();
    let want_lower = want.to_lowercase();
    find(&|w| {
        class_words(&w.class)
            .iter()
            .any(|c| c.to_lowercase() == want_lower)
    })
    .or_else(|| find(&|w| w.title.to_lowercase().contains(&want_lower)))
    .or_else(|| find(&|w| window_matches(w, want)))
    .or_else(|| {
        // Нечёткий подбор по заголовкам — как fuzzy_app_match на macOS.
        let titles: Vec<String> = windows.iter().map(|w| w.title.clone()).collect();
        fuzzy_app_match(want, &titles)
            .and_then(|hit| windows.iter().find(|w| w.title == hit).cloned())
    })
}

/// Запасной поиск через xdotool (когда wmctrl нет): окна ищутся по заголовку,
/// плюс проба по словарю алиасов — «хром» тоже найдётся. Класс окна xdotool
/// не отдаёт, поэтому сверяемся с заголовком.
fn find_window_xdotool(want: &str) -> Option<LinuxWindow> {
    if !command_exists("xdotool") {
        return None;
    }
    let mut patterns = vec![want.to_string()];
    if let Some(target) = alias(want) {
        patterns.push(target.to_string());
    }
    for pattern in patterns {
        if let Some(ids) = xdotool_search("--name", &pattern) {
            if let Some(id) = ids.into_iter().next() {
                let title = run("xdotool", &["getwindowname", &id]).unwrap_or_default();
                return Some(LinuxWindow {
                    id,
                    class: String::new(),
                    title,
                });
            }
        }
    }
    None
}

/// Найти окно приложения: основной путь — wmctrl, запасной — xdotool.
fn find_window(want: &str) -> Option<LinuxWindow> {
    // Список окон уже ходит через Hyprland IPC на Wayland — затвор на
    // наличие бинарника wmctrl тут не нужен и ломал поиск целиком.
    if let Some(window) = find_window_wmctrl(want) {
        return Some(window);
    }
    find_window_xdotool(want)
}

/// Запустить действие wmctrl над окном, вернуть человеческий итог.
fn wmctrl_action(id: &str, states_add: &[&str], states_remove: &[&str]) -> bool {
    let mut ok = true;
    if !states_remove.is_empty() {
        let spec = format!("remove,{}", states_remove.join(","));
        ok &= run_ok("wmctrl", &["-i", "-r", id, "-b", &spec]);
    }
    if !states_add.is_empty() {
        let spec = format!("add,{}", states_add.join(","));
        ok &= run_ok("wmctrl", &["-i", "-r", id, "-b", &spec]);
    }
    ok
}

/// Проверка состояния окна через _NET_WM_STATE (если есть xprop): «свернулось
/// ли», «развернулось ли» — по факту, а не по коду возврата.
fn window_state_has(id: &str, state: &str) -> Option<bool> {
    if !command_exists("xprop") {
        return None;
    }
    let out = run("xprop", &["-id", id, "_NET_WM_STATE"]).ok()?;
    Some(out.to_lowercase().contains(&state.to_lowercase()))
}

fn do_window_action(id: &str, name: &str, action: &str) -> String {
    let has_wmctrl = command_exists("wmctrl");
    let has_xdotool = command_exists("xdotool");

    match action {
        "minimize" => {
            // Пробуем оба инструмента: установленный xdotool может не сработать
            // (например, окно XWayland), и wmctrl остаётся запасным.
            let ok = (has_xdotool && run_ok("xdotool", &["windowminimize", id]))
                || (has_wmctrl && wmctrl_action(id, &["hidden"], &[]));
            if ok {
                format!("свернул окно {name}")
            } else {
                format!("Не удалось свернуть окно {name}")
            }
        }
        "maximize" => {
            if !has_wmctrl {
                return "Не могу развернуть окно без wmctrl — sudo apt install wmctrl".to_string();
            }
            let _ = run("wmctrl", &["-i", "-a", id]);
            let ok = wmctrl_action(id, &["maximized_vert", "maximized_horz"], &["hidden"]);
            match (ok, window_state_has(id, "_NET_WM_STATE_MAXIMIZED")) {
                (true, Some(true)) => format!("развернул окно {name}"),
                (true, Some(false)) => format!(
                    "Не удалось развернуть окно {name}: команда ушла, но состояние не изменилось"
                ),
                (true, None) => format!("развернул окно {name}"),
                (false, _) => format!("Не удалось развернуть окно {name}"),
            }
        }
        "fullscreen" => {
            if !has_wmctrl {
                return "Не могу включить полный экран без wmctrl — sudo apt install wmctrl"
                    .to_string();
            }
            let _ = run("wmctrl", &["-i", "-a", id]);
            let ok = wmctrl_action(id, &["fullscreen"], &[]);
            match (ok, window_state_has(id, "_NET_WM_STATE_FULLSCREEN")) {
                (true, Some(true)) => format!("включил полный экран {name}"),
                (true, Some(false)) => format!(
                    "Не удалось включить полный экран {name}: команда ушла, но состояние не изменилось"
                ),
                (true, None) => format!("включил полный экран {name}"),
                (false, _) => format!("Не удалось включить полный экран {name}"),
            }
        }
        "restore" => {
            if !has_wmctrl {
                return "Не могу вернуть окно без wmctrl — sudo apt install wmctrl".to_string();
            }
            let ok = wmctrl_action(
                id,
                &[],
                &["fullscreen", "maximized_vert", "maximized_horz", "hidden"],
            );
            let _ = run("wmctrl", &["-i", "-a", id]);
            if ok {
                format!("вернул окно {name} в обычный размер")
            } else {
                format!("Не удалось вернуть окно {name} в обычный размер")
            }
        }
        "close" => {
            let ok = (has_xdotool && run_ok("xdotool", &["windowclose", id]))
                || (has_wmctrl && run_ok("wmctrl", &["-i", "-c", id]));
            if ok {
                format!("закрыл окно {name}")
            } else {
                format!("Не удалось закрыть окно {name}")
            }
        }
        "focus" => {
            let ok = if has_wmctrl {
                run_ok("wmctrl", &["-i", "-a", id])
            } else {
                run_ok("xdotool", &["windowactivate", id])
            };
            if ok {
                format!("Переключился на {name}")
            } else {
                format!("Не удалось переключиться на {name}")
            }
        }
        other => format!("Не могу выполнить действие с окном: {other}"),
    }
}

fn window_list() -> String {
    let Some(windows) = wmctrl_windows() else {
        return missing_tools_message();
    };
    let shown: Vec<String> = windows
        .iter()
        .filter(|w| !w.title.trim().is_empty())
        .take(12)
        .map(|w| {
            let class_last = class_words(&w.class)
                .into_iter()
                .next_back()
                .unwrap_or_else(|| w.class.clone());
            format!("{class_last}: {}", w.title)
        })
        .collect();
    if shown.is_empty() {
        "Открытых окон не нашёл".to_string()
    } else {
        shown.join("; ")
    }
}

/// Окно подходит под просьбу: класс, заголовок или словарь алиасов.
fn window_app_matches(window: &LinuxWindow, want: &str) -> bool {
    let want_lower = want.to_lowercase();
    class_words(&window.class)
        .iter()
        .any(|c| c.to_lowercase() == want_lower)
        || window.title.to_lowercase().contains(&want_lower)
        || window_matches(window, want)
}

/// Все окна, подходящие под просьбу. На X11 — wmctrl, запасной путь —
/// xdotool. Возвращает пустой список, если окно не найдено.
fn windows_matching(want: &str) -> Vec<LinuxWindow> {
    if let Some(windows) = wmctrl_windows() {
        let hits: Vec<LinuxWindow> = windows
            .iter()
            .filter(|w| window_app_matches(w, want))
            .cloned()
            .collect();
        if !hits.is_empty() {
            return hits;
        }
        // Нечёткий подбор по заголовкам — как fuzzy_app_match на macOS.
        let titles: Vec<String> = windows.iter().map(|w| w.title.clone()).collect();
        if let Some(hit) = fuzzy_app_match(want, &titles) {
            return windows.into_iter().filter(|w| w.title == hit).collect();
        }
        // wmctrl не нашёл — возможно, окно видно только xdotool.
    }
    if !command_exists("xdotool") {
        return Vec::new();
    }
    let mut patterns = vec![want.to_string()];
    if let Some(target) = alias(want) {
        patterns.push(target.to_string());
    }
    let mut found: Vec<LinuxWindow> = Vec::new();
    for pattern in patterns {
        if let Some(ids) = xdotool_search("--name", &pattern) {
            for id in ids {
                if found.iter().any(|w| w.id == id) {
                    continue;
                }
                let title = run("xdotool", &["getwindowname", &id]).unwrap_or_default();
                found.push(LinuxWindow {
                    id,
                    class: String::new(),
                    title,
                });
            }
        }
    }
    found
}

/// Окно свёрнуто. На Hyprland это служебный воркспейс `special:hidden`;
/// на X11 — состояние `_NET_WM_STATE_HIDDEN` (если xprop отвечает).
fn window_is_minimized(window: &LinuxWindow) -> bool {
    if wayland_session_env() && command_exists("hyprctl") {
        return hypr_window_state(&window.id)
            .map(|(_, ws, _)| ws.starts_with("special"))
            .unwrap_or(false);
    }
    window_state_has(&window.id, "HIDDEN") == Some(true)
}

/// Имя приложения окна: из WM_CLASS берём человеческую часть
/// («google-chrome.Google-chrome» → «Google-chrome»).
fn window_app_name(window: &LinuxWindow) -> String {
    let class = window.class.trim();
    if class.is_empty() {
        return "окно без названия".to_string();
    }
    class.rsplit('.').next().unwrap_or(class).to_string()
}

/// Подпись окна для выбора: заголовок, а если он пуст — класс приложения.
fn window_label(window: &LinuxWindow) -> String {
    let title = window.title.trim();
    if !title.is_empty() {
        return title.to_string();
    }
    window_app_name(window)
}

/// Действие над окном тем путём, который соответствует сессии.
fn linux_window_action(window: &LinuxWindow, action: &str) -> String {
    if wayland_session_env() && command_exists("hyprctl") {
        hyprland_window_action(window, action)
    } else {
        do_window_action(&window.id, &window_label(window), action)
    }
}

/// Оконная команда без молчаливого выбора первого окна. Если под просьбу
/// подходит несколько окон, возвращается список — интерфейс предложит выбор.
pub fn window_control_checked(
    action: &str,
    app: Option<&str>,
) -> crate::platform::WindowControlOutcome {
    use crate::platform::{WindowControlOutcome, WindowTarget};

    if action == "list" {
        return WindowControlOutcome::Done(window_control(action, app));
    }

    // Действия без выбора окна: переключение столов и «свернуть всё».
    // Разбор «глагол номер» — номер приезжает внутри строки действия.
    let (verb, desktop) = split_action_number(action);
    if matches!(verb, "desktop" | "minimizeall") {
        if in_wayland_session() && command_exists("hyprctl") {
            let result = match verb {
                "desktop" => hypr_desktop_switch(desktop),
                _ => hypr_minimize_all(),
            };
            return WindowControlOutcome::Done(result);
        }
        return WindowControlOutcome::Done(
            "Столы и «свернуть всё» пока умею только на Hyprland (Wayland)".to_string(),
        );
    }

    let is_self = app.is_none() || app.is_some_and(crate::platform::is_self_window_target);
    let is_active = app.is_some_and(|wanted| {
        ["активное", "активную", "активный", "это", "текущее", "active"]
            .contains(&wanted.trim().to_lowercase().as_str())
    });

    let window = if is_self {
        wmctrl_windows().and_then(|windows| {
            windows.into_iter().find(|w| {
                class_words(&w.class)
                    .iter()
                    .any(|c| c.to_lowercase().contains("jarvis"))
            })
        })
    } else if is_active {
        if wayland_session_env() && command_exists("hyprctl") {
            hyprctl_active()
        } else {
            active_window().map(|(id, title)| LinuxWindow {
                id,
                class: String::new(),
                title,
            })
        }
    } else {
        let wanted = app.unwrap_or_default();
        let mut hits = windows_matching(wanted);
        if hits.is_empty() {
            return WindowControlOutcome::Done(format!(
                "Приложение {wanted} не найдено среди открытых{}",
                wayland_limitation()
            ));
        }
        // «Верни окно»: если какие-то окна свёрнуты, выбор — только среди
        // них; остальные и так видны и возвращать их нечего.
        if action == "restore" {
            let minimized: Vec<LinuxWindow> = hits
                .iter()
                .filter(|window| window_is_minimized(window))
                .cloned()
                .collect();
            if !minimized.is_empty() {
                hits = minimized;
            }
        }
        if hits.len() > 1 {
            let targets = hits
                .into_iter()
                .map(|window| WindowTarget {
                    app: window_app_name(&window),
                    minimized: window_is_minimized(&window),
                    title: window_label(&window),
                })
                .collect();
            return WindowControlOutcome::Ambiguous(targets);
        }
        hits.into_iter().next()
    };

    let Some(window) = window else {
        let name = app.unwrap_or("");
        if is_self {
            return WindowControlOutcome::Done(
                "Не нашёл своё окно — я в фоне. Скажи, чьё окно трогать".to_string(),
            );
        }
        if is_active {
            return WindowControlOutcome::Done(
                "Не нашёл активное приложение с окном".to_string(),
            );
        }
        return WindowControlOutcome::Done(format!(
            "Приложение {name} не найдено среди открытых{}",
            wayland_limitation()
        ));
    };
    WindowControlOutcome::Done(linux_window_action(&window, action))
}

/// Действие над конкретным окном, выбранным пользователем. Окно ищется
/// заново по заголовку и классу: список окон мог измениться, пока человек
/// читал варианты.
pub fn window_control_target(action: &str, app: &str, title: &str) -> String {
    let windows = wmctrl_windows().unwrap_or_default();
    let by_title = |window: &LinuxWindow| {
        window.title == title
            || (!title.trim().is_empty()
                && (window.title.contains(title) || title.contains(&window.title)))
    };
    let same_app = |window: &LinuxWindow| {
        app.trim().is_empty()
            || class_words(&window.class)
                .iter()
                .any(|c| c.eq_ignore_ascii_case(app))
            || window_matches(window, app)
    };
    let window = windows
        .iter()
        .find(|window| by_title(window) && same_app(window))
        .or_else(|| windows.iter().find(|window| by_title(window)))
        .cloned()
        .or_else(|| {
            // Запасной путь X11: wmctrl нет, но выбор строился через xdotool.
            find_window_xdotool(app).filter(|window| by_title(window))
        });
    match window {
        Some(window) => linux_window_action(&window, action),
        None => format!("Окно «{title}» уже закрыто или переименовано"),
    }
}

/// Действие над окном. `app = None` — окно самого Джарвиса; «активное» —
/// активное приложение (как на macOS: сворачивать чужое окно без просьбы —
/// неприятный сюрприз).
pub fn window_control(action: &str, app: Option<&str>) -> String {
    // Чтение списка окон на Wayland идёт через Hyprland IPC — wmctrl и
    // xdotool там не смотрят, но наблюдению это не мешает.
    if action == "list" {
        if let Some(windows) = hyprctl_windows() {
            let shown: Vec<String> = windows
                .iter()
                .filter(|w| !w.title.trim().is_empty())
                .take(12)
                .map(|w| {
                    let class_last = class_words(&w.class)
                        .into_iter()
                        .next_back()
                        .unwrap_or_else(|| w.class.clone());
                    format!("{class_last}: {}", w.title)
                })
                .collect();
            return if shown.is_empty() {
                "Открытых окон не нашёл".to_string()
            } else {
                shown.join("; ")
            };
        }
    }
    // Hyprland (Wayland): нативное IPC-управление. wmctrl/xdotool там слепы,
    // а диспетчеры hl.dsp.* закрывают весь набор: close/minimize/maximize/
    // fullscreen/restore/focus — с верификацией факта после действия.
    if in_wayland_session() && command_exists("hyprctl") {
        if let Some(result) = hyprland_window_control(action, app) {
            return result;
        }
    }

    // Изменения окон — инструменты X11: без DISPLAY им нечего смотреть.
    // Это не поломка, а отсутствие графической сессии — так и говорим.
    if std::env::var_os("DISPLAY").is_none() {
        return if in_wayland_session() {
            "Не могу управлять окнами в чистом Wayland: wmctrl и xdotool видят только X11-окна — войди в X11-сессию".to_string()
        } else {
            "Не могу управлять окнами: нет графической сессии (DISPLAY не задан)".to_string()
        };
    }
    if action == "list" {
        return window_list();
    }
    if !command_exists("wmctrl") && !command_exists("xdotool") {
        return missing_tools_message();
    }

    match app {
        Some(wanted) if crate::platform::is_self_window_target(wanted) => {
            return window_control(action, None);
        }
        Some(wanted)
            if [
                "активное",
                "активную",
                "активный",
                "это",
                "текущее",
                "active",
            ]
            .contains(&wanted.trim().to_lowercase().as_str()) =>
        {
            match active_window() {
                Some((id, title)) => {
                    let name = if title.trim().is_empty() {
                        "активным".to_string()
                    } else {
                        title
                    };
                    do_window_action(&id, &name, action)
                }
                None => "Не нашёл активное приложение с окном".to_string(),
            }
        }
        Some(wanted) => match find_window(wanted) {
            Some(window) => {
                let name = if window.title.trim().is_empty() {
                    wanted.to_string()
                } else {
                    window.title.clone()
                };
                do_window_action(&window.id, &name, action)
            }
            None => format!(
                "Приложение {wanted} не найдено среди открытых{}",
                wayland_limitation()
            ),
        },
        None => {
            // Своё окно: ищем среди окон окно с «jarvis» в классе.
            let own = std::env::current_exe()
                .ok()
                .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                .filter(|s| !s.is_empty())
                .and_then(|me| {
                    find_window(&me).or_else(|| {
                        wmctrl_windows()?.into_iter().find(|w| {
                            class_words(&w.class)
                                .iter()
                                .any(|c| c.to_lowercase().contains("jarvis"))
                        })
                    })
                });
            match own {
                Some(window) => do_window_action(&window.id, "Джарвис", action),
                None => "Не нашёл своё окно — я в фоне. Скажи, чьё окно свернуть".to_string(),
            }
        }
    }
}

// =========================================================================
// Музыка: playerctl (MPRIS)
// =========================================================================

fn playerctl_status() -> Option<String> {
    let out = run("playerctl", &["status"]).ok()?;
    // У нескольких плееров — по строке на каждого: считаем играющим, если
    // хоть один играет.
    if out.lines().any(|l| l.trim() == "Playing") {
        Some("Playing".to_string())
    } else {
        out.lines().next().map(|l| l.trim().to_string())
    }
}

/// Управление текущим плеером через MPRIS: работает и с браузерами
/// (YouTube, Яндекс Музыка во вкладке), и с нативными плеерами.
pub fn music_control(action: &str) -> Result<String, String> {
    if !command_exists("playerctl") {
        return Err(
            "Управление музыкой требует playerctl — sudo apt install playerctl".to_string(),
        );
    }
    let verb = match action {
        "play" => "play",
        "pause" | "stop" => "pause",
        "next" => "next",
        "previous" => "previous",
        other => return Err(format!("Не понимаю действие плеера: {other}")),
    };

    let before = match playerctl_status() {
        Some(status) => status,
        None => {
            return Err("Сейчас ничего не играет — управлять нечем".to_string());
        }
    };
    if matches!(action, "pause" | "stop") && before == "Paused" {
        return Err("Музыка уже на паузе".to_string());
    }
    if action == "play" && before == "Playing" {
        return Err("Музыка уже играет".to_string());
    }

    if let Err(e) = run("playerctl", &[verb]) {
        return Err(format!("Не удалось управлять плеером: {e}"));
    }

    // Состояние после команды: успешный код playerctl ещё не значит, что
    // плеер действительно остановился.
    let after = playerctl_status().unwrap_or_else(|| "неизвестно".to_string());
    let expected_change = match action {
        "play" => before != "Playing",
        "pause" | "stop" => before == "Playing",
        _ => true,
    };
    if !expected_change || (after == before && !matches!(action, "next" | "previous")) {
        return Err(format!(
            "Команда ушла в плеер, но состояние не изменилось ({after})"
        ));
    }

    let said = match action {
        "play" => "Включаю музыку",
        "pause" => "Поставил на паузу",
        "stop" => "Остановил музыку",
        "next" => "Следующий трек",
        _ => "Предыдущий трек",
    };
    Ok(said.to_string())
}
/// Статус плеера («Playing»/«Paused») без попытки управления — для наблюдения.
pub fn music_status() -> Option<String> {
    playerctl_status()
}

// =========================================================================
// Системные действия
// =========================================================================

pub fn set_volume(percent: u8) -> bool {
    if command_exists("pactl") {
        // Unmute обязателен: иначе «громкость 50» не даёт звука, если канал
        // был заглушён.
        return run_ok(
            "pactl",
            &["set-sink-volume", "@DEFAULT_SINK@", &format!("{percent}%")],
        ) && run_ok("pactl", &["set-sink-mute", "@DEFAULT_SINK@", "0"]);
    }
    if command_exists("amixer") {
        // Имя канала различается между звуковыми картами: Master, потом PCM.
        return run_ok(
            "amixer",
            &["-q", "sset", "Master", &format!("{percent}%"), "unmute"],
        ) || run_ok(
            "amixer",
            &["-q", "sset", "PCM", &format!("{percent}%"), "unmute"],
        );
    }
    false
}

/// Имя подключённого выхода в `xrandr --query` (строка «…connected …»).
fn parse_xrandr_connected(out: &str) -> Option<String> {
    out.lines()
        .find(|l| l.contains(" connected"))
        .and_then(|l| l.split_whitespace().next())
        .map(|s| s.to_string())
}

pub fn set_brightness(percent: u8) -> bool {
    if command_exists("brightnessctl") {
        return run_ok("brightnessctl", &["-q", "set", &format!("{percent}%")]);
    }
    if command_exists("xrandr") {
        if let Ok(out) = run("xrandr", &["--query"])
            && let Some(output) = parse_xrandr_connected(&out)
        {
            let v = (percent as f32 / 100.0).clamp(0.05, 1.0);
            return run_ok(
                "xrandr",
                &["--output", &output, "--brightness", &format!("{v}")],
            );
        }
    }
    false
}

/// Фактическое значение громкости (0..100) — постусловие set_volume.
/// Нет pactl/amixer или не разобрались — None (честное «не проверено»).
pub fn get_volume() -> Option<u8> {
    if command_exists("pactl") {
        let out = run("pactl", &["get-sink-volume", "@DEFAULT_SINK@"]);
        if let Ok(text) = out {
            // «Volume: front-left: 42949 / 65536 = 66% ...»
            if let Some(idx) = text.find('=') {
                let tail = &text[idx + 1..];
                let pct: String = tail
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(v) = pct.parse::<u8>() {
                    return Some(v.min(100));
                }
            }
        }
        return None;
    }
    if command_exists("amixer") {
        let out = run("amixer", &["get", "Master"]);
        if let Ok(text) = out {
            // «[playback 66%]» — берём первый процент в скобках
            if let Some(start) = text.find('[') {
                let tail = &text[start + 1..];
                let pct: String = tail
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(v) = pct.parse::<u8>() {
                    return Some(v.min(100));
                }
            }
        }
    }
    None
}

pub fn screenshot() -> Option<String> {
    let path = crate::tools::temp_path("jarvis_screenshot.png");
    // Старый файл от прошлого раза не должен сойти за свежий снимок:
    // иначе проверка «файл существует» прошла бы даже при неудаче утилиты.
    let _ = std::fs::remove_file(&path);
    let wayland = in_wayland_session();
    // Порядок: сначала то, что соответствует сессии, затем остальное.
    let mut candidates: Vec<(&str, Vec<&str>)> = Vec::new();
    if wayland && command_exists("grim") {
        candidates.push(("grim", vec![&path]));
    }
    if command_exists("gnome-screenshot") {
        candidates.push(("gnome-screenshot", vec!["-f", &path]));
    }
    if command_exists("spectacle") {
        candidates.push(("spectacle", vec!["-b", "-n", "-o", &path]));
    }
    if command_exists("scrot") {
        candidates.push(("scrot", vec![&path]));
    }
    if command_exists("grim") && !wayland {
        candidates.push(("grim", vec![&path]));
    }
    if command_exists("import") {
        candidates.push(("import", vec!["-window", "root", &path]));
    }

    for (bin, args) in candidates {
        if run_ok(bin, &args) && Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

pub fn lock_screen() -> bool {
    if command_exists("loginctl") && run_ok("loginctl", &["lock-session"]) {
        return true;
    }
    if command_exists("xdg-screensaver") && run_ok("xdg-screensaver", &["lock"]) {
        return true;
    }
    command_exists("gnome-screensaver-command") && run_ok("gnome-screensaver-command", &["-l"])
}

pub fn sleep_now() -> bool {
    command_exists("systemctl") && run_ok("systemctl", &["suspend"])
}

pub fn notify(title: &str, body: &str) -> bool {
    if !command_exists("notify-send") {
        return false;
    }
    run_ok(
        "notify-send",
        &[
            "--app-name=Jarvis",
            "--icon=dialog-information",
            title,
            body,
        ],
    )
}

pub fn clipboard_read() -> Option<String> {
    if in_wayland_session()
        && command_exists("wl-paste")
        && let Ok(out) = run("wl-paste", &["--no-newline"])
    {
        return Some(out);
    }
    if command_exists("xclip")
        && let Ok(out) = run("xclip", &["-selection", "clipboard", "-o"])
    {
        return Some(out);
    }
    if command_exists("xsel")
        && let Ok(out) = run("xsel", &["-ob"])
    {
        return Some(out);
    }
    None
}

pub fn clipboard_write(text: &str) -> bool {
    use std::io::Write;
    let write_via = |bin: &str, args: &[&str]| -> bool {
        if let Ok(mut child) = Command::new(bin)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(text.as_bytes());
            }
            return child.wait().map(|s| s.success()).unwrap_or(false);
        }
        false
    };
    if in_wayland_session() && command_exists("wl-copy") && write_via("wl-copy", &[]) {
        return true;
    }
    if command_exists("xclip") && write_via("xclip", &["-selection", "clipboard"]) {
        return true;
    }
    command_exists("xsel") && write_via("xsel", &["-ib"])
}

pub fn system_info() -> String {
    let mut info = Vec::new();
    if let Ok(release) = std::fs::read_to_string("/etc/os-release") {
        for line in release.lines() {
            if let Some(name) = line.strip_prefix("PRETTY_NAME=") {
                info.push(format!("OS:{}", name.trim_matches('"')));
                break;
            }
        }
    }
    if let Ok(out) = Command::new("uname").arg("-rm").output() {
        info.push(format!(
            "Kernel:{}",
            String::from_utf8_lossy(&out.stdout).trim()
        ));
    }
    if info.is_empty() {
        "Информация недоступна".to_string()
    } else {
        info.join(", ")
    }
}

// =========================================================================
// Нативные приложения Apple-экосистемы на Linux не существуют — честно
// =========================================================================

pub fn send_message(to: &str, _text: &str) -> Result<String, String> {
    Err(format!(
        "Отправка сообщений через системное приложение не поддерживается на Linux ({to})"
    ))
}

pub fn add_note(_title: &str, _body: &str) -> Result<String, String> {
    Err("Заметки системного приложения не поддерживаются на Linux".to_string())
}

pub fn create_event(_title: &str, _date: &str, _time: &str) -> Result<String, String> {
    Err("Календарь системного приложения не поддерживается на Linux".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wmctrl_lines_parse_into_windows() {
        let out = "0x05600007  1 google-chrome.Google-chrome  pav-home Google Chrome — Новости\n\
                   0x02e0000a  0 Navigator.Firefox     pav-home Реддит — Firefox\n";
        let windows = parse_wmctrl_lx(out);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].id, "0x05600007");
        assert_eq!(windows[0].class, "google-chrome.Google-chrome");
        assert_eq!(windows[0].title, "Google Chrome — Новости");
        assert_eq!(windows[1].class, "Navigator.Firefox");
    }

    #[test]
    fn wmctrl_matching_understands_russian_names() {
        let window = LinuxWindow {
            id: "0x1".into(),
            class: "google-chrome.Google-chrome".into(),
            title: "Google Chrome — Новости".into(),
        };
        assert!(window_matches(&window, "хром"));
        assert!(window_matches(&window, "chrome"));
        // «Яндекс Музыка» в заголовке — не хром.
        let yandex = LinuxWindow {
            id: "0x2".into(),
            class: "yandex-browser.Yandex-browser".into(),
            title: "Яндекс Музыка".into(),
        };
        assert!(!window_matches(&yandex, "хром"));
        assert!(window_matches(&yandex, "яндекс"));
    }

    #[test]
    fn xrandr_first_connected_output_wins() {
        let out = "Screen 0: minimum 320 x 200, current 1920 x 1080\n\
                   eDP-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis)\n\
                   DP-1 disconnected (normal left inverted right x axis y axis)\n\
                   HDMI-1 connected 2560x1440+1920+0\n";
        assert_eq!(parse_xrandr_connected(out), Some("eDP-1".to_string()));
        assert_eq!(parse_xrandr_connected("no outputs"), None);
    }

    #[test]
    fn action_number_split_and_desktop_digits() {
        assert_eq!(split_action_number("desktop 3"), ("desktop", Some("3")));
        assert_eq!(split_action_number("half_left"), ("half_left", None));
        assert_eq!(split_action_number(" minimizeall "), ("minimizeall", None));
        assert_eq!(desktop_number("3"), Some(3));
        assert_eq!(desktop_number("стол 3"), Some(3));
        assert_eq!(desktop_number("третий"), None);
        assert_eq!(desktop_number(""), None);
    }

    #[test]
    fn desktop_entry_parses_name_ru_and_exec() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("test-app.desktop");
        std::fs::write(
            &file,
            "[Desktop Entry]\nType=Application\nName=Test App\nName[ru]=Тестовое приложение\nExec=/usr/bin/testapp --flag %U\n",
        )
        .unwrap();
        let app = parse_desktop(&file, "test-app").expect("запись разобрана");
        assert_eq!(app.name, "Тестовое приложение");
        assert_eq!(app.exec_bin, "/usr/bin/testapp");

        // Hidden-записи не считаются приложениями.
        let hidden = dir.path().join("hidden.desktop");
        std::fs::write(
            &hidden,
            "[Desktop Entry]\nType=Application\nName=Hidden\nExec=x\nHidden=true\n",
        )
        .unwrap();
        assert!(parse_desktop(&hidden, "hidden").is_none());

        // Записи-демоны (foot --server) не попадают в каталог: у них нет
        // собственного окна, а модель видит их как правдоподобное имя.
        let daemon = dir.path().join("foot-server.desktop");
        std::fs::write(
            &daemon,
            "[Desktop Entry]\nType=Application\nName=Foot Server\nExec=foot --server\n",
        )
        .unwrap();
        assert!(parse_desktop(&daemon, "foot-server").is_none());

        // Обычное приложение, в имени которого «-server» — часть слова,
        // не выкидываем: правило точное, по целому слову.
        let normal = dir.path().join("openserver.desktop");
        std::fs::write(
            &normal,
            "[Desktop Entry]\nType=Application\nName=OpenServer Panel\nExec=/usr/bin/openserver %U\n",
        )
        .unwrap();
        assert!(parse_desktop(&normal, "openserver").is_some());

        // Exec=env VAR=... app — распространённая форма GNOME-записей:
        // имя процесса — уже после присваиваний, а не «env».
        let env_file = dir.path().join("env-app.desktop");
        std::fs::write(
            &env_file,
            "[Desktop Entry]\nType=Application\nName=Env App\n\
             Exec=env BAMF_DESKTOP_FILE_HINT=/x/y gnome-terminal --window\n",
        )
        .unwrap();
        let app = parse_desktop(&env_file, "env-app").expect("запись разобрана");
        assert_eq!(app.exec_bin, "gnome-terminal");
    }
}

#[cfg(test)]
mod hypr_tests {
    use super::*;

    #[test]
    fn hypr_clients_json_parses() {
        let out = r#"[
            {"address":"0x55f31ab2e770","hidden":false,"mapped":true,"class":"foot","title":"paveldev@omarchy:~"},
            {"address":"0x55f31a529a40","hidden":true,"mapped":true,"class":"foot","title":"скрытое"},
            {"address":"0xabc","hidden":false,"mapped":false,"class":"x","title":"не смаппено"}
        ]"#;
        let windows = parse_hypr_clients(out);
        assert_eq!(windows.len(), 1, "скрытые и не смаппенные окна — не наблюдение");
        assert_eq!(windows[0].class, "foot");
        assert_eq!(windows[0].title, "paveldev@omarchy:~");
    }

    #[test]
    fn hypr_clients_garbage_is_empty() {
        assert!(parse_hypr_clients("Invalid").is_empty());
        assert!(parse_hypr_clients("").is_empty());
    }

    #[test]
    fn hypr_active_json_parses() {
        let out = r#"{"address":"0x55f31ab2e770","class":"jarvis-app","title":"J.A.R.V.I.S"}"#;
        let v: serde_json::Value = serde_json::from_str(out).unwrap();
        assert_eq!(v.get("class").unwrap().as_str(), Some("jarvis-app"));
    }
}

