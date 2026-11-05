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

    if run_ok("pgrep", &["-x", &probe_name]) || app_running(&display) {
        return Ok(format!("{display} уже открыт"));
    }

    let launched = match &entry {
        Some(app) if command_exists("gtk-launch") => run_ok("gtk-launch", &[&app.id]),
        Some(app) if command_exists("gio") => run_ok("gio", &["launch", &app.path]),
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
        if run_ok("pgrep", &["-x", &probe_name]) || app_running(&display) {
            return Ok(format!("Открыл {display}"));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "Не удалось открыть {display}: приложение не запустилось"
    ))
}

/// Настоящее установленное приложение по русской/приблизительной просьбе.
/// `None` — такого приложения нет: подменять его файлом или сайтом нельзя.
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
    let probe = entry
        .as_ref()
        .map(|a| a.exec_bin.trim())
        .filter(|b| !b.is_empty())
        .map(|b| {
            Path::new(b)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| b.to_string())
        })
        .unwrap_or_else(|| display.clone());
    if !app_running(&display) && !app_running(&probe) {
        return false;
    }
    println!("🛑 Закрываю: {display}");
    run_ok("pkill", &["-x", &probe]) || run_ok("pkill", &["-f", &probe])
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

fn wmctrl_windows() -> Option<Vec<LinuxWindow>> {
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

/// Активное окно: id и заголовок. xdotool точнее; xprop — запасной путь.
fn active_window() -> Option<(String, String)> {
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
    if command_exists("wmctrl") {
        if let Some(window) = find_window_wmctrl(want) {
            return Some(window);
        }
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

/// Действие над окном. `app = None` — окно самого Джарвиса; «активное» —
/// активное приложение (как на macOS: сворачивать чужое окно без просьбы —
/// неприятный сюрприз).
pub fn window_control(action: &str, app: Option<&str>) -> String {
    // wmctrl и xdotool — инструменты X11: без DISPLAY им нечего смотреть.
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
