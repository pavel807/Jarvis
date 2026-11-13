//! macOS: реализация платформенного слоя поверх AppleScript, System Events
//! и системных утилит (`pmset`, `screencapture`, `pbcopy`).
//!
//! Здесь же — всё, что в этом коде было выстрадано жизнью и запротоколировано:
//! запросы разрешений приходят от самого Джарвиса (in-process NSAppleScript,
//! см. `crate::applescript`), успех подтверждается делом (состояние плеера
//! до/после, появление процесса после `open`), а отказы системы превращаются
//! в человеческие подсказки «где включить».

use crate::platform::{COMMON_ALIASES, fuzzy_app_match, lookup_alias, process_matches};
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// =========================================================================
// Словарь русских названий приложений (macOS)
// =========================================================================

/// Частые русские названия → фрагмент настоящего имени приложения/процесса.
/// Пользователь говорит «сверни хром», а процесс называется «Google Chrome».
/// Остальное покрывает транслитерация («телеграм» → «telegram»).
const APP_ALIASES: &[(&str, &str)] = &[
    ("браузер", "Safari"),
    ("сафари", "Safari"),
    ("музыка", "Music"),
    ("почта", "Mail"),
    ("мессенджер", "Messages"),
    ("сообщения", "Messages"),
    ("итерминал", "iTerm"),
    ("терминал", "Terminal"),
    ("калькулятор", "Calculator"),
    ("календарь", "Calendar"),
    ("заметки", "Notes"),
    ("карты", "Maps"),
    ("часы", "Clock"),
    ("напоминания", "Reminders"),
    ("фейстайм", "FaceTime"),
    ("фотографии", "Photos"),
    ("просмотр", "Preview"),
    ("настройки", "System Settings"),
];

/// Русское название из просьбы → каноничное имя, если оно есть в словарях
/// ОС и общем («гугл хром» → «Google Chrome»).
pub fn alias(word: &str) -> Option<&'static str> {
    lookup_alias(word, &[APP_ALIASES, COMMON_ALIASES])
}

// =========================================================================
// AppleScript
// =========================================================================

pub(crate) fn osascript(script: &str) -> Result<String, String> {
    // 1. Прямой путь: NSAppleScript в процессе Джарвиса. Системный запрос на
    //    автоматизацию приходит от самого приложения — его можно выдать один
    //    раз, и он действует на все последующие скрипты.
    if let Some(result) = crate::applescript::run(script) {
        return result.map(|out| out.trim().to_string());
    }

    // 2. Запасной путь: внешний osascript. Нужен, когда in-process режим
    //    отключён переменной окружения для отладки.
    match Command::new("osascript").args(["-e", script]).output() {
        Ok(out) if out.status.success() => {
            Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Err(if err.is_empty() {
                format!(
                    "osascript завершился с кодом {}",
                    out.status.code().unwrap_or(-1)
                )
            } else {
                // В сообщении AppleScript первая строка — сам текст ошибки,
                // позиция вроде «36:82:» только мешает.
                let line = err.lines().next().unwrap_or(&err);
                match line.split_once(": ") {
                    Some((_, msg)) if !msg.is_empty() => msg.to_string(),
                    _ => line.to_string(),
                }
            })
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Отказ macOS на управление приложением почти всегда означает отсутствие
/// разрешения, а не поломку. Переводим типовые случаи на человеческий и
/// сразу подсказываем, где включить.
pub(crate) fn friendly_access_error(err: &str) -> String {
    let lower = err.to_lowercase();
    if lower.contains("assistive access") || lower.contains("assistiveaccess") {
        "macOS не разрешает управлять окнами. Системные настройки → Конфиденциальность и безопасность → Универсальный доступ — включи Jarvis".into()
    } else if lower.contains("send keystrokes") || lower.contains("not allowed to send") {
        "macOS не разрешает нажимать клавиши. Системные настройки → Конфиденциальность и безопасность → Универсальный доступ — включи Jarvis".into()
    } else if lower.contains("not authorized") || lower.contains("-1743") {
        "macOS не разрешает автоматизацию этого приложения. Системные настройки → Конфиденциальность и безопасность → Автоматизация — разреши Jarvis".into()
    } else if lower.contains("privilege violation") || lower.contains("-10004") {
        "macOS не даёт доступ к этому приложению. Системные настройки → Конфиденциальность и безопасность → Автоматизация — разреши Jarvis".into()
    } else if lower.contains("javascript")
        || lower.contains("applescript") && lower.contains("allow")
    {
        "В браузере выключен JavaScript из AppleScript: «Вид → Разработка → Разрешить JavaScript из AppleScript»".into()
    } else if lower.contains("application isn't running") || lower.contains("-600") {
        "Приложение не запущено".into()
    } else {
        err.to_string()
    }
}

/// Отказ в автоматизации: одинаков для любого приложения, поэтому винить
/// конкретный браузер в нём нельзя — иначе пользователь пойдёт искать музыку
/// не там, где она играет.
fn is_permission_error(err: &str) -> bool {
    let lower = err.to_lowercase();
    lower.contains("privilege violation")
        || lower.contains("-10004")
        || lower.contains("not authorized")
        || lower.contains("-1743")
}

/// Общая подсказка про автоматизацию: одна причина — одно объяснение.
fn permission_hint() -> String {
    "macOS не даёт Jarvis управлять другими приложениями. Системные настройки → \
     Конфиденциальность и безопасность → Автоматизация — разреши Jarvis нужные \
     приложения (браузер и «Музыку»)"
        .to_string()
}

/// Экранирование строки для AppleScript-литерала.
pub(crate) fn applescript_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// =========================================================================
// Приложения
// =========================================================================

/// Запущено ли приложение: pgrep по имени процесса, затем по пути бандла —
/// у части приложений имя процесса не совпадает с названием
/// («Code» vs «Visual Studio Code»).
pub fn app_running(name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() {
        return false;
    }
    let exact = Command::new("pgrep")
        .args(["-x", name])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if exact {
        return true;
    }
    Command::new("pgrep")
        .args(["-f", &format!("{name}.app")])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Список установленных приложений: сканируем каталоги бандлов.
/// Результат кэшируется на минуту — установка приложения раз в сессию
/// не требует пересканирования на каждую просьбу.
fn bundle_names() -> Vec<String> {
    static CACHE: OnceLock<Mutex<Option<(Instant, Vec<String>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, names)) = guard.as_ref()
            && at.elapsed() < Duration::from_secs(60)
        {
            return names.clone();
        }
    }

    let mut bundles: Vec<String> = Vec::new();
    let mut roots: Vec<std::path::PathBuf> =
        vec!["/Applications".into(), "/System/Applications".into()];
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(std::path::PathBuf::from(home).join("Applications"));
    }
    for root in &roots {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let file = entry.file_name();
                if let Some(stem) = file.to_string_lossy().strip_suffix(".app") {
                    bundles.push(stem.to_string());
                }
            }
        }
    }
    *cache.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), bundles.clone()));
    bundles
}

/// Настоящее установленное приложение по русской/приблизительной просьбе.
/// `None` — такого приложения нет: вызывающий код не должен подменять его
/// файлом, папкой или сайтом.
/// Каталог установленных приложений для контекста агента.
/// Записи базы приложений: на этой платформе пока каталог не собирается.
pub fn app_db_entries() -> Vec<(String, String, String)> {
    Vec::new()
}

pub fn app_catalog() -> Vec<String> {
    let mut names = bundle_names();
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
    let bundles = bundle_names();
    let find = |pred: &dyn Fn(&String) -> bool| -> Option<String> {
        bundles.iter().find(|b| pred(b)).cloned()
    };
    for variant in crate::platform::name_variants(want) {
        let hit = find(&|b| b.to_lowercase() == variant.to_lowercase())
            .or_else(|| find(&|b| process_matches(b, &variant)))
            .or_else(|| alias(&variant).and_then(|target| find(&|b| process_matches(b, target))))
            // Нечёткий подбор последним: «энкубик слайсер некст» →
            // AnycubicSlicerNext. Точное вхождение транслита тут не работает —
            // слова просьбы и имя бандла различаются буквами (slaiser/slicer).
            .or_else(|| fuzzy_app_match(&variant, &bundles));
        if hit.is_some() {
            return hit;
        }
    }
    None
}

/// Настоящее имя приложения по русской/приблизительной просьбе: ищем по
/// каталогам приложений («хром» → «Google Chrome»). Ничего не нашлось —
/// возвращаем как есть: `open -a` честно сообщит, что приложения нет.
/// Положительные результаты кэшируются: имена бандлов за сессию не меняются.
pub fn app_resolve(name: &str) -> String {
    let want = name.trim();
    if want.is_empty() {
        return want.to_string();
    }

    static RESOLVED: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    let cache = RESOLVED.get_or_init(|| Mutex::new(HashMap::new()));
    let key = want.to_lowercase();
    if let Some(hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return hit.clone();
    }

    let result = match app_lookup(want) {
        Some(b) => {
            cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, b.clone());
            b
        }
        None => want.to_string(),
    };
    result
}

/// Открывает приложение. `Ok` — уже открыто или открылось (подтверждено
/// процессом); `Err` — честная причина.
pub fn app_open(name: &str) -> Result<String, String> {
    let name = app_resolve(name);
    if app_running(&name) {
        return Ok(format!("{name} уже открыт"));
    }
    // `open -a` завершается ошибкой, если приложения нет, — статус честный.
    match Command::new("open").args(["-a", &name]).status() {
        Ok(s) if s.success() => {
            // Статус «ок» ещё не гарантия: ждём появления процесса до 3 сек.
            for _ in 0..6 {
                if app_running(&name) {
                    return Ok(format!("Открыл {name}"));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(format!(
                "Не удалось открыть {name}: приложение не запустилось"
            ))
        }
        Ok(_) => Err(format!("Не удалось открыть {name}: приложение не найдено")),
        Err(e) => Err(format!("Не удалось открыть {name}: {e}")),
    }
}

/// Просит приложение закрыться — без ожидания.
///
/// Ключевой момент: `osascript ... to quit` нельзя ждать. Premiere с
/// несохранённым проектом показывает модальное окно и не завершается, а
/// `output()` висит до бесконечности — голосовой ассистент при этом молчит и
/// выглядит сломанным. Поэтому запрос отправляется в фоне и не блокирует.
/// `false` — приложение не запущено, закрывать нечего.
pub fn app_quit(name: &str) -> bool {
    let requested = name.trim();
    if requested.is_empty() {
        return false;
    }
    // «Закрой телеграм» → настоящий бандл Telegram, иначе AppleScript
    // ищет приложение с русским именем и молча ничего не делает.
    let resolved = app_lookup(requested)
        .or_else(|| alias(requested).map(str::to_string))
        .unwrap_or_else(|| requested.to_string());
    let app = resolved.trim();
    if app.is_empty() || !app_running(app) {
        return false;
    }
    println!("🛑 Закрываю: {app}");

    let script = format!("tell application \"{app}\" to quit");
    if let Err(e) = Command::new("osascript")
        .args(["-e", &script])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        eprintln!("⚠️ Не удалось отправить запрос на выход {app}: {e}");
        return false;
    }
    true
}

// =========================================================================
// Окна
// =========================================================================

/// Найти процесс по имени. Модель пишет «терминал», а процесс называется
/// `Terminal`, пользователь говорит «хром», а процесс — `Google Chrome`:
/// сначала точное совпадение, затем вхождение, затем словарь русских
/// названий и транслитерация.
/// `Err` — не удалось даже спросить System Events (например, нет разрешения
/// на автоматизацию): такую причину нельзя выдавать за «приложение не найдено»,
/// иначе пользователь ищет опечатку вместо галочки в Системных настройках.
fn find_process(name: &str) -> Result<Option<String>, String> {
    let want = name.trim();
    if want.is_empty() {
        return Ok(None);
    }
    let script = r#"tell application "System Events" to get name of every application process whose background only is false"#;
    let out = osascript(script)?;
    let names: Vec<String> = out
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let find = |pred: &dyn Fn(&String) -> bool| names.iter().find(|n| pred(n)).cloned();
    // Точное совпадение важнее вхождения: «Safari» и «Safari Technology
    // Preview» — разные приложения, и пользователь просил первое. Падежные
    // формы («телеграма», «хрома») проверяем отдельными вариантами.
    for variant in crate::platform::name_variants(want) {
        let hit = find(&|n| n.to_lowercase() == variant.to_lowercase())
            .or_else(|| find(&|n| n.to_lowercase().contains(&variant.to_lowercase())))
            .or_else(|| alias(&variant).and_then(|target| find(&|n| process_matches(n, target))))
            .or_else(|| find(&|n| process_matches(n, &variant)))
            .or_else(|| fuzzy_app_match(&variant, &names));
        if hit.is_some() {
            return Ok(hit);
        }
    }
    Ok(None)
}

fn active_process() -> Result<Option<String>, String> {
    let out = osascript(
        r#"tell application "System Events" to get name of first application process whose frontmost is true"#,
    )?;
    let name = out.trim().to_string();
    Ok((!name.is_empty() && name != "System Events").then_some(name))
}

/// Рабочая область экрана — для «развернуть на весь экран».
fn screen_bounds() -> Option<(i32, i32, i32, i32)> {
    let out = osascript(r#"tell application "Finder" to get bounds of window of desktop"#).ok()?;
    let nums: Vec<i32> = out
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    match nums.as_slice() {
        [x1, y1, x2, y2] => Some((*x1, *y1, *x2, *y2)),
        _ => None,
    }
}

/// Имя процесса самого Джарвиса (jarvis-app / jarvis-cli).
fn own_process_name() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    exe.file_stem()?.to_str().map(|s| s.to_string())
}

/// Действие над окном приложения.
///
/// `app = None` — окно самого Джарвиса: активным почти всегда является
/// приложение, которым человек прямо сейчас пользуется, и «сверни окно»,
/// применённое к нему, сворачивает Chrome посреди работы — неприятный
/// сюрприз. Явное «активное» адресует активное приложение.
///
/// Если у приложения несколько подходящих окон, функция не выбирает первое
/// молча: возвращается вопрос со списком. Инструмент превращает его в
/// интерактивный выбор (см. `choice`), а прямой вызов получает готовую фразу.
pub fn window_control(action: &str, app: Option<&str>) -> String {
    use crate::platform::WindowControlOutcome;
    match window_control_checked(action, app) {
        WindowControlOutcome::Done(text) => text,
        WindowControlOutcome::Ambiguous(targets) => {
            ambiguous_window_speech(action, app, &targets)
        }
    }
}

/// Оконная команда без молчаливого выбора первого окна.
pub fn window_control_checked(
    action: &str,
    app: Option<&str>,
) -> crate::platform::WindowControlOutcome {
    use crate::platform::{WindowControlOutcome, WindowTarget};

    // Список окон не требует конкретного приложения: показываем все.
    if action == "list" {
        return WindowControlOutcome::Done(window_list());
    }

    let (process, display) = match app {
        // «Сверни окно голосового помощника» — это моё окно, а не чужой
        // процесс с именем «голосовой помощник».
        Some(wanted) if crate::platform::is_self_window_target(wanted) => {
            match own_process_name().and_then(|me| find_process(&me).ok().flatten()) {
                Some(me) => (me, "Джарвис".to_string()),
                None => {
                    return WindowControlOutcome::Done(
                        "Моё окно не запущено — я в фоне. Скажи, чьё окно свернуть"
                            .to_string(),
                    );
                }
            }
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
            return WindowControlOutcome::Done(match active_process() {
                Ok(Some(p)) => window_run_index(&p, &p, action, None),
                Ok(None) => "Не нашёл активное приложение с окном".to_string(),
                Err(e) => format!("Не могу управлять окнами: {e}"),
            });
        }
        Some(wanted) => match find_process(wanted) {
            Ok(Some(process)) => (process.clone(), process),
            Ok(None) => {
                return WindowControlOutcome::Done(format!(
                    "Приложение {wanted} не найдено среди открытых"
                ));
            }
            Err(e) => {
                return WindowControlOutcome::Done(format!("Не могу управлять окнами: {e}"));
            }
        },
        None => match own_process_name() {
            Some(me) => match find_process(&me) {
                Ok(Some(me)) => (me, "Джарвис".to_string()),
                Ok(None) => {
                    return WindowControlOutcome::Done(
                        "Моё окно не запущено — я в фоне. Скажи, чьё окно свернуть"
                            .to_string(),
                    );
                }
                Err(e) => {
                    return WindowControlOutcome::Done(format!("Не могу управлять окнами: {e}"));
                }
            },
            None => {
                return WindowControlOutcome::Done(
                    "Не могу управлять окнами: не понял имя своего процесса".to_string(),
                );
            }
        },
    };

    let windows = match window_targets(&process) {
        Ok(windows) => windows,
        Err(e) => {
            return WindowControlOutcome::Done(format!(
                "Не могу управлять окнами: {}",
                friendly_access_error(&e)
            ));
        }
    };
    if windows.is_empty() {
        return WindowControlOutcome::Done(format!("У {display} нет открытых окон"));
    }

    // «Верни окно» при нескольких окнах неоднозначно только тогда, когда
    // свёрнутых несколько. Одно свёрнутое — это уже конкретная цель.
    let candidates: Vec<WindowTarget> = if action == "restore" {
        windows.iter().filter(|w| w.minimized).cloned().collect()
    } else {
        windows.clone()
    };
    if candidates.len() > 1 {
        return WindowControlOutcome::Ambiguous(candidates);
    }

    let index = candidates.first().and_then(|target| {
        windows
            .iter()
            .position(|window| window.title == target.title)
            .map(|index| index + 1)
    });
    WindowControlOutcome::Done(window_run_index(&process, &display, action, index))
}

/// Действие над конкретным окном, выбранным пользователем. Окно ищется
/// заново по заголовку: индексы на macOS меняются после сворачивания, и
/// запомненный номер мог бы указать на другое окно.
pub fn window_control_target(action: &str, app: &str, title: &str) -> String {
    let process = match find_process(app) {
        Ok(Some(process)) => process,
        Ok(None) => return format!("Приложение {app} больше не открыто"),
        Err(e) => return format!("Не могу управлять окнами: {e}"),
    };
    let windows = match window_targets(&process) {
        Ok(windows) => windows,
        Err(e) => {
            return format!(
                "Не могу управлять окнами: {}",
                friendly_access_error(&e)
            );
        }
    };
    let index = windows
        .iter()
        .position(|window| window.title == title)
        .or_else(|| {
            windows.iter().position(|window| {
                !window.title.trim().is_empty()
                    && (window.title.contains(title) || title.contains(&window.title))
            })
        })
        .map(|index| index + 1);
    match index {
        Some(index) => {
            let name = if title.trim().is_empty() {
                process.as_str()
            } else {
                title
            };
            window_run_index(&process, name, action, Some(index))
        }
        None => format!("Окно «{title}» уже закрыто или переименовано"),
    }
}

/// Список окон процесса: заголовок и состояние «свёрнуто».
fn window_targets(process: &str) -> Result<Vec<crate::platform::WindowTarget>, String> {
    let p = format!(
        "\"{}\"",
        process.replace('\\', "\\\\").replace('"', "\\\"")
    );
    let script = format!(
        r#"tell application "System Events" to tell process {p}
set out to ""
set sep to (ASCII character 31)
repeat with w in windows
  set wn to ""
  set mn to false
  try
    set wn to name of w
  end try
  try
    set mn to value of attribute "AXMinimized" of w
  end try
  set out to out & wn & sep & (mn as text) & linefeed
end repeat
return out
end tell"#
    );
    let out = osascript(&script)?;
    let mut windows = Vec::new();
    for line in out.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let (title, minimized) = line.split_once('\u{1f}').unwrap_or((line, "false"));
        windows.push(crate::platform::WindowTarget {
            app: process.to_string(),
            title: title.replace(['\n', '\r'], " ").trim().to_string(),
            minimized: minimized.trim().eq_ignore_ascii_case("true"),
        });
    }
    Ok(windows)
}

/// Человеческая фраза для прямого вызова `window_control`, когда окон
/// несколько. Интерактивный путь (`tools::Window`) использует `choice`.
fn ambiguous_window_speech(
    action: &str,
    app: Option<&str>,
    targets: &[crate::platform::WindowTarget],
) -> String {
    let who = app.unwrap_or("приложения");
    let list = targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            let label = if target.title.trim().is_empty() {
                "без названия"
            } else {
                target.title.as_str()
            };
            format!("{} — {label}", index + 1)
        })
        .collect::<Vec<_>>()
        .join("; ");
    let verb = match action {
        "minimize" => "свернуть",
        "close" => "закрыть",
        "maximize" => "развернуть",
        "fullscreen" => "открыть на весь экран",
        "restore" => "вернуть",
        "focus" => "выбрать",
        _ => "выполнить действие с",
    };
    format!("У {who} несколько окон: {list}. Какое {verb}? Уточни номер или название.")
}

/// Выполнить действие над окном процесса и превратить ответ в человеческую
/// фразу. `target` — имя процесса как есть, без кавычек: кавычки и
/// экранирование добавляются здесь, в одном месте. Раньше вызывающий код
/// оборачивал имя в кавычки, а функция экранировала их повторно — в AppleScript
/// уходило `\"zoom.us\"`, команда валилась с «syntax error» (-2741), и окно
/// не сворачивалось. `name` нужен только для текста ответа.
///
/// `index` — номер окна в исходном списке `windows` (с единицы). `None` —
/// прежнее поведение для одиночного окна/активного приложения.
fn window_run_index(target: &str, name: &str, action: &str, index: Option<usize>) -> String {
    let p = format!("\"{}\"", target.replace('\\', "\\\\").replace('"', "\\\""));
    // `set frontmost to true` — единственная валидная форма; `to frontmost
    // to true` давала -2740, которую никто не видел (ошибка активации
    // игнорировалась), и окно не поднималось перед действием.
    let tell =
        format!(r#"tell application "System Events" to tell process {p} to set frontmost to true"#);
    let _ = osascript(&tell);
    // Конкретное окно выбирается только тогда, когда пользователь его назвал.
    // Для одиночного окна и активного приложения поведение прежнее: window 1.
    let window = format!("window {}", index.unwrap_or(1));

    match action {
        "maximize" => {
            // Растягиваем на рабочую область экрана, но не включаем
            // полноэкранный режим: так остаётся видна панель меню и Dock —
            // обычно именно это и просят, говоря «разверни».
            let Some((x1, y1, x2, y2)) = screen_bounds() else {
                return "Не удалось узнать размер экрана".to_string();
            };
            let script = format!(
                r#"tell application "System Events" to tell process {p}
if (count of windows) = 0 then return "нет окон"
set value of attribute "AXFullScreen" of {window} to false
set value of attribute "AXMinimized" of {window} to false
set bounds of {window} to {{{x1}, {y1}, {x2}, {y2}}}
return "готово"
end tell"#
            );
            run_window_script(&script, name, "развернул")
        }
        "fullscreen" => {
            let script = format!(
                r#"tell application "System Events" to tell process {p}
if (count of windows) = 0 then return "нет окон"
set value of attribute "AXFullScreen" of {window} to true
return "готово"
end tell"#
            );
            run_window_script(&script, name, "включил полный экран")
        }
        "minimize" => {
            // Свёрнутое окно уезжает в конец списка windows. Проверять
            // «window 1» после действия нельзя: это уже следующее окно,
            // и честное сворачивание выглядело как неудача. Считаем
            // свёрнутые до и после — прирост и есть подтверждение.
            let script = format!(
                r#"tell application "System Events" to tell process {p}
if (count of windows) = 0 then return "нет окон"
try
  if value of attribute "AXMinimized" of {window} then return "already"
end try
set beforeMin to 0
repeat with w in windows
  try
    if value of attribute "AXMinimized" of w then set beforeMin to beforeMin + 1
  end try
end repeat
set value of attribute "AXMinimized" of {window} to true
delay 0.2
set afterMin to 0
repeat with w in windows
  try
    if value of attribute "AXMinimized" of w then set afterMin to afterMin + 1
  end try
end repeat
if afterMin > beforeMin then return "готово"
return "не подтвердил"
end tell"#
            );
            match osascript(&script) {
                Ok(out) if out.contains("нет окон") => format!("У {name} нет открытых окон"),
                Ok(out) if out.trim() == "already" => format!("Окно {name} уже свёрнуто"),
                Ok(out) if out.trim() == "готово" => format!("свернул окно {name}"),
                Ok(_) => {
                    format!("Не удалось свернуть окно {name}: macOS не подтвердил сворачивание")
                }
                Err(e) => format!("Ошибка: {}", friendly_access_error(&e)),
            }
        }
        "restore" => {
            // Свёрнутое окно — не «window 1»: оно в конце списка. Для
            // выбранного пользователем окна адресуем конкретный номер из
            // исходного списка; без выбора разворачиваем последнее свёрнутое.
            // В обоих случаях подтверждение — уменьшение числа свёрнутых.
            let script = match index {
                Some(target_index) => format!(
                    r#"tell application "System Events" to tell process {p}
if (count of windows) < {target_index} then return "нет окна"
set beforeMin to 0
repeat with w in windows
  try
    if value of attribute "AXMinimized" of w then set beforeMin to beforeMin + 1
  end try
end repeat
set wasMin to false
try
  set wasMin to value of attribute "AXMinimized" of window {target_index}
end try
if wasMin then
  try
    set value of attribute "AXFullScreen" of window {target_index} to false
  end try
  set value of attribute "AXMinimized" of window {target_index} to false
  delay 0.2
  set afterMin to 0
  repeat with w in windows
    try
      if value of attribute "AXMinimized" of w then set afterMin to afterMin + 1
    end try
  end repeat
  if afterMin < beforeMin then return "готово"
  return "не подтвердил"
end if
set wasFullscreen to false
try
  set wasFullscreen to value of attribute "AXFullScreen" of window {target_index}
end try
try
  set value of attribute "AXFullScreen" of window {target_index} to false
end try
if wasFullscreen then return "из полного экрана"
return "уже в обычном виде"
end tell"#
                ),
                None => format!(
                    r#"tell application "System Events" to tell process {p}
if (count of windows) = 0 then return "нет окон"
set beforeMin to 0
set targetIndex to 0
set idx to 0
repeat with w in windows
  set idx to idx + 1
  try
    if value of attribute "AXMinimized" of w then
      set beforeMin to beforeMin + 1
      set targetIndex to idx
    end if
  end try
end repeat
if targetIndex = 0 then
  set wasFullscreen to false
  try
    set wasFullscreen to value of attribute "AXFullScreen" of window 1
  end try
  try
    set value of attribute "AXFullScreen" of window 1 to false
  end try
  if wasFullscreen then return "из полного экрана"
  return "нет свёрнутых"
end if
try
  set value of attribute "AXFullScreen" of window targetIndex to false
end try
set value of attribute "AXMinimized" of window targetIndex to false
delay 0.2
set afterMin to 0
repeat with w in windows
  try
    if value of attribute "AXMinimized" of w then set afterMin to afterMin + 1
  end try
end repeat
if afterMin < beforeMin then return "готово"
return "не подтвердил"
end tell"#
                ),
            };
            match osascript(&script) {
                Ok(out) if out.contains("нет окон") => format!("У {name} нет открытых окон"),
                Ok(out) if out.contains("нет окна") => {
                    format!("Окно «{name}» уже закрыто")
                }
                Ok(out) if out.contains("из полного экрана") => {
                    format!("вернул окно {name} из полного экрана")
                }
                Ok(out) if out.contains("нет свёрнутых") => {
                    format!("У {name} нет свёрнутых окон")
                }
                Ok(out) if out.contains("уже в обычном виде") => {
                    format!("Окно {name} уже в обычном виде")
                }
                Ok(out) if out.trim() == "готово" => {
                    format!("вернул окно {name} из Dock")
                }
                Ok(_) => format!(
                    "Не удалось вернуть окно {name}: macOS не подтвердил разворачивание"
                ),
                Err(e) => format!("Ошибка: {}", friendly_access_error(&e)),
            }
        }
        "close" => {
            let script = format!(
                r#"tell application "System Events" to tell process {p}
if (count of windows) = 0 then return "нет окон"
close {window}
return "готово"
end tell"#
            );
            run_window_script(&script, name, "закрыл")
        }
        "focus" => match osascript(&tell) {
            Ok(_) => {
                // При выборе конкретного окна одного frontmost мало: поднимаем
                // его над остальными окнами приложения.
                if let Some(target_index) = index {
                    let raise = format!(
                        r#"tell application "System Events" to tell process {p}
try
  perform action "AXRaise" of window {target_index}
end try
set frontmost to true
end tell"#
                    );
                    let _ = osascript(&raise);
                }
                // `set frontmost` не гарантирует, что система успела сменить
                // активное приложение. Проверяем фактического frontmost.
                for _ in 0..6 {
                    std::thread::sleep(Duration::from_millis(100));
                    if let Ok(Some(active)) = active_process()
                        && (active.eq_ignore_ascii_case(target) || process_matches(&active, target))
                    {
                        return format!("Переключился на {name}");
                    }
                }
                format!(
                    "Не удалось вывести {name} на передний план: macOS не подтвердил переключение"
                )
            }
            Err(e) => format!("Ошибка: {}", friendly_access_error(&e)),
        },
        other => format!("Не могу выполнить действие с окном: {other}"),
    }
}

/// Выполнить скрипт над окном и превратить ответ в человеческую фразу.
fn run_window_script(script: &str, process: &str, done: &str) -> String {
    match osascript(script) {
        Ok(out) if out.contains("нет окон") => format!("У {process} нет открытых окон"),
        Ok(_) => format!("{done} окно {process}"),
        Err(e) => format!("Ошибка: {}", friendly_access_error(&e)),
    }
}

fn window_list() -> String {
    let script = r#"tell application "System Events"
set out to ""
repeat with p in (every application process whose background only is false)
try
set wn to name of windows of p
if (count of wn) > 0 then set out to out & (name of p) & ": " & (my joinList(wn)) & linefeed
end try
end repeat
return out
end tell

on joinList(lst)
set AppleScript's text item delimiters to ", "
set s to lst as text
set AppleScript's text item delimiters to ""
return s
end joinList"#;
    match osascript(script) {
        Ok(out) if out.trim().is_empty() => "Открытых окон не нашёл".to_string(),
        Ok(out) => out
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(12)
            .collect::<Vec<_>>()
            .join("; "),
        Err(e) => format!("Ошибка: {e}"),
    }
}

// =========================================================================
// Системные действия
// =========================================================================

pub fn set_volume(percent: u8) -> bool {
    Command::new("osascript")
        .args(["-e", &format!("set volume output volume {percent}")])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Фактическое значение громкости из системы (0..100) — постусловие set_volume.
pub fn get_volume() -> Option<u8> {
    let out = Command::new("osascript")
        .args(["-e", "output volume of (get volume settings)"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    text.parse::<f64>().ok().map(|v| v.round().clamp(0.0, 100.0) as u8)
}

pub fn set_brightness(percent: u8) -> bool {
    let v = (percent as f32 / 100.0).clamp(0.0, 1.0);
    // Утилита brightness из Homebrew — единственный скриптуемый путь
    // к яркости дисплея на macOS.
    Command::new("brightness")
        .arg(format!("{v}"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn screenshot() -> Option<String> {
    let path = crate::tools::temp_path("jarvis_screenshot.png");
    let ok = Command::new("screencapture")
        .args(["-x", &path])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    // Exit code screencapture — не доказательство: проверяем, что файл
    // реально создан и не пуст.
    let exists = ok
        && std::path::Path::new(&path)
            .metadata()
            .map(|m| m.len() > 0)
            .unwrap_or(false);
    exists.then_some(path)
}

pub fn lock_screen() -> bool {
    Command::new("pmset")
        .args(["displaysleepnow"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn sleep_now() -> bool {
    Command::new("pmset")
        .args(["sleepnow"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn notify(title: &str, body: &str) -> bool {
    let script = format!(
        "display notification \"{}\" with title \"{}\"",
        applescript_escape(body),
        applescript_escape(title),
    );
    Command::new("osascript")
        .args(["-e", &script])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn clipboard_read() -> Option<String> {
    if let Ok(out) = Command::new("pbpaste").output() {
        return Some(String::from_utf8_lossy(&out.stdout).to_string());
    }
    None
}

pub fn clipboard_write(text: &str) -> bool {
    use std::io::Write;
    if let Ok(mut child) = Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
    {
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(text.as_bytes());
        }
        return child.wait().map(|s| s.success()).unwrap_or(false);
    }
    false
}

pub fn system_info() -> String {
    let mut info = Vec::new();
    if let Ok(out) = Command::new("sw_vers").output() {
        let s = String::from_utf8_lossy(&out.stdout);
        for line in s.lines() {
            if let Some((k, v)) = line.split_once(':') {
                info.push(format!("{}:{}", k.trim(), v.trim()));
            }
        }
    }
    if let Ok(out) = Command::new("uname").arg("-m").output() {
        info.push(format!(
            "Arch:{}",
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
// Нативные приложения: Сообщения, Заметки, Календарь
// =========================================================================

/// Отправить сообщение через «Сообщения» (iMessage). Адресат — имя контакта,
/// email или телефон Apple ID. Отказы честные: нет аккаунта iMessage, нет
/// доступа к контакту, нет разрешения на автоматизацию.
pub fn send_message(to: &str, text: &str) -> Result<String, String> {
    let (to_e, text_e) = (applescript_escape(to), applescript_escape(text));
    let script = format!(
        r#"tell application "Messages"
set svc to 1st service whose service type = iMessage
set p to participant "{to_e}" of svc
send "{text_e}" to p
end tell"#
    );
    match osascript(&script) {
        Ok(_) => Ok(format!("Отправил сообщение {to}")),
        Err(e) => {
            let lower = e.to_lowercase();
            if lower.contains("can't get") || lower.contains("participant") {
                Err(format!(
                    "Не нашёл «{to}» в Сообщениях: нужен email/телефон Apple ID или точное имя контакта"
                ))
            } else {
                Err(format!(
                    "Не удалось отправить: {}",
                    friendly_access_error(&e)
                ))
            }
        }
    }
}

/// Новая заметка в «Заметках».
pub fn add_note(title: &str, body: &str) -> Result<String, String> {
    let script = format!(
        r#"tell application "Notes" to make new note with properties {{name:"{}", body:"{}"}}"#,
        applescript_escape(title),
        applescript_escape(body),
    );
    match osascript(&script) {
        Ok(_) => Ok(format!("Записал заметку «{title}»")),
        Err(e) => Err(format!(
            "Не удалось создать заметку: {}",
            friendly_access_error(&e)
        )),
    }
}

/// «ГГГГ-ММ-ДД» от модели → (год, месяц, день). Пусто или мусор — None.
fn parse_iso_date(date: &str) -> Option<(i64, u32, u32)> {
    let parts: Vec<&str> = date.trim().split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let year = parts[0].trim().parse::<i64>().ok()?;
    let month = parts[1].trim().parse::<u32>().ok()?;
    let day = parts[2].trim().parse::<u32>().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || !(1970..=3000).contains(&year) {
        return None;
    }
    Some((year, month, day))
}

/// «ЧЧ:ММ» от модели → (часы, минуты). Пусто или мусор — None.
fn parse_iso_time(time: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = time.trim().split(':').collect();
    if parts.len() != 2 {
        return None;
    }
    let h = parts[0].trim().parse::<u32>().ok()?;
    let m = parts[1].trim().parse::<u32>().ok()?;
    if h > 23 || m > 59 {
        return None;
    }
    Some((h, m))
}

/// Событие в «Календаре»: сегодня/указанная дата, указанное время или
/// ближайший круглый час, длительность час. Дата не проверяется на
/// существование (31 февраля) — Calendar сам нормализует.
pub fn create_event(title: &str, date: &str, time: &str) -> Result<String, String> {
    // Дата: из модели или сегодня. «Сегодня» достаём из AppleScript, чтобы
    // не тянуть в рантайм работу с часовыми поясами.
    let date_setup = match parse_iso_date(date) {
        Some((y, m, d)) => {
            // День ставим первым: иначе 31-е число «перекатывает» месяц.
            format!(
                "set day of sd to 1\nset year of sd to {y}\nset month of sd to {m}\nset day of sd to {d}\n"
            )
        }
        None => String::new(),
    };
    let time_setup = match parse_iso_time(time) {
        Some((h, min)) => format!("set hours of sd to {h}\nset minutes of sd to {min}\n"),
        None => String::new(), // останется текущее время — сдвигаем на час ниже
    };
    let script = format!(
        r#"tell application "Calendar"
set sd to current date
{date_setup}{time_setup}set seconds of sd to 0
if {time_given} then
-- время задано явно, ничего не двигаем
else
set sd to sd + 3600 - (minutes of sd) * 60
end if
set ed to sd + 3600
tell calendar 1 to make new event with properties {{summary:"{title_e}", start date:sd, end date:ed}}
end tell"#,
        time_given = parse_iso_time(time).is_some(),
        title_e = applescript_escape(title),
    );
    match osascript(&script) {
        Ok(_) => Ok(format!("Создал событие «{title}» в Календаре")),
        Err(e) => Err(format!(
            "Не удалось создать событие: {}",
            friendly_access_error(&e)
        )),
    }
}

// =========================================================================
// Музыка: нативная «Музыка» и вкладки браузеров
// =========================================================================

/// Какое приложение реально играет музыку: нативная «Музыка» или браузер.
///
/// Раньше управление шло только в «Музыку», и это давало молчаливую ложь:
/// `tell application "Music" to pause` завершается успешно, даже когда трек
/// не играет, поэтому Джарвис бодро отвечал «Поставил на паузу», а в браузере
/// (Яндекс Музыка, YouTube) музыка продолжала играть.
#[derive(Clone, Copy, PartialEq, Debug)]
enum MusicTarget {
    MusicApp,
    /// Браузер, в котором нашлось играющее медиа.
    BrowserTab(Browser),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Browser {
    Chrome,
    Safari,
    Edge,
    Brave,
    Arc,
    Yandex,
    Firefox,
}

impl Browser {
    /// Имя процесса для pgrep.
    fn process(self) -> &'static str {
        match self {
            Browser::Chrome => "Google Chrome",
            Browser::Safari => "Safari",
            Browser::Edge => "Microsoft Edge",
            Browser::Brave => "Brave Browser",
            Browser::Arc => "Arc",
            Browser::Yandex => "Yandex",
            Browser::Firefox => "firefox",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Browser::Chrome => "Chrome",
            Browser::Safari => "Safari",
            Browser::Edge => "Edge",
            Browser::Brave => "Brave",
            Browser::Arc => "Arc",
            Browser::Yandex => "Яндекс Браузер",
            Browser::Firefox => "Firefox",
        }
    }

    /// AppleScript, который отдаёт число играющих элементов медиа на активной
    /// вкладке. `execute ... javascript` поддерживают только WebKit/Chromium;
    /// Firefox так управлять нельзя, поэтому он только для честного отказа.
    fn script(self, js: &str) -> String {
        let js = js.replace('\\', "\\\\").replace('"', "\\\"");
        match self {
            Browser::Chrome => format!(
                r#"tell application "Google Chrome" to execute front window's active tab javascript "{js}""#
            ),
            Browser::Edge => format!(
                r#"tell application "Microsoft Edge" to execute front window's active tab javascript "{js}""#
            ),
            Browser::Brave => format!(
                r#"tell application "Brave Browser" to execute front window's active tab javascript "{js}""#
            ),
            Browser::Arc => format!(
                r#"tell application "Arc" to execute front window's active tab javascript "{js}""#
            ),
            Browser::Yandex => format!(
                r#"tell application "Yandex" to execute front window's active tab javascript "{js}""#
            ),
            // Safari отвечает объектом, а osascript ждёт текст — приводим к строке.
            Browser::Safari => format!(
                r#"tell application "Safari" to do JavaScript "{js}" in front document as text"#
            ),
            Browser::Firefox => String::new(),
        }
    }
}

/// Сколько элементов медиа сейчас играют в указанном браузере.
fn media_playing_count(browser: Browser) -> Option<usize> {
    if browser == Browser::Firefox || !app_running(browser.process()) {
        return None;
    }
    let out = osascript(&browser.script(
        "(function(){var m=document.querySelectorAll('video,audio');var n=0;\
         for(var i=0;i<m.length;i++){if(!m[i].paused&&!m[i].ended)n++}return n})()",
    ))
    .ok()?;
    out.trim().parse().ok()
}

/// Управляет медиа в браузере. `Some(Ok(count))` — сработало, вернулось число
/// переключённых элементов. `Some(Err)` — браузер нашёлся, но управление не
/// разрешено (нужна автоматизация или JavaScript из AppleScript).
fn control_browser_media(browser: Browser, verb: &str) -> Option<Result<usize, String>> {
    if browser == Browser::Firefox || !app_running(browser.process()) {
        return None;
    }
    let js = match verb {
        "pause" => {
            "(function(){var m=document.querySelectorAll('video,audio');var n=0;\
                  for(var i=0;i<m.length;i++){if(!m[i].paused&&!m[i].ended){m[i].pause();n++}}return n})()"
        }
        "stop" => {
            "(function(){var m=document.querySelectorAll('video,audio');var n=0;\
                  for(var i=0;i<m.length;i++){m[i].pause();try{m[i].currentTime=0}catch(e){}n++}return n})()"
        }
        _ => {
            "(function(){var m=document.querySelectorAll('video,audio');var n=0;\
              for(var i=0;i<m.length;i++){m[i].play();n++}return n})()"
        }
    };
    match osascript(&browser.script(js)) {
        Ok(out) => Some(
            out.trim()
                .parse()
                .map_err(|_| format!("ответ браузера: {out}")),
        ),
        Err(e) => Some(Err(friendly_access_error(&e))),
    }
}

/// Играет ли что-то прямо сейчас и где: «Музыка» или вкладка браузера.
fn current_music_target() -> Result<Option<MusicTarget>, String> {
    let mut blocked: Option<String> = None;
    for browser in [
        Browser::Chrome,
        Browser::Safari,
        Browser::Edge,
        Browser::Brave,
        Browser::Arc,
        Browser::Yandex,
    ] {
        if !app_running(browser.process()) {
            continue;
        }
        match media_playing_count(browser) {
            Some(0) => {}
            Some(_) => return Ok(Some(MusicTarget::BrowserTab(browser))),
            None => {
                // Причину неудачи читаем отдельным запросом: она решает, что
                // именно сказать пользователю.
                let reason = match osascript(&browser.script("1")) {
                    Ok(_) => format!(
                        "не удалось прочитать состояние вкладки в {}",
                        browser.title()
                    ),
                    Err(e) if is_permission_error(&e) => permission_hint(),
                    Err(e) => friendly_access_error(&e),
                };
                if blocked.is_none() {
                    blocked = Some(reason);
                }
            }
        }
    }

    if app_running("Music") {
        match osascript(r#"tell application "Music" to get player state as text"#) {
            Ok(state) => {
                let state = state.trim().to_string();
                if matches!(state.as_str(), "playing" | "paused") {
                    return Ok(Some(MusicTarget::MusicApp));
                }
                // Приложение открыто, но трека нет: это не источник, и
                // сообщать о нём как о музыке нельзя.
            }
            Err(e) => {
                let reason = if is_permission_error(&e) {
                    permission_hint()
                } else {
                    friendly_access_error(&e)
                };
                if blocked.is_none() {
                    blocked = Some(reason);
                }
            }
        }
    }

    match blocked {
        Some(reason) => Err(format!(
            "Музыка может играть в браузере или в «Музыке», но управлять не вышло: {reason}"
        )),
        None => Ok(None),
    }
}

/// Активное приложение для наблюдения: имя frontmost-процесса.
pub fn active_window() -> Option<(String, String)> {
    let name = active_process().ok().flatten()?;
    Some((name.clone(), name))
}

/// Статус плеера для наблюдения: Apple Music, если запущен.
pub fn music_status() -> Option<String> {
    if app_running("Music") {
        osascript(r#"tell application "Music" to get player state as text"#)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    } else {
        None
    }
}

/// Действие над плеером. «Стоп» — это пауза (для просьбы «выключи музыку»
/// пауза и остановка означают одно и то же). Состояние читается до и после
/// команды: успешный osascript ещё не значит, что плеер остановился.
pub fn music_control(action: &str) -> Result<String, String> {
    let pause_like = matches!(action, "pause" | "stop");
    let verb = match action {
        "play" => "play",
        "pause" | "stop" => "pause",
        "next" => "next",
        "previous" => "previous",
        other => return Err(format!("Не понимаю действие плеера: {other}")),
    };

    // Смотрим состояние ДО действия: иначе получается ложный успех, когда
    // плеер и так стоял, а пользователь ждал, что музыка замолчит.
    let target = current_music_target()?;
    let music_state = if app_running("Music") {
        osascript(r#"tell application "Music" to get player state as text"#)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    };

    match target {
        Some(MusicTarget::MusicApp) => {
            if pause_like && music_state.as_deref() == Some("paused") {
                return Err("Музыка уже на паузе".to_string());
            }
            if action == "play" && music_state.as_deref() == Some("playing") {
                return Err("Музыка уже играет".to_string());
            }
            let script = format!(r#"tell application "Music" to {verb}"#);
            match osascript(&script) {
                // Читаем состояние после команды: успешный osascript ещё не
                // значит, что плеер действительно остановился.
                Ok(_) => {
                    let after =
                        osascript(r#"tell application "Music" to get player state as text"#)
                            .ok()
                            .map(|s| s.trim().to_string());
                    let changed = after.as_deref() != music_state.as_deref();
                    if !changed {
                        return Err(format!(
                            "Команда ушла в «Музыку», но состояние не изменилось ({})",
                            after.as_deref().unwrap_or("неизвестно")
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
                Err(e) => Err(format!(
                    "Не удалось управлять Музыкой: {}",
                    friendly_access_error(&e)
                )),
            }
        }

        Some(MusicTarget::BrowserTab(browser)) => {
            let name = browser.title();
            match control_browser_media(browser, verb) {
                Some(Ok(n)) if n > 0 => {
                    let said = match action {
                        "play" => format!("Включаю музыку в {name}"),
                        "pause" | "stop" => format!("Поставил на паузу в {name}"),
                        "next" => format!("Переключаю трек в {name}"),
                        _ => format!("Предыдущий трек в {name}"),
                    };
                    Ok(said)
                }
                Some(Ok(_)) => Err(format!("В {name} ничего не играет — останавливать нечего")),
                Some(Err(e)) => Err(format!(
                    "Музыка играет в {name}, но macOS не даёт им управлять: {e}. \
                     Системные настройки → Конфиденциальность → Автоматизация: разреши Jarvis \
                     управлять {name}; в самом браузере — «Вид → Разработка → Разрешить \
                     JavaScript из AppleScript»"
                )),
                None => Err(format!(
                    "«Музыка» запущена, но трек в ней не выбран, а управлять {name} не вышло"
                )),
            }
        }

        None => {
            // Ничего не играет. Если приложение открыто — говорим прямо, что
            // останавливать нечего, вместо бодрого «Поставил на паузу».
            let hint = if app_running("Music") {
                "В «Музыке» не выбран трек и в браузере ничего не играет"
            } else {
                "Ни «Музыка», ни браузер сейчас ничего не играют"
            };
            Err(format!("{hint} — останавливать нечего"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_maps_common_names() {
        assert_eq!(alias("хром"), Some("Google Chrome"));
        assert_eq!(alias("гугл хром"), Some("Google Chrome"));
        assert_eq!(alias("браузер"), Some("Safari"));
        assert_eq!(alias("неизвестное приложение"), None);
        // «яндекс карты» — сайт, а не Apple Maps: суффикс «карты» через
        // границу слова не должен совпадать с алиасом.
        assert_eq!(alias("яндекс карты"), None);
        assert_eq!(alias("карты"), Some("Maps"));
    }

    #[test]
    fn iso_parsers_reject_garbage() {
        assert_eq!(parse_iso_date("2026-10-04"), Some((2026, 10, 4)));
        assert_eq!(parse_iso_date("завтра"), None);
        assert_eq!(parse_iso_date("2026-13-04"), None);
        assert_eq!(parse_iso_time("9:41"), Some((9, 41)));
        assert_eq!(parse_iso_time("25:00"), None);
        assert_eq!(parse_iso_time("утром"), None);
    }

    #[test]
    fn window_error_is_reported() {
        // Раньше неудача выглядела как успех, и модель повторяла команду.
        // В консольной сборке к «не найдено» может примешиваться отказ
        // разрешений — оба исхода честно провальные, оба должны сработать.
        let out = window_control("restore", Some("приложения-нет-такого"));
        assert!(
            out.contains("не найден") || out.contains("Не могу управлять окнами"),
            "получено: {out}"
        );
        assert!(!out.starts_with("готово"), "получено: {out}");
    }

    #[test]
    fn osascript_error_keeps_message() {
        // «this is not applescript» компилируется как сравнение и обычно
        // выполняется успешно (unwrap_err падал через раз). Нужен скрипт,
        // который гарантированно возбуждает ошибку выполнения.
        let err = osascript("error \"тест\"").unwrap_err();
        assert!(!err.is_empty());
        assert!(!err.starts_with("osascript завершился"), "получено: {err}");
    }

    #[test]
    fn browser_media_script_matches_the_browser() {
        // Chromium-браузеры получают javascript в активной вкладке, Safari —
        // другой синтаксис и приведение к тексту (иначе osascript отдаёт объект).
        let chrome = Browser::Chrome.script("return 1");
        assert!(chrome.contains("Google Chrome"), "{chrome}");
        assert!(chrome.contains("active tab javascript"), "{chrome}");
        let safari = Browser::Safari.script("return 1");
        assert!(safari.contains("do JavaScript"), "{safari}");
        assert!(safari.contains("as text"), "{safari}");
        // Firefox управлять так нельзя — скрипта нет, и мы обязаны это знать.
        assert!(Browser::Firefox.script("return 1").is_empty());
    }

    #[test]
    fn media_control_skips_browsers_that_are_not_running() {
        // Не запущенный браузер = None: osascript не зовём и чужую музыку
        // не трогаем, даже если она играет где-то ещё. Safari может успеть
        // запуститься между внешней и внутренней проверкой (живая машина) —
        // тогда вызов легитимен, и это не повод падать.
        if !app_running("Safari") {
            let outcome = control_browser_media(Browser::Safari, "pause");
            assert!(
                outcome.is_none() || app_running("Safari"),
                "Safari не запущен, но osascript всё равно звали: {outcome:?}"
            );
        }
        // Firefox не управляется никогда, даже если запущен.
        assert!(control_browser_media(Browser::Firefox, "pause").is_none());
    }

    /// Регрессия на живой случай: «поставь на паузу» при пустой «Музыке»
    /// отвечало «Поставил на паузу», хотя музыка продолжала играть в браузере.
    /// Ложный успех недопустим — лучше честное «останавливать нечего».
    #[test]
    fn music_control_never_fakes_success() {
        // Если что-то играет (или источник не читается), проверять ложный
        // успех нельзя: тест не должен трогать чужую музыку.
        match current_music_target() {
            Ok(None) => {}
            Ok(Some(target)) => {
                eprintln!("⚠️ играет {target:?} — проверка ложного успеха пропущена");
                return;
            }
            Err(reason) => {
                eprintln!("⚠️ источник не читается ({reason}) — проверка пропущена");
                return;
            }
        }
        let pause = music_control("pause");
        assert!(pause.is_err(), "пауза без музыки не может быть успехом");
        assert!(
            pause.unwrap_err().contains("нечего"),
            "нужна честная причина: нечего"
        );
        assert!(
            music_control("stop").is_err(),
            "стоп без музыки не может быть успехом"
        );
    }
}