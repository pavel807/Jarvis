//! Общий интерфейс управления клавиатурой и мышью.
//!
//! Использует штатные средства ОС или небольшие системные утилиты, не
//! добавляя тяжёлых C/C++ зависимостей в Rust-ядро. На macOS нужен cliclick,
//! на Linux — xdotool (X11) либо ydotool + ydotoold (Wayland), Windows
//! использует Win32 SendInput через встроенный PowerShell.

use serde::Serialize;
use std::process::{Command, Output};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DesktopAction {
    TypeText { text: String },
    PressKey { key: String },
    Hotkey { keys: Vec<String> },
    Click {
        x: i32,
        y: i32,
        button: MouseButton,
        count: u8,
    },
    Move { x: i32, y: i32 },
    Scroll { direction: ScrollDirection, amount: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

/// Выполняет одно действие в текущем активном окне/на рабочем столе.
pub fn execute(action: &DesktopAction) -> Result<String, String> {
    validate(action)?;
    #[cfg(target_os = "macos")]
    {
        return execute_macos(action);
    }
    #[cfg(target_os = "linux")]
    {
        return execute_linux(action);
    }
    #[cfg(target_os = "windows")]
    {
        return execute_windows(action);
    }
    #[allow(unreachable_code)]
    Err("Управление вводом поддерживается на macOS, Linux и Windows".into())
}

fn validate(action: &DesktopAction) -> Result<(), String> {
    match action {
        DesktopAction::TypeText { text } => {
            if text.is_empty() {
                return Err("Текст для ввода пустой".into());
            }
            if text.len() > 20_000 {
                return Err("За один вызов можно ввести не более 20 000 байт текста".into());
            }
            if text.contains('\0') {
                return Err("Текст содержит недопустимый нулевой символ".into());
            }
        }
        DesktopAction::PressKey { key } => {
            let normalized = normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))?;
            if is_modifier(&normalized) {
                return Err("Модификатор нельзя нажать отдельно: используй hotkey с основной клавишей".into());
            }
        }
        DesktopAction::Hotkey { keys } => {
            if !(2..=6).contains(&keys.len()) {
                return Err("Для сочетания укажи от двух до шести клавиш".into());
            }
            for key in keys {
                if normalize_key(key).is_none() {
                    return Err(format!("Неизвестная клавиша в сочетании: «{key}»"));
                }
            }
            if keys.iter().filter(|k| !is_modifier(k)).count() != 1 {
                return Err("В сочетании должны быть модификаторы и ровно одна основная клавиша".into());
            }
        }
        DesktopAction::Click { x, y, count, .. } => {
            validate_coordinates(*x, *y)?;
            if !(1..=3).contains(count) {
                return Err("Число кликов должно быть от одного до трёх".into());
            }
        }
        DesktopAction::Move { x, y } => validate_coordinates(*x, *y)?,
        DesktopAction::Scroll { amount, .. } => {
            if !(1..=50).contains(amount) {
                return Err("Прокрутка должна быть от одного до пятидесяти шагов".into());
            }
        }
    }
    Ok(())
}

fn validate_coordinates(x: i32, y: i32) -> Result<(), String> {
    if !(-32_768..=32_768).contains(&x) || !(-32_768..=32_768).contains(&y) {
        return Err("Координаты должны быть в диапазоне от −32768 до 32768".into());
    }
    Ok(())
}

fn compact_key(key: &str) -> String {
    key.trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Общая нормализация названий клавиш для русских/английских вариантов.
fn normalize_key(key: &str) -> Option<String> {
    let raw = compact_key(key);
    if raw.is_empty() {
        return None;
    }
    let normalized = match raw.as_str() {
        "ctrl" | "control" | "контрол" | "контрл" => "ctrl",
        "shift" | "шифт" => "shift",
        "alt" | "option" | "опция" | "альт" => "alt",
        "cmd" | "command" | "meta" | "win" | "windows" | "super" | "команда" => "super",
        "enter" | "return" | "ввод" | "энтер" => "enter",
        "esc" | "escape" | "эскейп" => "esc",
        "tab" | "таб" => "tab",
        "space" | "пробел" => "space",
        "backspace" | "back" => "backspace",
        "delete" | "del" => "delete",
        "insert" | "ins" => "insert",
        "home" => "home",
        "end" => "end",
        "pageup" | "pgup" => "pageup",
        "pagedown" | "pgdn" => "pagedown",
        "up" | "arrowup" | "вверх" | "стрелкавверх" => "up",
        "down" | "arrowdown" | "вниз" | "стрелкавниз" => "down",
        "left" | "arrowleft" | "влево" | "стрелкавлево" => "left",
        "right" | "arrowright" | "вправо" | "стрелкавправо" => "right",
        "capslock" => "capslock",
        "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8" | "f9" | "f10"
        | "f11" | "f12" => raw.as_str(),
        _ if raw.chars().count() == 1 && raw.chars().all(|c| c.is_ascii_alphanumeric()) => {
            return Some(raw);
        }
        _ => return None,
    };
    Some(normalized.to_string())
}

fn is_modifier(key: &str) -> bool {
    matches!(normalize_key(key).as_deref(), Some("ctrl" | "shift" | "alt" | "super"))
}

fn command_output(bin: &str, args: &[String]) -> Result<String, String> {
    let output = Command::new(bin)
        .args(args)
        .output()
        .map_err(|e| format!("Не удалось запустить {bin}: {e}"))?;
    output_text(bin, output)
}

fn output_text(bin: &str, output: Output) -> Result<String, String> {
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !stderr.is_empty() && (stderr.to_lowercase().contains("permission") || stderr.to_lowercase().contains("accessibility")) {
            return Err(format!("{bin}: {stderr}"));
        }
        Ok(if stdout.is_empty() { "Команда ввода отправлена системе".into() } else { stdout })
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let reason = if !stderr.is_empty() { stderr } else { stdout };
        Err(if reason.is_empty() {
            format!("{bin} завершился с кодом {}", output.status.code().unwrap_or(-1))
        } else {
            format!("{bin}: {}", reason.chars().take(240).collect::<String>())
        })
    }
}

#[cfg(target_os = "macos")]
fn execute_macos(action: &DesktopAction) -> Result<String, String> {
    let bin = ["/opt/homebrew/bin/cliclick", "/usr/local/bin/cliclick"]
        .iter()
        .find(|p| std::path::Path::new(p).is_file())
        .map(|s| (*s).to_string())
        .or_else(|| find_on_path("cliclick"))
        .ok_or_else(|| "Для мыши и клавиатуры macOS установи cliclick: brew install cliclick".to_string())?;

    let mut args: Vec<String> = vec!["-w".into(), "40".into()];
    match action {
        DesktopAction::TypeText { text } => args.push(format!("t:{text}")),
        DesktopAction::PressKey { key } => args.push(cliclick_press(key)?),
        DesktopAction::Hotkey { keys } => {
            let mut modifiers = Vec::new();
            let mut primary = None;
            for key in keys {
                let normalized = normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))?;
                if is_modifier(&normalized) {
                    modifiers.push(cliclick_modifier(&normalized)?.to_string());
                } else {
                    primary = Some(normalized);
                }
            }
            args.push(format!("kd:{}", modifiers.join(",")));
            let primary = primary.ok_or("В сочетании отсутствует основная клавиша")?;
            args.push(cliclick_press_normalized(&primary)?);
            args.push(format!("ku:{}", modifiers.join(",")));
        }
        DesktopAction::Click { x, y, button, count } => {
            match button {
                MouseButton::Left if *count == 1 => args.push(format!("c:{x},{y}")),
                MouseButton::Left if *count == 2 => args.push(format!("dc:{x},{y}")),
                MouseButton::Left => args.push(format!("tc:{x},{y}")),
                MouseButton::Right => {
                    for _ in 0..*count {
                        args.push(format!("rc:{x},{y}"));
                    }
                }
                MouseButton::Middle => return Err("cliclick не предоставляет надёжную команду для средней кнопки".into()),
            }
        }
        DesktopAction::Move { x, y } => args.push(format!("m:{x},{y}")),
        DesktopAction::Scroll { direction, amount } => {
            let key = match direction {
                ScrollDirection::Up => "page-up",
                ScrollDirection::Down => "page-down",
                ScrollDirection::Left | ScrollDirection::Right => {
                    return Err("cliclick не поддерживает горизонтальную прокрутку; используй Page Up / Page Down".into());
                }
            };
            for _ in 0..*amount {
                args.push(format!("kp:{key}"));
            }
        }
    }
    command_output(&bin, &args).map(|_| match action {
        // Конкретика в обратной связи: GUI-агент по ней понимает, что уже
        // сделано, и не повторяет одно и то же действие.
        DesktopAction::TypeText { text } => {
            let shown: String = text.chars().take(80).collect();
            format!("Введён текст: {shown}")
        }
        DesktopAction::PressKey { key } => format!("Нажата клавиша {key}"),
        DesktopAction::Hotkey { keys } => format!("Нажато сочетание {}", keys.join("+")),
        DesktopAction::Click { x, y, button, count } => format!(
            "Клик ({button:?}, x{count}) по координатам {x},{y}"
        ),
        DesktopAction::Move { x, y } => format!("Курсор перемещён в {x},{y}"),
        DesktopAction::Scroll { .. } => "Прокрутка отправлена как Page Up / Page Down".into(),
    }).map_err(|e| {
        let lower = e.to_lowercase();
        if lower.contains("not authorized") || lower.contains("accessibility") || lower.contains("assistive") {
            format!("macOS заблокировала ввод. Разреши Jarvis или cliclick в Системные настройки → Конфиденциальность и безопасность → Универсальный доступ. Подробности: {e}")
        } else {
            e
        }
    })
}

#[cfg(target_os = "macos")]
fn cliclick_modifier(key: &str) -> Result<&'static str, String> {
    match key {
        "ctrl" => Ok("ctrl"),
        "shift" => Ok("shift"),
        "alt" => Ok("alt"),
        "super" => Ok("cmd"),
        _ => Err(format!("«{key}» не является модификатором")),
    }
}

#[cfg(target_os = "macos")]
fn cliclick_press(key: &str) -> Result<String, String> {
    let normalized = normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))?;
    cliclick_press_normalized(&normalized)
}

#[cfg(target_os = "macos")]
fn cliclick_press_normalized(key: &str) -> Result<String, String> {
    if key.len() == 1 && key.as_bytes()[0].is_ascii_alphanumeric() {
        return Ok(format!("t:{key}"));
    }
    let name = match key {
        "ctrl" => "control",
        "super" => "command",
        "alt" => "option",
        "esc" => "esc",
        "pageup" => "page-up",
        "pagedown" => "page-down",
        other => other,
    };
    Ok(format!("kp:{name}"))
}

#[cfg(target_os = "macos")]
fn find_on_path(bin: &str) -> Option<String> {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths)
                .map(|p| p.join(bin))
                .find(|p| p.is_file())
                .map(|p| p.to_string_lossy().to_string())
        })
        .flatten()
}

#[cfg(target_os = "linux")]
fn execute_linux(action: &DesktopAction) -> Result<String, String> {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let ydotool = find_binary(&["ydotool", "/usr/bin/ydotool", "/usr/local/bin/ydotool"]);
    let xdotool = find_binary(&["xdotool", "/usr/bin/xdotool", "/usr/local/bin/xdotool"]);

    if wayland {
        if let Some(bin) = ydotool {
            return execute_ydotool(&bin, action);
        }
        return Err("Обнаружен Wayland. Для управления мышью и клавиатурой установи ydotool и запусти ydotoold; службе нужен доступ к /dev/uinput. xdotool в Wayland не управляет всем рабочим столом.".into());
    }
    if let Some(bin) = xdotool {
        return execute_xdotool(&bin, action);
    }
    if let Some(bin) = ydotool {
        return execute_ydotool(&bin, action);
    }
    Err("Не найдена утилита управления вводом. Для X11 установи xdotool; для Wayland — ydotool + ydotoold".into())
}

#[cfg(target_os = "linux")]
fn execute_xdotool(bin: &str, action: &DesktopAction) -> Result<String, String> {
    let mut args: Vec<String> = Vec::new();
    match action {
        DesktopAction::TypeText { text } => return paste_text_linux(bin, true, text),
        DesktopAction::PressKey { key } => args.extend(["key".into(), "--clearmodifiers".into(), xdotool_key(key)?]),
        DesktopAction::Hotkey { keys } => {
            let joined = keys.iter().map(|k| xdotool_key(k)).collect::<Result<Vec<_>, _>>()?.join("+");
            args.extend(["key".into(), "--clearmodifiers".into(), joined]);
        }
        DesktopAction::Click { x, y, button, count } => {
            args.extend(["mousemove".into(), "--sync".into(), x.to_string(), y.to_string()]);
            let button = match button { MouseButton::Left => "1", MouseButton::Middle => "2", MouseButton::Right => "3" };
            args.extend(["click".into(), "--repeat".into(), count.to_string(), "--delay".into(), "100".into(), button.into()]);
        }
        DesktopAction::Move { x, y } => args.extend(["mousemove".into(), "--sync".into(), x.to_string(), y.to_string()]),
        DesktopAction::Scroll { direction, amount } => {
            let button = match direction { ScrollDirection::Up => "4", ScrollDirection::Down => "5", ScrollDirection::Left => "6", ScrollDirection::Right => "7" };
            args.extend(["click".into(), "--repeat".into(), amount.to_string(), "--delay".into(), "60".into(), button.into()]);
        }
    }
    command_output(bin, &args).map(|_| "Ввод отправлен активному окну".into())
}


/// Вставляет Unicode-текст через системный буфер, а не раскладко-зависимые
/// keycodes: `ydotool type` и `xdotool type` не могут гарантировать кириллицу
/// при произвольной раскладке. Предыдущее текстовое содержимое восстанавливаем.
#[cfg(target_os = "linux")]
fn paste_text_linux(bin: &str, x11: bool, text: &str) -> Result<String, String> {
    let previous = crate::platform::clipboard_read().ok_or_else(|| {
        "Для надёжного ввода Unicode на Linux нужен текстовый буфер обмена. Установи wl-clipboard (Wayland) либо xclip/xsel (X11); текущий буфер не удалось прочитать, поэтому ввод отменён без его перезаписи.".to_string()
    })?;
    if !crate::platform::clipboard_write(text) {
        return Err("Не удалось временно записать текст в буфер обмена; ввод не выполнен".into());
    }

    let paste_args: Vec<String> = if x11 {
        vec!["key".into(), "--clearmodifiers".into(), "shift+Insert".into()]
    } else {
        // KEY_LEFTSHIFT=42, KEY_INSERT=110. Shift+Insert — системная
        // клавиатурная комбинация вставки, не зависящая от раскладки букв.
        vec!["key".into(), "42:1".into(), "110:1".into(), "110:0".into(), "42:0".into()]
    };
    let paste = command_output(bin, &paste_args);
    if paste.is_ok() {
        // Даём приложению обработать запрос selection/clipboard, прежде
        // чем возвращать предыдущее содержимое.
        std::thread::sleep(std::time::Duration::from_millis(450));
    }
    let restored = crate::platform::clipboard_write(&previous);

    match (paste, restored) {
        (Ok(_), true) => Ok("Unicode-текст вставлен; предыдущее текстовое содержимое буфера восстановлено".into()),
        (Ok(_), false) => Ok("Unicode-текст вставлен, но не удалось восстановить предыдущее содержимое буфера обмена".into()),
        (Err(e), true) => Err(e),
        (Err(e), false) => Err(format!("Вставка не удалась ({e}); также не удалось восстановить буфер обмена")),
    }
}

#[cfg(target_os = "linux")]
fn xdotool_key(key: &str) -> Result<String, String> {
    let normalized = normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))?;
    let mapped = match normalized.as_str() {
        "ctrl" => "ctrl",
        "super" => "Super_L",
        "alt" => "alt",
        "esc" => "Escape",
        "enter" => "Return",
        "backspace" => "BackSpace",
        "delete" => "Delete",
        "pageup" => "Page_Up",
        "pagedown" => "Page_Down",
        "up" => "Up",
        "down" => "Down",
        "left" => "Left",
        "right" => "Right",
        "capslock" => "Caps_Lock",
        other => other,
    };
    Ok(mapped.to_string())
}

#[cfg(target_os = "linux")]
fn execute_ydotool(bin: &str, action: &DesktopAction) -> Result<String, String> {
    let mut args: Vec<String> = Vec::new();
    match action {
        DesktopAction::TypeText { text } => return paste_text_linux(bin, false, text),
        DesktopAction::PressKey { key } => {
            args.push("key".into());
            args.extend(ydotool_key_event(key)?.split_whitespace().map(str::to_string));
        }
        DesktopAction::Hotkey { keys } => {
            let primary = keys.iter().find(|k| !is_modifier(k)).ok_or("В сочетании отсутствует основная клавиша")?;
            for key in keys.iter().filter(|k| is_modifier(k)) {
                args.push(format!("{}:1", ydotool_keycode(key)?));
            }
            let code = ydotool_keycode(primary)?;
            args.extend([format!("{code}:1"), format!("{code}:0")]);
            for key in keys.iter().filter(|k| is_modifier(k)).rev() {
                args.push(format!("{}:0", ydotool_keycode(key)?));
            }
            let mut call = vec!["key".to_string()];
            call.extend(args);
            return command_output(bin, &call).map(|_| "Сочетание клавиш отправлено активному окну".into());
        }
        DesktopAction::Click { x, y, button, count } => {
            args.extend(["mousemove".into(), "--absolute".into(), x.to_string(), y.to_string()]);
            command_output(bin, &args)?;
            args.clear();
            let code = match button { MouseButton::Left => "0xC0", MouseButton::Right => "0xC1", MouseButton::Middle => "0xC2" };
            args.extend(["click".into(), "--repeat".into(), count.to_string(), code.into()]);
        }
        DesktopAction::Move { x, y } => args.extend(["mousemove".into(), "--absolute".into(), x.to_string(), y.to_string()]),
        DesktopAction::Scroll { .. } => return Err("Установленная ветка ydotool не предоставляет надёжную команду колёсика; используй X11/xdotool или Page Up/Page Down".into()),
    }
    command_output(bin, &args).map(|_| "Ввод отправлен виртуальному устройству".into()).map_err(|e| {
        if e.to_lowercase().contains("socket") || e.to_lowercase().contains("connect") {
            format!("Не удалось подключиться к ydotoold. Запусти службу ydotoold и проверь доступ к /dev/uinput: {e}")
        } else { e }
    })
}

#[cfg(target_os = "linux")]
fn ydotool_key_event(key: &str) -> Result<String, String> {
    let code = ydotool_keycode(key)?;
    Ok(format!("{code}:1 {code}:0"))
}

#[cfg(target_os = "linux")]
fn ydotool_keycode(key: &str) -> Result<u16, String> {
    let normalized = normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))?;
    let code = match normalized.as_str() {
        "esc" => 1,
        "enter" => 28,
        "ctrl" => 29,
        "shift" => 42,
        "alt" => 56,
        "super" => 125,
        "tab" => 15,
        "space" => 57,
        "backspace" => 14,
        "delete" => 111,
        "insert" => 110,
        "home" => 102,
        "end" => 107,
        "pageup" => 104,
        "pagedown" => 109,
        "up" => 103,
        "left" => 105,
        "right" => 106,
        "down" => 108,
        "capslock" => 58,
        "a" => 30, "b" => 48, "c" => 46, "d" => 32, "e" => 18, "f" => 33,
        "g" => 34, "h" => 35, "i" => 23, "j" => 36, "k" => 37, "l" => 38,
        "m" => 50, "n" => 49, "o" => 24, "p" => 25, "q" => 16, "r" => 19,
        "s" => 31, "t" => 20, "u" => 22, "v" => 47, "w" => 17, "x" => 45,
        "y" => 21, "z" => 44,
        "1" => 2, "2" => 3, "3" => 4, "4" => 5, "5" => 6,
        "6" => 7, "7" => 8, "8" => 9, "9" => 10, "0" => 11,
        "f1" => 59, "f2" => 60, "f3" => 61, "f4" => 62, "f5" => 63, "f6" => 64,
        "f7" => 65, "f8" => 66, "f9" => 67, "f10" => 68, "f11" => 87, "f12" => 88,
        _ => return Err(format!("Клавиша «{key}» не поддерживается ydotool")),
    };
    Ok(code)
}

#[cfg(target_os = "linux")]
fn find_binary(candidates: &[&str]) -> Option<String> {
    for candidate in candidates {
        if candidate.contains('/') {
            if std::path::Path::new(candidate).is_file() {
                return Some((*candidate).to_string());
            }
        } else if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                let full = dir.join(candidate);
                if full.is_file() {
                    return Some(full.to_string_lossy().to_string());
                }
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn execute_windows(action: &DesktopAction) -> Result<String, String> {
    // На Windows нормализуем синонимы (Command/Return/стрелка-вниз и т.п.)
    // до передачи в PowerShell: Win32 получает только канонические имена.
    let canonical = match action {
        DesktopAction::PressKey { key } => DesktopAction::PressKey {
            key: normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))?,
        },
        DesktopAction::Hotkey { keys } => DesktopAction::Hotkey {
            keys: keys.iter().map(|key| normalize_key(key).ok_or_else(|| format!("Неизвестная клавиша «{key}»"))).collect::<Result<Vec<_>, _>>()?,
        },
        other => other.clone(),
    };
    let json = serde_json::to_string(&canonical).map_err(|e| format!("Не удалось сериализовать действие: {e}"))?;
    let encoded = base64(json.as_bytes());
    let script = format!(
        r#"
$ErrorActionPreference = 'Stop'
$payload = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}'))
$a = ConvertFrom-Json -InputObject $payload
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
[StructLayout(LayoutKind.Sequential)] public struct JDKBDINPUT {{ public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public UIntPtr dwExtraInfo; }}
[StructLayout(LayoutKind.Sequential)] public struct JDMOUSEINPUT {{ public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public UIntPtr dwExtraInfo; }}
[StructLayout(LayoutKind.Explicit)] public struct JDUNION {{ [FieldOffset(0)] public JDKBDINPUT ki; [FieldOffset(0)] public JDMOUSEINPUT mi; }}
[StructLayout(LayoutKind.Sequential)] public struct JDINPUT {{ public uint type; public JDUNION U; }}
public static class JarvisNativeInput {{
 [DllImport("user32.dll", SetLastError=true)] public static extern uint SendInput(uint nInputs, JDINPUT[] pInputs, int cbSize);
 [DllImport("user32.dll", SetLastError=true)] public static extern bool SetCursorPos(int X, int Y);
 public static void Key(ushort vk, bool down) {{ var i = new JDINPUT(); i.type=1; i.U.ki=new JDKBDINPUT(); i.U.ki.wVk=vk; i.U.ki.dwFlags=down?0u:2u; Send(new JDINPUT[]{{i}}); }}
 public static void Unicode(char ch, bool down) {{ var i = new JDINPUT(); i.type=1; i.U.ki=new JDKBDINPUT(); i.U.ki.wScan=(ushort)ch; i.U.ki.dwFlags=4u | (down?0u:2u); Send(new JDINPUT[]{{i}}); }}
 public static void Mouse(uint flags, uint data) {{ var i = new JDINPUT(); i.type=0; i.U.mi=new JDMOUSEINPUT(); i.U.mi.mouseData=data; i.U.mi.dwFlags=flags; Send(new JDINPUT[]{{i}}); }}
 static void Send(JDINPUT[] a) {{ if (SendInput((uint)a.Length,a,Marshal.SizeOf(typeof(JDINPUT))) != (uint)a.Length) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error()); }}
}}
'@ | Out-Null
function Get-Vk([string]$key) {{
 $k = ($key -replace '[^a-zA-Z0-9]', '').ToLowerInvariant()
 switch ($k) {{
  'ctrl' {{ return 0x11 }} 'shift' {{ return 0x10 }} 'alt' {{ return 0x12 }} 'super' {{ return 0x5B }} 'command' {{ return 0x5B }} 'meta' {{ return 0x5B }} 'win' {{ return 0x5B }} 'windows' {{ return 0x5B }}
  'enter' {{ return 0x0D }} 'esc' {{ return 0x1B }} 'tab' {{ return 0x09 }} 'space' {{ return 0x20 }}
  'backspace' {{ return 0x08 }} 'delete' {{ return 0x2E }} 'insert' {{ return 0x2D }} 'home' {{ return 0x24 }} 'end' {{ return 0x23 }}
  'pageup' {{ return 0x21 }} 'pagedown' {{ return 0x22 }} 'up' {{ return 0x26 }} 'down' {{ return 0x28 }} 'left' {{ return 0x25 }} 'right' {{ return 0x27 }} 'capslock' {{ return 0x14 }}
  default {{ if ($k -match '^f([1-9]|1[0-2])$') {{ return (0x70 + [int]$Matches[1] - 1) }}; if ($k -match '^[a-z0-9]$') {{ return [int][char]$k.ToUpperInvariant() }}; throw "Неизвестная клавиша: $key" }}
 }}
}}
function Send-Key([string]$key) {{ $vk = [ushort](Get-Vk $key); [JarvisNativeInput]::Key($vk,$true); [JarvisNativeInput]::Key($vk,$false) }}
switch ($a.type) {{
 'type_text' {{ foreach ($ch in $a.text.ToCharArray()) {{ [JarvisNativeInput]::Unicode($ch,$true); [JarvisNativeInput]::Unicode($ch,$false) }} }}
 'press_key' {{ Send-Key $a.key }}
 'hotkey' {{ $mods=@(); $main=$null; foreach ($key in $a.keys) {{ $n=($key -replace '[^a-zA-Z0-9]','').ToLowerInvariant(); if ($n -in @('ctrl','shift','alt','super','command','meta','win','windows')) {{ $mods += $key }} else {{ $main=$key }} }}; foreach($key in $mods) {{ [JarvisNativeInput]::Key([ushort](Get-Vk $key),$true) }}; Send-Key $main; for($i=$mods.Count-1;$i -ge 0;$i--) {{ [JarvisNativeInput]::Key([ushort](Get-Vk $mods[$i]),$false) }} }}
 'move' {{ if(-not [JarvisNativeInput]::SetCursorPos([int]$a.x,[int]$a.y)) {{ throw 'SetCursorPos failed' }} }}
 'click' {{ if(-not [JarvisNativeInput]::SetCursorPos([int]$a.x,[int]$a.y)) {{ throw 'SetCursorPos failed' }}; $down=0x0002; $up=0x0004; if($a.button -eq 'right') {{ $down=0x0008; $up=0x0010 }} elseif($a.button -eq 'middle') {{ $down=0x0020; $up=0x0040 }}; for($i=0;$i -lt [int]$a.count;$i++) {{ [JarvisNativeInput]::Mouse($down,0); [JarvisNativeInput]::Mouse($up,0); Start-Sleep -Milliseconds 70 }} }}
 'scroll' {{ $flags=0x0800; $data=120 * [int]$a.amount; if($a.direction -eq 'down') {{ $data=-$data }} elseif($a.direction -eq 'left' -or $a.direction -eq 'right') {{ $flags=0x1000; if($a.direction -eq 'left') {{ $data=-$data }} }}; [JarvisNativeInput]::Mouse([uint32]$flags,[uint32]($data -band 0xFFFFFFFF)) }}
 default {{ throw "Неизвестное действие: $($a.type)" }}
}}
'OK'
"#
    );
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &script])
        .output()
        .map_err(|e| format!("Не удалось запустить PowerShell: {e}"))?;
    output_text("PowerShell/Win32", output).map(|_| "Ввод отправлен активному окну".into())
}

#[cfg(target_os = "windows")]
fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { TABLE[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[(n & 63) as usize] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_common_key_names() {
        assert_eq!(normalize_key("CTRL"), Some("ctrl".into()));
        assert_eq!(normalize_key("Command"), Some("super".into()));
        assert_eq!(normalize_key("стрелка-вниз"), Some("down".into()));
        assert_eq!(normalize_key("F12"), Some("f12".into()));
        assert_eq!(normalize_key("unknown-key"), None);
    }

    #[test]
    fn rejects_invalid_hotkeys_and_text() {
        assert!(validate(&DesktopAction::Hotkey { keys: vec!["CTRL".into()] }).is_err());
        assert!(validate(&DesktopAction::Hotkey { keys: vec!["CTRL".into(), "SHIFT".into()] }).is_err());
        assert!(validate(&DesktopAction::TypeText { text: "".into() }).is_err());
    }

    #[test]
    fn accepts_a_typical_desktop_workflow_key_chord() {
        assert!(validate(&DesktopAction::Hotkey { keys: vec!["CTRL".into(), "SHIFT".into(), "S".into()] }).is_ok());
    }
}
