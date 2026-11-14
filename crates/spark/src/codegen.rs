use crate::ast::{Expr, Stmt};
use std::collections::HashSet;
use std::fs;
use std::process::{Command, Stdio};
use tempfile::Builder;

/// Системная прелюдия для плагинов-автоматизаций: аргументы командной
/// строки, shell, файлы, уведомления. Все функции возвращают String —
/// кодоген Spark'а оборачивает любой вызов в `.to_string()`.
const PRELUDE: &str = r#"#[allow(dead_code)]
fn spark_arg(n: i64) -> String {
    std::env::args().nth(n.max(1) as usize).unwrap_or_default()
}
#[allow(dead_code)]
fn spark_env(name: impl AsRef<str>) -> String {
    std::env::var(name.as_ref()).unwrap_or_default()
}
#[allow(dead_code)]
fn spark_run(cmd: impl AsRef<str>) -> String {
    // Вывод пишем во временный ФАЙЛ, а не в пайп: команда вида
    // `gtk-launch …` заменяет себя запускаемым приложением, которое живёт
    // долго — пайп никогда не закроется и чтение зависнет навсегда.
    let tmp = std::env::temp_dir().join(format!("spark-run-{}", std::process::id()));
    if let Ok(f) = fs::File::create(&tmp) {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd.as_ref())
            .stdout(Stdio::from(f))
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn();
        if let Ok(mut child) = child {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while std::time::Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        let out = fs::read_to_string(&tmp).unwrap_or_default();
                        let _ = fs::remove_file(&tmp);
                        return out.trim().to_string();
                    }
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            // Не завершилась за 10 с — не убиваем: процесс мог стать
            // запущенным приложением. Честно сообщаем об этом.
            let _ = fs::remove_file(&tmp);
            return String::new();
        }
    }
    String::new()
}
/// Запустить команду без ожидания: для открытия приложений
/// (`launch("gtk-launch org.gnome.Calculator")`).
#[allow(dead_code)]
fn spark_launch(cmd: impl AsRef<str>) -> String {
    let _ = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd.as_ref())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn();
    "ok".to_string()
}
#[allow(dead_code)]
fn spark_read(path: impl AsRef<str>) -> String {
    fs::read_to_string(path.as_ref()).unwrap_or_default()
}
#[allow(dead_code)]
fn spark_write(path: impl AsRef<str>, text: impl AsRef<str>) -> String {
    match fs::write(path.as_ref(), text.as_ref()) {
        Ok(_) => "ok".to_string(),
        Err(e) => format!("ошибка: {e}"),
    }
}
#[allow(dead_code)]
fn spark_notify(title: impl AsRef<str>, text: impl AsRef<str>) -> String {
    let _ = std::process::Command::new("notify-send")
        .arg("-a").arg("Jarvis")
        .arg(title.as_ref())
        .arg(text.as_ref())
        .spawn();
    "ok".to_string()
}
/// Речь плагина: строка «SAY: …» на stdout Джарвис озвучит голосом.
#[allow(dead_code)]
fn spark_say(text: impl AsRef<str>) -> String {
    println!("SAY: {}", text.as_ref());
    "ok".to_string()
}
#[allow(dead_code)]
fn input() -> String {
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
    line.trim().to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// Автоматизация ОС — аналог возможностей AutoHotkey: клавиатура, мышь,
// окна, столы, громкость, яркость, буфер обмена, скриншоты, сеть.
// Кроссплатформенно: Linux (Wayland/Hyprland + X11), macOS, Windows.
// ═══════════════════════════════════════════════════════════════════════

/// Запустить команду, вернуть stdout. С таймаутом и без риска зависнуть
/// на пайпе (вывод через временный файл). Для hyprctl, pactl, curl и пр.
#[allow(dead_code)]
fn spark_out(bin: &str, args: &[&str]) -> String {
    let tmp = std::env::temp_dir().join(format!("spark-out-{}", std::process::id()));
    if let Ok(f) = fs::File::create(&tmp) {
        if let Ok(mut child) = std::process::Command::new(bin)
            .args(args)
            .stdout(Stdio::from(f))
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()
        {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
            while std::time::Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        let out = fs::read_to_string(&tmp).unwrap_or_default();
                        let _ = fs::remove_file(&tmp);
                        return out.trim().to_string();
                    }
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(25)),
                    Err(_) => break,
                }
            }
            // curl завис / зависла сеть — убиваем, это не GUI-процесс.
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(&tmp);
        }
    }
    String::new()
}

/// Сочетание клавиш: spark_send_hotkey("CTRL+SHIFT+T"). Модификаторы через плюс,
/// основная клавиша последняя. Работает на любой платформе.
#[allow(dead_code)]
fn spark_send_hotkey(keys: impl AsRef<str>) -> String {
    let parts: Vec<String> = keys
        .as_ref()
        .split('+')
        .map(|p| p.trim().to_lowercase())
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return "пустое сочетание".to_string();
    }
    let primary = parts.last().cloned().unwrap_or_default();
    let mods: Vec<String> = parts[..parts.len() - 1].to_vec();
    #[cfg(target_os = "linux")]
    {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        if wayland {
            // wtype: паузы 100 мс обязательны — композитор склеивает
            // синтетику и теряет модификаторы.
            let mut args: Vec<String> = Vec::new();
            for m in &mods {
                let m = match m.as_str() {
                    "control" | "ctrl" | "контрол" => "ctrl",
                    "shift" | "шифт" => "shift",
                    "alt" => "alt",
                    "super" | "win" | "meta" => "logo",
                    other => other,
                };
                args.push("-M".into());
                args.push(m.into());
                args.push("-s".into());
                args.push("100".into());
            }
            args.push("-s".into());
            args.push("100".into());
            args.push("-P".into());
            args.push(primary.clone());
            args.push("-s".into());
            args.push("100".into());
            args.push("-p".into());
            args.push(primary);
            for m in mods.iter().rev() {
                let m = match m.as_str() {
                    "control" | "ctrl" | "контрол" => "ctrl",
                    "shift" | "шифт" => "shift",
                    "alt" => "alt",
                    "super" | "win" | "meta" => "logo",
                    other => other,
                };
                args.push("-s".into());
                args.push("100".into());
                args.push("-m".into());
                args.push(m.into());
            }
            let strs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            spark_out("wtype", &strs);
            return "ok".to_string();
        }
        let combo = parts.join("+");
        spark_out("xdotool", &["key", &combo]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        // keystroke "t" using {command down, shift down}
        let mac_mods: Vec<String> = mods
            .iter()
            .map(|m| match m.as_str() {
                "ctrl" | "control" => "control".to_string(),
                "super" | "win" | "meta" => "command".to_string(),
                other => other.to_string(),
            })
            .collect();
        let mods_str = mac_mods
            .iter()
            .map(|m| format!("{m} down"))
            .collect::<Vec<_>>()
            .join(", ");
        let script = if mods_str.is_empty() {
            format!("tell application \"System Events\" to keystroke \"{primary}\"")
        } else {
            format!(
                "tell application \"System Events\" to keystroke \"{primary}\" using {{{mods_str}}}"
            )
        };
        spark_out("osascript", &["-e", &script]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        // PowerShell SendKeys: ^ = ctrl, + = shift, % = alt
        let mut combo = String::new();
        for m in &mods {
            combo.push_str(match m.as_str() {
                "ctrl" | "control" => "^",
                "shift" => "+",
                "alt" => "%",
                _ => "",
            });
        }
        combo.push_str(&primary);
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{}')",
            combo.replace('\'', "''")
        );
        spark_out("powershell", &["-NoProfile", "-Command", &script]);
        return "ok".to_string();
    }
}

/// Напечатать текст в активное окно: spark_type_text("привет").
/// На Wayland вводит юникод нативно, без буфера обмена.
#[allow(dead_code)]
fn spark_type_text(text: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        if wayland {
            spark_out("wtype", &[text.as_ref()]);
            return "ok".to_string();
        }
        spark_out("xdotool", &["type", "--", text.as_ref()]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        let t = text.as_ref().replace('"', "\\\"");
        spark_out(
            "osascript",
            &["-e", &format!("tell application \"System Events\" to keystroke \"{t}\"")],
        );
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{}')",
            text.as_ref().replace('\'', "''")
        );
        spark_out("powershell", &["-NoProfile", "-Command", &script]);
        return "ok".to_string();
    }
}

/// Клик мышью по координатам: spark_mouse_click(500, 300). Требует ydotool
/// (Linux/Wayland), cliclick (macOS) — иначе честная ошибка.
#[allow(dead_code)]
fn spark_mouse_click(x: i64, y: i64) -> String {
    spark_mouse_action("click", x, y)
}
#[allow(dead_code)]
fn spark_mouse_right_click(x: i64, y: i64) -> String {
    spark_mouse_action("rclick", x, y)
}
/// Переместить курсор: spark_mouse_move(500, 300).
#[allow(dead_code)]
fn spark_mouse_move(x: i64, y: i64) -> String {
    spark_mouse_action("move", x, y)
}
/// Прокрутка: spark_mouse_scroll("down", 3).
#[allow(dead_code)]
fn spark_mouse_scroll(direction: impl AsRef<str>, amount: i64) -> String {
    let dir = direction.as_ref().to_lowercase();
    let n = amount.clamp(1, 20);
    #[cfg(target_os = "linux")]
    {
        let button = if dir.starts_with("вверх") || dir == "up" { "4" } else { "5" };
        for _ in 0..n {
            spark_out("ydotool", &["click", button]);
        }
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("cliclick", &["w:100"]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        return "прокрутка на Windows — через колесо мыши вручную".to_string();
    }
}
#[allow(dead_code)]
fn spark_mouse_action(kind: &str, x: i64, y: i64) -> String {
    #[cfg(target_os = "linux")]
    {
        // ydotool требует запущенный ydotoold с доступом к /dev/uinput.
        let probe = spark_out("pgrep", &["-x", "ydotoold"]);
        if probe.is_empty() {
            return "нужен ydotool: установи пакет и запусти ydotoold".to_string();
        }
        spark_out("ydotool", &["mousemove", "-a", &x.to_string(), &y.to_string()]);
        if kind == "move" {
            return "ok".to_string();
        }
        // 0xC0 = левый клик, 0xC1 = правый
        let code = if kind == "rclick" { "0xC1" } else { "0xC0" };
        spark_out("ydotool", &["click", code]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        let arg = match kind {
            "rclick" => format!("rc:{x},{y}"),
            "move" => format!("m:{x},{y}"),
            _ => format!("c:{x},{y}"),
        };
        spark_out("cliclick", &[&arg]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({x}, {y})"
        );
        spark_out("powershell", &["-NoProfile", "-Command", &script]);
        if kind == "move" {
            return "ok".to_string();
        }
        return "клик на Windows — клавиатурой или вручную".to_string();
    }
}

// ── Окна и столы ────────────────────────────────────────────────────────

/// Список открытых окон: spark_window_list() → строки «класс — заголовок».
#[allow(dead_code)]
fn spark_window_list() -> String {
    #[cfg(target_os = "linux")]
    {
        let raw = spark_out("hyprctl", &["-j", "clients"]);
        if !raw.is_empty() {
            return spark_scan_windows(&raw);
        }
        return String::new();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out(
            "osascript",
            &["-e", "tell application \"System Events\" to get name of every process whose background only is false"],
        )
    }
    #[cfg(target_os = "windows")]
    {
        spark_out(
            "powershell",
            &["-NoProfile", "-Command",
              "Get-Process | Where-Object {$_.MainWindowTitle} | ForEach-Object { $_.ProcessName + ' — ' + $_.MainWindowTitle }"],
        )
    }
}

/// Значение ключа в JSON-тексте без serde (hyprctl отдаёт JSON с
/// пробелами после двоеточий). Возвращает (значение, смещение после
/// значения) — чтобы скан мог продолжить с этого места.
#[allow(dead_code)]
fn spark_json_value(raw: &str, key: &str, from: usize) -> Option<(String, usize)> {
    let pat = format!("\"{key}\"");
    let pos = raw[from..].find(&pat)? + from;
    let colon = pos + pat.len() + raw[pos + pat.len()..].find(':')?;
    let rest = &raw[colon + 1..];
    let first = rest.find(|c: char| !c.is_whitespace())?;
    let rest = &rest[first..];
    if let Some(v) = rest.strip_prefix('"') {
        let end = v.find('"')?;
        Some((v[..end].to_string(), colon + 1 + first + end + 2))
    } else {
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        if end == 0 {
            return None;
        }
        Some((rest[..end].to_string(), colon + 1 + first + end))
    }
}

/// Вытащить пары «класс — заголовок» из JSON hyprctl без serde:
/// в сгенерированном бинарнике есть только std.
#[allow(dead_code)]
fn spark_scan_windows(raw: &str) -> String {
    let mut lines = Vec::new();
    let mut pos = 0;
    while let Some((class, after)) = spark_json_value(raw, "class", pos) {
        // Заголовок ищем рядом с классом (в том же объекте).
        if let Some((title, after_title)) = spark_json_value(raw, "title", after) {
            if !title.is_empty() {
                lines.push(format!("{class} — {title}"));
            }
            pos = after_title;
        } else {
            pos = after;
        }
    }
    lines.join("\n")
}

/// Найти адрес окна Hyprland по классу или заголовку (без serde).
#[allow(dead_code)]
fn spark_find_window_addr(name: &str) -> String {
    let raw = spark_out("hyprctl", &["-j", "clients"]);
    let want = name.to_lowercase();
    let mut pos = 0;
    while let Some((addr, after)) = spark_json_value(raw.as_str(), "address", pos) {
        // Класс и заголовок ищем до следующего объекта (до следующего
        // "address" или конца строки).
        let next = match spark_json_value(raw.as_str(), "address", after) {
            Some((_, n)) => n,
            None => raw.len(),
        };
        let window = &raw[after..next];
        let class = spark_json_value(window, "class", 0).map(|(v, _)| v).unwrap_or_default();
        let title = spark_json_value(window, "title", 0).map(|(v, _)| v).unwrap_or_default();
        if class.to_lowercase().contains(&want) || title.to_lowercase().contains(&want) {
            return addr;
        }
        pos = next;
    }
    String::new()
}

/// Фокус на окно по названию: spark_window_focus("chromium").
#[allow(dead_code)]
fn spark_window_focus(name: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        let name = name.as_ref();
        let addr = spark_find_window_addr(name);
        if addr.is_empty() {
            return format!("окно {name} не найдено");
        }
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.focus({{ window = \"address:{addr}\" }})")]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        return "фокус окон поддержан пока на Linux".to_string();
    }
}
/// Закрыть окно: spark_window_close("firefox").
#[allow(dead_code)]
fn spark_window_close(name: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        let name = name.as_ref();
        let addr = spark_find_window_addr(name);
        if addr.is_empty() {
            return format!("окно {name} не найдено");
        }
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.window.close({{ window = \"address:{addr}\" }})")]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        return "поддержано пока на Linux".to_string();
    }
}
/// Свернуть окно: spark_window_minimize("zed").
#[allow(dead_code)]
fn spark_window_minimize(name: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        let name = name.as_ref();
        let addr = spark_find_window_addr(name);
        if addr.is_empty() {
            return format!("окно {name} не найдено");
        }
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.window.move({{ workspace = \"special:hidden\", window = \"address:{addr}\" }})")]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        return "поддержано пока на Linux".to_string();
    }
}
/// Развернуть окно: spark_window_maximize("zed").
#[allow(dead_code)]
fn spark_window_maximize(name: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        let name = name.as_ref();
        let addr = spark_find_window_addr(name);
        if addr.is_empty() {
            return format!("окно {name} не найдено");
        }
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.window.fullscreen({{ window = \"address:{addr}\", mode = \"maximized\", action = \"toggle\" }})")]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        return "поддержано пока на Linux".to_string();
    }
}
/// Перенести окно на стол: spark_window_move_desktop("zed", 3).
#[allow(dead_code)]
fn spark_window_move_desktop(name: impl AsRef<str>, desktop: i64) -> String {
    #[cfg(target_os = "linux")]
    {
        let name = name.as_ref();
        let addr = spark_find_window_addr(name);
        if addr.is_empty() {
            return format!("окно {name} не найдено");
        }
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.window.move({{ workspace = \"{desktop}\", window = \"address:{addr}\" }})")]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (name, desktop);
        return "поддержано пока на Linux".to_string();
    }
}
/// Переключиться на стол: spark_desktop_switch(3).
#[allow(dead_code)]
fn spark_desktop_switch(n: i64) -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.focus({{ workspace = {n} }})")]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = n;
        return "поддержано пока на Linux".to_string();
    }
}
/// Следующий стол: spark_desktop_next().
#[allow(dead_code)]
fn spark_desktop_next() -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("hyprctl", &["dispatch", "hl.dsp.focus({ workspace = \"r+1\" })"]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        return "поддержано пока на Linux".to_string();
    }
}
/// Предыдущий стол: spark_desktop_prev().
#[allow(dead_code)]
fn spark_desktop_prev() -> String {
    #[cfg(target_os = "linux")]
    {
        let raw = spark_out("hyprctl", &["-j", "activeworkspace"]);
        let id = spark_json_value(&raw, "id", 0)
            .map(|(v, _)| v)
            .unwrap_or_else(|| "1".to_string());
        let n: i64 = id.parse().unwrap_or(1);
        if n <= 1 {
            return "ты уже на первом столе".to_string();
        }
        spark_out("hyprctl", &["dispatch", &format!("hl.dsp.focus({{ workspace = {} }})", n - 1)]);
        return "ok".to_string();
    }
    #[cfg(not(target_os = "linux"))]
    {
        return "поддержано пока на Linux".to_string();
    }
}

// ── Система: звук, яркость, экран, буфер, сеть, батарея ────────────────

/// Установить громкость: spark_volume_set(50). 0–100.
#[allow(dead_code)]
fn spark_volume_set(percent: i64) -> String {
    let p = percent.clamp(0, 100);
    #[cfg(target_os = "linux")]
    {
        spark_out("pactl", &["set-sink-volume", "@DEFAULT_SINK@", &format!("{p}%")]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("osascript", &["-e", &format!("set volume output volume {p}")]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        return "громкость на Windows — клавишами, средства .NET нет".to_string();
    }
}
/// Текущая громкость: spark_volume_get() → «50».
#[allow(dead_code)]
fn spark_volume_get() -> String {
    #[cfg(target_os = "linux")]
    {
        let raw = spark_out("pactl", &["get-sink-volume", "@DEFAULT_SINK@"]);
        let digits: String = raw.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
        return digits;
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("osascript", &["-e", "output volume of (get volume settings)"])
    }
    #[cfg(target_os = "windows")]
    {
        String::new()
    }
}
/// Включить или выключить звук: spark_volume_mute().
#[allow(dead_code)]
fn spark_volume_mute() -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("pactl", &["set-sink-mute", "@DEFAULT_SINK@", "toggle"]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("osascript", &["-e", "set volume output muted (not (output muted of (get volume settings)))"]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        return "на Windows — клавиша mute".to_string();
    }
}
/// Яркость экрана: spark_brightness_set(70). 0–100.
#[allow(dead_code)]
fn spark_brightness_set(percent: i64) -> String {
    let p = percent.clamp(1, 100);
    #[cfg(target_os = "linux")]
    {
        spark_out("brightnessctl", &["set", &format!("{p}%")]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        return "яркость на macOS — клавишами".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        return "яркость на Windows — клавишами".to_string();
    }
}
/// Скриншот в файл: spark_screenshot("/tmp/shot.png").
#[allow(dead_code)]
fn spark_screenshot(path: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("grim", &[path.as_ref()]);
        return fs::metadata(path.as_ref()).map(|_| "ok".to_string()).unwrap_or_else(|_| "grim не смог сделать снимок".to_string());
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("screencapture", &["-x", path.as_ref()]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        return "скриншот на Windows — Win+Shift+S".to_string();
    }
}
/// В буфер обмена: spark_clip_set("текст").
#[allow(dead_code)]
fn spark_clip_set(text: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("wl-copy", &[text.as_ref()]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("pbcopy", &[text.as_ref()]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        spark_out("powershell", &["-NoProfile", "-Command", &format!("Set-Clipboard -Value '{}'", text.as_ref().replace('\'', "''"))]);
        return "ok".to_string();
    }
}
/// Из буфера обмена: spark_clip_get().
#[allow(dead_code)]
fn spark_clip_get() -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("wl-paste", &[])
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("pbpaste", &[])
    }
    #[cfg(target_os = "windows")]
    {
        spark_out("powershell", &["-NoProfile", "-Command", "Get-Clipboard"])
    }
}
/// HTTP GET: spark_http_get("https://api.example.com/data") → тело ответа.
#[allow(dead_code)]
fn spark_http_get(url: impl AsRef<str>) -> String {
    spark_out("curl", &["-s", "-L", "--max-time", "12", url.as_ref()])
}
/// Открыть приложение: spark_app_open("firefox").
#[allow(dead_code)]
fn spark_app_open(name: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        spark_launch(&format!("gtk-launch {}", name.as_ref()));
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("open", &["-a", name.as_ref()]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        spark_out("cmd", &["/C", "start", name.as_ref()]);
        return "ok".to_string();
    }
}
/// Закрыть приложение: spark_app_quit("firefox").
#[allow(dead_code)]
fn spark_app_quit(name: impl AsRef<str>) -> String {
    #[cfg(target_os = "linux")]
    {
        spark_out("pkill", &["-x", name.as_ref()]);
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("osascript", &["-e", &format!("tell application \"{n}\" to quit", n = name.as_ref())]);
        return "ok".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        spark_out("taskkill", &["/IM", &format!("{}.exe", name.as_ref())]);
        return "ok".to_string();
    }
}
/// Заряд батареи в процентах: spark_battery() → «85».
#[allow(dead_code)]
fn spark_battery() -> String {
    #[cfg(target_os = "linux")]
    {
        // Читаем sysfs напрямую: работает и без upower.
        if let Ok(dir) = fs::read_dir("/sys/class/power_supply") {
            for entry in dir.flatten() {
                let p = entry.path();
                let fname = p
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_default();
                if fname.starts_with("BAT") {
                    let cap = fs::read_to_string(p.join("capacity")).unwrap_or_default();
                    let digits: String = cap.chars().filter(|c| c.is_ascii_digit()).collect();
                    if !digits.is_empty() {
                        return digits;
                    }
                }
            }
        }
        return String::new();
    }
    #[cfg(target_os = "macos")]
    {
        let raw = spark_out("pmset", &["-g", "batt"]);
        let digits: String = raw.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
        digits
    }
    #[cfg(target_os = "windows")]
    {
        spark_out("powershell", &["-NoProfile", "-Command",
            "(Get-WmiObject Win32_Battery).EstimatedChargeRemaining"])
    }
}
/// Размер экрана: spark_screen_size() → «1920x1080».
#[allow(dead_code)]
fn spark_screen_size() -> String {
    #[cfg(target_os = "linux")]
    {
        let raw = spark_out("hyprctl", &["-j", "monitors"]);
        let w = spark_json_value(&raw, "width", 0).map(|(v, _)| v).unwrap_or_else(|| "0".to_string());
        let h = spark_json_value(&raw, "height", 0).map(|(v, _)| v).unwrap_or_else(|| "0".to_string());
        format!("{w}x{h}")
    }
    #[cfg(target_os = "macos")]
    {
        spark_out("system_profiler", &["SPDisplaysDataType"])
            .lines()
            .find(|l| l.contains("Resolution"))
            .unwrap_or("")
            .trim()
            .to_string()
    }
    #[cfg(target_os = "windows")]
    {
        spark_out("powershell", &["-NoProfile", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; \"$(($b.Width))x$(($b.Height))\""])
    }
}
/// Пауза в миллисекундах: spark_sleep_ms(500).
#[allow(dead_code)]
fn spark_sleep_ms(ms: i64) -> String {
    std::thread::sleep(std::time::Duration::from_millis(ms.max(0) as u64));
    "ok".to_string()
}
"#;

/// Сгенерировать Rust-код из AST без вызова rustc: движок Джарвиса
/// компилирует плагины сам, CLI Spark — через эту же генерацию.
/// Контекст генерации: известные коллекции (`for x in items`) и функции
/// (`def f():` + `f()` без аргументов — вызов, а не константа).
pub struct Ctx {
    vecs: HashSet<String>,
    fns: HashSet<String>,
}

pub fn compile_to_string(ast: Vec<Stmt>) -> String {

    let mut rust_code = String::from("use std::io::{Read, Write};\nuse std::fs;\nuse std::process::Stdio;\n\n");

    let mut funcs = Vec::new();
    let mut classes = Vec::new();
    let mut main_stmts = Vec::new();

    for stmt in ast {
        match stmt {
            Stmt::Def { .. } => funcs.push(stmt),
            Stmt::Class { .. } => classes.push(stmt),
            Stmt::Import { .. } => main_stmts.push(stmt),
            _ => main_stmts.push(stmt),
        }
    }

    // Переменные-коллекции: `items = [...]` — чтобы `for x in items`
    // итерировал по ним, а не делал 0..items. Плюс имена функций: вызов
    // без аргументов — вызов, а не поле.
    let mut vecs = HashSet::new();
    let mut fns = HashSet::new();
    collect_vecs(&main_stmts, &mut vecs);
    for f in &funcs {
        if let Stmt::Def { name, body, .. } = f {
            fns.insert(name.clone());
            collect_vecs(body, &mut vecs);
        }
    }
    let ctx = Ctx { vecs, fns };

    rust_code.push_str(PRELUDE);
    rust_code.push('\n');

    for class in classes {
        rust_code.push_str(&gen_stmt(class, &ctx));
        rust_code.push('\n');
    }

    for func in funcs {
        rust_code.push_str(&gen_stmt(func, &ctx));
    }

    rust_code.push_str("fn main() {\n");
    for stmt in main_stmts {
        rust_code.push_str(&gen_stmt(stmt, &ctx));
    }
    rust_code.push_str("}\n");

    rust_code
}

/// Скомпилировать .sk-программу в бинарник: генерация + rustc.
/// Возвращает Err с выводом rustc, если сборка не удалась.
pub fn compile_to_binary(ast: Vec<Stmt>, output: &str, opt_level: i32) -> Result<(), String> {
    let rust_code = compile_to_string(ast);

    let tmp_dir = Builder::new().prefix("spark_build").tempdir().unwrap();
    let rs_path = tmp_dir.path().join("main.rs");

    fs::write(&rs_path, &rust_code).expect("Failed to write Rust file");

    let opt_args: Vec<&str> = match opt_level {
        1 => vec!["-C", "opt-level=1"],
        2 => vec!["-C", "opt-level=2"],
        3 => vec!["-C", "opt-level=3"],
        _ => vec!["-C", "opt-level=3"],
    };

    let out = Command::new("rustc")
        .args(&opt_args)
        .arg(rs_path.to_str().unwrap())
        .arg("-o")
        .arg(output)
        .output();

    match out {
        Ok(o) if o.status.success() => {
            println!("✅ Spark скомпилировался в: {} (opt-level={})", output, opt_level);
            Ok(())
        }
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).into_owned()),
        Err(e) => Err(format!("rustc не запустился: {e}")),
    }
}

fn gen_stmt(stmt: Stmt, ctx: &Ctx) -> String {
    match stmt {
        Stmt::Assign { target, value } => {
            format!("    let {} = {};\n", target, gen_expr(value, ctx))
        }
        Stmt::AugAssign { target, op, value } => {
            format!("    {} {} = {};\n", target, op, gen_expr(value, ctx))
        }
        Stmt::Expr(expr) => {
            let e = gen_expr(expr, ctx);
            if e.starts_with("println!") {
                format!("    {};\n", e)
            } else if e.contains(" = ") {
                format!("    let {};\n", e)
            } else {
                format!("    println!(\"{{}}\", {});\n", e)
            }
        }
        Stmt::Print(expr) => {
            format!("    println!(\"{{}}\", {});\n", gen_expr(expr, ctx))
        }
        Stmt::If { condition, then_branch, elif_branches, else_branch } => {
            let mut code = format!("    if {} {{\n", gen_expr(condition, ctx));
            for s in then_branch {
                code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
            }
            code.push_str("    }");
            for (cond, body) in elif_branches {
                code.push_str(&format!("\n    else if {} {{\n", gen_expr(cond, ctx)));
                for s in body {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }");
            }
            if let Some(else_stmts) = else_branch {
                code.push_str(" else {\n");
                for s in else_stmts {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }");
            }
            code.push('\n');
            code
        }
        Stmt::While { condition, body } => {
            let mut code = format!("    while {} {{\n", gen_expr(condition, ctx));
            for s in body {
                code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
            }
            code.push_str("    }\n");
            code
        }
        Stmt::For { var, iter, body, else_body } => {
            let iter_str = gen_expr(iter.clone(), ctx);
            // Коллекция для for: литерал массива/кортежа, переменная,
            // помеченная как коллекция (`items = [...]`), или выражение,
            // уже содержащее диапазон/итератор. Иначе — счётчик 0..N.
            let collection = matches!(iter, Expr::Array(_) | Expr::Tuple(_))
                || matches!(&iter, Expr::Ident(name) if ctx.vecs.contains(name))
                || iter_str.contains("0..")
                || iter_str.contains(".iter()")
                || iter_str.contains("range");
            let mut code = if collection {
                format!("    for {} in {} {{\n", var, iter_str)
            } else {
                format!("    for {} in 0..{} {{\n", var, iter_str)
            };
            for s in body {
                code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
            }
            code.push_str("    }");
            if let Some(else_stmts) = else_body {
                code.push_str(" else {\n");
                for s in else_stmts {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }");
            }
            code.push('\n');
            code
        }
        Stmt::Def { name, args, body } => {
            let params: Vec<String> = args.iter().map(|(s, _)| format!("{}: &str", s)).collect();
            let mut code = format!("fn {}({}) -> String {{\n", name, params.join(", "));
            for s in body {
                code.push_str(&gen_stmt(s, ctx));
            }
            code.push_str("}\n");
            code
        }
        Stmt::Class { name, bases: _, body } => {
            let mut struct_code = format!("struct {} {{\n", name);
            let mut impl_code = format!("impl {} {{\n", name);
            let mut has_new = false;
            
            for stmt in body {
                match stmt {
                    Stmt::Def { name: func_name, args, body: func_body } => {
                        if func_name == name {
                            let params: Vec<String> = args.iter().map(|(s, _)| s.clone()).collect();
                            impl_code.push_str(&format!("    pub fn new({}) -> Self {{\n", params.join(", ")));
                            for s in func_body {
                                impl_code.push_str(&gen_stmt(s, ctx));
                            }
                            impl_code.push_str("    }\n");
                            has_new = true;
                        } else {
                            let params: Vec<String> = args.iter().map(|(s, _)| s.clone()).collect();
                            impl_code.push_str(&format!("    pub fn {}({}) {{\n", func_name, params.join(", ")));
                            for s in func_body {
                                impl_code.push_str(&gen_stmt(s, ctx));
                            }
                            impl_code.push_str("    }\n");
                        }
                    }
                    Stmt::Assign { target, value: _ } => {
                        struct_code.push_str(&format!("    pub {}: i32,\n", target));
                    }
                    _ => {}
                }
            }
            
            if !has_new {
                impl_code.push_str(&format!("    pub fn new() -> Self {{\n",));
                impl_code.push_str(&format!("        {} {{ }}\n", name));
                impl_code.push_str("    }\n");
            }
            
            struct_code.push_str("}\n\n");
            impl_code.push_str("}\n");
            
            struct_code + &impl_code
        }
        Stmt::Try { body, except_branches, else_branch, finally_body } => {
            let mut code = String::new();
            code.push_str("    {\n");
            
            for s in body {
                code.push_str(&gen_stmt(s, ctx));
            }
            
            if !except_branches.is_empty() {
                for (exc_type, alias, exc_body) in except_branches {
                    let _type_str = exc_type.unwrap_or_else(|| "Box<dyn std::error::Error>".to_string());
                    let alias_str = alias.unwrap_or_else(|| "e".to_string());
                    code.push_str(&format!("    }}\n    Err({}) => {{\n", alias_str));
                    for s in exc_body {
                        code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                    }
                    code.push_str("    }\n");
                }
            }
            
            if let Some(else_stmts) = else_branch {
                code.push_str("    Ok(_) => {\n");
                for s in else_stmts {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }\n");
            }
            
            if let Some(finally_stmts) = finally_body {
                for s in finally_stmts {
                    code.push_str(&gen_stmt(s, ctx));
                }
            }
            
            code.push_str("    }\n");
            code
        }
        Stmt::Raise(expr) => {
            if let Some(e) = expr {
                format!("    panic!(\"{{}}\", {});\n", gen_expr(e, ctx))
            } else {
                "    panic!(\"Raised exception\");\n".to_string()
            }
        }
        Stmt::Assert(cond, msg) => {
            if let Some(m) = msg {
                format!("    assert!({}, \"{{}}\", {});\n", gen_expr(cond, ctx), gen_expr(m, ctx))
            } else {
                format!("    assert!({});\n", gen_expr(cond, ctx))
            }
        }
        Stmt::Return(expr) => {
            if let Some(e) = expr {
                let expr_str = gen_expr(e, ctx);
                if !expr_str.contains('=') {
                    // Def'ы Spark возвращают String: &str/число приводим.
                    if expr_str.starts_with('"') || expr_str.ends_with(".to_string()")
                        || expr_str.starts_with("format!") || expr_str.starts_with("String::")
                    {
                        format!("    {}\n", expr_str)
                    } else {
                        format!("    return {}.to_string();\n", expr_str)
                    }
                } else {
                    format!("    let {};\n", expr_str)
                }
            } else {
                "    return String::new();\n".to_string()
            }
        }
        Stmt::Break => "    break;\n".to_string(),
        Stmt::Continue => "    continue;\n".to_string(),
        Stmt::Pass => "".to_string(),
        Stmt::Global(vars) => {
            format!("    // global {}\n", vars.join(", "))
        }
        Stmt::Nonlocal(vars) => {
            format!("    // nonlocal {}\n", vars.join(", "))
        }
        Stmt::Import { module, items, alias } => {
            if let Some(a) = alias {
                format!("    // import {} as {}\n", module, a)
            } else if let Some(items) = items {
                format!("    // from {} import {}\n", module, items.join(", "))
            } else {
                format!("    // import {}\n", module)
            }
        }
        Stmt::With { items, body } => {
            let mut code = String::new();
            for (expr, alias) in items {
                let var = alias.unwrap_or_else(|| "_".to_string());
                code.push_str(&format!("    let {} = {};\n", var, gen_expr(expr, ctx)));
            }
            for s in body {
                code.push_str(&gen_stmt(s, ctx));
            }
            code
        }
    }
}


/// Системные built-ins плагинов: имена в Spark → функции прелюдии.
fn builtin(fname: &str, args: &[String]) -> Option<String> {
    let call = |n: &str| {
        if args.is_empty() {
            format!("{n}().to_string()")
        } else {
            format!("{n}({}).to_string()", args.join(", "))
        }
    };
    match fname {
        "arg" | "аргумент" => Some(call("spark_arg")),
        "env" | "переменная" => Some(call("spark_env")),
        "run" | "выполнить" => Some(call("spark_run")),
        "launch" | "запустить" => Some(call("spark_launch")),
        "read_file" | "прочитать" => Some(call("spark_read")),
        "write_file" | "записать" => Some(call("spark_write")),
        "notify" | "уведомить" => Some(call("spark_notify")),
        "say" | "сказать" => Some(call("spark_say")),
        // ── Автоматизация ОС, аналог AutoHotkey (v0.3) ──
        "send_hotkey" | "сочетание" | "хоткей" => Some(call("spark_send_hotkey")),
        "type_text" | "напечатать" | "напечатать_текст" => Some(call("spark_type_text")),
        "mouse_click" | "клик" => Some(call("spark_mouse_click")),
        "mouse_right_click" | "правый_клик" => Some(call("spark_mouse_right_click")),
        "mouse_move" | "движение_мыши" => Some(call("spark_mouse_move")),
        "mouse_scroll" | "прокрутка" => Some(call("spark_mouse_scroll")),
        "window_list" | "окна" => Some(call("spark_window_list")),
        "window_focus" | "фокус" => Some(call("spark_window_focus")),
        "window_close" | "закрыть_окно" => Some(call("spark_window_close")),
        "window_minimize" | "свернуть_окно" => Some(call("spark_window_minimize")),
        "window_maximize" | "развернуть_окно" => Some(call("spark_window_maximize")),
        "window_move_desktop" | "перенести_окно" => Some(call("spark_window_move_desktop")),
        "desktop_switch" | "стол" => Some(call("spark_desktop_switch")),
        "desktop_next" | "следующий_стол" => Some(call("spark_desktop_next")),
        "desktop_prev" | "предыдущий_стол" => Some(call("spark_desktop_prev")),
        "volume_set" | "громкость" => Some(call("spark_volume_set")),
        "volume_get" => Some(call("spark_volume_get")),
        "volume_mute" | "без_звука" => Some(call("spark_volume_mute")),
        "brightness_set" | "яркость" => Some(call("spark_brightness_set")),
        "screenshot" | "скриншот" => Some(call("spark_screenshot")),
        "clip_set" | "в_буфер" => Some(call("spark_clip_set")),
        "clip_get" | "буфер" => Some(call("spark_clip_get")),
        "http_get" | "запрос" => Some(call("spark_http_get")),
        "app_open" | "открыть_приложение" => Some(call("spark_app_open")),
        "app_quit" | "закрыть_приложение" => Some(call("spark_app_quit")),
        "battery" | "батарея" => Some(call("spark_battery")),
        "screen_size" | "экран_размер" => Some(call("spark_screen_size")),
        "sleep_ms" | "пауза" => Some(call("spark_sleep_ms")),
        // str(x) — Python-привычка: привести что угодно к строке.
        "str" if !args.is_empty() => Some(format!("format!(\"{{}}\", {})", args[0])),
        // int(x) — аргументы плагина приходят строками, арифметика
        // требует числа. Нечисло → 0 (плагин сам решает, что с этим делать).
        "int" if !args.is_empty() => {
            Some(format!("({}.to_string().parse::<i64>().unwrap_or(0))", args[0]))
        }
        _ => None,
    }
}

/// Собрать имена переменных, присвоенным значением которых был массив,
/// кортеж или вызов run: по ним `for x in имя` итерирует коллекцию.
fn collect_vecs(stmts: &[Stmt], out: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign { target, value } => {
                if matches!(value, Expr::Array(_) | Expr::Tuple(_)) {
                    out.insert(target.clone());
                }
            }
            Stmt::If { then_branch, elif_branches, else_branch, .. } => {
                collect_vecs(then_branch, out);
                for (_, body) in elif_branches {
                    collect_vecs(body, out);
                }
                if let Some(b) = else_branch {
                    collect_vecs(b, out);
                }
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => collect_vecs(body, out),
            _ => {}
        }
    }
}

fn indent_stmt(stmt: &str) -> String {
    stmt.lines()
        .map(|line| format!("    {}", line))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn gen_expr(expr: Expr, ctx: &Ctx) -> String {
    match expr {
        Expr::Number(n) => n.to_string(),
        Expr::Float(f) => f.to_string(),
        Expr::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Expr::Ident(i) => i,
        Expr::Bool(b) => if b { "true" } else { "false" }.to_string(),
        Expr::None => "None".to_string(),
        Expr::Array(els) => {
            let parts: Vec<String> = els.into_iter().map(|e| gen_expr(e, ctx)).collect();
            format!("vec![{}]", parts.join(", "))
        }
        Expr::Tuple(els) => {
            let parts: Vec<String> = els.into_iter().map(|e| gen_expr(e, ctx)).collect();
            format!("({})", parts.join(", "))
        }
        Expr::Dict(pairs) => {
            let parts: Vec<String> = pairs.iter()
                .map(|(k, v)| format!("{} => {}", gen_expr(k.clone(), ctx), gen_expr(v.clone(), ctx)))
                .collect();
            format!("std::collections::HashMap::from([{}])", parts.join(", "))
        }
        Expr::Set(els) => {
            let parts: Vec<String> = els.into_iter().map(|e| gen_expr(e, ctx)).collect();
            format!("std::collections::HashSet::from([{}])", parts.join(", "))
        }
        Expr::BinaryOp(left, op, right) => {
            if op == "+" {
                let l = gen_expr(*left, ctx);
                let r = gen_expr(*right, ctx);
                // Строковый контекст: литерал или то, что уже String
                // (вызовы Spark-функций кодоген оборачивает в .to_string()).
                let l_str = l.starts_with('"') || l.ends_with(".to_string()")
                    || l.starts_with("format!(");
                let r_str = r.starts_with('"') || r.ends_with(".to_string()")
                    || r.starts_with("format!(");
                if l_str || r_str {
                    format!("format!(\"{{}}{{}}\", {}, {})", l, r)
                } else {
                    format!("{} + {}", l, r)
                }
            } else {
                format!("({} {} {})", gen_expr(*left, ctx), op, gen_expr(*right, ctx))
            }
        }
        Expr::UnaryOp(op, expr) => {
            format!("({}{})", op, gen_expr(*expr, ctx))
        }
        Expr::Compare(left, op, right) => {
            format!("({} {} {})", gen_expr(*left, ctx), op, gen_expr(*right, ctx))
        }
        Expr::Index { array, index } => {
            format!("{}[{}]", gen_expr(*array, ctx), gen_expr(*index, ctx))
        }
        Expr::MethodCall { receiver, method, args } => {
            let recv = gen_expr(*receiver, ctx);
            let a: Vec<String> = args.iter().map(|x| gen_expr(x.clone(), ctx)).collect();
            // Питоновские строковые методы → Rust; неизвестные — как есть
            // (плагин может звать и rust-методы напрямую).
            match method.as_str() {
                "upper" => format!("{recv}.to_uppercase()"),
                "lower" => format!("{recv}.to_lowercase()"),
                "trim" | "strip" => format!("{recv}.trim().to_string()"),
                "lstrip" => format!("{recv}.trim_start().to_string()"),
                "rstrip" => format!("{recv}.trim_end().to_string()"),
                "replace" if a.len() == 2 => format!("{recv}.replace({}, {})", a[0], a[1]),
                "contains" if a.len() == 1 => format!("{recv}.contains({})", a[0]),
                "startswith" if a.len() == 1 => format!("{recv}.starts_with({})", a[0]),
                "endswith" if a.len() == 1 => format!("{recv}.ends_with({})", a[0]),
                "split" if a.len() == 1 => format!(
                    "{recv}.split({}).map(|s| s.to_string()).collect::<Vec<String>>()", a[0]),
                "join" if a.len() == 1 => format!("{}.join(&{})", a[0], recv),
                "len" => format!("{recv}.chars().count()"),
                "find" | "index" if a.len() == 1 => format!(
                    "{recv}.find({}).map(|i| i as i64).unwrap_or(-1)", a[0]),
                _ => format!("{}.{}({})", recv, method, a.join(", ")),
            }
        }
        Expr::Call { func, args } => {
            let fname = gen_expr(*func, ctx);
            let args_str: Vec<String> = args.iter().map(|a| gen_expr(a.clone(), ctx)).collect();
            
            if fname == "print" {
                if args_str.is_empty() {
                    "()".to_string()
                } else {
                    format!("println!(\"{{}}\", {})", args_str.join(", "))
                }
            } else if fname == "range" {
                if args_str.len() == 1 {
                    format!("0..{}", args_str[0])
                } else if args_str.len() == 2 {
                    format!("{}..{}", args_str[0], args_str[1])
                } else {
                    format!("0..1")
                }
            } else if let Some(mapped) = builtin(&fname, &args_str) {
                mapped
            } else if args_str.is_empty() {
                if ctx.fns.contains(&fname) {
                    format!("{}().to_string()", fname)
                } else {
                    format!("{}.to_string()", fname)
                }
            } else {
                format!("{}({}).to_string()", fname, args_str.join(", "))
            }
        }
        Expr::Lambda { args, body } => {
            let params: Vec<String> = args.iter().map(|s| s.clone()).collect();
            let body_str = gen_expr(*body, ctx);
            format!("|{}| {}", params.join(", "), body_str)
        }
        Expr::Ternary { cond, then, else_ } => {
            format!(
                "if {} {{ {} }} else {{ {} }}",
                gen_expr(*cond, ctx),
                gen_expr(*then, ctx),
                gen_expr(*else_, ctx)
            )
        }
        Expr::Input(_prompt) => {
            format!("{{ input() }}")
        }
    }
}