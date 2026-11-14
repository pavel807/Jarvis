//! Системная прелюдия Spark: она живёт двойной жизнью.
//! Как модуль крейта — интерпретатор вызывает её функции напрямую,
//! поэтому плагины работают без rustc. Как текст — include_str! вкладывает
//! её в сгенерированный код плагина. Файл обязан быть самодостаточным
//! (свои use) и не иметь внутренних атрибутов `#!` — в сгенерированном
//! коде он стоит не в начале крейта.

use std::fs;
use std::process::Stdio;

#[allow(dead_code)]
pub(crate) fn spark_arg(n: i64) -> String {
    std::env::args().nth(n.max(1) as usize).unwrap_or_default()
}
#[allow(dead_code)]
pub(crate) fn spark_env(name: impl AsRef<str>) -> String {
    std::env::var(name.as_ref()).unwrap_or_default()
}
#[allow(dead_code)]
pub(crate) fn spark_run(cmd: impl AsRef<str>) -> String {
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
pub(crate) fn spark_launch(cmd: impl AsRef<str>) -> String {
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
pub(crate) fn spark_read(path: impl AsRef<str>) -> String {
    fs::read_to_string(path.as_ref()).unwrap_or_default()
}
#[allow(dead_code)]
pub(crate) fn spark_write(path: impl AsRef<str>, text: impl AsRef<str>) -> String {
    match fs::write(path.as_ref(), text.as_ref()) {
        Ok(_) => "ok".to_string(),
        Err(e) => format!("ошибка: {e}"),
    }
}
#[allow(dead_code)]
pub(crate) fn spark_notify(title: impl AsRef<str>, text: impl AsRef<str>) -> String {
    let _ = std::process::Command::new("notify-send")
        .arg("-a").arg("Jarvis")
        .arg(title.as_ref())
        .arg(text.as_ref())
        .spawn();
    "ok".to_string()
}
/// Речь плагина: строка «SAY: …» на stdout Джарвис озвучит голосом.
#[allow(dead_code)]
pub(crate) fn spark_say(text: impl AsRef<str>) -> String {
    println!("SAY: {}", text.as_ref());
    "ok".to_string()
}
#[allow(dead_code)]
pub(crate) fn input() -> String {
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
pub(crate) fn spark_out(bin: &str, args: &[&str]) -> String {
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
pub(crate) fn spark_send_hotkey(keys: impl AsRef<str>) -> String {
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
pub(crate) fn spark_type_text(text: impl AsRef<str>) -> String {
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
pub(crate) fn spark_mouse_click(x: i64, y: i64) -> String {
    spark_mouse_action("click", x, y)
}
#[allow(dead_code)]
pub(crate) fn spark_mouse_right_click(x: i64, y: i64) -> String {
    spark_mouse_action("rclick", x, y)
}
/// Переместить курсор: spark_mouse_move(500, 300).
#[allow(dead_code)]
pub(crate) fn spark_mouse_move(x: i64, y: i64) -> String {
    spark_mouse_action("move", x, y)
}
/// Прокрутка: spark_mouse_scroll("down", 3).
#[allow(dead_code)]
pub(crate) fn spark_mouse_scroll(direction: impl AsRef<str>, amount: i64) -> String {
    #[cfg(target_os = "linux")]
    {
        // Переменные внутри cfg-блока: на macOS/Windows они не нужны,
        // а объявление снаружи давало unused-предупреждения в CI.
        let dir = direction.as_ref().to_lowercase();
        let n = amount.clamp(1, 20);
        let button = if dir.starts_with("вверх") || dir == "up" { "4" } else { "5" };
        for _ in 0..n {
            spark_out("ydotool", &["click", button]);
        }
        return "ok".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (direction, amount);
        return "прокрутка на macOS — жесты трекпада".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = (direction, amount);
        return "прокрутка на Windows — через колесо мыши вручную".to_string();
    }
}
#[allow(dead_code)]
pub(crate) fn spark_mouse_action(kind: &str, x: i64, y: i64) -> String {
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
pub(crate) fn spark_window_list() -> String {
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
pub(crate) fn spark_json_value(raw: &str, key: &str, from: usize) -> Option<(String, usize)> {
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
pub(crate) fn spark_scan_windows(raw: &str) -> String {
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
pub(crate) fn spark_find_window_addr(name: &str) -> String {
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
pub(crate) fn spark_window_focus(name: impl AsRef<str>) -> String {
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
pub(crate) fn spark_window_close(name: impl AsRef<str>) -> String {
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
pub(crate) fn spark_window_minimize(name: impl AsRef<str>) -> String {
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
pub(crate) fn spark_window_maximize(name: impl AsRef<str>) -> String {
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
pub(crate) fn spark_window_move_desktop(name: impl AsRef<str>, desktop: i64) -> String {
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
pub(crate) fn spark_desktop_switch(n: i64) -> String {
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
pub(crate) fn spark_desktop_next() -> String {
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
pub(crate) fn spark_desktop_prev() -> String {
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
pub(crate) fn spark_volume_set(percent: i64) -> String {
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
pub(crate) fn spark_volume_get() -> String {
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
pub(crate) fn spark_volume_mute() -> String {
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
pub(crate) fn spark_brightness_set(percent: i64) -> String {
    #[cfg(target_os = "linux")]
    {
        let p = percent.clamp(1, 100);
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
pub(crate) fn spark_screenshot(path: impl AsRef<str>) -> String {
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
pub(crate) fn spark_clip_set(text: impl AsRef<str>) -> String {
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
pub(crate) fn spark_clip_get() -> String {
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
pub(crate) fn spark_http_get(url: impl AsRef<str>) -> String {
    spark_out("curl", &["-s", "-L", "--max-time", "12", url.as_ref()])
}
/// Открыть приложение: spark_app_open("firefox").
#[allow(dead_code)]
pub(crate) fn spark_app_open(name: impl AsRef<str>) -> String {
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
pub(crate) fn spark_app_quit(name: impl AsRef<str>) -> String {
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
pub(crate) fn spark_battery() -> String {
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
pub(crate) fn spark_screen_size() -> String {
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
pub(crate) fn spark_sleep_ms(ms: i64) -> String {
    std::thread::sleep(std::time::Duration::from_millis(ms.max(0) as u64));
    "ok".to_string()
}
