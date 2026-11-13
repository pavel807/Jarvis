//! Windows: реализация платформенного слоя поверх PowerShell (Windows
//! PowerShell 5.1 есть на любой системе) и Win32 через Add-Type.
//!
//! Строки, приходящие от пользователя (имена, текст буфера, заголовки окон),
//! передаются в PowerShell только в UTF-8 → base64: консольные кодировки
//! (cp1251/cp866) иначе превращают русский текст в мусор, а кавычки — в
//! инъекцию в скрипт. Вывод PowerShell всегда переключается на UTF-8.
//!
//! Музыка: GlobalSystemMediaTransportControls (SMTC) — тот же механизм, что
//! у системных кнопок медиа; если SMTC недоступен, честно деградируем до
//! нажатия медиа-клавиши, не выдавая это за проверенный успех.

use crate::platform::{COMMON_ALIASES, fuzzy_app_match, lookup_alias, process_matches};
use std::process::{Command, Stdio};
use std::time::Duration;

// =========================================================================
// Словарь русских названий приложений (Windows)
// =========================================================================

const APP_ALIASES: &[(&str, &str)] = &[
    ("блокнот", "notepad"),
    ("проводник", "explorer"),
    ("терминал", "wt"),
    ("калькулятор", "calc"),
    ("панель управления", "control"),
];

pub fn alias(word: &str) -> Option<&'static str> {
    lookup_alias(word, &[APP_ALIASES, COMMON_ALIASES])
}

// =========================================================================
// Базовая инфраструктура PowerShell
// =========================================================================

/// UTF-8 → base64 (RFC 4648): единственный безопасный способ протащить
/// русский текст и кавычки внутрь PowerShell-скрипта.
pub fn b64(s: &str) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn ps_command(script: &str) -> Command {
    let full = format!(
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;$ErrorActionPreference='Stop';{script}"
    );
    let mut cmd = Command::new("powershell");
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        &full,
    ]);
    cmd
}

/// Выполняет PowerShell и возвращает stdout. Всё, что не уложилось в код 0
/// или напечатало пустоту при явной ошибке, — Err с человеческой причиной.
fn ps_run(script: &str) -> Result<String, String> {
    let out = ps_command(script)
        .output()
        .map_err(|e| format!("powershell недоступен: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        let line = err
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("powershell завершился с ошибкой")
            .trim()
            .to_string();
        Err(line)
    }
}

/// `$w = <base64-декодированная строка>` — прелюдия каждого скрипта.
fn want_var(b64_payload: &str) -> String {
    format!("$want=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_payload}'));")
}

/// Общий кусок скрипта: найти процесс по имени.
/// Точное совпадение ProcessName, затем все токены просьбы в ProcessName или
/// заголовке окна. `$procs` должен быть уже определён вызывающим скриптом.
const FIND_PROC: &str = r#"
$tokens=@($want.ToLower().Split(' ') | Where-Object { $_.Length -ge 2 });
$hit=$null;
$hit=@($procs | Where-Object { $_.ProcessName.ToLower() -eq $want.ToLower() }) | Select-Object -First 1;
if (-not $hit -and $tokens.Count -gt 0) {
$hit=@($procs | Where-Object { $p=$_; $pn=$p.ProcessName.ToLower(); $tt=(''+$p.MainWindowTitle).ToLower(); @($tokens | Where-Object { -not ($pn.Contains($_) -or $tt.Contains($_)) }).Count -eq 0 }) | Select-Object -First 1;
}
"#;

// =========================================================================
// Win32: общий заголовок P/Invoke для окон
// =========================================================================

const WIN32_HEADER: &str = r#"
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class JWin {
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
}
'@
"#;

/// SW_* константы ShowWindowAsync.
fn sw_flag(action: &str) -> Option<u32> {
    match action {
        "minimize" => Some(6),
        // Полноэкранный режим — прерогатива самого приложения; честный
        // максимум, который мы можем гарантировать, — разворот окна.
        "maximize" | "fullscreen" => Some(3),
        "restore" => Some(9),
        _ => None,
    }
}

// =========================================================================
// Приложения
// =========================================================================

/// Список из меню «Пуск»: (отображаемое имя, AppID). Кэш на сессию.
fn start_apps() -> Vec<(String, String)> {
    static CACHE: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            ps_run("Get-StartApps | ForEach-Object { ('{0}|{1}' -f $_.Name, $_.AppID) }")
                .map(|out| parse_start_apps(&out))
                .unwrap_or_default()
        })
        .clone()
}

/// Разбор вывода Get-StartApps: «Имя|AppID» по строке.
pub fn parse_start_apps(out: &str) -> Vec<(String, String)> {
    out.lines()
        .filter_map(|line| {
            let (name, appid) = line.split_once('|')?;
            let name = name.trim();
            if name.is_empty() || appid.trim().is_empty() {
                return None;
            }
            Some((name.to_string(), appid.trim().to_string()))
        })
        .collect()
}

fn find_start_app(want: &str) -> Option<(String, String)> {
    let apps = start_apps();
    for variant in crate::platform::name_variants(want) {
        let want_lower = variant.to_lowercase();
        let hit = apps
            .iter()
            .find(|(name, _)| name.to_lowercase() == want_lower)
            .cloned()
            .or_else(|| {
                alias(&variant).and_then(|target| {
                    apps.iter()
                        .find(|(name, _)| process_matches(name, target))
                        .cloned()
                })
            })
            .or_else(|| {
                apps.iter()
                    .find(|(name, _)| process_matches(name, &variant))
                    .cloned()
            })
            .or_else(|| {
                let names: Vec<String> = apps.iter().map(|(n, _)| n.clone()).collect();
                fuzzy_app_match(&variant, &names)
                    .and_then(|hit| apps.iter().find(|(name, _)| *name == hit).cloned())
            });
        if hit.is_some() {
            return hit;
        }
    }
    None
}

/// Запущено ли приложение: tasklist по точному имени и по маске (быстро),
/// затем сопоставление по процессам через PowerShell (для отображаемых имён
/// вроде «Google Chrome» при процессе chrome.exe).
pub fn app_running(name: &str) -> bool {
    let base = name.trim().trim_end_matches(".exe");
    if base.is_empty() {
        return false;
    }
    let tasklist = |filter: &str| -> bool {
        Command::new("tasklist")
            .args(["/FI", filter, "/NH"])
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .to_lowercase()
                    .contains(&format!("{}.exe", base.to_lowercase()))
            })
            .unwrap_or(false)
    };
    if tasklist(&format!("IMAGENAME eq {base}.exe")) {
        return true;
    }
    if base.chars().count() >= 4 && tasklist(&format!("IMAGENAME eq {base}*")) {
        return true;
    }

    // Отображаемое имя: сверяем токены с именами процессов.
    let script = format!(
        "{}{}if ($hit) {{ 'YES' }} else {{ 'NO' }}",
        want_var(&b64(base)),
        r#"
$procs=@(Get-Process);
$tokens=@($want.ToLower().Split(' ') | Where-Object { $_.Length -ge 4 });
$hit=$null;
if ($tokens.Count -gt 0) {
$hit=@($procs | Where-Object { $p=$_; $pn=$p.ProcessName.ToLower(); @($tokens | Where-Object { $pn.Contains($_) }).Count -gt 0 }) | Select-Object -First 1;
}
"#
    );
    ps_run(&script)
        .map(|out| out.trim() == "YES")
        .unwrap_or(false)
}

/// Настоящее установленное приложение из меню «Пуск». `None` — такого
/// приложения нет: подменять его файлом или сайтом нельзя.
/// Каталог установленных приложений для контекста агента.
/// Записи базы приложений: на этой платформе пока каталог не собирается.
pub fn app_db_entries() -> Vec<(String, String, String)> {
    Vec::new()
}

pub fn app_catalog() -> Vec<String> {
    let mut names: Vec<String> = start_apps().into_iter().map(|(n, _)| n).collect();
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
    find_start_app(want).map(|(display, _)| display)
}

/// Имя приложения по русской/приблизительной просьбе (для быстрых путей).
pub fn app_resolve(name: &str) -> String {
    let want = name.trim();
    if want.is_empty() {
        return want.to_string();
    }
    if let Some(display) = app_lookup(want) {
        return display;
    }
    if let Some(target) = alias(want) {
        return target.to_string();
    }
    want.to_string()
}

fn ps_start_direct(exe: &str) -> bool {
    let script = format!(
        "{}try {{ Start-Process $want -ErrorAction Stop; 'STARTED' }} catch {{ 'NOMATCH' }}",
        want_var(&b64(exe))
    );
    matches!(ps_run(&script).as_deref(), Ok("STARTED"))
}

/// ProgId браузера по умолчанию из реестра → исполняемый файл (chrome,
/// msedge…) или SHELL — «определить точно не вышло, откроем через оболочку».
fn default_browser_prog() -> Option<String> {
    let script = r#"
$prog=(Get-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice' -ErrorAction Stop).ProgId;
if ($prog -like '*Chrome*') { 'chrome' }
elseif ($prog -like '*Edge*' -or $prog -like '*MSEdge*') { 'msedge' }
elseif ($prog -like '*Firefox*') { 'firefox' }
elseif ($prog -like '*Opera*') { 'opera' }
elseif ($prog -like '*Brave*') { 'brave' }
elseif ($prog -like '*Yandex*') { 'browser' }
else { 'SHELL' }
"#;
    ps_run(script)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Открыть браузер по умолчанию: известный процесс — с подтверждением,
/// иначе — через оболочку по URL (она сама вызовет браузер по умолчанию).
fn open_default_browser() -> Result<String, String> {
    let prog = default_browser_prog().unwrap_or_else(|| "SHELL".to_string());
    if prog == "SHELL" {
        let script = "try { Start-Process 'https://example.com' -ErrorAction Stop; 'STARTED' } catch { 'NOMATCH' }";
        return if ps_run(script).as_deref() == Ok("STARTED") {
            Ok("Открыл браузер".to_string())
        } else {
            Err("Не удалось открыть браузер по умолчанию".to_string())
        };
    }
    if !ps_start_direct(&prog) {
        return Err(format!("Не удалось открыть браузер по умолчанию ({prog})"));
    }
    // Как и с приложениями: статус запуска — не гарантия, ждём процесс.
    for _ in 0..6 {
        if app_running(&prog) {
            return Ok("Открыл браузер".to_string());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "Не удалось открыть браузер по умолчанию ({prog}): процесс не появился"
    ))
}

/// Открывает приложение: прямой Start-Process (App Paths), затем запись из
/// меню «Пуск» (shell:AppsFolder). Успех подтверждается процессом, кроме
/// пути через AppsFolder: там AppID не даёт имени процесса, и запуск
/// считается успешным по факту передачи оболочке.
pub fn app_open(name: &str) -> Result<String, String> {
    let want = name.trim();
    if want.is_empty() {
        return Err("Пустое имя приложения".to_string());
    }
    if crate::text::compact_name(want) == "браузер" {
        return open_default_browser();
    }

    let canonical = alias(want).unwrap_or(want).to_string();

    // 1. Имя без пробелов — прямой запуск (chrome, notepad, msedge…).
    if !canonical.contains(' ') && !canonical.contains('.') {
        if app_running(&canonical) {
            return Ok(format!("{canonical} уже открыт"));
        }
        if ps_start_direct(&canonical) {
            for _ in 0..6 {
                if app_running(&canonical) {
                    return Ok(format!("Открыл {canonical}"));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            return Err(format!(
                "Не удалось открыть {canonical}: приложение не запустилось"
            ));
        }
    }

    // 2. Отображаемое имя из меню «Пуск».
    if let Some((display, appid)) = find_start_app(&canonical) {
        if app_running(&display) {
            return Ok(format!("{display} уже открыт"));
        }
        let script = format!(
            "try {{ Start-Process ('shell:AppsFolder\' + '{appid}') -ErrorAction Stop; 'STARTED' }} catch {{ 'NOMATCH' }}"
        );
        if ps_run(&script).as_deref() == Ok("STARTED") {
            return Ok(format!("Открыл {display}"));
        }
    }

    // 3. Последний токен: «Google Chrome» → chrome.
    if let Some(last) = canonical.split_whitespace().last()
        && last != canonical
        && ps_start_direct(last)
    {
        for _ in 0..6 {
            if app_running(last) {
                return Ok(format!("Открыл {last}"));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    Err(format!(
        "Не удалось открыть {want}: приложение не найдено среди установленных"
    ))
}

/// Просит приложение закрыться: вежливое CloseMainWindow; приложения без
/// главного окна (трей: Telegram, Discord) закрываются Stop-Process — иначе
/// «закрой X» для них не работал бы вообще. `false` — процесс не найден.
pub fn app_quit(name: &str) -> bool {
    let requested = name.trim();
    if requested.is_empty() {
        return false;
    }
    let app = app_lookup(requested)
        .or_else(|| alias(requested).map(str::to_string))
        .unwrap_or_else(|| requested.to_string());
    let script = format!(
        "{}{}{}",
        want_var(&b64(&app)),
        r#"
$procs=@(Get-Process);
"#,
        r#"
$tokens=@($want.ToLower().Split(' ') | Where-Object { $_.Length -ge 3 });
$hit=@($procs | Where-Object { $_.ProcessName.ToLower() -eq $want.ToLower() }) | Select-Object -First 1;
if (-not $hit) {
$hit=@($procs | Where-Object { $p=$_; $pn=$p.ProcessName.ToLower(); @($tokens | Where-Object { -not $pn.Contains($_) }).Count -eq 0 }) | Select-Object -First 1;
}
if (-not $hit) { 'NOTFOUND' }
else {
if (-not $hit.CloseMainWindow()) { Stop-Process -Id $hit.Id -ErrorAction SilentlyContinue | Out-Null; 'FORCED' }
else { 'SENT' }
}
"#
    );
    match ps_run(&script) {
        Ok(out) if out.trim() == "NOTFOUND" => false,
        Ok(out) => {
            println!("🛑 Закрываю: {app} ({})", out.trim());
            true
        }
        Err(e) => {
            eprintln!("⚠️ Не удалось отправить запрос на выход {app}: {e}");
            false
        }
    }
}

// =========================================================================
// Окна
// =========================================================================

/// Общий скрипт над окном найденного процесса. `$hit` — процесс, `$act` —
/// действие; печатает «имя|заголовок» или NOTFOUND.
const WINDOW_ACT: &str = r#"
if (-not $hit) { 'NOTFOUND' }
else {
$h=$hit.MainWindowHandle;
if ($act -eq 'minimize') { [JWin]::ShowWindowAsync($h, 6) | Out-Null }
elseif ($act -eq 'maximize' -or $act -eq 'fullscreen') { [JWin]::ShowWindowAsync($h, 3) | Out-Null }
elseif ($act -eq 'restore') { [JWin]::ShowWindowAsync($h, 9) | Out-Null }
elseif ($act -eq 'close') { $hit.CloseMainWindow() | Out-Null }
elseif ($act -eq 'focus') { [JWin]::SetForegroundWindow($h) | Out-Null }
('{0}|{1}' -f $hit.ProcessName, $hit.MainWindowTitle)
}
"#;

fn act_on_want(want: &str, action: &str) -> Result<(String, String), String> {
    // `action` приходит только из фиксированного набора window_control —
    // инъекция через него невозможна.
    let script = format!(
        "{}{}\n$act='{}';\n$procs=@(Get-Process | Where-Object {{ $_.MainWindowHandle -ne 0 }});\n{}{}",
        WIN32_HEADER,
        want_var(&b64(want)),
        action,
        FIND_PROC,
        WINDOW_ACT
    );
    let out = ps_run(&script)?;
    if out.trim() == "NOTFOUND" {
        return Err(want.to_string());
    }
    let (proc, title) = out.split_once('|').unwrap_or((out.as_str(), ""));
    Ok((proc.trim().to_string(), title.trim().to_string()))
}

fn act_on_own_window(action: &str) -> String {
    let me = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
        .unwrap_or_default();
    if me.is_empty() {
        return "Моё окно не запущено — я в фоне. Скажи, чьё окно свернуть".to_string();
    }
    match act_on_want(&me, action) {
        Ok((name, _)) => window_speech(action, &name),
        Err(_) => "Моё окно не запущено — я в фоне. Скажи, чьё окно свернуть".to_string(),
    }
}

fn act_on_active(action: &str) -> String {
    let script = format!(
        "{}{}",
        WIN32_HEADER,
        r#"
$h=[JWin]::GetForegroundWindow();
$pid2=[uint32]0; [void][JWin]::GetWindowThreadProcessId($h, [ref]$pid2);
$pr=Get-Process -Id $pid2 -ErrorAction SilentlyContinue;
if (-not $pr) { 'NONE' }
else {
$act=__ACT__;
if ($act -eq 'minimize') { [JWin]::ShowWindowAsync($h, 6) | Out-Null }
elseif ($act -eq 'maximize' -or $act -eq 'fullscreen') { [JWin]::ShowWindowAsync($h, 3) | Out-Null }
elseif ($act -eq 'restore') { [JWin]::ShowWindowAsync($h, 9) | Out-Null }
elseif ($act -eq 'close') { $pr.CloseMainWindow() | Out-Null }
elseif ($act -eq 'focus') { [JWin]::SetForegroundWindow($h) | Out-Null }
$pr.ProcessName
}
"#
        .replace("__ACT__", &format!("'{action}'"))
    );
    match ps_run(&script) {
        Ok(out) if out.trim() == "NONE" => "Не нашёл активное приложение с окном".to_string(),
        Ok(out) => window_speech(action, out.trim()),
        Err(e) => format!("Ошибка: {e}"),
    }
}

fn window_speech(action: &str, name: &str) -> String {
    match action {
        "minimize" => format!("свернул окно {name}"),
        "maximize" => format!("развернул окно {name}"),
        // На Windows «полный экран» — режим самого приложения (F11 и т.п.);
        // максимум, который можно гарантировать снаружи, — разворот окна.
        "fullscreen" => format!("развернул окно {name}"),
        "restore" => format!("вернул окно {name} в обычный размер"),
        "close" => format!("закрыл окно {name}"),
        "focus" => format!("Переключился на {name}"),
        other => format!("Не могу выполнить действие с окном: {other}"),
    }
}

/// Действие над окном браузера: сначала браузер по умолчанию из реестра,
/// затем — распространённые, пока какой-нибудь не окажется среди открытых.
/// Имя «браузер» само по себе процесс не называет, поэтому вслепую не ищем.
fn act_on_browser_window(action: &str) -> String {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(prog) = default_browser_prog().filter(|p| p != "SHELL") {
        candidates.push(prog);
    }
    for c in ["chrome", "msedge", "firefox", "brave", "opera", "browser"] {
        if !candidates.iter().any(|x| x == c) {
            candidates.push(c.to_string());
        }
    }
    for candidate in candidates {
        if let Ok((name, _)) = act_on_want(&candidate, action) {
            return window_speech(action, &name);
        }
    }
    "Не нашёл окно браузера среди открытых".to_string()
}

fn window_list() -> String {
    let script = "Get-Process | Where-Object { $_.MainWindowTitle } | Select-Object -First 12 | ForEach-Object { ('{0}: {1}' -f $_.ProcessName, $_.MainWindowTitle) }";
    match ps_run(script) {
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

pub fn window_control(action: &str, app: Option<&str>) -> String {
    if action == "list" {
        return window_list();
    }
    if sw_flag(action).is_none() && action != "close" && action != "focus" {
        return format!("Не могу выполнить действие с окном: {action}");
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
            act_on_active(action)
        }
        // «Сверни окно браузера» — окно браузера по умолчанию, а не нечто
        // первое попавшееся.
        Some(wanted) if crate::text::compact_name(wanted) == "браузер" => {
            act_on_browser_window(action)
        }
        Some(wanted) => {
            let target = alias(wanted).unwrap_or(wanted).to_string();
            match act_on_want(&target, action) {
                Ok((name, _)) => window_speech(action, &name),
                Err(_) => format!("Приложение {wanted} не найдено среди открытых"),
            }
        }
        None => act_on_own_window(action),
    }
}

/// Оконная команда без выбора: Win32 через PowerShell отдаёт только главное
/// окно процесса (`MainWindowTitle`), поэтому отдельных окон одного
/// приложения здесь не видно — предлагать выбор не из чего.
/// Категорий .desktop на этой ОС нет: терминал ищется по именам.
pub fn terminal_category_names() -> Vec<String> {
    Vec::new()
}

pub fn window_control_checked(
    action: &str,
    app: Option<&str>,
) -> crate::platform::WindowControlOutcome {
    crate::platform::WindowControlOutcome::Done(window_control(action, app))
}

/// Действие над выбранным окном. На Windows список окон не различает окна
/// одного процесса, так что адресуем приложение целиком.
pub fn window_control_target(action: &str, app: &str, _title: &str) -> String {
    window_control(action, Some(app))
}

// =========================================================================
// Музыка: SMTC (кнопки медиа системы)
// =========================================================================

/// Какую клавишу нажать, если SMTC недоступен. VK-коды медиа-клавиш.
fn media_vk(action: &str) -> u8 {
    match action {
        "next" => 0xB0,
        "previous" => 0xB1,
        "stop" => 0xB2,
        _ => 0xB3, // play/pause
    }
}

fn send_media_key(action: &str) -> bool {
    let script = format!(
        r#"
Add-Type -Namespace JB -Name K -MemberDefinition '[DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, System.UIntPtr dwExtraInfo);'
try {{
[JB.K]::keybd_event({vk},0,0,[System.UIntPtr]::Zero);
[JB.K]::keybd_event({vk},0,2,[System.UIntPtr]::Zero);
'KEY'
}} catch {{ 'FAIL' }}
"#,
        vk = media_vk(action)
    );
    matches!(ps_run(&script).as_deref(), Ok("KEY"))
}

/// Управление текущим медиа-плеером через SMTC: состояние читается до и
/// после команды — как на macOS, чтобы «Поставил на паузу» не звучало при
/// тишине. SMTC недоступен — честное нажатие медиа-клавиши.
/// Наблюдение на Windows: честная заглушка — активное окно по имени
/// не читается, плеер по SMTC требует отдельного чтения. Отчёт скажет
/// «данных нет», а не выдумает.
pub fn active_window() -> Option<(String, String)> {
    None
}

pub fn music_status() -> Option<String> {
    None
}

pub fn music_control(action: &str) -> Result<String, String> {
    let action_word = match action {
        "play" => "TryPlayAsync",
        "pause" | "stop" => "TryPauseAsync",
        "next" => "TrySkipNextAsync",
        "previous" => "TrySkipPreviousAsync",
        other => return Err(format!("Не понимаю действие плеера: {other}")),
    };

    let script = format!(
        r#"
try {{
Add-Type -AssemblyName System.Runtime.WindowsRuntime | Out-Null;
$asTaskGeneric=([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0];
function Await($op, $t) {{
$task=$asTaskGeneric.MakeGenericMethod($t).Invoke($null, @($op));
$task.Wait(-1) | Out-Null;
$task.Result
}};
$mgrType=[Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager,Windows.Media.Control,ContentType=WindowsRuntime];
$mgr=Await ($mgrType::RequestAsync()) $mgrType;
$session=$mgr.GetCurrentSession();
if (-not $session) {{ 'NONE' }}
else {{
$before=''+$session.GetPlaybackInfo().PlaybackStatus;
$null=Await ($session.{action_word}()) ([bool]);
$after=''+$session.GetPlaybackInfo().PlaybackStatus;
('{{0}}|{{1}}' -f $before, $after)
}}
}} catch {{ 'NOAPI' }}
"#
    );

    match ps_run(&script)?.trim() {
        "NONE" => Err("Сейчас ничего не играет — управлять нечем".to_string()),
        "NOAPI" => {
            if send_media_key(action) {
                let said = match action {
                    "play" => "Отправил плееру команду «играть»",
                    "pause" => "Отправил плееру команду «пауза»",
                    "stop" => "Отправил плееру команду «стоп»",
                    "next" => "Отправил плееру команду «следующий трек»",
                    _ => "Отправил плееру команду «предыдущий трек»",
                };
                Ok(said.to_string())
            } else {
                Err("Не удалось отправить команду плееру".to_string())
            }
        }
        line => {
            let (before, after) = line
                .split_once('|')
                .ok_or_else(|| format!("Неожиданный ответ системы: {line}"))?;
            let (before, after) = (before.trim(), after.trim());
            if matches!(action, "pause" | "stop") && before == "Paused" {
                return Err("Музыка уже на паузе".to_string());
            }
            if action == "play" && before == "Playing" {
                return Err("Музыка уже играет".to_string());
            }
            let expected_change = match action {
                "play" => before != "Playing",
                "pause" | "stop" => before != "Paused" && before != "Stopped",
                _ => true,
            };
            if expected_change && !matches!(action, "next" | "previous") && after == before {
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
    }
}

// =========================================================================
// Системные действия
// =========================================================================

/// Громкость через CoreAudio (IAudioEndpointVolume): абсолютные проценты.
const VOLUME_CS: &str = r#"
Add-Type @'
using System;
using System.Runtime.InteropServices;
namespace JA {
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject { }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int NotImpl1();
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice ppDevice);
}
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(ref Guid iid, int dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
}
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
    int RegisterControlChangeNotify(IntPtr pNotify);
    int UnregisterControlChangeNotify(IntPtr pNotify);
    int GetChannelCount(out uint pnChannelCount);
    int SetMasterVolumeLevel(float fLevelDB, ref Guid pguidEventContext);
    int SetMasterVolumeLevelScalar(float fLevel, ref Guid pguidEventContext);
    int SetChannelVolumeLevel(uint nChannel, float fLevelDB, ref Guid pguidEventContext);
    int SetChannelVolumeLevelScalar(uint nChannel, float fLevel, ref Guid pguidEventContext);
    int SetMute([MarshalAs(UnmanagedType.Bool)] bool bMute, ref Guid pguidEventContext);
    int GetMute(out bool pbMute);
}
public static class VolumeControl {
    public static void Set(float scalar) {
        var enumerator = (IMMDeviceEnumerator)(object)new MMDeviceEnumeratorComObject();
        IMMDevice device;
        Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(0, 1, out device));
        Guid iid = typeof(IAudioEndpointVolume).GUID;
        object o;
        Marshal.ThrowExceptionForHR(device.Activate(ref iid, 23, IntPtr.Zero, out o));
        var vol = (IAudioEndpointVolume)o;
        Guid ctx = Guid.Empty;
        Marshal.ThrowExceptionForHR(vol.SetMasterVolumeLevelScalar(scalar, ref ctx));
        Marshal.ThrowExceptionForHR(vol.SetMute(false, ref ctx));
    }
}
}
'@
"#;

pub fn set_volume(percent: u8) -> bool {
    let scalar = (percent as f32 / 100.0).clamp(0.0, 1.0);
    let script = format!("{VOLUME_CS}[JA.VolumeControl]::Set({scalar:.2}f); 'OK'");
    matches!(ps_run(&script).as_deref(), Ok("OK"))
}

/// Readback громкости на Windows не реализован: честное «не проверено»
/// вместо выдуманного значения.
pub fn get_volume() -> Option<u8> {
    None
}

pub fn set_brightness(percent: u8) -> bool {
    let script = format!(
        r#"
$m=@(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop);
if ($m.Count -eq 0) {{ throw 'нет управляемого дисплея' }};
$m | ForEach-Object {{ $null = $_.WmiSetBrightness(1, {percent}) }};
'OK'
"#
    );
    matches!(ps_run(&script).as_deref(), Ok("OK"))
}

pub fn screenshot() -> Option<String> {
    let path = crate::tools::temp_path("jarvis_screenshot.png");
    // Старый файл не должен сойти за свежий снимок.
    let _ = std::fs::remove_file(&path);
    let script = format!(
        r#"
Add-Type -AssemblyName System.Windows.Forms,System.Drawing | Out-Null;
Add-Type -Namespace JD -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();' | Out-Null;
[JD.Dpi]::SetProcessDPIAware() | Out-Null;
$b=[System.Windows.Forms.SystemInformation]::VirtualScreen;
$bmp=New-Object System.Drawing.Bitmap $b.Width, $b.Height;
$g=[System.Drawing.Graphics]::FromImage($bmp);
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size);
$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{}'));
$bmp.Save($p, [System.Drawing.Imaging.ImageFormat]::Png);
'OK'
"#,
        b64(&path)
    );
    match ps_run(&script) {
        Ok(_) if std::path::Path::new(&path).exists() => Some(path),
        _ => None,
    }
}

pub fn lock_screen() -> bool {
    Command::new("rundll32")
        .args(["user32.dll,LockWorkStation"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn sleep_now() -> bool {
    // 0,1,0 = приостановить, не переходить в спящий режим принудительно
    // и не требовать подтверждения.
    Command::new("rundll32")
        .args(["powrprof.dll,SetSuspendState", "0,1,0"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn notify(title: &str, body: &str) -> bool {
    // Процесс живёт ~6 секунд, чтобы всплывающее уведомление успело
    // появиться; запускается в фоне, чтобы не тормозить ответ.
    let script = format!(
        r#"
Add-Type -AssemblyName System.Windows.Forms | Out-Null;
$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{}'));
$b=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{}'));
$ni=New-Object System.Windows.Forms.NotifyIcon;
$ni.Icon=[System.Drawing.SystemIcons]::Information;
$ni.Visible=$true;
$ni.ShowBalloonTip(5000, $t, $b, [System.Windows.Forms.ToolTipIcon]::Info);
Start-Sleep -Milliseconds 5500;
$ni.Dispose();
"#,
        b64(title),
        b64(body)
    );
    ps_command(&script)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .is_ok()
}

pub fn clipboard_read() -> Option<String> {
    ps_run("Get-Clipboard -Raw").ok()
}

pub fn clipboard_write(text: &str) -> bool {
    let script = format!(
        "Set-Clipboard -Value ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{}'))); 'OK'",
        b64(text)
    );
    matches!(ps_run(&script).as_deref(), Ok("OK"))
}

pub fn system_info() -> String {
    let script = r#"
$os=Get-CimInstance Win32_OperatingSystem;
('OS:{0} {1}, Arch:{2}' -f $os.Caption.Trim(), $os.Version, $env:PROCESSOR_ARCHITECTURE)
"#;
    ps_run(script).unwrap_or_else(|_| "Информация недоступна".to_string())
}

// =========================================================================
// Приложения Apple-экосистемы на Windows не существуют — честно
// =========================================================================

pub fn send_message(to: &str, _text: &str) -> Result<String, String> {
    Err(format!(
        "Отправка сообщений через системное приложение не поддерживается на Windows ({to})"
    ))
}

pub fn add_note(_title: &str, _body: &str) -> Result<String, String> {
    Err("Заметки системного приложения не поддерживаются на Windows".to_string())
}

pub fn create_event(_title: &str, _date: &str, _time: &str) -> Result<String, String> {
    Err("Календарь системного приложения не поддерживается на Windows".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encodes_utf8() {
        assert_eq!(b64("hello"), "aGVsbG8=");
        assert_eq!(b64(""), "");
        assert_eq!(b64("hi!"), "aGkh");
        // Кириллица — два байта UTF-8 на букву: «ё» → 0xD1 0x91 → «0ZE=».
        assert_eq!(b64("ё"), "0ZE=");
    }

    #[test]
    fn start_apps_parse_pairs() {
        let out = "Google Chrome|Chrome\n\
                   Calculator|Microsoft.WindowsCalculator_8wekyb3d8bbwe!App\n\
                   мусор без разделителя\n";
        let apps = parse_start_apps(out);
        assert_eq!(apps.len(), 2);
        assert_eq!(apps[0].0, "Google Chrome");
        assert_eq!(apps[1].1, "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App");
    }

    #[test]
    fn media_vk_maps_actions() {
        assert_eq!(media_vk("next"), 0xB0);
        assert_eq!(media_vk("previous"), 0xB1);
        assert_eq!(media_vk("stop"), 0xB2);
        assert_eq!(media_vk("pause"), 0xB3);
        assert_eq!(media_vk("play"), 0xB3);
    }
}