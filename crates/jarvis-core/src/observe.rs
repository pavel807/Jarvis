//! Слой наблюдения среды: ассистент видит систему, а не догадывается о ней.
//!
//! Снимок собирается из дешёвых чтений платформенного слоя (окна, активное
//! окно, плеер, буфер, громкость) и вызывается перед действием и после
//! каждого шага агентного цикла — перечитывание после шага обязательная
//! часть цикла, а не опция. Каждая позиция честная: не удалось прочитать —
//! так и пишем, пустота не выдаётся за «сообщений нет».

use crate::platform;

/// Непрочитанное, увиденное в заголовке окна: приложения-мессенджеры и
/// браузеры выносят счётчик в заголовок («(3) Telegram», «Почта — 12»).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unread {
    pub app: String,
    pub count: u32,
    /// Откуда число: заголовок окна — единственный честный источник на
    /// этой платформе; выдумывать числа из других мест запрещено.
    pub source: &'static str,
}

/// Снимок состояния среды на момент чтения.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    /// Окна с заголовками: (приложение, заголовок).
    pub windows: Vec<(String, String)>,
    /// Заголовок активного окна, если получилось прочитать.
    pub active: Option<String>,
    /// Состояние плеера («Playing», «Paused») — без выдуманного трека.
    pub player: Option<String>,
    /// Начало буфера обмена (первые 100 символов, переносы схлопнуты).
    pub clipboard: Option<String>,
    pub volume: Option<u8>,
    pub unread: Vec<Unread>,
    /// false — наблюдение недоступно (нет графической сессии, нет
    /// утилит): всё остальное тогда пусто и это отдельный явный факт.
    pub available: bool,
}

/// Счётчик непрочитанных в заголовке окна: «(3) Telegram», «Telegram (3)»,
/// «(12) Входящие — Почта». Возвращает только настоящее число из скобок,
/// стоящее у края заголовка.
pub fn unread_from_title(title: &str) -> Option<u32> {
    let t = title.trim();
    let take = |s: &str| -> Option<u32> {
        let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == s.len() && !s.is_empty() {
            s.parse().ok()
        } else {
            None
        }
    };
    // Префикс «(3) …» — типовой у мессенджеров.
    if let Some(rest) = t.strip_prefix('(') {
        if let Some(close) = rest.find(')') {
            if let Some(n) = take(&rest[..close]) {
                return Some(n);
            }
        }
    }
    // Суффикс «… (3)» — типовой у браузеров и почты.
    if let Some(open) = t.rfind(" (") {
        let inner = &t[open + 2..];
        if let Some(close) = inner.find(')') {
            if close == inner.len() - 1 {
                if let Some(n) = take(&inner[..close]) {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Приложение для счётчика: из заголовка, если класс пуст, иначе класс.
fn unread_entries(windows: &[(String, String)]) -> Vec<Unread> {
    let mut out = Vec::new();
    for (app, title) in windows {
        if let Some(count) = unread_from_title(title) {
            let name = if app.trim().is_empty() {
                title.clone()
            } else {
                app.clone()
            };
            out.push(Unread {
                app: name,
                count,
                source: "заголовок окна",
            });
        }
    }
    out
}

/// Снимок среды: только чтения, ничего не меняет.
pub fn snapshot() -> Snapshot {
    let list = platform::window_control("list", None);
    // Честная недоступность: платформенный слой возвращает текст-причину,
    // а не список — принять его за окна нельзя.
    let unavailable = list.starts_with("Не могу")
        || list.contains("нет графической сессии")
        || list.contains("требует wmctrl");
    let mut windows: Vec<(String, String)> = Vec::new();
    if !unavailable && list != "Открытых окон не нашёл" {
        for entry in list.split("; ") {
            if let Some((app, title)) = entry.split_once(": ") {
                windows.push((app.trim().to_string(), title.trim().to_string()));
            }
        }
    }
    let active = platform::active_window().map(|(_, title)| title.trim().to_string());
    let player = platform::music_status().filter(|s| {
        !s.is_empty() && s != "неизвестно"
    });
    let clipboard = platform::clipboard_read().map(|s| {
        let flat: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
        let flat = flat.trim().to_string();
        flat.chars().take(100).collect()
    });
    Snapshot {
        unread: unread_entries(&windows),
        available: !unavailable,
        windows,
        active,
        player,
        clipboard,
        volume: platform::get_volume(),
    }
}

/// Короткий текст снимка для контекста модели: то, что видно системе
/// сейчас. Каждая пропущенная позиция объяснена, а не спрятана.
pub fn brief(s: &Snapshot) -> String {
    if !s.available {
        return "ОБСТАНОВКА: наблюдение недоступно — нет доступа к окнам (графическая сессия или wmctrl/xdotool). Опирайся на другие инструменты и честно скажи, если для цели нужны окна.".to_string();
    }
    let mut lines: Vec<String> = Vec::new();
    if s.windows.is_empty() {
        lines.push("Окна: открытых окон с заголовками не нашёл".to_string());
    } else {
        let shown: Vec<String> = s
            .windows
            .iter()
            .take(12)
            .map(|(app, t)| format!("{app} — «{t}»"))
            .collect();
        lines.push(format!("Окна: {}", shown.join("; ")));
    }
    match &s.active {
        Some(a) => lines.push(format!("Активное окно: «{a}»")),
        None => lines.push("Активное окно: не удалось определить".to_string()),
    }
    match &s.player {
        Some(p) => lines.push(format!("Плеер: {p}")),
        None => lines.push("Плеер: данных нет".to_string()),
    }
    match &s.clipboard {
        Some(c) if !c.is_empty() => lines.push(format!("Буфер: «{c}»")),
        _ => lines.push("Буфер: пуст или нечитаем".to_string()),
    }
    if let Some(v) = s.volume {
        lines.push(format!("Громкость: {v}%"));
    }
    if s.unread.is_empty() {
        lines.push("Непрочитанные: счётчиков в заголовках окон нет (это не значит, что сообщений нет — если нужен точный счётчик, открой приложение и прочитай элемент через ui_read)".to_string());
    } else {
        let list: Vec<String> = s
            .unread
            .iter()
            .map(|u| format!("{} — {} ({})", u.app, u.count, u.source))
            .collect();
        lines.push(format!("Непрочитанные: {}", list.join("; ")));
    }
    format!("ОБСТАНОВКА:\n{}", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unread_prefix_and_suffix() {
        assert_eq!(unread_from_title("(3) Telegram"), Some(3));
        assert_eq!(unread_from_title("Telegram Desktop (12)"), Some(12));
        // Число в середине заголовка — не счётчик непрочитанных: настоящий
        // счётчик приложения ставят у края. Середину не выдумываем.
        assert_eq!(unread_from_title("Почта — Входящие (2) — Gmail"), None);
    }

    #[test]
    fn no_unread_in_plain_titles() {
        assert_eq!(unread_from_title("Google — Веб-поиск"), None);
        assert_eq!(unread_from_title("(в разработке) черновик"), None);
        // Число не у края — не счётчик непрочитанных.
        assert_eq!(unread_from_title("страница (5) из 10"), None);
        assert_eq!(unread_from_title(""), None);
    }

    #[test]
    fn big_counts_parse() {
        assert_eq!(unread_from_title("(999+) WhatsApp"), None); // не число — не выдумываем
        assert_eq!(unread_from_title("(42) Чат"), Some(42));
    }

    #[test]
    fn brief_marks_unavailable_explicitly() {
        let s = Snapshot {
            available: false,
            ..Default::default()
        };
        let b = brief(&s);
        assert!(b.contains("наблюдение недоступно"));
    }

    #[test]
    fn brief_lists_unread_and_gaps_honestly() {
        let s = Snapshot {
            available: true,
            windows: vec![("Telegram".into(), "(3) Telegram".into())],
            active: None,
            player: None,
            clipboard: None,
            volume: Some(45),
            unread: vec![Unread {
                app: "Telegram".into(),
                count: 3,
                source: "заголовок окна",
            }],
        };
        let b = brief(&s);
        assert!(b.contains("Telegram — 3"));
        assert!(b.contains("Активное окно: не удалось определить"));
        assert!(b.contains("Громкость: 45%"));
    }
}
