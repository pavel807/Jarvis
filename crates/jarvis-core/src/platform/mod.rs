//! Платформенный слой: одно действие — одна логика, три реализации.
//!
//! Каждая ОС-реализация (`macos`, `linux`, `windows`) предоставляет один и
//! тот же набор функций: окна, приложения, музыка, громкость, яркость,
//! скриншот, блокировка, сон, уведомления, буфер обмена, системная
//! информация. Вызывающий код (`tools`, `commands`, `llm`) не знает, под
//! какой ОС он работает, — он получает либо готовую фразу для озвучки,
//! либо честную причину неудачи.
//!
//! Правила слоя:
//! - успех подтверждается делом, а не кодом возврата: открытие приложения
//!   проверяется процессом, скриншот — существованием файла, музыка —
//!   изменением состояния плеера;
//! - если действие на этой ОС невозможно, возвращается не «ОК», а причина
//!   с подсказкой, что установить или включить;
//! - общая часть (сопоставление имён, алиасы, подбор по сходству) живёт
//!   здесь и покрыта тестами, которые идут на любой ОС.

#[cfg(target_os = "macos")]
pub(crate) mod macos;
#[cfg(target_os = "macos")]
use macos as imp;

// Реализации Linux и Windows зависят только от std (утилиты вызываются
// процессами), поэтому компилируются на всех ОС: так их чистая логика —
// разбор вывода wmctrl/playerctl/tasklist, словари, base64 — проверяется
// тестами на любой машине разработчика, а не только на целевой ОС.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod linux;
#[cfg(target_os = "linux")]
use linux as imp;

#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
mod windows;
#[cfg(target_os = "windows")]
use windows as imp;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod unsupported;
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
use unsupported as imp;

pub use imp::*;

/// Конкретное окно-кандидат. `app` — имя процесса/приложения, по которому
/// действие найдёт окно повторно (индексы окон на macOS меняются после
/// сворачивания, поэтому запоминать номер окна нельзя).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowTarget {
    pub app: String,
    pub title: String,
    /// На macOS и Hyprland состояние известно; на остальных ОС — `false`.
    pub minimized: bool,
}

/// Результат оконной команды. `Ambiguous` — у приложения несколько окон,
/// и выбор нужно оставить пользователю, а не делать первое попавшееся.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowControlOutcome {
    Done(String),
    Ambiguous(Vec<WindowTarget>),
}

pub mod shell;
pub mod desktop;

// =========================================================================
// Словарь русских названий приложений
// =========================================================================

/// Кросс-платформенные алиасы: продукт называется одинаково на всех ОС.
/// Значение — каноничное имя, по которому каждая ОС ищет своё приложение.
pub const COMMON_ALIASES: &[(&str, &str)] = &[
    ("гуглхром", "Google Chrome"),
    ("хром", "Google Chrome"),
    ("телеграм", "Telegram"),
    ("телега", "Telegram"),
    ("ватсап", "WhatsApp"),
    ("зум", "Zoom"),
    ("слак", "Slack"),
    ("дискорд", "Discord"),
    ("нотион", "Notion"),
    ("фигма", "Figma"),
    ("спотифай", "Spotify"),
    ("файрфокс", "Firefox"),
    ("фаерфокс", "Firefox"),
    ("файерфокс", "Firefox"),
    ("файтфокс", "Firefox"),
    ("файрофокс", "Firefox"),
    ("зед", "Zed"),
    ("вскод", "Code"),
    ("код", "Code"),
    ("икскод", "Xcode"),
    ("эксель", "Excel"),
    ("ворд", "Word"),
    ("пауэрпоинт", "PowerPoint"),
    ("фотошоп", "Photoshop"),
    ("иллюстратор", "Illustrator"),
    ("премьер", "Premiere"),
    ("афтерэффекты", "After Effects"),
    ("дэвинсинк", "DaVinci Resolve"),
];

/// Просьба открыть терминал («терминал», «консоль», «командная строка»).
/// Терминал нельзя маппить на одно имя приложения: на разных системах
/// это foot, kitty, alacritty, Terminal… Поэтому просьба — категория,
/// а приложение резолвится по установленным эмуляторам (см.
/// `terminal_app_name`).
pub fn is_terminal_request(name: &str) -> bool {
    let c = crate::text::compact_name(name);
    ["терминал", "терминалл", "консоль", "команднаястрока", "shell", "шэл"]
        .iter()
        .any(|k| c.contains(k))
}

/// Первый установленный эмулятор терминала из известного списка. Порядок —
/// от современных к дедовским; возврат — каноничное имя .desktop.
pub fn terminal_app_name(installed: &[String]) -> Option<String> {
    const CANDIDATES: &[&str] = &[
        "foot", "footclient", "ghostty", "kitty", "alacritty", "wezterm",
        "konsole", "gnome-terminal", "gnome-terminal-server", "xfce4-terminal",
        "tilix", "terminator", "qterminal", "lxterminal", "xterm", "st", "urxvt",
    ];
    // Демоны и службы — не то, что видит пользователь: «Foot Server»
    // открывает фоновый процесс без окна, а просили терминал.
    let is_daemon = |n: &str| {
        let l = n.to_lowercase();
        l.contains("server") || l.contains("daemon")
    };
    // Нормализация для сравнения: «GNOME Terminal» и «gnome-terminal» —
    // одно и то же, дефисы и пробелы не различаем.
    let norm = |n: &str| n.to_lowercase().replace(['-', '_'], " ");
    // Первый проход: точное имя («Foot» == foot). Второй: подстрока
    // («GNOME Terminal» содержит gnome terminal), но не демон.
    for cand in CANDIDATES {
        if let Some(hit) = installed
            .iter()
            .find(|n| !is_daemon(n) && norm(n) == *cand)
        {
            return Some(hit.clone());
        }
    }
    for cand in CANDIDATES {
        let cand = cand.replace('-', " ");
        if let Some(hit) = installed
            .iter()
            .find(|n| !is_daemon(n) && norm(n).contains(&cand))
        {
            return Some(hit.clone());
        }
    }
    None
}

/// Окно самого Джарвиса: пользователь называет его «Jarvis», «Джарвис»,
/// «голосовой помощник», «ассистент». Такие цели нельзя отправлять в
/// `find_process` как имя чужого приложения: процесс называется `jarvis-app`,
/// а «голосового помощника» вообще не совпадёт ни с одним процессом.
pub fn is_self_window_target(target: &str) -> bool {
    let compact = crate::text::compact_name(target);
    if compact.chars().count() < 3 {
        return false;
    }
    const OWN: &[&str] = &[
        "jarvis",
        "джарвис",
        "джавиз",
        "джарвиз",
        "джарис",
        "жарвис",
        "ярвис",
        "assistant",
        "ассистент",
        "голосовойпомощник",
        "голосовогопомощника",
        "голосовомупомощнику",
        "помощник",
        "помощника",
        "помощнику",
    ];
    OWN.iter()
        .any(|word| compact == *word || compact.contains(word) || word.contains(&compact))
}

/// Грубая основа русского слова: снимает падежное окончание, чтобы
/// «телеграма», «хрома», «музыку» сходились с алиасами «телеграм», «хром»,
/// «музыка». Латинские имена и слишком короткие слова не трогаем.
pub fn ru_stem(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.chars().any(|c| c.is_ascii_alphabetic()) {
        return lower;
    }
    let count = lower.chars().count();
    const SUFFIXES: &[&str] = &[
        "ами", "ями", "ов", "ев", "ом", "ем", "ой", "ей", "ах", "ях", "а", "я", "у", "ю", "е", "ы",
        "и", "ь",
    ];
    for suffix in SUFFIXES {
        if lower.ends_with(suffix) {
            let rest = count.saturating_sub(suffix.chars().count());
            if rest >= 3 {
                return lower[..lower.len() - suffix.len()].to_string();
            }
        }
    }
    lower
}

/// Формы просьбы для сопоставления: как сказано и с грубой основой. Первая
/// форма всегда исходная — точное имя приложения важнее догадки.
pub fn name_variants(name: &str) -> Vec<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut out = vec![trimmed.to_string()];
    let stemmed = trimmed
        .split_whitespace()
        .map(ru_stem)
        .collect::<Vec<_>>()
        .join(" ");
    if !stemmed.is_empty() && stemmed != trimmed.to_lowercase() {
        out.push(stemmed);
    }
    out
}

/// Поиск в списках алиасов: точное совпадение всей фразы — приоритет;
/// нечёткий префикс/суффикс допускается только для однословных просьб,
/// иначе «яндекс карты» ловится на алиас «карты» и открывает не то.
pub fn lookup_alias(
    word: &str,
    tables: &[&[(&'static str, &'static str)]],
) -> Option<&'static str> {
    let wc = crate::text::compact_name(word);
    if wc.is_empty() {
        return None;
    }
    for table in tables {
        if let Some((_, target)) = table.iter().find(|(alias, _)| wc == *alias) {
            return Some(target);
        }
    }
    // Падежная форма: «телеграма» → основа «телеграм» → алиас «телеграм».
    let stem = ru_stem(&wc);
    for table in tables {
        if let Some((_, target)) = table.iter().find(|(alias, _)| ru_stem(alias) == stem) {
            return Some(target);
        }
    }
    if word.trim().contains(char::is_whitespace) {
        return None;
    }
    for table in tables {
        if let Some((_, target)) = table
            .iter()
            .find(|(alias, _)| wc.starts_with(alias) || wc.ends_with(alias))
        {
            return Some(target);
        }
    }
    None
}

// =========================================================================
// Сопоставление имён приложений (общая логика для всех ОС)
// =========================================================================

/// Совпадает ли имя приложения/процесса с просьбой: компактно и через
/// транслитерацию, чтобы «Zed» ↔ «зед» и «Telegram» ↔ «телеграм» сходились.
pub fn process_matches(candidate: &str, want: &str) -> bool {
    let wc = crate::text::compact_name(want);
    let cc = crate::text::compact_name(candidate);
    if wc.is_empty() || cc.is_empty() {
        return false;
    }
    if cc == wc || cc.contains(&wc) {
        return true;
    }
    // Падежная форма: «телеграма» → «телеграм», «хрома» → «хром».
    let wc_stem = ru_stem(&wc);
    if !wc_stem.is_empty() && (cc == wc_stem || cc.contains(&wc_stem)) {
        return true;
    }
    let ct = crate::text::compact_name(&crate::text::translit_ru(candidate));
    let wt = crate::text::compact_name(&crate::text::translit_ru(want));
    let wt_stem = crate::text::compact_name(&crate::text::translit_ru(&wc_stem));
    ct == wt
        || ct.contains(&wt)
        || (!wt_stem.is_empty() && (ct == wt_stem || ct.contains(&wt_stem)))
}

/// Акронимы держим целиком: «Arduino IDE» → [arduino, ide] — иначе «IDE»
/// распадается на i/d/e и «электронный дневник» совпадает с ним по первым
/// буквам (реальный баг: дневник открывал Arduino IDE).
pub fn app_name_tokens(name: &str) -> Vec<String> {
    let chars: Vec<char> = name.chars().collect();
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() || c == '-' || c == '_' || c == '.' {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let prev = if i > 0 { chars[i - 1] } else { c };
        let next_lower = chars.get(i + 1).copied().is_some_and(char::is_lowercase);
        // Граница слова: строчная → Заглавная или Заглавная-Заглавная → строчная.
        let starts_new = c.is_uppercase()
            && !cur.is_empty()
            && (prev.is_lowercase() || (prev.is_uppercase() && next_lower));
        if starts_new {
            tokens.push(std::mem::take(&mut cur));
        }
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

/// Нечёткий подбор приложения: токены просьбы (транслит) сверяем с токенами
/// имени приложения через word_similarity, в том числе с ОКНАМИ-ПРЕФИКСАМИ
/// токена («фокс» ≈ «fox» в «firefox»). Уверенность: два совпавших слова
/// (порог 0.6), один сильный токен (0.72) или очень близкое однословное
/// совпадение (0.8). Сильный одиночный выбирает кандидата только с запасом
/// от второго лучшего — иначе «пиджин» уехал бы в похожее название.
pub fn fuzzy_app_match(want: &str, candidates: &[String]) -> Option<String> {
    let want_tokens: Vec<String> = want
        .split_whitespace()
        .map(|w| crate::text::compact_name(&crate::text::translit_ru(w)))
        .filter(|w| w.chars().count() >= 4)
        .collect();
    if want_tokens.is_empty() {
        return None;
    }

    // Сходство токена просьбы с ЛУЧШИМ окном токена приложения. STT режет
    // имя на слова в произвольных местах («фаер фокс» = fire + fox), поэтому
    // сравниваем со ВСЕМИ окнами подстрок длины просьбы ±1: «фокс» находит
    // «fox» внутри «firefox» (0.75), хотя целое слово даёт лишь 0.14.
    let token_sim = |wt: &str, app_tokens: &[String]| -> f32 {
        let mut best = 0.0f32;
        let wt_len = wt.chars().count();
        for at_raw in app_tokens.iter().filter(|at| at.chars().count() >= 2) {
            // Фонетика латинского x: произносится как «кс», и STT пишет его
            // двумя буквами («фокс» = fox). Разворачиваем x → ks, чтобы окна
            // сравнивались со звучанием, а не с буквами.
            let expanded = at_raw.replace('x', "ks");
            let at = expanded.as_str();
            best = best.max(crate::text::word_similarity(wt, at));
            let chars: Vec<char> = at.chars().collect();
            for size in wt_len.saturating_sub(1).max(3)..=wt_len + 1 {
                if size >= chars.len() {
                    break;
                }
                for start in 0..=(chars.len() - size) {
                    let window: String = chars[start..start + size].iter().collect();
                    best = best.max(crate::text::word_similarity(wt, &window));
                }
            }
        }
        best
    };

    let mut scored: Vec<(usize, f32, usize, &String)> = Vec::new();
    for candidate in candidates {
        let app_tokens = app_name_tokens(candidate);
        let mut hits = 0usize;
        let mut best_sim = 0.0f32;
        for wt in &want_tokens {
            let sim = token_sim(wt, &app_tokens);
            if sim >= 0.6
                || (wt.len() >= 4
                    && app_tokens.iter().any(|at| {
                        at.starts_with(wt.as_str()) || wt.starts_with(at.as_str())
                    }))
            {
                hits += 1;
            }
            best_sim = best_sim.max(sim);
        }
        // Сильный одиночный токен считается за два: «фаер фокс» имеет одно
        // честное совпадение (fox), но его достаточно для выбора цели.
        // Реальные попадания сохраняются отдельно — маржа против близкого
        // второго кандидата оценивается по ним.
        let effective = if best_sim >= 0.72 { hits.max(2) } else { hits };
        let single_strong = want_tokens.len() == 1 && best_sim >= 0.8;
        if effective >= 2 || single_strong {
            scored.push((effective, best_sim, hits, candidate));
        }
    }
    // Ранжирование по баллам, при равенстве — по сходству. Единственная
    // опасность — выигравший по одиночному сильному токену кандидат с
    // сомнительным запасом: если второй лучший почти так же близок, честнее
    // отказаться, чем открыть не то приложение.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
    match scored.first() {
        Some((_, top_sim, real_hits, name)) => {
            // Победа одним токеном при почти равном втором кандидате —
            // неоднозначность: честнее не угадывать, чем открыть не то.
            if *real_hits < 2
                && let Some((_, second_sim, _, _)) = scored.get(1)
                && top_sim - second_sim < 0.1
            {
                return None;
            }
            Some((*name).clone())
        }
        None => None,
    }
}

// =========================================================================
// Проверка наличия внешней утилиты
// =========================================================================

/// Есть ли бинарь в PATH. Ответ кэшируется: проверки идут в горячих путях
/// (каждое действие с окном/громкостью), а PATH за сессию не меняется.
pub fn command_exists(bin: &str) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(bin) {
        return hit;
    }
    let hit = std::env::var_os("PATH")
        .map(|paths| search_path_in(bin, &paths))
        .unwrap_or(false);
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(bin.to_string(), hit);
    hit
}

/// Поиск исполняемого файла по списку каталогов PATH (чистая функция —
/// тестируется без подмены окружения процесса).
pub fn search_path_in(bin: &str, paths: &std::ffi::OsStr) -> bool {
    for dir in std::env::split_paths(paths) {
        if dir.join(bin).is_file() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_app_names_resolve() {
        assert!(process_matches("Google Chrome", "Google Chrome"));
        assert!(process_matches("Google Chrome", "chrome"));
        assert!(process_matches("Telegram", "телеграм"));
        assert!(process_matches("Zed", "зед"));
        assert!(process_matches("Adobe Premiere Pro 2026", "Premiere"));
        // «Safari» и «Safari Technology Preview» различаем по точному совпадению:
        // неточное вхождение не должно превращать сафари в превью-версию.
        assert!(!process_matches("Safari", "Safari Technology Preview"));
    }

    #[test]
    fn common_alias_maps_across_platforms() {
        // Общий словарь не знает мак-специфичных приложений: их добавляют
        // таблицы самих ОС, а не общая часть.
        assert_eq!(
            lookup_alias("хром", &[COMMON_ALIASES]),
            Some("Google Chrome")
        );
        assert_eq!(
            lookup_alias("гугл хром", &[COMMON_ALIASES]),
            Some("Google Chrome")
        );
        assert_eq!(lookup_alias("телега", &[COMMON_ALIASES]), Some("Telegram"));
        assert_eq!(
            lookup_alias("неизвестное приложение", &[COMMON_ALIASES]),
            None
        );
        // «яндекс карты» — сайт, а не приложение: суффикс «карты» через
        // границу слова не должен совпадать с алиасом.
        let mac = &[("карты", "Maps")][..];
        assert_eq!(lookup_alias("яндекс карты", &[mac, COMMON_ALIASES]), None);
        assert_eq!(lookup_alias("карты", &[mac, COMMON_ALIASES]), Some("Maps"));
        // Таблица ОС имеет приоритет над общей.
        let os = &[("хром", "Chromium")][..];
        assert_eq!(
            lookup_alias("хром", &[os, COMMON_ALIASES]),
            Some("Chromium")
        );
    }

    #[test]
    fn app_name_tokens_split_camel_and_spaces() {
        assert_eq!(
            app_name_tokens("AnycubicSlicerNext"),
            vec!["anycubic", "slicer", "next"]
        );
        assert_eq!(app_name_tokens("Google Chrome"), vec!["google", "chrome"]);
        // Акроним не распадается на буквы — иначе «электронный дневник»
        // матчится с «IDE» по первым буквам (реальный регресс).
        assert_eq!(app_name_tokens("Arduino IDE"), vec!["arduino", "ide"]);
        assert_eq!(app_name_tokens("VSCode"), vec!["vs", "code"]);
    }

    #[test]
    fn fuzzy_match_finds_slicer_by_voice_spelling() {
        let bundles = vec![
            "Google Chrome".to_string(),
            "AnycubicSlicerNext".to_string(),
            "Safari".to_string(),
        ];
        // «энкубик слайсер некст» — так STT слышит Anycubic Slicer Next:
        // точного вхождения транслита нет, но два токена сходятся.
        assert_eq!(
            fuzzy_app_match("энкубик слайсер некст", &bundles).as_deref(),
            Some("AnycubicSlicerNext")
        );
    }

    #[test]
    fn fuzzy_match_requires_confidence() {
        let bundles = vec!["Mail".to_string(), "Numbers".to_string()];
        // Одно слово без близкого совпадения — не подбираем ничего.
        assert_eq!(fuzzy_app_match("почта новая", &bundles), None);
    }

    #[test]
    fn fuzzy_match_rejects_school_diary() {
        let bundles = vec![
            "Arduino IDE".to_string(),
            "Google Chrome".to_string(),
            "Safari".to_string(),
        ];
        // «электронный дневник» — не приложение: раньше совпадал с токенами
        // акронима IDE по первым буквам и открывал Arduino IDE.
        assert_eq!(fuzzy_app_match("электронный дневник", &bundles), None);
    }

    #[test]
    fn search_path_finds_and_rejects() {
        // Сам тест-бинарник лежит на диске: если передать каталог с ним —
        // имя найдётся, если пустой каталог — нет.
        let exe = std::env::current_exe().expect("путь теста");
        let dir = exe.parent().expect("каталог теста");
        let bin = exe.file_name().unwrap();
        assert!(search_path_in(bin.to_str().unwrap(), dir.as_os_str()));
        assert!(!search_path_in("jarvis-нет-такого-бинаря", dir.as_os_str()));
    }
}

#[cfg(test)]
mod alias_tests {
    use super::imp::alias;

    #[test]
    fn firefox_stt_garbles_resolve() {
        // STT искажает «Firefox» по-разному: все частые варианты должны
        // находить приложение, а не уезжать в выдуманный инструмент.
        for said in ["файрфокс", "фаер фокс", "файт фокс", "файерфокс", "файрофокс"] {
            assert_eq!(alias(said), Some("Firefox"), "алиас для «{said}» потерян");
        }
    }
}

#[cfg(test)]
mod fuzzy_tests {
    use super::fuzzy_app_match;

    #[test]
    fn stt_garbles_resolve_without_aliases() {
        // Универсальность: фонетические искажения STT должны находить
        // приложение численно, а не по зашитому списку алиасов.
        let apps = vec![
            "Firefox".to_string(),
            "Google Chrome".to_string(),
            "Telegram".to_string(),
            "Vim".to_string(),
            "Blender".to_string(),
            "Discord".to_string(),
            "Code - OSS".to_string(),
        ];
        for (said, want) in [
            ("фаер фокс", "Firefox"),
            ("файр фокс", "Firefox"),
            ("гогл хром", "Google Chrome"),
            ("дискорд", "Discord"),
            ("блэндер", "Blender"),
            ("файт фокс", "Firefox"),
            ("фаер фокс", "Firefox"),
        ] {
            assert_eq!(
                fuzzy_app_match(said, &apps).as_deref(),
                Some(want),
                "«{said}» должен находить {want}"
            );
        }
    }

    #[test]
    fn garbage_does_not_open_anything() {
        // Честный отказ важнее угадывания: мусор не открывает приложения.
        let apps = vec![
            "Firefox".to_string(),
            "Google Chrome".to_string(),
            "Telegram".to_string(),
        ];
        for said in ["пиджин", "заплатка", "совещание в три часа"] {
            assert_eq!(
                fuzzy_app_match(said, &apps),
                None,
                "«{said}» не должно ни во что превращаться"
            );
        }
    }

    #[test]
    fn close_call_between_two_apps_is_refused() {
        // Два кандидата почти одинаково близки — честнее не угадывать.
        let apps = vec!["Foxit Reader".to_string(), "Foxmail".to_string()];
        assert_eq!(fuzzy_app_match("фокс", &apps), None);
    }
}

#[cfg(test)]
mod terminal_tests {
    use super::*;

    #[test]
    fn terminal_request_recognized() {
        assert!(is_terminal_request("терминал"));
        assert!(is_terminal_request("открой консоль"));
        assert!(is_terminal_request("командная строка"));
        assert!(!is_terminal_request("термометр"));
        assert!(!is_terminal_request("докер"));
    }

    #[test]
    fn terminal_resolves_to_installed_emulator() {
        // Каталог этой машины: foot. Резолвер обязан выбрать эмулятор,
        // а не первый попавшийся (Docker не входит в кандидаты).
        let installed = vec![
            "Docker".to_string(),
            "Foot Server".to_string(),
            "Foot".to_string(),
            "Zed".to_string(),
        ];
        assert_eq!(
            terminal_app_name(&installed),
            Some("Foot".to_string()),
            "демон Foot Server не терминал"
        );
        // Без точного — подстрока: «GNOME Terminal» подходит.
        let installed = vec!["GNOME Terminal".to_string(), "Docker".to_string()];
        assert_eq!(terminal_app_name(&installed), Some("GNOME Terminal".to_string()));
        // Без терминалов в каталоге — честный None, никакого выдуманного.
        let installed = vec!["Docker".to_string(), "Zed".to_string()];
        assert_eq!(terminal_app_name(&installed), None);
    }
}
