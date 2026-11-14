//! Обучаемая память команд: «фраза → успешное действие».
//!
//! Нейросеть без LLM и без внешних моделей: фраза превращается в эмбеддинг —
//! символьные n-граммы (3/2/1) со знаковым feature hashing на blake3 и
//! L2-нормализацией, — ближайший сосед среди заученных фраз с cosine ≥ TAU
//! воспроизводит сохранённое действие мгновенно и офлайн, без запроса к LLM.
//!
//! Правила безопасности (§5 autocore) — до нейросети:
//! заучивается только то, что уже выполнилось успешно и входит в белый
//! список обратимых инструментов. Shell, файлы, отправка наружу — никогда:
//! память не умеет больше, чем пользователь уже разрешил голосом.

use crate::tools::{self, ToolCall, ToolParse};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Размер эмбеддинга и веса n-грамм — как в autocore::hsr (проверено
/// калибровкой на живых фразах).
const DIM: usize = 256;
const W_TRI: f32 = 1.0;
const W_BI: f32 = 0.5;
const W_UNI: f32 = 0.25;

/// Порог воспроизведения: ниже — отказ, и фраза идёт обычным путём.
/// Калиброван на живых парах: перифразы одного намерения дают 0.73–0.83
/// («открой калькулятор, пожалуйста» = 0.80), смена объекта («открой
/// блокнот») — 0.35, опасные пары («закрой окно» vs «закрой вкладку») —
/// 0.45–0.50. TAU 0.7 — посередине зазора.
pub const TAU: f32 = 0.7;

/// Память не бесконечна: старые записи вытесняются.
const MAX_ENTRIES: usize = 100;

/// Инструменты, которые можно заучивать: обратимые действия на машине.
/// Файлы, shell, отправка сообщений — вне списка, сколько бы раз ни
/// повторились.
const CACHEABLE: &[&str] = &[
    "open_app",
    "open_url",
    "open_path",
    "window",
    "set_volume",
    "set_brightness",
    "set_timer",
    "set_alarm",
    "alarm_stop",
    "screenshot",
    "lock_screen",
    "open_utils",
    "close_utils",
    "wait",
    "notify",
];

struct Entry {
    phrase: String,
    vec: [f32; DIM],
    block: String,
}

fn path() -> PathBuf {
    crate::paths::base_dir().join("learned_commands.json")
}

static LOCK: Mutex<()> = Mutex::new(());

/// Векторы пересчитываются при загрузке: файл хранит только фразы и блоки.
fn read_entries_at(path: &std::path::Path) -> Vec<Entry> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(raw) = serde_json::from_str::<Vec<(String, String)>>(&content) else {
        return Vec::new();
    };
    raw.into_iter()
        .map(|(phrase, block)| {
            let vec = featurize(&phrase);
            Entry { phrase, vec, block }
        })
        .collect()
}

fn write_entries_at(path: &std::path::Path, list: &[Entry]) {
    if let Ok(json) = serde_json::to_string_pretty(
        &list
            .iter()
            .map(|e| (&e.phrase, &e.block))
            .collect::<Vec<_>>(),
    ) {
        let _ = fs::write(path, json);
    }
}

/// Нормализация фразы для точного совпадения и стабильных эмбеддингов.
fn normalize(phrase: &str) -> String {
    phrase
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn hash64(bytes: &[u8]) -> u64 {
    // splitmix64 поверх FNV-1a: у чистого FNV слабое перемешивание
    // младших бит (реальная проблема SimHash), splitmix лечит.
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h = h.wrapping_add(0x9e3779b97f4a7c15);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

fn touch(v: &mut [f32; DIM], gram: &str, weight: f32) {
    let h = hash64(gram.as_bytes());
    let idx = (h % DIM as u64) as usize;
    let sign = if (h >> 63) & 1 == 1 { 1.0 } else { -1.0 };
    v[idx] += sign * weight;
}

/// Эмбеддинг фразы: символьные n-граммы 3/2/1 со знаковым feature hashing,
/// затем L2-нормализация. Идентичные фразы дают cosine 1.0, перефразировки
/// держатся близко, несвязанные — далеко.
fn featurize(text: &str) -> [f32; DIM] {
    let norm = normalize(text);
    let chars: Vec<char> = norm.chars().collect();
    let mut v = [0f32; DIM];
    if chars.is_empty() {
        return v;
    }
    if chars.len() < 3 {
        touch(&mut v, &norm, W_TRI);
    } else {
        for w in 0..=chars.len() - 3 {
            let gram: String = chars[w..w + 3].iter().collect();
            touch(&mut v, &gram, W_TRI);
        }
    }
    if chars.len() >= 2 {
        for w in 0..=chars.len() - 2 {
            let gram: String = chars[w..w + 2].iter().collect();
            touch(&mut v, &gram, W_BI);
        }
    } else {
        touch(&mut v, &norm, W_BI);
    }
    for c in &chars {
        touch(&mut v, &c.to_string(), W_UNI);
    }
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > 0.0 {
        for x in &mut v {
            *x /= len;
        }
    }
    v
}

fn cosine(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Многосоставные фразы не заучиваем и не воспроизводим: «открой хром и
/// поставь таймер» слишком похожа на «открой хром», а повтор целиком —
/// это не то, что просили.
fn is_multi_intent(phrase: &str) -> bool {
    let t = format!(" {phrase} ");
    [" и ", " потом ", " затем ", " заодно ", " а потом "]
        .iter()
        .any(|m| t.contains(m))
}

/// Вежливость и заполняющие слова: они не отличают команды.
const TOKEN_STOP: &[&str] = &[
    "пожалуйста", "давай", "давайте", "мне", "сэр", "джарвис", "ну", "вот",
    "быстро", "сейчас", "ещё", "еще", "снова", "опять", "только",
];

/// Отличительные слова фразы: существительные, названия сайтов и
/// приложений, числа. Глагол-обёртка («открой») тоже отличает действие.
fn significant_tokens(phrase: &str) -> Vec<String> {
    normalize(phrase)
        .split(' ')
        .filter(|w| {
            !w.is_empty()
                && !TOKEN_STOP.contains(w)
                && (w.chars().any(|c| c.is_ascii_digit()) || w.chars().count() >= 5)
        })
        .map(str::to_string)
        .collect()
}

/// Ствол слова: первые 5 букв — склеивают падежи (вкладку/вкладки,
/// музыка/музыку), не склеивают разные названия.
fn stem_of(token: &str) -> String {
    token.chars().take(5).collect()
}

/// Каждый отличительный токен заученной фразы обязан присутствовать в
/// новой. Реальный случай без этой охраны: «открой вкладку яндекс» имела
/// cosine 0.70 к заученному «...кинпоиск» — открылся Кинопоиск. У ютуба
/// было 0.72: порогом это не лечится, лечение — совпадение объекта.
fn stems_match(stored: &str, new: &str) -> bool {
    let req = significant_tokens(stored);
    if req.is_empty() {
        return true;
    }
    let have: Vec<String> = significant_tokens(new);
    req.iter()
        .all(|r| have.iter().any(|h| stem_of(h) == stem_of(r)))
}

/// Запомнить успешное действие. `raw_tool_text` — сырой ответ модели с
/// tool-блоком; при воспроизведении он разбирается тем же парсером.
/// Возвращает true, если запись принята.
pub fn remember(phrase: &str, raw_tool_text: &str) -> bool {
    remember_at(&path(), phrase, raw_tool_text)
}

fn remember_at(path: &std::path::Path, phrase: &str, raw_tool_text: &str) -> bool {
    let ToolParse::Call(call) = tools::parse(raw_tool_text) else {
        return false;
    };
    let name = tools::tool_name_public(&call);
    // Плагины Spark — авторские автоматизации пользователя: повторяемы
    // и безопасны так же, как их первый голосовой запуск.
    let cacheable = CACHEABLE.contains(&name.as_str()) || name.starts_with("plugin_");
    if !cacheable {
        return false;
    }
    let norm = normalize(phrase);
    if norm.chars().count() < 4 || is_multi_intent(&norm) {
        return false;
    }
    let block = raw_tool_text.trim().to_string();
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = read_entries_at(path);
    // Точная фраза перезаписывает себя: действие могло уточниться.
    list.retain(|e| e.phrase != norm);
    list.push(Entry {
        phrase: norm,
        vec: featurize(phrase),
        block,
    });
    if list.len() > MAX_ENTRIES {
        list.remove(0);
    }
    write_entries_at(path, &list);
    true
}

/// Ближайший сосед среди заученных фраз; ниже TAU — отказ (обычный путь).
pub fn lookup(phrase: &str) -> Option<ToolCall> {
    lookup_at(&path(), phrase)
}

fn lookup_at(path: &std::path::Path, phrase: &str) -> Option<ToolCall> {
    let norm = normalize(phrase);
    if norm.chars().count() < 4 || is_multi_intent(&norm) {
        return None;
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let list = read_entries_at(path);
    let target = featurize(phrase);
    // Охрана объекта: чужие сайты/приложения/числа не воспроизводятся,
    // как бы близко ни казались по эмбеддингу.
    let mut best: Option<(f32, &Entry)> = None;
    for e in &list {
        if !stems_match(&e.phrase, &norm) {
            continue;
        }
        let s = cosine(&target, &e.vec);
        if best.map(|(b, _)| s > b).unwrap_or(true) {
            best = Some((s, e));
        }
    }
    let (score, entry) = best?;
    if score < TAU {
        return None;
    }
    println!(
        "🧠 Память команд: «{phrase}» ≈ «{}» ({score:.2}) — без LLM",
        entry.phrase
    );
    match tools::parse(&entry.block) {
        ToolParse::Call(call) => Some(call),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "jarvis-learned-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn identical_and_similar_phrases_match() {
        let a = featurize("открой калькулятор");
        let b = featurize("открой калькулятор");
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-4, "идентичные = 1.0");
        // Перефразировка того же намерения держится близко.
        let c = featurize("открой калькулятор пожалуйста");
        assert!(cosine(&a, &c) >= TAU, "перифраз ≥ TAU");
        // Несвязанная фраза и смена объекта — далеко: память не выдаст
        // «открой блокнот» за «открой калькулятор».
        let d = featurize("поставь будильник на семь утра");
        assert!(cosine(&a, &d) < TAU, "несвязанные не матчатся");
        let e = featurize("открой блокнот");
        assert!(cosine(&a, &e) < TAU, "другой объект — не наш домен");
    }

    #[test]
    fn multi_intent_phrases_are_skipped() {
        assert!(is_multi_intent("открой хром и поставь таймер"));
        assert!(is_multi_intent("закрой окна потом включи музыку"));
        assert!(!is_multi_intent("открой калькулятор"));
        // Союз внутри слова не считается («и» в «иллюзия»).
        assert!(!is_multi_intent("включи иллюзию"));
    }

    #[test]
    fn remember_accepts_only_whitelisted_tools() {
        let mut file = temp_dir("whitelist");
        file.push("learned_commands.json");
        // Shell — вне белого списка: память его не заучит никогда.
        assert!(!remember_at(
            &file,
            "покажи файл",
            r#"<tool>{"name":"shell","args":{"cmd":"cat /etc/passwd"}}</tool>"#,
        ));
        assert!(!file.exists(), "ничего не записано");
        // Обратимое действие — заучивается.
        assert!(remember_at(
            &file,
            "открой калькулятор",
            r#"<tool>{"name":"open_app","args":{"name":"omacalc"}}</tool>"#,
        ));
        // Не tool-блок — не заучивается.
        assert!(!remember_at(&file, "просто текст", "ответ без блоков"));
        // Тот же смысл другой фразой — воспроизводится тем же парсером.
        let call = lookup_at(&file, "открой калькулятор, пожалуйста").expect("найдено");
        assert!(matches!(call, ToolCall::OpenApp { .. }));
        // Несвязанная фраза — отказ памяти (пойдёт обычным путём).
        assert!(lookup_at(&file, "поставь будильник на восемь").is_none());

        // Живая жалоба: «открой вкладку яндекс» имела 0.70 к заученному
        // «открой вкладку кинопоиск» — открылся чужой сайт. Объект
        // (название сайта) обязан совпадать, как бы близко ни было.
        let mut sites = temp_dir("sites");
        sites.push("learned_commands.json");
        assert!(remember_at(
            &sites,
            "открой вкладку кинопоиск",
            r#"<tool>{"name":"open_url","args":{"url":"https://www.kinopoisk.ru"}}</tool>"#,
        ));
        assert!(lookup_at(&sites, "открой вкладку яндекс").is_none());
        assert!(lookup_at(&sites, "открой вкладку ютуб").is_none());
        // Числа тоже отличительные: стол 2 не воспроизводится за стол 3.
        let mut desks = temp_dir("desks");
        desks.push("learned_commands.json");
        assert!(remember_at(
            &desks,
            "открой стол 2",
            r#"<tool>{"name":"window","args":{"action":"desktop","desktop":"2"}}</tool>"#,
        ));
        assert!(lookup_at(&desks, "открой стол 3").is_none());
        assert!(lookup_at(&desks, "открой стол 2").is_some());
    }
}
