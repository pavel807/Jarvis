//! Обучаемый кэш сайтов: «название → URL».
//!
//! Ничего не хардкодим: пока нейросеть разрешает незнакомую просьбу
//! (`web_search` → `open_url` или сразу `open_url`), мы запоминаем пару.
//! При повторе Jarvis открывает сайт мгновенно, вообще не трогая LLM.
//!
//! Файл `${JARVIS_HOME}/sites.json` — обычный объект `{ "ключ": "url" }`,
//! его можно смотреть и править руками.

use crate::paths;
use crate::text::word_similarity;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Запись кэша — из процесса может писать агентный цикл, сериализуем.
static LOCK: Mutex<()> = Mutex::new(());

/// Слова-обёртки вокруг названия сайта: их выкидываем из ключа.
const STOP: &[&str] = &[
    "открой",
    "откройте",
    "открыть",
    "открывай",
    "зайди",
    "зайти",
    "заходи",
    "перейди",
    "перейти",
    "переходи",
    "го",
    "давай",
    "покажи",
    "дай",
    "включи",
    "сайт",
    "сайта",
    "сайте",
    "сайтэ",
    "страница",
    "страницу",
    "страницы",
    "страничку",
    "интернет",
    "интернете",
    "мне",
    "пожалуйста",
    "на",
    "в",
    "по",
    "к",
    "с",
    "и",
    "ну",
    "вот",
    "этот",
    "эту",
    "это",
];

/// Путь к файлу кэша.
pub fn path() -> PathBuf {
    paths::base_dir().join("sites.json")
}

fn read_map() -> BTreeMap<String, String> {
    read_map_at(&path())
}

fn read_map_at(path: &std::path::Path) -> BTreeMap<String, String> {
    let Ok(content) = fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    serde_json::from_str(&content).unwrap_or_default()
}

fn write_map(map: &BTreeMap<String, String>) -> Result<()> {
    write_map_at(&path(), map)
}

fn write_map_at(path: &std::path::Path, map: &BTreeMap<String, String>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let body = serde_json::to_string_pretty(map).context("сериализация кэша сайтов")?;
    // Пишем через временный файл, чтобы не потерять кэш при сбое.
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, body).with_context(|| format!("Не удалось записать {:?}", tmp))?;
    fs::rename(&tmp, path).with_context(|| format!("Не удалось заменить {:?}", path))?;
    Ok(())
}

/// Обрезает частые русские окончания, чтобы «айпадом» совпал с «айпад».
fn stem(word: &str) -> String {
    const ENDINGS: &[&str] = &[
        "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ой", "ей", "ом", "ем", "ах", "ях",
        "ам", "ям", "ую", "юю", "ый", "ий", "ое", "ее", "а", "я", "у", "ю", "е", "и", "ы", "о",
    ];
    let w = word.to_lowercase();
    for end in ENDINGS {
        if w.chars().count() > end.chars().count() + 2 && w.ends_with(end) {
            return w[..w.len() - end.len()].to_string();
        }
    }
    w
}

/// Значимые слова запроса: без действий и служебных слов.
pub fn content_words(input: &str) -> Vec<String> {
    input
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .filter(|w| !is_stop(w))
        .map(stem)
        .filter(|w| !w.is_empty())
        .collect()
}

/// Стоп-слово, в том числе с опечаткой распознавания: «откройт» ≈ «открой».
fn is_stop(word: &str) -> bool {
    if STOP.contains(&word) {
        return true;
    }
    // Короткие слова сравниваем только точно, чтобы не выкинуть «год», «дом».
    word.chars().count() >= 4
        && STOP
            .iter()
            .any(|s| s.chars().count() >= 4 && word_similarity(word, s) >= 0.8)
}

/// Ключ кэша из запроса. `None`, если значимых слов не осталось.
fn key_of(input: &str) -> Option<String> {
    let words = content_words(input);
    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

/// Поисковый запрос по названию сайта: значимые слова без обёрток.
pub fn search_query(input: &str) -> String {
    content_words(input).join(" ")
}

/// Ищет в кэше сайт по запросу. Возвращает URL, если все значимые слова
/// запроса совпали с ключом (с учётом опечаток/окончаний).
pub fn lookup(input: &str) -> Option<String> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    best_match(&read_map(), input)
}

fn best_match(map: &BTreeMap<String, String>, input: &str) -> Option<String> {
    let query = content_words(input);
    if query.is_empty() {
        return None;
    }

    let mut best: Option<(usize, String)> = None;
    for (key, url) in map {
        let key_words: Vec<&str> = key.split(' ').filter(|w| !w.is_empty()).collect();
        // Полное совпадение по числу слов: «эппл» не должен перехватывать
        // «эппл почта» — это уже другой запрос, пусть решает модель.
        if key_words.is_empty() || key_words.len() != query.len() {
            continue;
        }
        let all_match = key_words.iter().all(|kw| {
            query
                .iter()
                .any(|qw| qw == kw || word_similarity(qw, kw) >= 0.72)
        });
        if all_match {
            let score = key_words.len();
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, url.clone()));
            }
        }
    }
    best.map(|(_, url)| url)
}

/// Запоминает «название → URL». Повторная запись перезаписывает.
pub fn remember(input: &str, url: &str) {
    let Some(key) = key_of(input) else {
        return;
    };
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return;
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut map = read_map();
    if map.get(&key).map(String::as_str) == Some(url) {
        return;
    }
    map.insert(key, url.trim().to_string());
    if let Err(e) = write_map(&map) {
        eprintln!("⚠️ Не удалось сохранить кэш сайтов: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_strips_wrappers_and_stems() {
        assert_eq!(
            key_of("открой сайт с айпадом про").as_deref(),
            Some("айпад про")
        );
        assert_eq!(key_of("открой эппл").as_deref(), Some("эппл"));
        assert_eq!(key_of("зайди на ютюб").as_deref(), Some("ютюб"));
        assert_eq!(key_of("откройт сайт сдэк шоп").as_deref(), Some("сдэк шоп"));
        assert_eq!(key_of("ну открой мне сайт").as_deref(), None);
    }

    #[test]
    fn lookup_matches_inflected_words() {
        // Эмулируем содержимое кэша без файловой системы.
        let query = content_words("открой страницу с айпадом про");
        let key = content_words("открой сайт айпад про");
        assert_eq!(query, key, "окончания должны нормализоваться");
    }

    #[test]
    fn roundtrip_via_temp_file() {
        let dir = std::env::temp_dir().join("jarvis-site-cache-test");
        let _ = fs::remove_dir_all(&dir);
        let file = dir.join("sites.json");

        let mut map = BTreeMap::new();
        map.insert(
            "айпад про".to_string(),
            "https://www.apple.com/ipad-pro".to_string(),
        );
        map.insert("эппл".to_string(), "https://www.apple.com".to_string());
        write_map_at(&file, &map).unwrap();

        let loaded = read_map_at(&file);
        assert_eq!(
            best_match(&loaded, "открой страницу с айпадом про").as_deref(),
            Some("https://www.apple.com/ipad-pro"),
            "фраза про iPad Pro должна дать точный раздел, а не корень сайта"
        );
        assert_eq!(
            best_match(&loaded, "открой эппл").as_deref(),
            Some("https://www.apple.com")
        );
        assert_eq!(best_match(&loaded, "открой ютюб"), None);
        // «эппл почта» — другой запрос, не должен тянуться к «эппл».
        assert_eq!(best_match(&loaded, "открой эппл почту"), None);

        let _ = fs::remove_dir_all(&dir);
    }
}