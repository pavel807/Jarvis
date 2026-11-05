//! TextNormalizer: превращает произвольный текст (обычные фразы, числа,
//! аббревиатуры, Markdown, пути, версии) в устную русскую речь для TTS.
//!
//! Пайплайн [`normalize`]:
//! 1. [`markdown_to_speech`] — структурная обработка Markdown: заголовки,
//!    списки, таблицы, ссылки, цитаты, код. Разметка не читается вслух
//!    буквально; длинные фрагменты кода сжимаются («фрагмент кода»).
//! 2. [`apply_tech_pronunciation`] — аббревиатуры (CPU, GPU, API…),
//!    версии (2.1.3 → «два точка один точка три»), названия программ
//!    (Premiere Pro → «премьер про»). Словарь расширяется пользователем
//!    через `pronunciations.toml` в каталоге данных (перечитывается при
//!    каждом запуске, правка не требует пересборки).
//! 3. [`crate::text::normalize_for_tts`] — числа, даты, время, единицы.
//!
//! Транскрипт ASR не затрагивается: этот модуль стоит только на пути
//! LLM/сообщение → TTS.

use std::collections::BTreeMap;
use std::sync::LazyLock;

/// Полная нормализация текста для озвучивания.
pub fn normalize(input: &str) -> String {
    let md = markdown_to_speech(input);
    let pronounced = apply_tech_pronunciation(&md);
    let numbers = crate::text::normalize_for_tts(&pronounced);
    // Схлопнуть пробелы и пустые фразы; не терять вопросы/восклицания.
    numbers.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------- Markdown

/// Сколько символов кода произносить вслух — остальное сворачивается.
const MAX_CODE_CHARS: usize = 200;

/// Структурная обработка Markdown → говоримый текст.
pub fn markdown_to_speech(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_code = false;
    let mut code: Vec<&str> = Vec::new();
    let mut list_started = false;

    for line in input.lines() {
        let trimmed = line.trim();

        // Блоки кода ```lang\n...\n```
        if let Some(rest) = trimmed.strip_prefix("```") {
            if in_code {
                // Закрывающая строка: выверить накопленный код.
                let body = code.join("\n");
                in_code = false;
                code.clear();
                emit_code(&mut out, &body, rest);
                out.push(' ');
            } else {
                in_code = true;
                code.clear();
            }
            continue;
        }
        if in_code {
            code.push(line);
            continue;
        }

        // Пустая строка — разрыв секции.
        if trimmed.is_empty() {
            list_started = false;
            continue;
        }

        // Горизонтальная линия — пропуск.
        if trimmed.len() >= 3 && trimmed.chars().all(|c| matches!(c, '-' | '=' | '*' | '_' | ' ')) {
            continue;
        }

        // Заголовки: «## Заголовок» → «Заголовок.» с паузой после точки.
        if let Some(rest) = trimmed.strip_prefix('#') {
            let heading = rest.trim_start_matches('#').trim();
            if !heading.is_empty() {
                if out.ends_with(' ') {
                    out.pop();
                }
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&inline_markdown(heading));
                out.push_str(". ");
            }
            list_started = false;
            continue;
        }

        // Списки: «- пункт», «* пункт», «1. пункт».
        let bullet = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("• "));
        let ordered = ordered_list_item(trimmed);
        if let Some(item) = bullet.or(ordered) {
            if !list_started {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str("Пункты: ");
                list_started = true;
            } else {
                out.push_str(", ");
            }
            out.push_str(&inline_markdown(item.trim_end_matches(['.', ';']).trim()));
            continue;
        }
        list_started = false;

        // Таблица: «| a | b |» → каждая строка «a — b».
        if trimmed.starts_with('|') && trimmed.ends_with('|') {
            let cells: Vec<&str> =
                trimmed[1..trimmed.len() - 1].split('|').map(str::trim).collect();
            if cells.iter().all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')) {
                continue; // строка-разделитель
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&cells.join(" — "));
            out.push_str(". ");
            continue;
        }

        // Цитаты: «> текст» → просто текст.
        let quote = trimmed.strip_prefix("> ");
        if let Some(q) = quote {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&inline_markdown(q));
            out.push(' ');
            continue;
        }

        // Обычный абзац.
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&inline_markdown(trimmed));
        out.push(' ');
    }
    if in_code && !code.is_empty() {
        // Незакрытый код-блок — тоже сворачиваем по правилам кода.
        let body = code.join("\n");
        emit_code(&mut out, &body, "");
    }
    out.trim().to_string()
}

/// Элемент нумерованного списка: «1. пункт», «2) пункт».
fn ordered_list_item(line: &str) -> Option<&str> {
    let mut chars = line.chars();
    let first = chars.next()?;
    if !first.is_ascii_digit() {
        return None;
    }
    let rest = line.trim_start_matches(|c: char| c.is_ascii_digit());
    rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))
}

/// Код: короткий — читаем по правилам произношения, длинный — сворачиваем.
fn emit_code(out: &mut String, body: &str, _lang: &str) {
    let single_line = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.len() <= MAX_CODE_CHARS {
        out.push_str("Код: ");
        out.push_str(&inline_markdown(&single_line));
        out.push('.');
    } else {
        out.push_str("Здесь фрагмент кода.");
    }
}

/// Внутристрочные конструкции: **жирный**, *курсив*, `код`,
/// [текст](ссылка), ![alt](img).
fn inline_markdown(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Ссылки и картинки: [text](url), ![alt](url)
        if chars[i] == '[' || (chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '[') {
            if let Some((text, rest)) = link_text(&chars, i) {
                let text = text.trim();
                if !text.is_empty() {
                    if !out.is_empty() && !out.ends_with(' ') {
                        out.push(' ');
                    }
                    out.push_str(text);
                }
                i = rest;
                continue;
            }
        }
        // Код в строке: `...`
        if chars[i] == '`' {
            if let Some(end) = find_char(&chars, i + 1, '`') {
                let code: String = chars[i + 1..end].iter().collect();
                if !code.trim().is_empty() {
                    if !out.is_empty() && !out.ends_with(' ') {
                        out.push(' ');
                    }
                    out.push_str(&code.trim());
                }
                i = end + 1;
                continue;
            }
        }
        // **жирный**, *курсив*, _курсив_
        if chars[i] == '*' || chars[i] == '_' {
            let double = i + 1 < chars.len() && chars[i + 1] == chars[i];
            let marker = if double { 2 } else { 1 };
            if let Some(end) = find_substring(&chars, i + marker, &vec![chars[i]; marker]) {
                let inner: String = chars[i + marker..end].iter().collect();
                if !inner.trim().is_empty() {
                    out.push_str(inner.trim());
                    out.push(' ');
                }
                i = end + marker;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    // Остаточные одиночные '`' и '*': подчистить, чтобы не читались.
    out.trim().replace('`', "").replace('*', "").trim().to_string()
}

/// Текст ссылки/картинки: возвращает (текст, индекс после закрывающей скобки).
fn link_text(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start;
    if chars[i] == '!' {
        i += 1;
    }
    if chars.get(i) != Some(&'[') {
        return None;
    }
    let close = find_char(chars, i + 1, ']')?;
    // за ] должна идти (url)
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let url_close = find_char(chars, close + 2, ')')?;
    let text: String = chars[i + 1..close].iter().collect();
    Some((text, url_close + 1))
}

fn find_char(chars: &[char], from: usize, c: char) -> Option<usize> {
    (from..chars.len()).find(|&i| chars[i] == c)
}

fn find_substring(chars: &[char], from: usize, pattern: &[char]) -> Option<usize> {
    (from..chars.len()).position(|i| {
        pattern.iter().enumerate().all(|(k, p)| chars.get(i + k) == Some(p))
    })
    .map(|p| p + from)
}

// ------------------------------------------------- Технологические слова

/// Аббревиатуры и названия: как писать их кириллицей, чтобы русский голос
/// Silero произносил их естественно.
const BUILTIN_PRONUNCIATION: &[(&str, &str)] = &[
    // Сокращения: побуквенные читаются по-русски.
    ("CPU", "си-пи-ю"),
    ("GPU", "ги-пи-ю"),
    ("RAM", "рам"),
    ("API", "а-пи-и"),
    ("URL", "юрл"),
    ("HTTP", "аш-ти-ти-пи"),
    ("HTTPS", "аш-ти-ти-пэ-эс"),
    ("ASR", "а-эс-эр"),
    ("TTS", "ти-ти-эс"),
    ("LLM", "эл-эл-эм"),
    ("SSH", "эс-эс-аш"),
    ("SSD", "эс-эс-ди"),
    ("HDD", "аш-ди-ди"),
    ("USB", "ю-эс-би"),
    ("IP", "ай-пи"),
    ("OS", "о-эс"),
    ("OK", "окей"),
    // Названия программ и брендов (слышимая форма, не транслитерация).
    ("Premiere Pro", "премьер про"),
    ("Premiere", "премьер"),
    ("GitHub", "гитхаб"),
    ("Git", "гит"),
    ("macOS", "мак ос"),
    ("M4 Pro", "эм четыре про"),
    ("Rust", "раст"),
    ("Python", "пайтон"),
    ("Chrome", "хром"),
    ("Safari", "сафари"),
    ("Finder", "файндер"),
    ("Telegram", "телеграм"),
    ("WhatsApp", "ватсап"),
    ("Figma", "фигма"),
    ("Spotify", "спотифай"),
    ("YouTube", "ютуб"),
    ("JetBrains", "джет брейнс"),
    ("VS Code", "вс код"),
    ("Xcode", "экс-код"),
    // Джарвис — единообразное имя помощника.
    ("Jarvis", "Джарвис"),
];

/// Пользовательский словарь произношений: `pronunciations.toml`
/// в каталоге данных Jarvis. Формат:
/// ```toml
/// [pronunciations]
/// "Моя программа" = "моя программа"
/// ```
fn user_pronunciations() -> BTreeMap<String, String> {
    let path = crate::paths::base_dir().join("pronunciations.toml");
    let mut map = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return map;
    };
    let value: Result<toml::Table, _> = toml::from_str(&text);
    if let Ok(table) = value {
        if let Some(section) = table.get("pronunciations").and_then(|v| v.as_table()) {
            for (k, v) in section {
                if let Some(replacement) = v.as_str() {
                    map.insert(k.clone(), replacement.to_string());
                }
            }
        }
    }
    map
}

/// Каталог, который пользователь просил править вручную (кэш после первой
/// загрузки: файл читается один раз на процесс, перезапуск подхватит правки).
static USER_PRONUNCIATIONS: LazyLock<BTreeMap<String, String>> =
    LazyLock::new(user_pronunciations);

/// Возвращает путь к пользовательскому словарю произношений (для UI-подсказки).
pub fn pronunciations_path() -> std::path::PathBuf {
    crate::paths::base_dir().join("pronunciations.toml")
}

/// Подстановка названий/аббревиатур: сначала пользовательский словарь,
/// затем встроенный. Совпадение без учёта регистра, длинные ключи — раньше
/// коротких (Premiere Pro до Premiere).
pub fn apply_tech_pronunciation(input: &str) -> String {
    static RULES: LazyLock<Vec<(String, String)>> = LazyLock::new(|| {
        let mut rules: Vec<(String, String)> = USER_PRONUNCIATIONS
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        rules.extend(
            BUILTIN_PRONUNCIATION
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string())),
        );
        // Длинные ключи — раньше коротких.
        rules.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
        rules
    });

    // Версии раньше общих правил числа: «версия 2.1.3» не должна стать
    // дробью («две целых одна десятая»).
    let mut s = apply_version_pronunciation(input);

    for (from, to) in RULES.iter() {
        // Границы слова вместо look-around: крейт regex не поддерживает
        // look-behind/look-ahead, а \b не совпадает внутри «открытки» —
        // соседние буквы (включая кириллицу) словесные.
        let re = regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(from)))
            .expect("валидный шаблон произношения");
        // Замена — литеральный текст; `$` в значениях словаря не поддерживается.
        s = re.replace_all(&s, to.replace('$', " ")).to_string();
    }
    s
}

/// Версии вида 1.2.3 / v1.2 → «один точка два точка три» (буква v опускается).
fn apply_version_pronunciation(input: &str) -> String {
    static VERSION_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\b[vV]?(\d{1,3}(?:\.\d{1,3}){1,3})\b").unwrap()
    });
    VERSION_RE
        .replace_all(input, |caps: &regex::Captures| {
            caps[1]
                .split('.')
                .map(|part| part.to_string())
                .collect::<Vec<_>>()
                .join(" точка ")
        })
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_headings_and_lists() {
        let md = "# Утро\n\n## Погода\n- Дождь\n- Ветер\n\n1. Первый\n2. Второй";
        let speech = markdown_to_speech(md);
        assert!(speech.contains("Утро."));
        assert!(speech.contains("Погода."));
        assert!(speech.contains("Пункты: Дождь, Ветер"));
        assert!(speech.contains("Первый, Второй"));
        assert!(!speech.contains('#'));
    }

    #[test]
    fn markdown_links_and_code() {
        let md = "Смотри [документацию](https://example.com) и `cargo build`.\n\n```rust\nlet длинный = 1;\n```\n\n**Важно**: не читать звёздочки.";
        let speech = markdown_to_speech(md);
        assert!(speech.contains("Смотри документацию и cargo build."));
        assert!(speech.contains("Код: let длинный = 1;"));
        assert!(speech.contains("Важно"));
        assert!(!speech.contains('*'));
        assert!(!speech.contains('`'));
        assert!(!speech.contains("example.com"));
    }

    #[test]
    fn long_code_collapsed() {
        let body = "let x = 1; ".repeat(40);
        let md = format!("```\n{body}```");
        let speech = markdown_to_speech(&md);
        assert!(speech.contains("фрагмент кода"));
        assert!(!speech.contains("let x"));
    }

    #[test]
    fn abbreviations() {
        assert_eq!(
            apply_tech_pronunciation("CPU загружен на 50%"),
            "си-пи-ю загружен на 50%"
        );
        assert_eq!(apply_tech_pronunciation("API доступен"), "а-пи-и доступен");
        // Внутри слова не подменяется.
        assert_eq!(apply_tech_pronunciation("открытка"), "открытка");
    }

    #[test]
    fn app_names() {
        let s = apply_tech_pronunciation("Открой Premiere Pro");
        assert_eq!(s, "Открой премьер про");
        let s = apply_tech_pronunciation("Обновился на GitHub");
        assert_eq!(s, "Обновился на гитхаб");
    }

    #[test]
    fn versions_not_fractions() {
        let s = normalize("Версия 2.1.3 вышла");
        assert!(s.contains("два точка один точка три"), "{s}");
        assert!(!s.contains("целых"), "{s}");
    }

    #[test]
    fn mixed_everything() {
        let md = "## Погода\n- Дождь 60%, ветер 15 м/с\n- Запусти Premiere Pro";
        let s = normalize(md);
        assert!(s.contains("Пункты: Дождь"), "{s}");
        assert!(s.contains("процентов"), "{s}");
        assert!(s.contains("метров в секунду"), "{s}");
        assert!(s.contains("премьер про"), "{s}");
    }
}
