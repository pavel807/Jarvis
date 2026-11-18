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
//! 4. [`apply_latin_fallback`] — страховка читаемости: русский синтез
//!    CIS не знает ни цифр, ни латиницы, поэтому всё уцелевшее (домены,
//!    акронимы, коды вида M4/b1g, слова вроде Ubuntu) переводится в
//!    кириллицу, а цифры внутри слов — в слова.
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
    // Страховка читаемости: цифры и латиница заменяются на русские слова —
    // всё уцелевшее (домены, коды, модели с цифрами) переводим в кириллицу.
    let speakable = apply_latin_fallback(&numbers);
    // Схлопнуть пробелы и пустые фразы; не терять вопросы/восклицания.
    speakable.split_whitespace().collect::<Vec<_>>().join(" ")
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
/// синтез произносит их естественно.
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
    ("ChatGPT", "чат гэ-пэ-тэ"),
    ("GPT", "гэ-пэ-тэ"),
    ("AI", "эй-ай"),
    ("Qwen", "квэн"),
    ("PDF", "пэ-дэ-эф"),
    ("Docker", "докер"),
    ("Linux", "линукс"),
    ("Wi-Fi", "вай-фай"),
    ("WiFi", "вай-фай"),
    ("npm", "эн-пэ-эм"),
    ("iOS", "и-ос"),
    ("M1", "эм один"),
    ("M2", "эм два"),
    ("M3", "эм три"),
    ("M4", "эм четыре"),
    // Слова разработчиков: без словаря латиница выпадает из речи молча
    // (редкая в русских ответах), фолбэк-транслит ниже грубее.
    ("cargo", "карго"),
    ("build", "билд"),
    ("bug", "баг"),
    ("deploy", "деплой"),
    ("commit", "коммит"),
    ("merge", "мерж"),
    ("branch", "бранч"),
    ("release", "релиз"),
    ("cache", "кэш"),
    ("localhost", "локалхост"),
    ("server", "сервер"),
    ("GigaChat", "гига чат"),
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

    // Словарь раньше всего: длинные ключи («M4 Pro» → «эм четыре про»)
    // успевают до разделения приклеенных цифр.
    let mut s = input.to_string();
    for (from, to) in RULES.iter() {
        // Границы слова вместо look-around: крейт regex не поддерживает
        // look-behind/look-ahead, а \b не совпадает внутри «открытки» —
        // соседние буквы (включая кириллицу) словесные.
        let re = regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(from)))
            .expect("валидный шаблон произношения");
        // Замена — литеральный текст; `$` в значениях словаря не поддерживается.
        s = re.replace_all(&s, to.replace('$', " ")).to_string();
    }

    // Приклеенные цифры отделяем от букв: \b не работает между «n» и «3»,
    // поэтому «qwen3.5», «х264» и «M4» без этого невидимы для правил чисел
    // и молча выпадали бы из речи.
    s = split_glued_alnum(&s);

    // Версии — последними, уже с пробелами вокруг числа: «0.8b» после
    // сплита получает границы и читается «ноль точка восемь», а не дробью.
    apply_version_pronunciation(&s)
}

/// Цифра↔буква → цифра, пробел, буква (в обе стороны, кириллица и латиница).
fn split_glued_alnum(input: &str) -> String {
    static DIGIT_LETTER_RE: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"(\p{Nd})(\p{L})").unwrap());
    static LETTER_DIGIT_RE: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"(\p{L})(\p{Nd})").unwrap());
    let s = DIGIT_LETTER_RE.replace_all(input, "$1 $2");
    LETTER_DIGIT_RE.replace_all(&s, "$1 $2").to_string()
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

// ------------------------------------------------- Латиница и хвосты цифр

/// Русский синтез не читает вслух ни цифры, ни почти ни одну
/// латинской буквы (только «h»): всё непереваренное молча выпадает из
/// речи. Словарь произношений и числа закрывают большую часть текста,
/// эта страховка добирает остаток: домены («github.com»), акронимы
/// («SVG»), коды («M4», «b1g», «x264»), слова («Ubuntu») и цифры,
/// приклеенные к словам («qwen3.5» — regex-границы их не ловят).
pub fn apply_latin_fallback(input: &str) -> String {
    input
        .split_whitespace()
        .map(spoken_token)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Имена латинских букв — для побуквенного чтения акронимов и кодов.
const LETTER_NAMES: &[(char, &str)] = &[
    ('a', "а"),
    ('b', "бэ"),
    ('c', "цэ"),
    ('d', "дэ"),
    ('e', "е"),
    ('f', "эф"),
    ('g', "гэ"),
    ('h', "аш"),
    ('i', "и"),
    ('j', "джей"),
    ('k', "ка"),
    ('l', "эль"),
    ('m', "эм"),
    ('n', "эн"),
    ('o', "о"),
    ('p', "пэ"),
    ('q', "ку"),
    ('r', "эр"),
    ('s', "эс"),
    ('t', "тэ"),
    ('u', "у"),
    ('v', "вэ"),
    ('w', "дабл-ю"),
    ('x', "икс"),
    ('y', "игрек"),
    ('z', "зэт"),
];

fn letter_name(c: char) -> Option<&'static str> {
    LETTER_NAMES
        .iter()
        .find(|(letter, _)| *letter == c.to_ascii_lowercase())
        .map(|(_, name)| *name)
}

/// Токен → говоримое. Токен мог разрастись («qwen3.5» → «qwen3 точка 5»),
/// поэтому внутри обрабатываем каждое слово по отдельности.
fn spoken_token(token: &str) -> String {
    let has_latin = token.chars().any(|c| c.is_ascii_alphabetic());
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    if !has_latin && !has_digit {
        return token.to_string();
    }
    if has_latin && let Some(spoken) = spoken_domain(token) {
        return spoken;
    }
    // Цифры вокруг точки внутри слова («qwen3.5») — это версия/дробь:
    // разделяем словами, иначе точка выпадет и цифры слипнутся.
    let expanded = if has_latin {
        static DOT_RE: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"(\d)[.,](\d)").unwrap());
        DOT_RE.replace_all(token, "$1 точка $2").to_string()
    } else {
        token.to_string()
    };
    expanded
        .split_whitespace()
        .map(spoken_word)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Разбор слова: прогоны однородных символов (латиница/цифры/прочее)
/// с сохранением разделителей и порядка.
fn spoken_word(word: &str) -> String {
    let mut out = String::with_capacity(word.len() * 3);
    let mut run = String::new();
    for c in word.chars() {
        if c.is_alphanumeric() {
            run.push(c);
        } else {
            flush_run(&run, &mut out);
            run.clear();
            out.push(c);
        }
    }
    flush_run(&run, &mut out);
    out
}

/// Однородный сегмент прогона.
enum Segment {
    /// Латинские буквы.
    Latin(String),
    /// Цифры.
    Digits(String),
    /// Кириллица и прочие буквы — остаются как есть.
    Other(String),
}

fn segments(run: &str) -> Vec<Segment> {
    let classify = |c: char| {
        if c.is_ascii_alphabetic() {
            0
        } else if c.is_ascii_digit() {
            1
        } else {
            2
        }
    };
    let mut segs: Vec<Segment> = Vec::new();
    let mut current = String::new();
    let mut current_class = classify(run.chars().next().unwrap_or(' '));
    for c in run.chars() {
        let class = classify(c);
        if class != current_class && !current.is_empty() {
            segs.push(make_segment(current_class, std::mem::take(&mut current)));
        }
        current_class = class;
        current.push(c);
    }
    if !current.is_empty() {
        segs.push(make_segment(current_class, current));
    }
    segs
}

fn make_segment(class: u8, text: String) -> Segment {
    match class {
        0 => Segment::Latin(text),
        1 => Segment::Digits(text),
        _ => Segment::Other(text),
    }
}

fn flush_run(run: &str, out: &mut String) {
    if run.is_empty() {
        return;
    }
    // После дефиса пробел не нужен: «5-0» остаётся «пять-ноль» — дефис
    // Синтез читает паузой.
    if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('-') {
        out.push(' ');
    }
    let segs = segments(run);
    // Акроним капсом без цифр: SVG → эс-вэ-гэ (словарь отработал раньше).
    if let [Segment::Latin(word)] = segs.as_slice() {
        if word.chars().count() >= 2 && word.chars().all(|c| c.is_ascii_uppercase()) {
            let spoken = word
                .chars()
                .filter_map(letter_name)
                .collect::<Vec<_>>()
                .join("-");
            out.push_str(&spoken);
            return;
        }
    }
    // Код из одиночных латинских букв и чисел: M4, b1g, x264 — читаем
    // по порядку: буквы по именам, числа словами.
    let code_style = segs.iter().any(|s| matches!(s, Segment::Latin(_)))
        && segs.iter().all(|s| match s {
            Segment::Latin(w) => w.chars().count() == 1,
            _ => true,
        });
    let mut pieces: Vec<String> = Vec::new();
    for seg in &segs {
        match seg {
            Segment::Latin(w) if code_style => {
                let letter = w.chars().next().unwrap_or('x');
                pieces.push(letter_name(letter).unwrap_or("икс").to_string());
            }
            Segment::Latin(w) => pieces.push(translit_word(w)),
            Segment::Digits(d) => pieces.push(spoken_number(d)),
            Segment::Other(o) => pieces.push(o.clone()),
        }
    }
    out.push_str(&pieces.join(" "));
}

/// Числа словами; длинные прогоны и ведущие нули (коды, «007») — по цифрам.
fn spoken_number(digits: &str) -> String {
    if digits.len() >= 11 || (digits.len() > 1 && digits.starts_with('0')) {
        crate::text::digits_one_by_one(digits)
    } else {
        crate::text::number_words(digits)
    }
}

/// Домен или адрес: «github.com» → «гитхаб точка ком». Последняя часть —
/// «зона» из одних букв, это отсекает версии вида «0.8b».
fn spoken_domain(token: &str) -> Option<String> {
    let body = token.trim_start_matches(['(', '"', '«', '[']);
    let lead = &token[..token.len() - body.len()];
    let core = body.trim_end_matches(['.', ',', ';', ':', '!', '?', ')', '"', '»', ']']);
    let tail = &body[core.len()..];
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let plain = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '-');
    if !parts.iter().all(|p| plain(p)) {
        return None;
    }
    let zone = parts[parts.len() - 1];
    if zone.chars().count() < 2 || !zone.chars().all(char::is_alphabetic) {
        return None;
    }
    let spoken = parts
        .iter()
        .map(|p| {
            if p.chars().count() == 1 {
                letter_name(p.chars().next().unwrap_or('x'))
                    .unwrap_or("точка")
                    .to_string()
            } else {
                translit_word(p)
            }
        })
        .collect::<Vec<_>>()
        .join(" точка ");
    Some(format!("{lead}{spoken}{tail}"))
}

/// Грубая фонетическая транслитерация латинского слова кириллицей.
/// Цель не точность, а читаемость: латиница в русской речи звучит лучше
/// кириллицей.
/// Тонкие случаи правятся словарём произношений (`pronunciations.toml`).
fn translit_word(word: &str) -> String {
    // Частые диграфы разбираем до одиночных букв.
    const DIGRAPHS: &[(&str, &str)] = &[
        ("ch", "ч"),
        ("sh", "ш"),
        ("th", "т"),
        ("ph", "ф"),
        ("ck", "к"),
        ("oo", "у"),
        ("ee", "и"),
        ("ou", "ау"),
        ("ai", "ай"),
    ];
    const SINGLES: &[(char, &str)] = &[
        ('a', "а"),
        ('b', "б"),
        ('c', "к"),
        ('d', "д"),
        ('e', "е"),
        ('f', "ф"),
        ('g', "г"),
        ('h', "х"),
        ('i', "и"),
        ('j', "дж"),
        ('k', "к"),
        ('l', "л"),
        ('m', "м"),
        ('n', "н"),
        ('o', "о"),
        ('p', "п"),
        ('q', "к"),
        ('r', "р"),
        ('s', "с"),
        ('t', "т"),
        ('u', "у"),
        ('v', "в"),
        ('w', "в"),
        ('x', "кс"),
        ('z', "з"),
    ];
    let lower: Vec<char> = word.to_lowercase().chars().collect();
    let mut out = String::with_capacity(lower.len() * 2);
    let mut i = 0;
    while i < lower.len() {
        let digraph = DIGRAPHS.iter().find(|(two, _)| {
            lower[i] == two.chars().next().unwrap()
                && lower.get(i + 1) == Some(&two.chars().nth(1).unwrap())
        });
        if let Some((_, spoken)) = digraph {
            out.push_str(spoken);
            i += 2;
            continue;
        }
        let c = lower[i];
        // Немое «e» на конце: make → мак, code → код.
        if c == 'e' && i + 1 == lower.len() && lower.len() > 2 && !is_english_vowel(lower[i - 1]) {
            i += 1;
            continue;
        }
        if c == 'y' {
            // «y» в начале — согласная (youtube → ютуб), внутри — гласная.
            out.push_str(if i == 0 { "й" } else { "и" });
        } else if let Some((_, spoken)) = SINGLES.iter().find(|(letter, _)| *letter == c) {
            out.push_str(spoken);
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

fn is_english_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
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
    fn latin_words_are_transliterated() {
        // Словарь покрывает частое, фолбэк — остальное: латиницу
        // молча выбрасывает, так хотя бы читаемо.
        let s = normalize("Скачай vim и запусти sync");
        assert!(s.contains("вим"), "{s}");
        assert!(s.contains("синк"), "{s}");
    }

    #[test]
    fn models_and_codes_are_spoken() {
        let s = normalize("Модель qwen3.5-0.8b и папка b1g, кодек х264");
        assert!(
            s.contains("квен три точка пять-ноль точка восемь бэ"),
            "{s}"
        );
        assert!(s.contains("бэ один гэ"), "{s}");
        assert!(s.contains("двести шестьдесят четыре"), "{s}");
        // Дефис внутри имени модели — не «минус».
        assert!(!s.contains("минус"), "{s}");
    }

    #[test]
    fn acronyms_and_domains_are_spelled() {
        let s = normalize("Открой SVG на web.site");
        assert!(s.contains("эс-вэ-гэ"), "{s}");
        assert!(s.contains("веб точка сит"), "{s}");
    }

    #[test]
    fn mixed_case_words_and_mac_names() {
        let s = normalize("Обновился на Ubuntu22, чип M4");
        assert!(s.contains("убунту двадцать два"), "{s}");
        assert!(s.contains("эм четыре"), "{s}");
    }

    #[test]
    fn russian_digits_inside_words() {
        // Приклеенные к кириллице цифры regex-границы не ловят — раньше
        // они выпадали из речи молча.
        let s = normalize("формат 16х9 и версия 0.8b");
        assert!(s.contains("шестнадцать х девять"), "{s}");
        assert!(s.contains("ноль точка восемь бэ"), "{s}");
    }

    #[test]
    fn leading_zero_codes_are_digit_by_digit() {
        let s = apply_latin_fallback("код 007");
        assert_eq!(s, "код ноль ноль семь");
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
