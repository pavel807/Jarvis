//! Нормализация и сопоставление текста: чистка, fuzzy-метрики, валидация.

/// Нормализация: только буквы/цифры/пробелы, lowercase, схлопнутые пробелы.
pub fn clean_text(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Очистка текста от Markdown и спецсимволов перед подачей в TTS.
pub fn clean_tts_text(input: &str) -> String {
    input
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .replace(['*', '#', '`', '_'], "")
        .replace("  ", " ")
        .trim()
        .to_string()
}

/// Отсекаем обращения "джарвис", "джавиз" и вводные фразы для получения чистого промпта.
/// Варианты wake word, которые STT записывает по-разному.
const WAKE_VARIANTS: &[&str] = &[
    "джарвис",
    "джавиз",
    "джарвиз",
    "джарис",
    "жарвис",
    "ярвис",
    "jarvis",
    "эй",
    "привет",
    "слушай",
    "пожалуйста",
];

pub fn clean_wakewords_and_prompt(text: &str) -> String {
    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    let mut start_idx = 0;
    while start_idx < words.len() {
        if WAKE_VARIANTS.contains(&words[start_idx]) {
            start_idx += 1;
        } else {
            break;
        }
    }

    if start_idx < words.len() {
        words[start_idx..].join(" ")
    } else {
        cleaned
    }
}

/// Фраза состоит только из wake-слов («Джарвис», даже записанный STT как
/// «жарвис»). В окне прямого слушания это не вопрос — молчим и слушаем дальше.
pub fn is_wake_only(text: &str) -> bool {
    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    !words.is_empty() && words.iter().all(|w| WAKE_VARIANTS.contains(w))
}

/// Слова-связки без содержания. Фраза только из них — обрывок чужой речи
/// («а после», «ну вот»), а не вопрос: в LLM он превращается в выдумку
/// ни о чём. Да/нет сознательно не здесь: это осмысленные ответы.
const FILLER_WORDS: &[&str] = &[
    "а",
    "и",
    "но",
    "же",
    "ли",
    "бы",
    "ну",
    "вот",
    "это",
    "эта",
    "итак",
    "типа",
    "после",
    "до",
    "про",
    "для",
    "кстати",
    "просто",
    "так",
    "там",
    "тут",
    "здесь",
    "ещё",
    "еще",
    "уже",
    "всё",
    "все",
    "вообще",
    "короче",
    "значит",
    "поэтому",
    "причём",
    "причем",
];

pub fn is_filler_only(text: &str) -> bool {
    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    !words.is_empty() && words.iter().all(|w| FILLER_WORDS.contains(w))
}

/// Достаёт фразу после триггера «скажи/повтори/...» для режима повторения.
pub fn extract_repeat_text(text: &str) -> Option<String> {
    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    let triggers = ["скажи", "повтори", "произнеси", "озвучь"];

    for trigger in &triggers {
        if let Some(pos) = words.iter().position(|&w| w == *trigger)
            && pos + 1 < words.len()
        {
            let repeat_phrase = words[pos + 1..].join(" ");
            if !repeat_phrase.trim().is_empty() {
                return Some(repeat_phrase);
            }
        }
    }

    None
}

/// Маркеры, по которым пользователь однозначно просит вернуться к прежней
/// теме. Учитываем написание без «ё»: STV часто её не слышит.
const RESUME_STRONG: [&str; 20] = [
    "о чем говорили",
    "о чём говорили",
    "что мы говорили",
    "что мы обсуждали",
    "о чем мы говорили",
    "о чём мы говорили",
    "на чем остановились",
    "на чём остановились",
    "на чем мы остановились",
    "на чём мы остановились",
    "где мы остановились",
    "мы остановились на",
    "о чём речь шла",
    "о чем речь шла",
    "то что ты говорил",
    "то что ты сказал",
    "вернись к теме",
    "вернуться к теме",
    "вернись к прошлой теме",
    "вернись к тому что обсуждали",
];

/// Слабые маркеры: сами по себе они двусмысленны («мы забыли купить хлеб» —
/// это новая просьба), поэтому работают только в короткой фразе на конце.
const RESUME_WEAK: [&str; 6] = [
    "забыли",
    "забыл",
    "забыла",
    "вернись",
    "продолжи",
    "возвращайся",
];

/// Длина фразы, в которой слабый маркер ещё считается возвратом к теме.
const RESUME_WEAK_MAX_WORDS: usize = 4;

/// Достаёт просьбу вернуться к прежней теме: «мы это забыли», «о чём
/// говорили», «продолжи».
///
/// Возвращает очищенную фразу целиком — она нужна как метка для журнала.
pub fn extract_topic_resume(text: &str) -> Option<String> {
    let cleaned = clean_wakewords_and_prompt(text);
    if cleaned.is_empty() {
        return None;
    }

    if RESUME_STRONG.iter().any(|m| cleaned.contains(m)) {
        return Some(cleaned);
    }

    // Слабый маркер принимаем, только если он почти в конце короткой фразы:
    // «забыли» — это про забытую тему, а «забыли купить хлеб» — новая просьба.
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let near_end = words.iter().rposition(|w| RESUME_WEAK.contains(w));
    match near_end {
        Some(pos) if words.len() <= RESUME_WEAK_MAX_WORDS && pos + 2 >= words.len() => {
            Some(cleaned)
        }
        _ => None,
    }
}

pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let v1: Vec<char> = s1.chars().collect();
    let v2: Vec<char> = s2.chars().collect();

    let len1 = v1.len();
    let len2 = v2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut cache: Vec<usize> = (0..=len2).collect();

    for (i, &c1) in v1.iter().enumerate() {
        let mut result = i + 1;
        let mut distance_frame = i;

        for (j, &c2) in v2.iter().enumerate() {
            let old_distance = result;
            result = distance_frame;
            distance_frame = cache[j + 1];

            if c1 != c2 {
                result = result.min(distance_frame).min(old_distance) + 1;
            }

            cache[j + 1] = result;
        }
    }

    cache[len2]
}

/// Русские буквы → латиница («сверни телеграм»: «телеграм» → «telegram»).
/// Нужно для сопоставления русской речи с именами приложений и процессов.
pub fn translit_ru(word: &str) -> String {
    let mut out = String::new();
    for c in word.to_lowercase().chars() {
        match c {
            'а' => out.push('a'),
            'б' => out.push('b'),
            'в' => out.push('v'),
            'г' => out.push('g'),
            'д' => out.push('d'),
            'е' | 'ё' | 'э' => out.push('e'),
            'ж' => out.push_str("zh"),
            'з' => out.push('z'),
            'и' | 'й' => out.push('i'),
            'к' => out.push('k'),
            'л' => out.push('l'),
            'м' => out.push('m'),
            'н' => out.push('n'),
            'о' => out.push('o'),
            'п' => out.push('p'),
            'р' => out.push('r'),
            'с' => out.push('s'),
            'т' => out.push('t'),
            'у' => out.push('u'),
            'ф' => out.push('f'),
            'х' => out.push('h'),
            'ц' => out.push_str("ts"),
            'ч' => out.push_str("ch"),
            'ш' | 'щ' => out.push_str("sh"),
            'ы' => out.push('y'),
            'ю' => out.push_str("yu"),
            'я' => out.push_str("ya"),
            'ъ' | 'ь' => {}
            _ if c.is_ascii_alphanumeric() => out.push(c),
            _ => {}
        }
    }
    out
}

/// Имя без регистра, пробелов и знаков: «Google Chrome» и «google-chrome»
/// сравниваются одинаково. Для сопоставления приложений и процессов.
/// Чинит кириллические буквы в СМЕШАННЫХ идентификаторах: модель пишет
/// «Саfari» с кириллической «С» — фонетическая транслитерация даёт
/// «safari», и процесс находится. Чистые скрипты («терминал», "Safari")
/// не трогаем: русский текст нужен как есть.
pub fn fix_lookalikes(s: &str) -> String {
    let cyr = s
        .chars()
        .filter(|c| ('А'..='я').contains(c) || matches!(c, 'Ё' | 'ё'))
        .count();
    let lat = s.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if cyr == 0 || lat == 0 {
        return s.to_string();
    }
    translit_ru(s)
}

pub fn compact_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

pub fn word_similarity(w1: &str, w2: &str) -> f32 {
    let max_len = w1.chars().count().max(w2.chars().count());
    if max_len == 0 {
        return 1.0;
    }

    let dist = levenshtein_distance(w1, w2);
    1.0 - (dist as f32 / max_len as f32)
}

/// Отсекает шумовые/служебные расшифровки, чтобы не реагировать на мусор STT.
pub fn is_valid(text: &str) -> bool {
    let t = clean_text(text);
    if t.chars().count() <= 1 {
        return false;
    }

    let invalid = [
        "music",
        "silence",
        "noise",
        "thank you",
        "yes",
        "no",
        "угу",
        "ага",
        "да",
        "нет",
    ];

    !invalid.iter().any(|x| t == *x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_strips_punctuation_and_case() {
        assert_eq!(clean_text("Привет, Джарвис!"), "привет джарвис");
        assert_eq!(clean_text("  много   пробелов "), "много пробелов");
    }

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("кот", "код"), 1);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn word_similarity_bounds() {
        assert!((word_similarity("код", "код") - 1.0).abs() < 1e-6);
        assert_eq!(word_similarity("", ""), 1.0);
        assert!(word_similarity("кот", "код") > 0.6);
        assert!(word_similarity("кот", "алгоритм") < 0.3);
    }

    #[test]
    fn wake_words_are_stripped() {
        assert_eq!(
            clean_wakewords_and_prompt("Джарвис, какая погода"),
            "какая погода"
        );
        assert_eq!(
            clean_wakewords_and_prompt("эй привет джавиз открой ютуб"),
            "открой ютуб"
        );
        assert_eq!(
            clean_wakewords_and_prompt("джарвис джарвис"),
            "джарвис джарвис"
        );
    }

    #[test]
    fn bare_wake_word_is_detected() {
        // STT часто записывает «Джарвис» криво — «жарвис» это тоже wake word.
        assert!(is_wake_only("жарвис"));
        assert!(is_wake_only("джарвис"));
        assert!(is_wake_only("эй джарвис привет"));
        assert!(!is_wake_only("джарвис какая погода"));
        assert!(!is_wake_only("погода"));
        assert!(!is_wake_only(""));
    }

    #[test]
    fn fragments_of_filler_words_are_not_questions() {
        assert!(is_filler_only("а после"));
        assert!(is_filler_only("ну вот и всё"));
        assert!(is_filler_only("значит так"));
        assert!(!is_filler_only("а какая погода"));
        assert!(!is_filler_only("послезавтра"));
        assert!(!is_filler_only("да"));
        assert!(!is_filler_only(""));
    }

    #[test]
    fn repeat_text_extraction() {
        assert_eq!(
            extract_repeat_text("скажи привет мир"),
            Some("привет мир".into())
        );
        assert_eq!(extract_repeat_text("скажи"), None);
        assert_eq!(extract_repeat_text("открой youtube"), None);
    }

    #[test]
    fn validity_filters_short_and_noise() {
        assert!(!is_valid("да"));
        assert!(!is_valid("угу"));
        assert!(is_valid("открой ютуб"));
    }

    #[test]
    fn strong_resume_markers() {
        for phrase in [
            "о чём мы говорили",
            "о чем говорили",
            "что мы обсуждали",
            "на чём мы остановились",
            "вернись к теме",
            "продолжи то, что ты говорил",
        ] {
            assert_eq!(
                extract_topic_resume(phrase),
                Some(clean_text(phrase)),
                "{phrase} должен распознаваться как возврат к теме"
            );
        }
    }

    #[test]
    fn wake_words_are_stripped_from_resume() {
        assert_eq!(
            extract_topic_resume("Джарвис, слушай, о чём говорили"),
            Some("о чём говорили".into())
        );
    }

    #[test]
    fn short_weak_resume_markers() {
        assert_eq!(extract_topic_resume("забыли"), Some("забыли".into()));
        assert_eq!(
            extract_topic_resume("мы это забыли"),
            Some("мы это забыли".into())
        );
        assert_eq!(extract_topic_resume("вернись"), Some("вернись".into()));
    }

    #[test]
    fn forgotten_thing_is_not_a_resume() {
        // Новая просьба, а не возврат к теме.
        assert_eq!(extract_topic_resume("мы забыли купить хлеб"), None);
        assert_eq!(extract_topic_resume("забыл включить свет в спальне"), None);
        assert_eq!(
            extract_topic_resume("продолжи то что я говорил вчера про ремонт"),
            None
        );
    }

    #[test]
    fn ordinary_questions_are_not_resume() {
        assert_eq!(extract_topic_resume("что такое марс"), None);
        assert_eq!(extract_topic_resume("открой youtube"), None);
        assert_eq!(extract_topic_resume("расскажи про погоду"), None);
    }

    #[test]
    fn tts_text_cleanup() {
        assert_eq!(
            clean_tts_text("**Заголовок**\n\n- пункт"),
            "Заголовок - пункт"
        );
    }
}

// ============================================================
// Нормализация текста для TTS: числа словами, единицы, даты,
// годы, ударения. Silero не читает «2 м/с» и «2026» так, как
// это принято в устной речи — готовим текст заранее.
// ============================================================

const ONES_M: [&str; 20] = [
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
];
const ONES_F: [&str; 20] = [
    "ноль",
    "одна",
    "две",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
];
const TENS: [&str; 10] = [
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
];
const HUNDREDS: [&str; 10] = [
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
];

#[derive(Clone, Copy, PartialEq)]
pub enum Gender {
    Masc,
    Fem,
}

/// Падеж для порядковых числительных (мужской род, единственное число).
#[derive(Clone, Copy)]
pub enum GramCase {
    Nom,
    Gen,
    Dat,
    Instr,
    Prep,
}

impl GramCase {
    fn idx(self) -> usize {
        match self {
            GramCase::Nom => 0,
            GramCase::Gen => 1,
            GramCase::Dat => 2,
            GramCase::Instr => 3,
            GramCase::Prep => 4,
        }
    }
}

/// Формы порядковых числительных м.р. ед.ч.: [Nom, Gen, Dat, Instr, Prep].
/// Порядковые в датах склоняются только последним словом: «две тысячи
/// двадцать шестого» — не «двух тысяч двадцати шестого».
const ORD_M_1_19: [[&str; 5]; 20] = [
    ["нулевой", "нулевого", "нулевому", "нулевым", "нулевом"],
    ["первый", "первого", "первому", "первым", "первом"],
    ["второй", "второго", "второму", "вторым", "втором"],
    ["третий", "третьего", "третьему", "третьим", "третьем"],
    [
        "четвёртый",
        "четвёртого",
        "четвёртому",
        "четвёртым",
        "четвёртом",
    ],
    ["пятый", "пятого", "пятому", "пятым", "пятом"],
    ["шестой", "шестого", "шестому", "шестым", "шестом"],
    ["седьмой", "седьмого", "седьмому", "седьмым", "седьмом"],
    ["восьмой", "восьмого", "восьмому", "восьмым", "восьмом"],
    ["девятый", "девятого", "девятому", "девятым", "девятом"],
    ["десятый", "десятого", "десятому", "десятым", "десятом"],
    [
        "одиннадцатый",
        "одиннадцатого",
        "одиннадцатому",
        "одиннадцатым",
        "одиннадцатом",
    ],
    [
        "двенадцатый",
        "двенадцатого",
        "двенадцатому",
        "двенадцатым",
        "двенадцатом",
    ],
    [
        "тринадцатый",
        "тринадцатого",
        "тринадцатому",
        "тринадцатым",
        "тринадцатом",
    ],
    [
        "четырнадцатый",
        "четырнадцатого",
        "четырнадцатому",
        "четырнадцатым",
        "четырнадцатом",
    ],
    [
        "пятнадцатый",
        "пятнадцатого",
        "пятнадцатому",
        "пятнадцатым",
        "пятнадцатом",
    ],
    [
        "шестнадцатый",
        "шестнадцатого",
        "шестнадцатому",
        "шестнадцатым",
        "шестнадцатом",
    ],
    [
        "семнадцатый",
        "семнадцатого",
        "семнадцатому",
        "семнадцатым",
        "семнадцатом",
    ],
    [
        "восемнадцатый",
        "восемнадцатого",
        "восемнадцатому",
        "восемнадцатым",
        "восемнадцатом",
    ],
    [
        "девятнадцатый",
        "девятнадцатого",
        "девятнадцатому",
        "девятнадцатым",
        "девятнадцатом",
    ],
];

const ORD_M_TENS: [[&str; 5]; 10] = [
    ["", "", "", "", ""],
    ["", "", "", "", ""],
    [
        "двадцатый",
        "двадцатого",
        "двадцатому",
        "двадцатым",
        "двадцатом",
    ],
    [
        "тридцатый",
        "тридцатого",
        "тридцатому",
        "тридцатым",
        "тридцатом",
    ],
    [
        "сороковой",
        "сорокового",
        "сороковому",
        "сороковым",
        "сороковом",
    ],
    [
        "пятидесятый",
        "пятидесятого",
        "пятидесятому",
        "пятидесятым",
        "пятидесятом",
    ],
    [
        "шестидесятый",
        "шестидесятого",
        "шестидесятому",
        "шестидесятым",
        "шестидесятом",
    ],
    [
        "семидесятый",
        "семидесятого",
        "семидесятому",
        "семидесятым",
        "семидесятом",
    ],
    [
        "восьмидесятый",
        "восьмидесятого",
        "восьмидесятому",
        "восьмидесятым",
        "восьмидесятом",
    ],
    [
        "девяностый",
        "девяностого",
        "девяностому",
        "девяностым",
        "девяностом",
    ],
];

const ORD_M_HUNDREDS: [[&str; 5]; 10] = [
    ["", "", "", "", ""],
    ["сотый", "сотого", "сотому", "сотым", "сотом"],
    [
        "двухсотый",
        "двухсотого",
        "двухсотому",
        "двухсотым",
        "двухсотом",
    ],
    [
        "трёхсотый",
        "трёхсотого",
        "трёхсотому",
        "трёхсотым",
        "трёхсотом",
    ],
    [
        "четырёхсотый",
        "четырёхсотого",
        "четырёхсотому",
        "четырёхсотым",
        "четырёхсотом",
    ],
    [
        "пятисотый",
        "пятисотого",
        "пятисотому",
        "пятисотым",
        "пятисотом",
    ],
    [
        "шестисотый",
        "шестисотого",
        "шестисотому",
        "шестисотым",
        "шестисотом",
    ],
    [
        "семисотый",
        "семисотого",
        "семисотому",
        "семисотым",
        "семисотом",
    ],
    [
        "восьмисотый",
        "восьмисотого",
        "восьмисотому",
        "восьмисотым",
        "восьмисотом",
    ],
    [
        "девятисотый",
        "девятисотого",
        "девятисотому",
        "девятисотым",
        "девятисотом",
    ],
];

/// Порядковое числительное мужского рода от 0 до 999.
/// 26 → «двадцать шестой»; 26 в родительном → «двадцать шестого».
fn ordinal_1_999(n: u64, case: GramCase) -> String {
    let i = case.idx();
    if n < 20 {
        return ORD_M_1_19[n as usize][i].to_string();
    }
    if n < 100 {
        let t = (n / 10) as usize;
        let u = (n % 10) as usize;
        if u == 0 {
            return ORD_M_TENS[t][i].to_string();
        }
        // Склоняется только последнее слово: «двадцать шестого».
        return format!("{} {}", TENS[t], ORD_M_1_19[u][i]);
    }
    let h = (n / 100) as usize;
    let rest = n % 100;
    if rest == 0 {
        return ORD_M_HUNDREDS[h][i].to_string();
    }
    // Склоняется только последнее слово: «девятьсот девяносто девятого».
    format!("{} {}", HUNDREDS[h], ordinal_1_999(rest, case))
}

/// Порядковое числительное года: 2026 -> «две тысячи двадцать шестой»
/// (родительный: «две тысячи двадцать шестого»). Склоняется только
/// последнее слово, как и принято в устной речи.
pub fn year_ordinal_words(n: u64, case: GramCase) -> String {
    if n < 1000 {
        return ordinal_1_999(n, case);
    }
    if n < 2000 {
        let rest = n - 1000;
        if rest == 0 {
            return match case {
                GramCase::Nom => "тысячный",
                GramCase::Gen => "тысячного",
                GramCase::Dat => "тысячному",
                GramCase::Instr => "тысячным",
                GramCase::Prep => "тысячном",
            }
            .to_string();
        }
        return format!("тысяча {}", ordinal_1_999(rest, case));
    }
    if n < 3000 {
        let rest = n - 2000;
        if rest == 0 {
            return match case {
                GramCase::Nom => "двухтысячный",
                GramCase::Gen => "двухтысячного",
                GramCase::Dat => "двухтысячному",
                GramCase::Instr => "двухтысячным",
                GramCase::Prep => "двухтысячном",
            }
            .to_string();
        }
        return format!("две тысячи {}", ordinal_1_999(rest, case));
    }
    // 3000..9999 — общий случай (для годов за пределами реальных дат).
    let th = n / 1000;
    let rest = n % 1000;
    let head = format!("{} тысяч", num_to_words(th, Gender::Fem));
    if rest == 0 {
        return head;
    }
    format!("{} {}", head, ordinal_1_999(rest, case))
}

/// Число прописью (до миллионов), с согласованием рода.
pub fn num_to_words(n: u64, gender: Gender) -> String {
    fn inner(n: u64, gender: Gender) -> String {
        let ones = |i: usize| match gender {
            Gender::Masc => ONES_M[i],
            Gender::Fem => ONES_F[i],
        };
        if n < 20 {
            return ones(n as usize).to_string();
        }
        if n < 100 {
            let t = TENS[(n / 10) as usize];
            let rest = n % 10;
            if rest == 0 {
                return t.to_string();
            }
            return format!("{t} {}", ones(rest as usize));
        }
        if n < 1000 {
            let h = HUNDREDS[(n / 100) as usize];
            let rest = n % 100;
            if rest == 0 {
                return h.to_string();
            }
            return format!("{h} {}", inner(rest, gender));
        }
        if n < 1_000_000 {
            let th = n / 1000;
            let form = if th % 10 == 1 && th % 100 != 11 {
                "тысяча"
            } else if (2..=4).contains(&(th % 10)) && !(11..=14).contains(&(th % 100)) {
                "тысячи"
            } else {
                "тысяч"
            };
            let rest = n % 1000;
            if th == 1 && rest == 0 {
                return "тысяча".to_string();
            }
            if rest == 0 {
                return format!("{} {}", inner(th, Gender::Fem), form);
            }
            return format!(
                "{} {} {}",
                inner(th, Gender::Fem),
                form,
                inner(rest, gender)
            );
        }
        let m = n / 1_000_000;
        let form = if m % 10 == 1 && m % 100 != 11 {
            "миллион"
        } else if (2..=4).contains(&(m % 10)) && !(11..=14).contains(&(m % 100)) {
            "миллиона"
        } else {
            "миллионов"
        };
        let rest = n % 1_000_000;
        if rest == 0 {
            return format!("{} {}", inner(m, Gender::Masc), form);
        }
        format!(
            "{} {} {}",
            inner(m, Gender::Masc),
            form,
            inner(rest, gender)
        )
    }

    inner(n, gender)
}

/// Форма существительного по числу: (один, два, пять).
pub fn plural_form(n: u64, one: &str, few: &str, many: &str) -> String {
    if n % 10 == 1 && n % 100 != 11 {
        one.to_string()
    } else if (2..=4).contains(&(n % 10)) && !(11..=14).contains(&(n % 100)) {
        few.to_string()
    } else {
        many.to_string()
    }
}

/// Словарь произношения: слово -> как писать для TTS.
/// Используйте, чтобы чинить ударения отдельных слов.
pub fn apply_pronunciation(input: &str) -> String {
    use std::sync::LazyLock;

    // ё восстанавливаем для частых слов — синтез читает её правильно.
    static MAP: LazyLock<Vec<(regex::Regex, &'static str)>> = LazyLock::new(|| {
        [
            ("ее", "её"),
            ("еще", "ещё"),
            ("все таки", "всё-таки"),
            ("что то", "что-то"),
            ("как то", "как-то"),
            ("где то", "где-то"),
            ("сколько нибудь", "сколько-нибудь"),
        ]
        .iter()
        .map(|(from, to)| {
            let re = regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(from))).unwrap();
            (re, *to)
        })
        .collect()
    });

    let mut s = input.to_string();
    for (re, to) in MAP.iter() {
        s = re.replace_all(&s, *to).to_string();
    }
    s
}

/// Порядковое числительное среднего рода (для дат): 24 -> «двадцать четвёртое».
pub fn date_ordinal_words(n: u64) -> String {
    const ORD: [&str; 20] = [
        "нулевое",
        "первое",
        "второе",
        "третье",
        "четвёртое",
        "пятое",
        "шестое",
        "седьмое",
        "восьмое",
        "девятое",
        "десятое",
        "одиннадцатое",
        "двенадцатое",
        "тринадцатое",
        "четырнадцатое",
        "пятнадцатое",
        "шестнадцатое",
        "семнадцатое",
        "восемнадцатое",
        "девятнадцатое",
    ];
    const TENS_ORD: [&str; 10] = [
        "",
        "",
        "двадцатое",
        "тридцатое",
        "сороковое",
        "пятидесятое",
        "шестидесятое",
        "семидесятое",
        "восьмидесятое",
        "девяностое",
    ];
    if n < 20 {
        return ORD[n as usize].to_string();
    }
    let t = (n / 10) as usize;
    let rest = n % 10;
    if rest == 0 {
        return TENS_ORD[t].to_string();
    }
    format!("{} {}", TENS[t], ORD[rest as usize])
}

const MONTHS_GENITIVE: &[&str] = &[
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
];

// 1970-01-01 — четверг.
const WEEKDAYS: &[&str] = &[
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
    "понедельник",
    "вторник",
    "среда",
];

const MONTHS_NOMINATIVE: &[&str] = &[
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
];

fn epoch_days() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
        .div_euclid(86_400)
}

/// Сегодняшняя дата по-русски: «24 октября 2026, пятница».
pub fn today_russian() -> String {
    let days = epoch_days();
    let (y, m, d) = civil_from_days(days);
    let weekday = WEEKDAYS[days.rem_euclid(7) as usize];
    let month = MONTHS_GENITIVE[(m - 1) as usize];
    format!("{d} {month} {y}, {weekday}")
}

/// Мгновенные проценты без LLM: «сколько будет 15 процентов от 2400».
/// Арифметика маленькой модели нестабильна (то 360, то 636) — считаем сами.
/// Число-слово по префиксу (STT пишет падежи как попало): (значение, разряд).
/// Разряд: 0 — единицы/десятки (прибавляем), 1 — сотни (прибавляем),
/// 2 — тысячи (умножаем накопленное 1..=999). Цифры тоже принимаем.
fn ru_number_word(w: &str) -> Option<(f64, u8)> {
    if let Ok(d) = w.parse::<f64>() {
        return Some((d, 0));
    }
    if w == "сто" {
        return Some((100.0, 1));
    }
    const TABLE: &[(&str, f64, u8)] = &[
        ("одиннадцат", 11.0, 0),
        ("двенадцат", 12.0, 0),
        ("тринадцат", 13.0, 0),
        ("четырнадцат", 14.0, 0),
        ("пятнадцат", 15.0, 0),
        ("шестнадцат", 16.0, 0),
        ("семнадцат", 17.0, 0),
        ("восемнадцат", 18.0, 0),
        ("девятнадцат", 19.0, 0),
        ("двадцат", 20.0, 0),
        ("тридцат", 30.0, 0),
        ("сорок", 40.0, 0),
        ("пятьдесят", 50.0, 0),
        ("пятидесят", 50.0, 0),
        ("шестьдесят", 60.0, 0),
        ("шестидесят", 60.0, 0),
        ("семьдесят", 70.0, 0),
        ("семидесят", 70.0, 0),
        ("восемьдесят", 80.0, 0),
        ("восьмидесят", 80.0, 0),
        ("девяност", 90.0, 0),
        ("двест", 200.0, 1),
        ("двухсот", 200.0, 1),
        ("двухст", 200.0, 1),
        ("двумст", 200.0, 1),
        ("трист", 300.0, 1),
        ("трёхсот", 300.0, 1),
        ("трёхст", 300.0, 1),
        ("тремст", 300.0, 1),
        ("четырест", 400.0, 1),
        ("четырёхсот", 400.0, 1),
        ("четырёхст", 400.0, 1),
        ("пятьсот", 500.0, 1),
        ("пятисот", 500.0, 1),
        ("пятист", 500.0, 1),
        ("шестьсот", 600.0, 1),
        ("шестисот", 600.0, 1),
        ("шестист", 600.0, 1),
        ("семьсот", 700.0, 1),
        ("семисот", 700.0, 1),
        ("семист", 700.0, 1),
        ("восемьсот", 800.0, 1),
        ("восьмисот", 800.0, 1),
        ("восьмист", 800.0, 1),
        ("девятьсот", 900.0, 1),
        ("девятисот", 900.0, 1),
        ("девятист", 900.0, 1),
        ("тысяч", 0.0, 2),
        ("один", 1.0, 0),
        ("одн", 1.0, 0),
        ("два", 2.0, 0),
        ("две", 2.0, 0),
        ("двух", 2.0, 0),
        ("трём", 3.0, 0),
        ("трёх", 3.0, 0),
        ("четырём", 4.0, 0),
        ("четырёх", 4.0, 0),
        ("пяти", 5.0, 0),
        ("шести", 6.0, 0),
        ("семи", 7.0, 0),
        ("восьми", 8.0, 0),
        ("девяти", 9.0, 0),
        ("десяти", 10.0, 0),
        ("три", 3.0, 0),
        ("четыр", 4.0, 0),
        ("пят", 5.0, 0),
        ("шест", 6.0, 0),
        ("сем", 7.0, 0),
        ("восем", 8.0, 0),
        ("девят", 9.0, 0),
        ("десят", 10.0, 0),
        ("сот", 100.0, 1),
    ];
    let mut best: Option<(usize, f64, u8)> = None;
    for (stem, value, kind) in TABLE {
        if w.starts_with(stem) && best.is_none_or(|(len, _, _)| stem.len() > len) {
            best = Some((stem.len(), *value, *kind));
        }
    }
    best.map(|(_, v, k)| (v, k))
}

/// Последовательность чисел-слов в число: «две тысячи четырёхсот» → 2400,
/// «двадцать пять» → 25. Числа-слова должны идти подряд.
fn ru_number_seq(words: &[&str]) -> Option<f64> {
    let mut total: f64 = 0.0;
    let mut current: f64 = 0.0;
    let mut saw_any = false;
    for w in words {
        let Some((value, kind)) = ru_number_word(w) else {
            if saw_any {
                break;
            }
            continue;
        };
        saw_any = true;
        match kind {
            2 => {
                total += current.max(1.0) * 1000.0;
                current = 0.0;
            }
            _ => current += value,
        }
    }
    if !saw_any {
        return None;
    }
    let result = total + current;
    (result > 0.0).then_some(result)
}

/// Мгновенные проценты без LLM: «сколько будет 15 процентов от 2400» и
/// «сколько будет пятнадцать процентов от двух тысяч четырёхсот».
/// Арифметика маленькой модели нестабильна (то 360, то 636) — считаем сами.
pub fn percent_answer(text: &str) -> Option<String> {
    let t = text.to_lowercase().replace(',', ".");
    let pct_pos = t.find("процент")?;
    let before: Vec<&str> = t[..pct_pos].split_whitespace().collect();

    // Проценты: до трёх числовых слов вплотную перед словом «процент».
    let mut pct_seq: Vec<&str> = Vec::new();
    for w in before.iter().rev() {
        if ru_number_word(w).is_some() {
            pct_seq.push(w);
            if pct_seq.len() >= 3 {
                break;
            }
        } else if !pct_seq.is_empty() {
            break;
        }
    }
    pct_seq.reverse();
    let pct = ru_number_seq(&pct_seq)?;
    if !(0.0..=100.0).contains(&pct) {
        return None;
    }

    // База — числа после «от».
    let rest = &t[pct_pos + "процент".len()..];
    let from_pos = rest.find("от")?;
    let base_tokens: Vec<&str> = rest[from_pos + "от".len()..].split_whitespace().collect();
    let base = ru_number_seq(&base_tokens)?;

    let value = base * pct / 100.0;
    let value_str = if (value - value.round()).abs() < 0.005 {
        format!("{}", value.round() as i64)
    } else {
        format!("{value:.2}")
    };
    let pct_str = if (pct - pct.round()).abs() < 0.005 {
        format!("{}", pct.round() as i64)
    } else {
        format!("{pct:.2}")
    };
    let base_str = if (base - base.round()).abs() < 0.005 {
        format!("{}", base.round() as i64)
    } else {
        format!("{base:.2}")
    };
    Some(format!(
        "{pct_str} процентов от {base_str} — это {value_str}."
    ))
}

/// Длительность из речи: «на двадцать секунд», «на 5 минут», «на полчаса»,
/// «через полторы минуты» → секунды. Используется быстрым путём таймера,
/// чтобы «поставь таймер на N» не зависел от арифметики модели.
pub fn parse_duration(text: &str) -> Option<u64> {
    let t = text.to_lowercase().replace(',', " ");
    // Особые дробные формы без числа.
    if t.contains("полчаса") {
        return Some(1800);
    }
    if t.contains("полминут") {
        return Some(30);
    }
    let words: Vec<&str> = t.split_whitespace().collect();
    let is_unit = |w: &str| -> Option<u64> {
        let w = w.trim_matches(|c: char| !c.is_alphabetic());
        if w.starts_with("секунд") || w.starts_with("секундo") {
            Some(1)
        } else if w.starts_with("минут") {
            Some(60)
        } else if w.starts_with("час") {
            Some(3600)
        } else {
            None
        }
    };
    for (i, w) in words.iter().enumerate() {
        if let Some(mult) = is_unit(w) {
            // Числа-слова/цифры вплотную перед единицей.
            let mut seq: Vec<&str> = Vec::new();
            let mut j = i;
            while j > 0 {
                let prev = words[j - 1];
                let pw = prev.trim_matches(|c: char| !c.is_alphanumeric() && c != '.');
                if ru_number_word(pw).is_some() {
                    seq.insert(0, pw);
                    j -= 1;
                    if seq.len() >= 3 {
                        break;
                    }
                } else {
                    break;
                }
            }
            if let Some(n) = ru_number_seq(&seq) {
                let secs = (n * mult as f64).round() as u64;
                if secs > 0 {
                    return Some(secs);
                }
            }
        }
    }
    None
}

/// Мгновенные календарные ответы без LLM: «какой день будет послезавтра»,
/// «какое завтра число», «какой сегодня день недели». Арифметика календаря
/// маленькой модели не даётся (она честно отвечает «не знаю») — считаем сами:
/// мгновенно, без веб-поиска и без ошибок.
///
/// Слова вопроса ищем по целым словам, а не подстрокой: «год» живёт внутри
/// «город» и «погода», «день» — внутри «деньги», и подстрока превращала
/// «какой город» в «Сегодня 2026 год».
pub fn calendar_answer(text: &str) -> Option<String> {
    let t = text.to_lowercase();
    let words: Vec<&str> = t
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
        .filter(|w| !w.is_empty())
        .collect();

    // «В каком году родился Дональд Трамп?» — вопрос о событии, а не о
    // текущей дате. Раньше «каком» + «году» давали «Сейчас 2026 год».
    if asks_about_event(&t, &words) {
        return None;
    }

    let has = |forms: &[&str]| words.iter().any(|w| forms.contains(w));
    let which = has(&[
        "какой", "какая", "какое", "какие", "какого", "каком", "какой-то",
    ]);

    let day_question = which && has(&["день", "дня", "дню", "днем", "дни", "дней"]);
    let date_question = which && has(&["дата", "дате", "даты", "число", "числа", "числе", "числу"]);
    let month_question = which && has(&["месяц", "месяца", "месяцу", "месяце"]);
    let year_question = which && has(&["год", "года", "году", "годе", "годов", "лет"]);
    if !(day_question || date_question || month_question || year_question) {
        return None;
    }

    let offset: i64 = if t.contains("послезавтра") {
        2
    } else if t.contains("позавчера") {
        -2
    } else if t.contains("завтра") {
        1
    } else if t.contains("вчера") {
        -1
    } else {
        0
    };

    let days = epoch_days() + offset;
    let (y, m, d) = civil_from_days(days);
    let weekday = WEEKDAYS[days.rem_euclid(7) as usize];
    let when = match offset {
        0 => "Сегодня",
        1 => "Завтра",
        2 => "Послезавтра",
        -1 => "Вчера",
        _ => "Позавчера",
    };
    let verb = match offset {
        0 => "",
        -1 | -2 => " было",
        _ => " будет",
    };

    if year_question {
        // «Завтра 2026 год» — бессмыслица: от «завтра» год не меняется.
        // Другим он становится только через границу 31 декабря.
        let this_year = civil_from_days(epoch_days()).0;
        return Some(if y == this_year {
            format!("Сейчас {y} год.")
        } else {
            format!("{when}{verb} {y} год.")
        });
    }
    if month_question {
        return Some(format!(
            "{when}{verb} {}.",
            MONTHS_NOMINATIVE[(m - 1) as usize]
        ));
    }
    if date_question && !day_question {
        return Some(format!(
            "{when}{verb} {d} {} {y} года.",
            MONTHS_GENITIVE[(m - 1) as usize]
        ));
    }
    // «Какой день» — даём и день недели, и дату: и то и другое полезно.
    if offset == 0 {
        Some(format!(
            "Сегодня {weekday}, {d} {} {y} года.",
            MONTHS_GENITIVE[(m - 1) as usize]
        ))
    } else {
        Some(format!(
            "{when} {weekday}, {d} {} {y} года.",
            MONTHS_GENITIVE[(m - 1) as usize]
        ))
    }
}

/// Вопрос о событии/человеке, а не о текущем календаре: «в каком году родился
/// Трамп», «когда умер Пушкин». Маркеры — глаголы и отглагольные формы,
/// которые не встречаются в вопросах «какой сейчас год/месяц/день».
fn asks_about_event(text: &str, words: &[&str]) -> bool {
    const EVENT_MARKERS: &[&str] = &[
        "родился",
        "родилась",
        "родились",
        "рождения",
        "родившийся",
        "умер",
        "умерла",
        "умерли",
        "скончался",
        "скончалась",
        "погиб",
        "погибла",
        "основан",
        "основана",
        "основано",
        "основали",
        "изобрел",
        "изобрела",
        "изобрели",
        "изобретён",
        "изобретен",
        "построен",
        "построена",
        "построили",
        "написан",
        "написана",
        "написали",
        "создан",
        "создана",
        "создали",
        "открыт",
        "открыта",
        "открыли",
        "вышел",
        "вышла",
        "вышло",
        "вышли",
        "произошло",
        "произошла",
        "произошли",
        "случилось",
        "случился",
        "случилась",
        "начался",
        "началась",
        "началось",
        "закончился",
        "закончилась",
        "закончилось",
        "избран",
        "избрана",
        "стал",
        "стала",
        "стали",
    ];
    words.iter().any(|word| EVENT_MARKERS.contains(word))
        || text.contains("сколько лет")
        || text.contains("в каком веке")
        || text.contains("какого года")
}

/// Алгоритм civil_from_days (H. Hinnant): дни от эпохи -> (год, месяц, день).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

fn parse_num(raw: &str) -> (u64, bool) {
    let cleaned = raw.replace(',', ".");
    let has_frac = cleaned.contains('.');
    let int_part: u64 = cleaned
        .split('.')
        .next()
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);
    (int_part, has_frac)
}

fn number_words(raw: &str) -> String {
    let cleaned = raw.replace(',', ".");
    match cleaned.split_once('.') {
        Some((int_part, frac)) => {
            let n: u64 = int_part.parse().unwrap_or(0);
            let mut out = format!("{} целых", num_to_words(n, Gender::Fem));
            for d in frac.chars() {
                if let Some(d) = d.to_digit(10) {
                    out.push(' ');
                    out.push_str(ONES_F[d as usize]);
                }
            }
            out
        }
        None => num_to_words(cleaned.parse().unwrap_or(0), Gender::Masc),
    }
}

/// Главная функция: превращает цифры и обозначения единиц в устную русскую речь.
pub fn normalize_for_tts(input: &str) -> String {
    use std::sync::LazyLock;

    static SIGN_RE: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"(?i)([+\-−])\s*(\d)").unwrap());

    static UNIT_RULES: LazyLock<Vec<(regex::Regex, [&'static str; 3])>> = LazyLock::new(|| {
        [
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*м\s*/\s*с\b",
                ["метр в секунду", "метра в секунду", "метров в секунду"],
            ),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*км\s*/\s*ч\b",
                ["километр в час", "километра в час", "километров в час"],
            ),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*км\s*/\s*с\b",
                [
                    "километр в секунду",
                    "километра в секунду",
                    "километров в секунду",
                ],
            ),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*(?:°|градус(?:ов|а)?)\s*[cс]\b",
                ["градус Цельсия", "градуса Цельсия", "градусов Цельсия"],
            ),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*(?:%|процент(?:ов|а)?)",
                ["процент", "процента", "процентов"],
            ),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*кг\b",
                ["килограмм", "килограмма", "килограммов"],
            ),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*км\b",
                ["километр", "километра", "километров"],
            ),
            (r"(?i)(\d+(?:[.,]\d+)?)\s*м\b", ["метр", "метра", "метров"]),
            (
                r"(?i)(\d+(?:[.,]\d+)?)\s*(?:₽|руб(?:лей|ля|.)?)",
                ["рубль", "рубля", "рублей"],
            ),
        ]
        .iter()
        .map(|(pattern, forms)| (regex::Regex::new(pattern).unwrap(), *forms))
        .collect()
    });

    static TIME_RE: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\b(\d{1,2}):(\d{2})\b").unwrap());

    static NUM_RE: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\b\d+(?:[.,]\d+)?\b").unwrap());

    // Даты с опциональным годом: «24 октября», «24 октября 2026»,
    // «24 октября 2026 года», «3-го мая 2026 г.». Год всегда превращается
    // в порядковое числительное родительного падежа + «года».
    static DATE_FULL_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        let months = MONTHS_GENITIVE.join("|");
        regex::Regex::new(&format!(
            r"(?i)\b(\d{{1,2}})(?:-?\s*го)?\s+({months})(?:\s+(\d{{4}})(?:\s+(?:года|год|г\.|г))?)?"
        ))
        .unwrap()
    });

    // Отдельные годы с явным существительным: «2026 год/года/году/годом».
    // Падеж определяется по слову после числа; для «году» берём
    // предложный («в 2026 году») — самый частый контекст.
    static YEAR_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(\d{4})\s+(год|года|году|годом|г\.|г)\b").unwrap()
    });

    let mut s = input.to_string();

    // Знаки чисел: «+» не произносим, «−» читаем как «минус».
    s = SIGN_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let sign = caps[1].chars().next().unwrap();
            if sign == '+' {
                caps[2].to_string()
            } else {
                format!("минус {}", &caps[2])
            }
        })
        .to_string();

    // Единицы с числом: формы существительного по правилам plural_form.
    for (re, forms) in UNIT_RULES.iter() {
        s = re
            .replace_all(&s, |caps: &regex::Captures| {
                let (n, has_frac) = parse_num(&caps[1]);
                let form = if has_frac {
                    forms[1].to_string()
                } else {
                    plural_form(n, forms[0], forms[1], forms[2])
                };
                let words = if has_frac {
                    number_words(&caps[1])
                } else {
                    num_to_words(n, Gender::Masc)
                };
                format!("{words} {form}")
            })
            .to_string();
    }

    // Даты: «24 октября» → «двадцать четвёртое октября»;
    // «24 октября 2026» → «…октября две тысячи двадцать шестого года».
    s = DATE_FULL_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let day: u64 = caps[1].parse().unwrap_or(0);
            let month = &caps[2];
            let mut out = format!("{} {}", date_ordinal_words(day), month);
            if let Some(y) = caps.get(3) {
                let year: u64 = y.as_str().parse().unwrap_or(0);
                out.push(' ');
                out.push_str(&year_ordinal_words(year, GramCase::Gen));
                out.push_str(" года");
            }
            out
        })
        .to_string();

    // Годы с явным словом: «2026 год/года/году/годом». Раньше они
    // читались количественным («две тысячи двадцать шесть лет») —
    // в русском это порядковое, падеж по существительному.
    s = YEAR_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let y: u64 = caps[1].parse().unwrap_or(0);
            let tail = caps[2].to_lowercase();
            let (case, noun) = match tail.as_str() {
                "года" => (GramCase::Gen, "года"),
                "году" => (GramCase::Prep, "году"),
                "годом" => (GramCase::Instr, "годом"),
                _ => (GramCase::Nom, "год"),
            };
            format!("{} {}", year_ordinal_words(y, case), noun)
        })
        .to_string();

    // Время ЧЧ:ММ -> «двенадцать тридцать».
    s = TIME_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let h: u64 = caps[1].parse().unwrap_or(0);
            let m: u64 = caps[2].parse().unwrap_or(0);
            format!(
                "{} {}",
                num_to_words(h, Gender::Masc),
                num_to_words(m, Gender::Fem)
            )
        })
        .to_string();

    // Оставшиеся числа словами.
    s = NUM_RE
        .replace_all(&s, |caps: &regex::Captures| number_words(&caps[0]))
        .to_string();

    let s = apply_pronunciation(&s);
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tts_normalize_tests {
    use super::*;

    #[test]
    fn units_are_spoken() {
        assert_eq!(
            normalize_for_tts("ветер 2 м/с"),
            "ветер два метра в секунду"
        );
        assert_eq!(
            normalize_for_tts("ветер 1 м/с"),
            "ветер один метр в секунду"
        );
        assert_eq!(
            normalize_for_tts("ветер 5 м/с"),
            "ветер пять метров в секунду"
        );
        assert_eq!(normalize_for_tts("90 км/ч"), "девяносто километров в час");
        assert_eq!(normalize_for_tts("+21°C"), "двадцать один градус Цельсия");
        assert_eq!(normalize_for_tts("50%"), "пятьдесят процентов");
        assert_eq!(normalize_for_tts("5 км"), "пять километров");
    }

    #[test]
    fn numbers_are_spoken() {
        assert_eq!(normalize_for_tts("21 год"), "двадцать один год");
        assert_eq!(normalize_for_tts("1000 рублей"), "тысяча рублей");
        assert_eq!(normalize_for_tts("12:30"), "двенадцать тридцать");
        assert_eq!(normalize_for_tts("2,5 км"), "две целых пять километра");
    }

    #[test]
    fn pronunciation_map() {
        assert_eq!(
            normalize_for_tts("она еще все таки сделает"),
            "она ещё всё-таки сделает"
        );
    }

    #[test]
    fn dates_are_ordinal() {
        assert_eq!(
            normalize_for_tts("сегодня 24 октября"),
            "сегодня двадцать четвёртое октября"
        );
        assert_eq!(normalize_for_tts("1 мая праздник"), "первое мая праздник");
        assert_eq!(normalize_for_tts("встреча 3-го мая"), "встреча третье мая");
    }

    #[test]
    fn dates_with_year_are_genitive_ordinal() {
        // «2026» после месяца -> порядковое родительного + «года».
        assert_eq!(
            normalize_for_tts("сегодня 24 октября 2026"),
            "сегодня двадцать четвёртое октября две тысячи двадцать шестого года"
        );
        // С явным «года» — то же самое, «года» не дублируется.
        assert_eq!(
            normalize_for_tts("1 января 2025 года"),
            "первое января две тысячи двадцать пятого года"
        );
        // «г.» — тоже съедаем.
        assert_eq!(
            normalize_for_tts("3-го мая 1999 г."),
            "третье мая тысяча девятьсот девяносто девятого года"
        );
    }

    #[test]
    fn standalone_years_are_ordinal() {
        // «в 2026 году» — предложный: «в две тысячи двадцать шестом году».
        assert_eq!(
            normalize_for_tts("в 2026 году"),
            "в две тысячи двадцать шестом году"
        );
        // «с 2024 года» — родительный.
        assert_eq!(
            normalize_for_tts("с 2024 года"),
            "с две тысячи двадцать четвёртого года"
        );
        // «2026 год» — именительный.
        assert_eq!(
            normalize_for_tts("наступил 2026 год"),
            "наступил две тысячи двадцать шестой год"
        );
        // Круглый год: 2000 -> «двухтысячный».
        assert_eq!(normalize_for_tts("в 2000 году"), "в двухтысячном году");
    }

    #[test]
    fn today_russian_sane() {
        let today = today_russian();
        // «24 октября 2026, четверг» — месяц и день недели на месте.
        assert!(MONTHS_GENITIVE.iter().any(|m| today.contains(m)));
        assert!(today.contains("202"));
        assert!(WEEKDAYS.iter().any(|w| today.contains(w)));
    }

    #[test]
    fn calendar_day_offset_is_deterministic() {
        let a = calendar_answer("какой день будет послезавтра").expect("ответ");
        assert!(a.starts_with("Послезавтра"), "получено: {a}");
        assert!(WEEKDAYS.iter().any(|w| a.contains(w)), "получено: {a}");
        assert!(a.contains("20"), "год не попал в ответ: {a}");
    }

    #[test]
    fn calendar_today_weekday() {
        let a = calendar_answer("а какой сегодня день недели").expect("ответ");
        assert!(a.starts_with("Сегодня"), "получено: {a}");
    }

    #[test]
    fn calendar_words_not_adjacent() {
        // «какой сейчас год»: слова вопроса не стоят подряд — раньше такой
        // вопрос проваливался в LLM и ловил выдуманный tool-блок.
        let a = calendar_answer("какой сейчас год").expect("ответ");
        assert!(a.contains("год"), "получено: {a}");
        let b = calendar_answer("какой сейчас месяц").expect("ответ");
        assert!(
            MONTHS_NOMINATIVE.iter().any(|m| b.contains(m)),
            "получено: {b}"
        );
        let c = calendar_answer("а какое сейчас число").expect("ответ");
        assert!(c.contains("года"), "получено: {c}");
    }

    #[test]
    fn parse_duration_word_numbers() {
        // STT пишет числа словами — «на двадцать секунд» раньше ловилось
        // как «открой таймер», а не постановка.
        assert_eq!(
            parse_duration("поставь таймер на двадцать секунд"),
            Some(20)
        );
        assert_eq!(parse_duration("поставь таймер на 5 минут"), Some(300));
        assert_eq!(parse_duration("таймер на полчаса"), Some(1800));
        assert_eq!(parse_duration("на полминуты"), Some(30));
        assert_eq!(
            parse_duration("поставь таймер на полтора часа"),
            None // полтора не поддерживаем — уйдёт в LLM
        );
        assert_eq!(parse_duration("расскажи про таймеры"), None);
    }

    #[test]
    fn percent_answer_computes_deterministically() {
        assert_eq!(
            percent_answer("сколько будет 15 процентов от 2400").as_deref(),
            Some("15 процентов от 2400 — это 360.")
        );
        assert_eq!(
            percent_answer("посчитай 20 процентов от 150").as_deref(),
            Some("20 процентов от 150 — это 30.")
        );
        // STT пишет числа словами: «двадцать секунд» ловится таймером,
        // «пятнадцать процентов от двух тысяч четырёхсот» — процентами.
        assert_eq!(
            percent_answer("сколько будет пятнадцать процентов от двух тысяч четырёхсот")
                .as_deref(),
            Some("15 процентов от 2400 — это 360.")
        );
        assert_eq!(
            percent_answer("двадцать пять процентов от 80").as_deref(),
            Some("25 процентов от 80 — это 20.")
        );
        // Не проценты — None (вопрос уходит в LLM).
        assert_eq!(percent_answer("расскажи про проценты в банке"), None);
    }

    #[test]
    fn calendar_number_tomorrow() {
        let a = calendar_answer("какое завтра число").expect("ответ");
        assert!(a.starts_with("Завтра будет"), "получено: {a}");
        assert!(
            MONTHS_GENITIVE.iter().any(|m| a.contains(m)),
            "получено: {a}"
        );
    }

    #[test]
    fn non_calendar_questions_pass_through() {
        assert!(calendar_answer("какая сегодня погода").is_none());
        assert!(calendar_answer("кто такая лариса долина").is_none());
        assert!(calendar_answer("сколько будет дважды два").is_none());
        // Регрессия на подстроки: «год» внутри «город»/«погода», «день»
        // внутри «деньги» раньше выдавали «Сегодня 2026 год» не по теме.
        assert!(calendar_answer("какой город столица австралии").is_none());
        assert!(calendar_answer("какая погода в городе").is_none());
        assert!(calendar_answer("сколько денег нужно").is_none());
        assert!(calendar_answer("какой сегодня праздник").is_none());
        // Вопрос о событии, а не о текущей дате: «каком» + «году» раньше
        // перехватывались календарём и давали «Сейчас 2026 год».
        assert!(calendar_answer("в каком году родился дональд трамп").is_none());
        assert!(calendar_answer("в каком году умер пушкин").is_none());
        assert!(calendar_answer("в каком году основали москву").is_none());
        assert!(calendar_answer("какого года выпуска эта машина").is_none());
        // А обычные вопросы про «сейчас» по-прежнему считаются сами.
        assert!(calendar_answer("какой сейчас год").is_some());
        assert!(calendar_answer("какой сегодня день").is_some());
    }

    #[test]
    fn calendar_year_ignores_day_offset() {
        let this_year = civil_from_days(epoch_days()).0;
        let a = calendar_answer("какой завтра год").expect("ответ");
        // «Завтра 2026 год» — абсурд: завтра год тот же самый.
        assert_ne!(a, format!("Завтра {this_year} год."), "получено: {a}");
    }
}

#[cfg(test)]
mod speed_tests {
    use super::*;

    /// После правил в типичном ответе про погоду не должно остаться цифр —
    /// иначе включается лишний проход LLM перед озвучкой.
    #[test]
    fn weather_answer_needs_no_neural_pass() {
        let out = normalize_for_tts(
            "В Москве облачно, температура около -3°C, ветер 5 м/с, к вечеру до -7°C.",
        );
        assert!(
            !out.chars().any(|c| c.is_ascii_digit()),
            "остались цифры: {out}"
        );
        assert!(out.contains("минус три градуса"), "{out}");
        assert!(out.contains("пять метров в секунду"), "{out}");
    }
}