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
        .replace('*', "")
        .replace('#', "")
        .replace('`', "")
        .replace('_', "")
        .replace("  ", " ")
        .trim()
        .to_string()
}

/// Отсекаем обращения "джарвис", "джавиз" и вводные фразы для получения чистого промпта.
pub fn clean_wakewords_and_prompt(text: &str) -> String {
    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    let wake_variants = [
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

    let mut start_idx = 0;
    while start_idx < words.len() {
        if wake_variants.contains(&words[start_idx]) {
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

/// Достаёт фразу после триггера «скажи/повтори/...» для режима повторения.
pub fn extract_repeat_text(text: &str) -> Option<String> {
    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    let triggers = ["скажи", "повтори", "произнеси", "озвучь"];

    for trigger in &triggers {
        if let Some(pos) = words.iter().position(|&w| w == *trigger) {
            if pos + 1 < words.len() {
                let repeat_phrase = words[pos + 1..].join(" ");
                if !repeat_phrase.trim().is_empty() {
                    return Some(repeat_phrase);
                }
            }
        }
    }

    None
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
        assert_eq!(clean_wakewords_and_prompt("Джарвис, какая погода"), "какая погода");
        assert_eq!(clean_wakewords_and_prompt("эй привет джавиз открой ютуб"), "открой ютуб");
        assert_eq!(clean_wakewords_and_prompt("джарвис джарвис"), "джарвис джарвис");
    }

    #[test]
    fn repeat_text_extraction() {
        assert_eq!(extract_repeat_text("скажи привет мир"), Some("привет мир".into()));
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
    fn tts_text_cleanup() {
        assert_eq!(clean_tts_text("**Заголовок**\n\n- пункт"), "Заголовок - пункт");
    }
}

// ============================================================
// Нормализация текста для TTS: числа словами, единицы, ударения.
// Piper/espeak не умеет читать «2 м/с» и цифры — готовим текст заранее.
// ============================================================

const ONES_M: [&str; 20] = [
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
];
const ONES_F: [&str; 20] = [
    "ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
];
const TENS: [&str; 10] = [
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
    "восемьдесят", "девяносто",
];
const HUNDREDS: [&str; 10] = [
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
    "восемьсот", "девятьсот",
];

#[derive(Clone, Copy, PartialEq)]
pub enum Gender {
    Masc,
    Fem,
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
            return format!("{} {} {}", inner(th, Gender::Fem), form, inner(rest, gender));
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
        format!("{} {} {}", inner(m, Gender::Masc), form, inner(rest, gender))
    }

    let s = inner(n, gender);
    s
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
/// Используйте, чтобы чинить ударения отдельных слов (espeak-ng не умеет
/// различать омографы; тут помогает запись «как слышится»).
pub fn apply_pronunciation(input: &str) -> String {
    // ё восстанавливаем для частых слов — Piper читает её правильно.
    const MAP: &[(&str, &str)] = &[
        ("ее", "её"),
        ("еще", "ещё"),
        ("все таки", "всё-таки"),
        ("зато", "зато"),
        ("что то", "что-то"),
        ("как то", "как-то"),
        ("где то", "где-то"),
        ("сколько нибудь", "сколько-нибудь"),
    ];
    let mut s = input.to_string();
    for (from, to) in MAP {
        let re = regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(from))).unwrap();
        s = re.replace_all(&s, *to).to_string();
    }
    s
}

fn parse_num(raw: &str) -> (u64, bool) {
    let cleaned = raw.replace(',', ".");
    let has_frac = cleaned.contains('.');
    let int_part: u64 = cleaned.split('.').next().unwrap_or("0").parse().unwrap_or(0);
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
    let mut s = input.to_string();

    // Знаки чисел: «+» не произносим, «−» читаем как «минус».
    let sign_re = regex::Regex::new(r"(?i)([+\-−])\s*(\d)").unwrap();
    s = sign_re
        .replace_all(&s, |caps: &regex::Captures| {
            let sign = caps[1].chars().next().unwrap();
            if sign == '+' {
                caps[2].to_string()
            } else {
                format!("минус {}", &caps[2])
            }
        })
        .to_string();

    // Единицы с числом: (регекс, формы существительного, род числа не нужен —
    // мужской род числительного подходит для всех этих единиц).
    let unit_rules: &[(&str, [&str; 3])] = &[
        (r"(?i)(\d+(?:[.,]\d+)?)\s*м\s*/\s*с\b", ["метр в секунду", "метра в секунду", "метров в секунду"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*км\s*/\s*ч\b", ["километр в час", "километра в час", "километров в час"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*км\s*/\s*с\b", ["километр в секунду", "километра в секунду", "километров в секунду"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:°|градус(?:ов|а)?)\s*[cс]\b", ["градус Цельсия", "градуса Цельсия", "градусов Цельсия"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:%|процент(?:ов|а)?)", ["процент", "процента", "процентов"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*кг\b", ["килограмм", "килограмма", "килограммов"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*км\b", ["километр", "километра", "километров"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*м\b", ["метр", "метра", "метров"]),
        (r"(?i)(\d+(?:[.,]\d+)?)\s*(?:₽|руб(?:лей|ля|.)?)", ["рубль", "рубля", "рублей"]),
    ];

    for (pattern, forms) in unit_rules {
        let re = regex::Regex::new(pattern).unwrap();
        s = re
            .replace_all(&s, |caps: &regex::Captures| {
                let (n, has_frac) = parse_num(&caps[1]);
                // Дробные числа: родительный единственный («полтора метра»).
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

    // Время ЧЧ:ММ -> «двенадцать тридцать».
    let time_re = regex::Regex::new(r"\b(\d{1,2}):(\d{2})\b").unwrap();
    s = time_re
        .replace_all(&s, |caps: &regex::Captures| {
            let h: u64 = caps[1].parse().unwrap_or(0);
            let m: u64 = caps[2].parse().unwrap_or(0);
            format!("{} {}", num_to_words(h, Gender::Masc), num_to_words(m, Gender::Fem))
        })
        .to_string();

    // Оставшиеся числа словами.
    let num_re = regex::Regex::new(r"\b\d+(?:[.,]\d+)?\b").unwrap();
    s = num_re
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
        assert_eq!(normalize_for_tts("ветер 2 м/с"), "ветер два метра в секунду");
        assert_eq!(normalize_for_tts("ветер 1 м/с"), "ветер один метр в секунду");
        assert_eq!(normalize_for_tts("ветер 5 м/с"), "ветер пять метров в секунду");
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
        assert_eq!(normalize_for_tts("она еще все таки сделает"), "она ещё всё-таки сделает");
    }
}
