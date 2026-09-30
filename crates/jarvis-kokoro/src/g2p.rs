//! Русский G2P-фронтенд для kokoro-ru: текст -> IPA-фенемы -> токены.
//!
//! Порт чистых правил `ru_g2p.py` из zaakirio/kokoro-ru (Apache-2.0):
//! орфоэпические перезаписи (-ого/-ево, чн->шн и т.п.) применяются к тексту
//! до фонемизации; нормализация выходных символов espeak, позиционная
//! редукция безударных гласных (аканье) и удлинение щ — после.
//! Нейросетевой ударитель RUAccent (Python/torch) не портирован — ударение
//! ставит встроенный espeak-ng; на бытовых фразах качество сопоставимое.

use espeak_rs::text_to_phonemes;
use std::sync::Mutex;

/// espeak-rs держит глобальное состояние и не потокобезопасен.
static ESPEAK_MUTEX: Mutex<()> = Mutex::new(());

/// Выход espeak -> словарь kokoro-ru. Замечено на ru-эспике (см. ru_g2p.py).
const NORMALIZE: &[(&str, &str)] = &[
    ("u\"", "u"), // централизованный u протекает как "u+quote" (чу́вство)
    ("ɭ", "l"),   // твёрдое л от espeak
    ("ɵ", "o"),   // ядро ударной ё (jɵ -> jo)
    ("ʑ", "ʒ"),   // редкая ассимиляция через морфему
    ("ʐ", "ʒ"),   // защитно; для ru не встречается
    ("ʧʲ", "ʧ"),  // ч изначально мягкая — лишняя палатализация
];

// -ого/-его читается -ово/-ево. Исключения — наречия и заимствования, где г
// произносится.
const OGO_BLACKLIST: &[&str] = &[
    "много", "немного", "намного", "ненамного", "строго", "нестрого", "настрого", "дорого",
    "недорого", "полого", "убого", "лего", "диего", "ого", "огого",
];

// Орфоэпические перезаписи, которые правила espeak не покрывают.
const RESPELL_SUBSTR: &[(&str, &str)] = &[
    ("солнц", "сонц"),  // со́лнце; солнечный без ц и с л
    ("чувств", "чуств"),
    ("здравств", "здраств"),
    ("счастлив", "счаслив"),
    ("завистлив", "завислив"),
    ("участлив", "учаслив"),
    ("совестлив", "совеслив"),
];

const RESPELL_WORD: &[(&str, &str)] = &[
    ("дорого", "дорога"),
    ("недорого", "недорога"),
    ("настрого", "настрога"),
    ("конечно", "конешно"),
    ("скучно", "скушно"),
    ("скучный", "скушный"),
    ("нарочно", "нарошно"),
    ("яичница", "яишница"),
    ("скворечник", "скворешник"),
    ("девичник", "девишник"),
    ("двоечник", "двоешник"),
    ("троечник", "троешник"),
    ("прачечная", "прашечная"),
    ("горчичник", "горчишник"),
    ("пустячный", "пустяшный"),
];

const VOWELS: &str = "aɑoeiuyʌəɪɐɛɨ";
const REDUCIBLE: &str = "aɑoʌə";
const STRESS_MARKS: &str = "ˈˌ";

fn is_cyrillic_word_char(c: char) -> bool {
    c.is_ascii_alphabetic() || ('а'..='я').contains(&c) || c == 'ё' || c.is_uppercase()
}

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

/// -ого/-его -> -ово/-ево на конце слова, минус исключения из blacklist.
fn respell_ogo(word: &str) -> String {
    let bare = word.to_lowercase();
    if OGO_BLACKLIST.contains(&bare.as_str()) {
        return word.to_string();
    }
    let chars: Vec<char> = word.chars().collect();
    // Ищем суффикс "ого"/"его" в конце (регистронезависимо по сути).
    let n = chars.len();
    if n < 3 {
        return word.to_string();
    }
    let ends_ogo = (chars[n - 3] == 'о' || chars[n - 3] == 'О')
        && (chars[n - 2] == 'г' || chars[n - 2] == 'Г')
        && (chars[n - 1] == 'о' || chars[n - 1] == 'О');
    let ends_ego = (chars[n - 3] == 'е' || chars[n - 3] == 'Е')
        && (chars[n - 2] == 'г' || chars[n - 2] == 'Г')
        && (chars[n - 1] == 'о' || chars[n - 1] == 'О');
    if ends_ogo || ends_ego {
        // г -> в, гласная суффикса сохраняется: "большего" -> "большево".
        let chars: Vec<char> = word.chars().collect();
        let stem: String = chars[..chars.len() - 2].iter().collect();
        format!("{}во", stem)
    } else {
        word.to_string()
    }
}

fn respell_word(word: &str) -> String {
    let bare = word.to_lowercase();
    if let Some((_, target)) = RESPELL_WORD.iter().find(|(k, _)| *k == bare) {
        return (*target).to_string();
    }
    let word = respell_ogo(word);
    let bare = word.to_lowercase();
    let mut out = word;
    for (old, new) in RESPELL_SUBSTR {
        if bare.contains(old) {
            out = out.replace(old, new);
        }
    }
    out
}

/// Орфоэпические перезаписи на уровне текста (только кириллические слова).
pub fn respell(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    for c in text.chars() {
        if is_cyrillic_word_char(c) {
            word.push(c);
        } else {
            if !word.is_empty() {
                out.push_str(&respell_word(&word));
                word.clear();
            }
            out.push(c);
        }
    }
    if !word.is_empty() {
        out.push_str(&respell_word(&word));
    }
    out
}

/// Позиционная редукция безударных гласных (аканье) на одном "слове" IPA.
/// Порт `_reduce_token` из ru_g2p.py — позиционные правила над espeak-выходом.
fn reduce_token(tok: &str) -> String {
    let chars: Vec<char> = tok.chars().collect();
    let vowels: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|&(_, &c)| is_vowel(c))
        .map(|(i, _)| i)
        .collect();
    if vowels.is_empty() {
        return tok.to_string();
    }

    // Знак ударения стоит перед слоговым началом — помечает следующую гласную.
    // Ложное вторичное ударение на -ться/-тся игнорируем (после первичного).
    let primary = chars.iter().position(|&c| c == 'ˈ');
    let mut stressed: Vec<usize> = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        if !STRESS_MARKS.contains(c) {
            continue;
        }
        if c == 'ˌ' && primary.is_some_and(|p| i > p) {
            continue;
        }
        if let Some(&nxt) = vowels.iter().find(|&&j| j > i) {
            stressed.push(nxt);
        }
    }
    let mut stressed_ordinals: Vec<usize> = stressed
        .iter()
        .map(|&i| vowels.iter().position(|&v| v == i).unwrap_or(usize::MAX))
        .collect();
    stressed_ordinals.sort_unstable();

    let mut out: Vec<char> = chars.clone();
    for (ordinal, &i) in vowels.iter().enumerate() {
        if stressed.contains(&i) || !REDUCIBLE.contains(out[i]) {
            continue;
        }
        if i > 0 && (out[i - 1] == 'ʲ' || out[i - 1] == 'j') {
            out[i] = 'ɪ';
            continue;
        }
        let next_stress = stressed_ordinals.iter().find(|&&s| s > ordinal).copied();
        let absolute_initial = i == 0;
        out[i] = if next_stress == Some(ordinal + 1) || absolute_initial {
            'ɐ'
        } else {
            'ə'
        };
    }

    // Убираем артефактные вторичные ударения после первичного.
    if let Some(p) = primary {
        let filtered: Vec<char> = out
            .iter()
            .enumerate()
            .filter(|&(k, &c)| !(c == 'ˌ' && k > p))
            .map(|(_, c)| *c)
            .collect();
        return filtered.into_iter().collect();
    }
    out.into_iter().collect()
}

/// щ/сч/зч — долгая [ɕː]; espeak выдаёт краткую [ɕ] и никогда не ставит ː.
fn lengthen_sch(ps: &str) -> String {
    // ɕ, за которым не следует ː, заменяем на ɕː.
    let mut out = String::with_capacity(ps.len() + 8);
    let chars: Vec<char> = ps.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        out.push(c);
        if c == 'ɕ' && chars.get(i + 1) != Some(&'ː') {
            out.push('ː');
        }
    }
    out
}

/// Фонемизация уже готового текста: espeak (ru) -> нормализация -> редукция.
fn phonemize_text(text: &str) -> String {
    let ps = {
        let _guard = ESPEAK_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        text_to_phonemes(text, "ru", None, true, false)
            .unwrap_or_default()
            .join("")
    };
    let mut ps = ps.to_string();
    for (old, new) in NORMALIZE {
        ps = ps.replace(old, new);
    }
    let reduced: Vec<String> = ps.split(' ').map(reduce_token).collect();
    lengthen_sch(&reduced.join(" "))
}

/// Текст -> IPA-фенемы в словаре kokoro-ru.
/// espeak скобки трактует как границы пауз — приводим остальные к ним.
pub fn phonemize(text: &str) -> String {
    let text = text.replace(['[', '{'], "(").replace([']', '}'], ")");
    let lowered = text.to_lowercase();
    phonemize_text(&respell(&lowered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::tokenize;

    #[test]
    fn ogo_respell() {
        assert_eq!(respell_ogo("большего"), "большево");
        assert_eq!(respell_ogo("много"), "много"); // из blacklist
        assert_eq!(respell_ogo("стол"), "стол");
    }

    #[test]
    fn sch_lengthening() {
        assert_eq!(lengthen_sch("ɕ"), "ɕː");
        assert_eq!(lengthen_sch("ɕː"), "ɕː");
        assert_eq!(lengthen_sch("aɕb"), "aɕːb");
    }

    #[test]
    fn vowel_reduction_positions() {
        // Ударная гласная не редуцируется, заударная -> ə.
        assert_eq!(reduce_token("mˈarka"), "mˈarkə");
        // Первая предударная ɐ не трогается (уже редуцированный символ).
        assert_eq!(reduce_token("vɐdˈa"), "vɐdˈa");
    }

    #[test]
    fn tokenize_skips_unknown() {
        assert_eq!(tokenize("a!"), vec![43, 5]);
        assert!(tokenize("漢字").is_empty());
    }
}
