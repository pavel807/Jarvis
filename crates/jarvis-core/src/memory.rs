//! Долговременная память о пользователе — по логике OpenJarvis: после
//! диалога фоновый экстрактор вычленяет устойчивые факты (предпочтения,
//! проекты, ограничения) и складывает в JSON-стор; при следующем запросе
//! факты подмешиваются в системный промпт. Стор намеренно маленький:
//! дедуп по нормализованному тексту, потолок на число фактов, отказо-
//! устойчивость — любая ошибка экстракции деградирует в «фактов нет».

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Сколько фактов держим максимум: промпт не бесконечный.
pub const MAX_FACTS: usize = 200;
/// Сколько фактов уходит в промпт (самые свежие).
const PROMPT_FACTS: usize = 20;
/// Лимит длины одного факта — как в OpenJarvis.
const FACT_CHARS: usize = 200;

/// Один факт: краткое устойчивое утверждение о пользователе.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub text: String,
    /// ISO-дата добавления.
    pub added: String,
}

/// Файл стора: рядом с settings.toml, переживает перезапуски.
fn store_path() -> PathBuf {
    crate::paths::base_dir().join("memory.json")
}

/// Прочитать все факты (нет файла — пусто).
pub fn all() -> Vec<Fact> {
    load(&store_path())
}

fn load(path: &Path) -> Vec<Fact> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Нормализация для дедупа: нижний регистр, схлопнутые пробелы.
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Добавить факты, вернуть сколько реально добавлено (дубликаты и пустые
/// отброшены). Свежие — в начало списка.
pub fn add_facts(facts: &[String]) -> usize {
    add_facts_at(&store_path(), facts)
}

fn add_facts_at(path: &Path, facts: &[String]) -> usize {
    let mut all = load(path);
    let mut added = 0;
    for fact in facts {
        let text = fact.trim().chars().take(FACT_CHARS).collect::<String>();
        if text.chars().count() < 3 {
            continue;
        }
        let norm = normalize(&text);
        if all.iter().any(|f| normalize(&f.text) == norm) {
            continue;
        }
        all.insert(
            0,
            Fact {
                text,
                added: crate::text::today_russian(),
            },
        );
        added += 1;
    }
    all.truncate(MAX_FACTS);
    if added > 0 || !path.exists() {
        save_at(path, all);
    }
    added
}

/// Удалить факт по тексту (точное совпадение после нормализации).
pub fn remove_fact(text: &str) -> bool {
    let norm = normalize(text);
    let mut all = all();
    let before = all.len();
    all.retain(|f| normalize(&f.text) != norm);
    let removed = all.len() < before;
    if removed {
        save(all);
    }
    removed
}

/// Очистить память целиком (настройки/отладка).
pub fn clear() {
    save(Vec::new());
}

fn save(facts: Vec<Fact>) {
    save_at(&store_path(), facts);
}

/// Атомарная запись: tmp + rename, чтобы обрыв не оставил битый стор.
fn save_at(path: &Path, facts: Vec<Fact>) {
    let tmp = path.with_extension("json.tmp");
    if let Ok(file) = std::fs::File::create(&tmp)
        && serde_json::to_writer_pretty(file, &facts).is_ok()
    {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Блок фактов для системного промпта: пустая память — пустая строка.
pub fn prompt_block() -> String {
    prompt_block_from(&all())
}

fn prompt_block_from(facts: &[Fact]) -> String {
    if facts.is_empty() {
        return String::new();
    }
    let listed: Vec<String> = facts
        .iter()
        .take(PROMPT_FACTS)
        .map(|f| format!("  — {}", f.text))
        .collect();
    format!(
        "\n\nИЗВЕСТНО О ПОЛЬЗОВАТЕЛЕ (долговременная память, учитывай в ответах):\n{}\n",
        listed.join("\n")
    )
}

/// Промпт-экстрактор: логика OpenJarvis — только устойчивые факты,
/// ответ строго JSON-массивом, ничего достойного — [].
pub const EXTRACT_SYSTEM_PROMPT: &str = "Ты извлекаешь долговременные факты о пользователе из одной реплики диалога. Факт — то, что стабильно во времени и пригодится в будущих разговорах: предпочтения, личные данные, цели, текущие проекты, ограничения, близкие люди. Разовые детали задачи, светская болтовня и слова ассистента не считаются. Ответь ТОЛЬКО JSON-массивом коротких строк (каждая до 200 символов, по-русски). Если запоминать нечего — ответь [].";

/// Разбор ответа экстрактора: массив строк; мусор — пустой список.
pub fn parse_facts(reply: &str) -> Vec<String> {
    let trimmed = reply.trim();
    let start = match trimmed.find('[') {
        Some(i) => i,
        None => return Vec::new(),
    };
    let end = match trimmed.rfind(']') {
        Some(i) if i > start => i + 1,
        _ => return Vec::new(),
    };
    serde_json::from_str::<Vec<String>>(&trimmed[start..end]).unwrap_or_default()
}

/// Фоновая экстракция из завершённого обмена: никогда не паникуем и не
/// блокируем диалог — ошибки просто пишутся в stderr.
pub fn extract_async(shared: crate::llm::SharedLlm, user: String, assistant: String) {
    // Пустая реплика или слишком короткая — модели делать нечего.
    if user.trim().chars().count() < 8 {
        return;
    }
    std::thread::spawn(move || {
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let Ok(reply) = crate::llm::generate_blocking(
            &shared,
            EXTRACT_SYSTEM_PROMPT,
            &[],
            &format!("Реплика пользователя: {user}\nОтвет ассистента: {assistant}"),
            &cancel,
        ) else {
            return; // модель занята/недоступна — память в этот раз не пополняется
        };
        let facts = parse_facts(&reply);
        if facts.is_empty() {
            return;
        }
        let added = add_facts(&facts);
        if added > 0 {
            println!("🧠 Память: +{added} факт(ов)");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jarvis-memory-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("memory.json")
    }

    #[test]
    fn facts_dedupe_cap_and_order() {
        let path = temp_store("dedupe");
        assert_eq!(add_facts_at(&path, &["Любит короткие ответы".into()]), 1);
        // Дубль в другом регистре/пробелах не добавляется.
        assert_eq!(add_facts_at(&path, &["любит  короткие ответы".into()]), 0);
        // Пустышки и огрызки не добавляются.
        assert_eq!(add_facts_at(&path, &["  ".into(), "ок".into()]), 0);
        // Свежее — в начало.
        add_facts_at(&path, &["Работает над Jarvis".into()]);
        let all = load(&path);
        assert_eq!(all[0].text, "Работает над Jarvis");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn long_fact_truncated() {
        let path = temp_store("truncate");
        let long = "х".repeat(500);
        add_facts_at(&path, &[long]);
        let all = load(&path);
        assert_eq!(all[0].text.chars().count(), FACT_CHARS);
    }

    #[test]
    fn parse_facts_accepts_array_and_rejects_garbage() {
        assert_eq!(parse_facts("[\"а\", \"б\"]"), vec!["а", "б"]);
        assert!(parse_facts("Вот что я запомнил: [\"проект Jarvis\"]").len() == 1);
        assert!(parse_facts("[]").is_empty());
        assert!(parse_facts("мимо кассы").is_empty());
        assert!(parse_facts("[битый json").is_empty());
    }

    #[test]
    fn prompt_block_lists_facts_and_empty_memory_is_silent() {
        assert_eq!(prompt_block_from(&[]), "");
        let block = prompt_block_from(&[Fact {
            text: "Тестовый факт для блока".into(),
            added: "тест".into(),
        }]);
        assert!(block.contains("ИЗВЕСТНО О ПОЛЬЗОВАТЕЛЕ"));
        assert!(block.contains("Тестовый факт для блока"));
        // Кап на число фактов в промпте.
        let many: Vec<Fact> = (0..50)
            .map(|i| Fact { text: format!("факт номер {i}"), added: String::new() })
            .collect();
        assert_eq!(prompt_block_from(&many).lines().count(), 2 + PROMPT_FACTS + 1);
    }

    #[test]
    fn remove_fact_by_normalized_text() {
        let path = temp_store("remove");
        add_facts_at(&path, &["Любит короткие ответы".into()]);
        // Удаление через публичный API работает с реальным стором — здесь
        // проверяем только нормализацию поиска на уровне списка.
        let all = load(&path);
        assert!(all.iter().any(|f| normalize(&f.text) == normalize("Любит короткие ответы")));
    }
}
