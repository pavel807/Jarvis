//! Бюджет промптов в токенах.
//!
//! Размер системного промпта — это не эстетика, а время до первого звука:
//! модель читает его на каждом шаге цикла. Раньше action-промпт со всеми 32
//! инструментами и подробными правилами занимал 1662 токена против 250
//! токенов ответа.
//!
//! Здесь только счётчик. Потолки проверяет `tests/prompt_budget.rs`: это
//! интеграционный тест, он сам выставляет `JARVIS_HOME` и не зависит от того,
//! откуда запущен `cargo test`.

use std::path::Path;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

static TOKENIZER: OnceLock<Option<Tokenizer>> = OnceLock::new();

/// Токенизатор активной модели. `None` — модели нет на диске.
fn tokenizer() -> Option<&'static Tokenizer> {
    TOKENIZER
        .get_or_init(|| {
            let dir = crate::paths::models_dir().join(crate::llm::config::model_name());
            load_from(&dir)
        })
        .as_ref()
}

fn load_from(dir: &Path) -> Option<Tokenizer> {
    let path = dir.join("tokenizer.json");
    if !path.exists() {
        return None;
    }
    Tokenizer::from_file(&path).ok()
}

/// Сколько токенов занимает текст. `None` — токенизатора нет.
pub fn count(text: &str) -> Option<usize> {
    let tok = tokenizer()?;
    tok.encode(text, false).ok().map(|e| e.get_ids().len())
}