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
//!
//! Счётчик одинаков на всех платформах (включая iOS — candle-бэкенд там
//! работает, токенизатор приходит с моделью): модели нет на диске — None.

use std::path::Path;
use std::sync::OnceLock;

static TOKENIZER: OnceLock<Option<tokenizers::Tokenizer>> = OnceLock::new();

/// Токенизатор активной модели. `None` — модели нет на диске.
fn tokenizer() -> Option<&'static tokenizers::Tokenizer> {
    TOKENIZER
        .get_or_init(|| {
            let models = crate::paths::models_dir();
            let name = crate::llm::config::model_name();
            // uzu-модели скачиваются в models/uzu/<имя>/, candle — в models/<имя>/.
            let exact = if models.join("uzu").join(&name).is_dir() {
                models.join("uzu").join(&name)
            } else {
                models.join(&name)
            };
            if let Some(tok) = load_from(&exact) {
                return Some(tok);
            }
            // Точная модель не скачана (например, в тестах применён дефолт
            // статики, а не settings.toml). Для счёта бюджета подходит
            // токенизатор любой скачанной модели той же семьи — беру первый
            // найденный в models/uzu/*/.
            let uzu = models.join("uzu");
            if let Ok(entries) = std::fs::read_dir(&uzu) {
                for entry in entries.flatten() {
                    if entry.path().is_dir()
                        && let Some(tok) = load_from(&entry.path())
                    {
                        return Some(tok);
                    }
                }
            }
            None
        })
        .as_ref()
}

fn load_from(dir: &Path) -> Option<tokenizers::Tokenizer> {
    let path = dir.join("tokenizer.json");
    if !path.exists() {
        return None;
    }
    tokenizers::Tokenizer::from_file(&path).ok()
}

/// Сколько токенов занимает текст. `None` — токенизатора нет.
pub fn count(text: &str) -> Option<usize> {
    let tok = tokenizer()?;
    tok.encode(text, false).ok().map(|e| e.get_ids().len())
}
