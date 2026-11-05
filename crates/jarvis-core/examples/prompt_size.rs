//! Замер размера промптов в токенах: сколько стоит префилл под конкретную
//! просьбу. Печатает таблицу, чтобы правки промпта не раздували отклик.
//!
//! Запуск: JARVIS_HOME=. cargo run --release --example prompt_size -p jarvis-core

use anyhow::Result;
use tokenizers::Tokenizer;

fn tokens(tok: &Tokenizer, text: &str) -> Result<usize> {
    let encoding = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    Ok(encoding.get_ids().len())
}

fn main() -> Result<()> {
    let dir = jarvis_core::paths::models_dir().join(jarvis_core::llm::config::model_name());
    let tok =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;

    let chat = jarvis_core::llm::CHAT_SYSTEM_PROMPT;
    let web = jarvis_core::llm::WEB_SYSTEM_PROMPT;
    let catalog = jarvis_core::commands::prompt_catalog();

    println!("{:<34} {:>7} {:>6}", "промпт", "символы", "токены");
    println!("{}", "-".repeat(50));
    let row = |name: &str, text: &str| -> Result<()> {
        println!(
            "{:<34} {:>7} {:>6}",
            name,
            text.chars().count(),
            tokens(&tok, text)?
        );
        Ok(())
    };

    row("chat", chat)?;
    row("web", web)?;
    row("каталог команд", &catalog)?;
    println!("{}", "-".repeat(50));
    for (name, prompt) in [
        ("действие: без среза", ""),
        ("действие: сверни зум", "сверни зум"),
        ("действие: таймер", "поставь таймер на 10 минут"),
        ("действие: открой хром", "открой хром"),
        ("действие: напиши сообщение", "отправь сообщение маме"),
        (
            "действие: много срезов",
            "сверни зум, поставь таймер и отправь сообщение",
        ),
    ] {
        row(name, &jarvis_core::router::action_prompt_for(prompt))?;
    }
    Ok(())
}