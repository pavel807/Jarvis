//! Проба чат-ветки LLM: тот же системный промпт, что собирает `llm::ask`,
//! те же вопросы, на которых пользователь поймал отказы.
//!
//! Запуск: cargo run --release -p jarvis-core --example chat_probe [old] [l]
//! По умолчанию — промпт, который отправляет бэкенду uzu
//! (UZU_CHAT_SYSTEM_PROMPT); аргумент `old` — старый жёсткий промпт для
//! сравнения. Аргумент `l` переключает модель на 8-битную qwen3.5-2b-mirai-l.

use anyhow::Result;
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::config as llm_config;
use jarvis_core::llm::{self, SharedLlmState};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let old_prompt = args.iter().any(|a| a == "old");
    let l8 = args.iter().any(|a| a == "l");

    let (events_tx, events_rx) = channel::<JarvisEvent>();
    std::thread::spawn(move || {
        for ev in events_rx {
            match ev {
                JarvisEvent::LlmStatus { status } => println!("🧠 LLM: {status}"),
                JarvisEvent::Log { message } => println!("ℹ️  {message}"),
                JarvisEvent::Error { message } => eprintln!("❗ {message}"),
                _ => {}
            }
        }
    });

    let settings = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())?;
    settings.apply();
    if l8 {
        jarvis_core::llm::config::set_model_name("qwen3.5-2b-mirai-l");
    }
    println!(
        "ℹ️  llm_backend = {}, llm_model = {}, промпт = {}",
        llm_config::backend(),
        llm_config::model_name(),
        if old_prompt {
            "old (CHAT_SYSTEM_PROMPT)"
        } else {
            "uzu (UZU_CHAT_SYSTEM_PROMPT)"
        }
    );

    let shared = Arc::new(SharedLlmState::new());
    llm::spawn_llm_loader(&shared, &events_tx);

    let cancel = Arc::new(AtomicBool::new(false));
    let mut turns: Vec<(String, String)> = Vec::new();

    let asked = |shared: &Arc<SharedLlmState>,
                 history: &[(String, String)],
                 question: &str,
                 cancel: &Arc<AtomicBool>|
     -> String {
        // Как в llm::ask: промпт честности выбирается под бэкенд; `old`
        // принуждает старый жёсткий вариант для сравнения.
        let base = if old_prompt {
            llm::CHAT_SYSTEM_PROMPT.to_string()
        } else {
            llm::UZU_CHAT_SYSTEM_PROMPT.to_string()
        };
        let system = format!(
            "{}\nСегодняшняя дата: {}.\nЕсли просьба совпадает с командой пользователя, \
             вызови run_command с её именем; иначе просто ответь.",
            base,
            jarvis_core::text::today_russian()
        );
        let started = Instant::now();
        let answer = llm::generate_blocking(shared, &system, history, question, cancel)
            .map_err(|e| eprintln!("❗ {e:#}"))
            .unwrap_or_default();
        println!("\n❓ {question}");
        println!("🤖 {answer}");
        println!("⏱  {:.1?}", started.elapsed());
        answer
    };

    for question in [
        "Что такое язык раст?",
        "Что такое Rust?",
        "Сколько будет восемь умножить на тринадцать?",
        "Кто написал «Мастера и Маргариту»?",
    ] {
        let answer = asked(&shared, &turns, question, &cancel);
        turns.push((question.to_string(), answer));
    }

    Ok(())
}