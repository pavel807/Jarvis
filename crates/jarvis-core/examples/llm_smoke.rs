//! Смоук-тест LLM-бэкенда из settings.toml: инициализация и один вопрос-ответ.
//! Модели не качаются сами: если выбранной модели нет на диске, увидишь
//! честную ошибку — скачай её в приложении (Настройки → Модель).
//!
//! Запуск: cargo run --release -p jarvis-core --example llm_smoke

use anyhow::Result;
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::config as llm_config;
use jarvis_core::llm::{self, SharedLlmState};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::time::Instant;

fn main() -> Result<()> {
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
    println!(
        "ℹ️  llm_backend = {}, llm_model = {}",
        llm_config::backend(),
        llm_config::model_name()
    );

    let shared = Arc::new(SharedLlmState::new());
    llm::spawn_llm_loader(&shared, &events_tx);

    let cancel = Arc::new(AtomicBool::new(false));
    // Два хода с настоящей историей: второй проверяет дельта-префилл сессии
    // (общий префикс system+история переиспользуется), третий — что модель
    // по инструкцией промпта выдаёт tool-блок в нашем формате.
    let mut turns: Vec<(String, String)> = Vec::new();
    let asked = |shared: &Arc<SharedLlmState>,
                 system: &'static str,
                 history: &[(String, String)],
                 question: &str,
                 cancel: &Arc<AtomicBool>|
     -> String {
        let started = Instant::now();
        let answer = llm::generate_blocking(shared, system, history, question, cancel)
            .map_err(|e| eprintln!("❗ {e:#}"))
            .unwrap_or_default();
        println!("\n❓ {question}");
        println!("🤖 {answer}");
        println!("⏱  {:.1?}", started.elapsed());
        answer
    };

    let a1 = asked(
        &shared,
        llm::CHAT_SYSTEM_PROMPT,
        &turns,
        "Привет! Кто ты?",
        &cancel,
    );
    turns.push(("Привет! Кто ты?".to_string(), a1.clone()));
    let a2 = asked(
        &shared,
        llm::CHAT_SYSTEM_PROMPT,
        &turns,
        "Что я только что спросил? Ответь одной фразой.",
        &cancel,
    );
    turns.push((
        "Что я только что спросил? Ответь одной фразой.".to_string(),
        a2,
    ));
    let a3 = asked(
        &shared,
        llm::CHAT_SYSTEM_PROMPT,
        &turns,
        "Поставь таймер на 5 минут",
        &cancel,
    );
    println!("🔧 tool-блок в ответе: {}", a3.contains("<tool>"));

    Ok(())
}