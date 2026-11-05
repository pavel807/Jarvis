//! Проба action-флоу через полный `llm::ask` (роутер → tool → ответ):
//! проверяет, что после веб-поиска модель отвечает по сниппетам, а не выдумывает.
//!
//! Запуск: cargo run --release -p jarvis-core --example action_probe "вопрос"

use anyhow::Result;
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::{self, SharedLlmState};
use jarvis_core::tts::AudioMessage;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};

fn main() -> Result<()> {
    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "расскажи про игру руми куб".to_string());

    let settings = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())?;
    settings.apply();

    let (events_tx, events_rx) = channel::<JarvisEvent>();
    let (audio_tx, audio_rx) = channel::<AudioMessage>();

    std::thread::spawn(move || {
        for ev in events_rx {
            match ev {
                JarvisEvent::LlmChunk { text } => print!("{text}"),
                JarvisEvent::LlmDone { text } => println!("\n✅ ГОТОВО: {text}"),
                JarvisEvent::ActionStarted { tool, summary } => {
                    println!("▶️ {tool}: {summary}")
                }
                JarvisEvent::ActionResult { ok, summary } => {
                    println!("🔧 [{ok}] {summary}")
                }
                JarvisEvent::Error { message } => eprintln!("❗ {message}"),
                JarvisEvent::LlmStatus { status } => println!("🧠 LLM: {status}"),
                JarvisEvent::Log { message } => println!("ℹ️  {message}"),
                _ => {}
            }
        }
    });
    std::thread::spawn(move || for _ in audio_rx {});

    let shared = Arc::new(SharedLlmState::new());
    llm::spawn_llm_loader(&shared, &events_tx);
    let history = Arc::new(Mutex::new(Vec::new()));
    let cancel = Arc::new(AtomicBool::new(false));

    // Зеркало llm_ask: свежие данные и рекомендации — сразу веб-флоу.
    let explicit_web = jarvis_core::router::explicit_web_request(&question);
    let fresh_info = jarvis_core::web::looks_like_fresh_info(&question)
        && !jarvis_core::router::calendar_question(&question);
    let mut system = llm::CHAT_SYSTEM_PROMPT;
    let mut prompt = question;
    if explicit_web || fresh_info {
        let query = jarvis_core::router::web_query_from(&prompt).unwrap_or_else(|| prompt.clone());
        println!("🌐 веб-флоу: {query}");
        if let Ok(hits) = jarvis_core::web::search(&query, 5)
            && let Some(web_prompt) = jarvis_core::web::build_summary_prompt(&query, &hits)
        {
            system = llm::WEB_SYSTEM_PROMPT;
            prompt = web_prompt;
        }
    }
    llm::ask(shared, system, prompt, history, audio_tx, events_tx, cancel);

    // Вопрос с поиском занимает до ~20 c; ждём с запасом и выходим.
    std::thread::sleep(std::time::Duration::from_secs(60));
    Ok(())
}
