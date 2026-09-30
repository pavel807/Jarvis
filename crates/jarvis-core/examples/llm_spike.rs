//! Спайк: проверка Qwen/Qwen3-1.7B через candle (Metal) на разных языках.
//!
//! Запуск: cargo run --release --example llm_spike -p jarvis-core

use jarvis_core::llm::{self, SharedLlmState};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("== Спайк: {} ==", llm::config::HF_MODEL_ID);

    let shared: llm::SharedLlm = Arc::new(SharedLlmState::new());
    let (events_tx, events_rx) = channel();

    std::thread::spawn(move || {
        for ev in events_rx {
            println!("[событие] {ev:?}");
        }
    });

    // Низкая температура — чтобы отличать ошибку реализации от стохастики.
    llm::config::update_generation_params(llm::config::GenerationParams {
        temperature: 0.3,
        ..Default::default()
    });

    llm::spawn_llm_loader(&shared, &events_tx);
    let cancel = Arc::new(AtomicBool::new(false));

    for (lang, question) in [
        ("EN", "What is the capital of France? Answer briefly."),
        ("ZH", "中国的首都是哪里？简单回答。"),
        ("RU", "Какая столица России? Ответь кратко."),
    ] {
        let started = Instant::now();
        match llm::generate_blocking(&shared, llm::SYSTEM_PROMPT, &[], question, &cancel) {
            Ok(answer) => {
                println!("\n=== {lang} ({:.1?}) ===", started.elapsed());
                println!("{}", answer.trim());
            }
            Err(e) => println!("\n=== {lang} ОШИБКА: {e} ==="),
        }
    }

    Ok(())
}
