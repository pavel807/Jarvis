//! Замер LLM: префилл и время до первого куска речи (то, что слышит
//! пользователь). Сравнивает короткий чат-промпт и веб-сводку из 5/3
//! результатов — видно, сколько стоит контекст поиска.
//!
//! Запуск: cargo run --release --example llm_bench -p jarvis-core
//! Бэкенд и модель берутся из settings.toml (`llm_backend`, `llm_model`);
//! первые два аргумента переопределяют их для сравнения:
//! `... --example llm_bench -- uzu qwen3.5-0.8b`.

use anyhow::Result;
use jarvis_core::llm::{self, ChatTurn, SharedLlmState, config::GenerationParams};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn web_prompt(hits: usize, snippet_chars: usize) -> String {
    let mut context = String::new();
    for i in 0..hits {
        let snippet = "В Москве сегодня облачно, температура около минус трёх градусов, ветер пять метров в секунду, к вечеру похолодает до минус семи."
            .chars()
            .take(snippet_chars)
            .collect::<String>();
        context.push_str(&format!(
            "[{}] Погода в Москве на сегодня — прогноз\n{}\n\n",
            i + 1,
            snippet
        ));
    }
    format!(
        "Результаты поиска в интернете по запросу «какая сегодня погода в москве»:\n\n{context}На основе этих результатов ответь на вопрос пользователя. Кратко: 1-3 предложения, только суть, на русском языке."
    )
}

fn run(engine: &mut llm::LlmEngine, name: &str, prompt: &str) -> Result<()> {
    let started = Instant::now();
    let mut first_chunk: Option<std::time::Duration> = None;
    let mut chunks = 0usize;
    let cancel = AtomicBool::new(false);
    let text = engine.generate_streaming_with_params(
        llm::WEB_SYSTEM_PROMPT,
        &[] as &[ChatTurn],
        prompt,
        Some(GenerationParams {
            temperature: 0.2,
            top_p: 0.9,
            max_tokens: 120,
            ..Default::default()
        }),
        &cancel,
        |chunk| {
            chunks += 1;
            if first_chunk.is_none() {
                first_chunk = Some(started.elapsed());
                println!(
                    "   первый кусок TTS: {:?} ({} симв.)",
                    started.elapsed(),
                    chunk.chars().count()
                );
            }
        },
        |_| {},
    )?;
    println!(
        "  {name}: всего {:?}, кусков TTS {chunks}, ответ {} симв.: {}",
        started.elapsed(),
        text.chars().count(),
        text.chars().take(90).collect::<String>()
    );
    Ok(())
}

fn main() -> Result<()> {
    // Настройки из settings.toml, затем необязательные переопределения
    // аргументами: `llm_bench [backend] [model]`.
    let settings = jarvis_core::settings::Settings::load(&jarvis_core::paths::settings_file())?;
    settings.apply();
    let mut args = std::env::args().skip(1);
    if let Some(backend) = args.next() {
        llm::config::set_backend(&backend);
    }
    if let Some(model) = args.next() {
        llm::config::set_model_name(&model);
    }
    println!(
        "ℹ️  llm_backend = {}, llm_model = {}",
        llm::config::backend(),
        llm::config::model_name()
    );

    let shared: llm::SharedLlm = Arc::new(SharedLlmState::new());
    let (tx, _rx) = std::sync::mpsc::channel();
    llm::spawn_llm_loader(&shared, &tx);
    println!("⏳ жду модель...");
    let t = Instant::now();
    let mut guard = loop {
        let guard = shared.engine.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            break guard;
        }
        if shared.failed.load(std::sync::atomic::Ordering::SeqCst) {
            anyhow::bail!("модель не загрузилась");
        }
        drop(guard);
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    println!("модель готова за {:?}", t.elapsed());
    let engine = guard.as_mut().expect("модель");

    // Прогревающий прогон (прогрев Metal/кэшей)
    run(engine, "прогрев", "Скажи одно слово: привет.")?;

    run(engine, "чат (короткий промпт)", "Как дела?")?;
    run(engine, "веб 5×220 симв.", &web_prompt(5, 220))?;
    run(engine, "веб 3×140 симв.", &web_prompt(3, 140))?;
    run(engine, "веб 3×90 симв.", &web_prompt(3, 90))?;
    Ok(())
}