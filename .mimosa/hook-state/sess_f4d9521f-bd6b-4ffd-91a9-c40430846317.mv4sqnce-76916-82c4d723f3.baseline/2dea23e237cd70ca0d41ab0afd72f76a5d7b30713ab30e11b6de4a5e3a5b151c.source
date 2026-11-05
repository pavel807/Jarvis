//! Замер скорости поиска: «как было» (новый агент на каждый запрос, без
//! прогрева и кэша) против текущей реализации.
//!
//! Запуск: cargo run --release --example web_bench -p jarvis-core

use anyhow::Result;
use std::time::{Duration, Instant};

const SEARCH_URL: &str = "https://lite.duckduckgo.com/lite/";
const USER_AGENT: &str = "JarvisVoiceAssistant/2.0 (personal assistant)";

/// Старое поведение: свежий агент и таймаут 6 с на каждый запрос.
fn legacy_search(query: &str) -> Result<usize> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(6)))
        .build()
        .new_agent();
    let url = format!("{}?q={}", SEARCH_URL, urlencoding::encode(query));
    let response = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept-Language", "ru-RU,ru;q=0.9,en;q=0.8")
        .call()?;
    let html = response.into_body().read_to_string()?;
    Ok(html.len())
}

fn main() -> Result<()> {
    let query = "какая сегодня погода в москве";

    println!("--- старое поведение (новый агент каждый раз) ---");
    for i in 1..=3 {
        let t = Instant::now();
        let n = legacy_search(query)?;
        println!("  запрос {i}: {:.0?} ({} байт HTML)", t.elapsed(), n);
    }

    println!("\n--- прогрев ---");
    let t = Instant::now();
    jarvis_core::web::warm_up();
    std::thread::sleep(Duration::from_millis(1200));
    println!("  warm_up вызван за {:.0?} (фон)", t.elapsed());

    println!("\n--- новое поведение ---");
    for i in 1..=3 {
        let t = Instant::now();
        let hits = jarvis_core::web::search(query, 5)?;
        println!(
            "  запрос {i}: {:.0?} ({} результатов)",
            t.elapsed(),
            hits.len()
        );
    }

    // Другой запрос — проверяем, что кэш не мешает свежим
    let t = Instant::now();
    let hits = jarvis_core::web::search("курс доллара к рублю", 5)?;
    println!(
        "\n  новый запрос: {:.0?} ({} результатов)",
        t.elapsed(),
        hits.len()
    );
    Ok(())
}
