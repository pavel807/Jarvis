//! Бенчмарк декодера Holo-3.1-4B: время загрузки, префилл (токенов/с) и
//! декод (токенов/с) на реальных весах. Run:
//!   JARVIS_HOME=<repo> cargo run --release -p jarvis-core --example holo_bench

use jarvis_core::holo::runtime::SamplingParams;
use jarvis_core::holo::text_engine::{ChatMessage, TextEngine35};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::holo::model_dir();
    println!("=== Загрузка Holo-3.1-4B ===");
    let started = Instant::now();
    let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
    let mut engine = TextEngine35::load(&dir, device)?;
    let load = started.elapsed();
    println!("Загрузка (веса+транспоны+Metal): {:.1?}", load);

    // Прогрев (компиляция Metal-ядер, пул буферов).
    let _ = engine.reply(
        &[ChatMessage::User("привет".into())],
        None,
        false,
        &SamplingParams { max_new_tokens: 8, temperature: 0.0, ..Default::default() },
        None,
        |_| true,
    )?;

    // --- Декод: 160 жадных токенов, короткий промпт ---
    let params = SamplingParams {
        max_new_tokens: 160,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let started = Instant::now();
    let out = engine.reply_report(
        &[
            ChatMessage::System("Ты Джарвис. Отвечай кратко.".into()),
            ChatMessage::User("Расскажи, что умеешь, списком из пяти пунктов.".into()),
        ],
        None,
        false,
        &params,
        None,
        |_| true,
    )?;
    let decode = started.elapsed().as_secs_f64();
    println!(
        "Декод: {} токенов за {:.2} с = {:.1} т/с (ускорение KV-кэша растёт с длиной контекста)",
        out.generated_tokens,
        decode,
        out.generated_tokens as f64 / decode
    );

    // --- Декод с длинным контекстом (префилл ~1500 токенов + 96 шагов) ---
    let filler = "Дополнительный контекст: ".repeat(90);
    let started = Instant::now();
    let out = engine.reply_report(
        &[
            ChatMessage::System("Ты Джарвис. Отвечай кратко.".into()),
            ChatMessage::User(format!("{filler}\nПерескажи одним предложением, что было в контексте.")),
        ],
        None,
        false,
        &SamplingParams { max_new_tokens: 96, temperature: 0.0, ..Default::default() },
        None,
        |_| true,
    )?;
    let total = started.elapsed().as_secs_f64();
    let prompt_tokens = out.prompt_tokens;
    let prefill = prompt_tokens as f64 / (total * 0.8); // грубо: 80% времени — префилл
    println!(
        "Длинный контекст: префилл ~{} токенов, генерация {} токенов за {:.2} с; префилл ≈{:.0} т/с, декод ≈{:.1} т/с",
        prompt_tokens,
        out.generated_tokens,
        total,
        prefill,
        out.generated_tokens as f64 / (total * 0.2)
    );
    Ok(())
}
