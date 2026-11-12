//! Проверка активационной фразы: Silero синтезирует её разными дикторами,
//! CTC-детектор проверяет, слышит ли модель фразу буквально.
//!
//! Запуск: cargo run -p jarvis-core --example wake_phrase_check -- "Компьютер"

use anyhow::{Context, Result};

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let phrase = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Джарвис".to_string());
    let stt = std::sync::Arc::new(std::sync::Mutex::new(jarvis_core::stt::Stt::load(
        &jarvis_core::paths::models_dir(),
    )?));

    println!("🎙️ Синтезирую «{phrase}» разными дикторами…");
    let report = jarvis_core::wake_phrase::generate_and_check(&stt, &phrase)
        .with_context(|| format!("проверка фразы «{phrase}»"))?;

    println!(
        "✅ Услышано {}/{} вариантов (худший счёт {:.3})",
        report.detected, report.total, report.worst_score
    );
    println!(
        "📐 Рекомендованная длина окна детектора: {} мс",
        report.window_ms
    );
    println!("📂 Записи: {}", report.samples_dir);
    if report.detected < report.total {
        println!(
            "⚠️ Модель различает не все произнесения — проще выбрать другую фразу."
        );
    }
    Ok(())
}
