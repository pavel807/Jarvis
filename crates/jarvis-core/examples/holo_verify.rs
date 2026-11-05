//! Live-проверка Holo-3.1-4B: карта весов → загрузка гибридного декодера на
//! Metal → русский и английский чат. Run:
//!   JARVIS_HOME=<repo> cargo run --release -p jarvis-core --example holo_verify

use jarvis_core::holo::{
    HybridConfig, HybridLayerType, TextEngine35, expected_weights_hybrid, model_dir,
    text_engine::ChatMessage, verify_checkpoint_files_hybrid, weight_files,
};
use jarvis_core::holo::runtime::SamplingParams;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = model_dir();
    println!("Каталог модели: {}", dir.display());
    let files = weight_files(&dir)?;
    for f in &files {
        println!("Шард: {} ({} МБ)", f.display(), std::fs::metadata(f)?.len() / 1_000_000);
    }

    let cfg = HybridConfig::load(&dir)?;
    println!(
        "config OK: qwen3_5, {} сл. ({} delta-net + {} attention), hidden {}, vocab {}, \
         DeltaNet {}K/{}V, head_dim {}, rotary {}, MRoPE {:?}, tie={}",
        cfg.text.num_hidden_layers,
        cfg.text.layer_types.iter().filter(|t| **t == HybridLayerType::LinearAttention).count(),
        cfg.text.layer_types.iter().filter(|t| **t == HybridLayerType::FullAttention).count(),
        cfg.text.hidden_size,
        cfg.text.vocab_size,
        cfg.text.linear_num_key_heads,
        cfg.text.linear_num_value_heads,
        cfg.text.head_dim,
        cfg.rotary_dim(),
        cfg.text.mrope_sections,
        cfg.tie_word_embeddings,
    );

    let report = verify_checkpoint_files_hybrid(&cfg, &files)?;
    println!(
        "Проверка весов: тензоров={}, ок={}, пропущено={}, лишних={}, несовпадений={}",
        report.tensors, report.ok(), report.missing.len(), report.unexpected.len(), report.mismatches.len()
    );
    let _ = expected_weights_hybrid;
    if !report.ok() {
        anyhow::bail!("Чекпоинт не соответствует карте весов");
    }

    println!("\n=== Загрузка на Metal (~9,7 ГБ, mmap) ===");
    let started = Instant::now();
    let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
    let mut engine = TextEngine35::load(&dir, device)?;
    println!("Загружена за {:.1?}", started.elapsed());

    let params = SamplingParams {
        max_new_tokens: 256,
        temperature: 0.0,
        ..SamplingParams::default()
    };

    println!("\n=== Чат, жадный ===");
    let started = Instant::now();
    let ru = engine.reply(
        &[
            ChatMessage::System("Ты Джарвис — голосовой ассистент. Отвечай кратко и по-русски.".into()),
            ChatMessage::User("Привет! Кто ты и что умеешь?".into()),
        ],
        None,
        false,
        &params,
        None,
        |_| true,
    )?;
    println!("Ответ RU ({:.1} с): {ru}", started.elapsed().as_secs_f64());

    let en = engine.reply(
        &[
            ChatMessage::System("You are Jarvis, a helpful assistant.".into()),
            ChatMessage::User("What is the capital of France? Answer in one sentence.".into()),
        ],
        None,
        false,
        &params,
        None,
        |_| true,
    )?;
    println!("Ответ EN: {en}");

    println!("\n✅ Holo-3.1-4B живая проверка пройдена");
    Ok(())
}
