//! jarvis-core: движок голосового ассистента (wake word, STT, TTS, LLM, команды).

/// Тюнинг рантайма до первой тензорной операции. candle на сотнях мелких
/// операций энкодера/фронтенда теряет до 20× на оверхеде rayon-пула из
/// многих потоков (замер: 12 потоков — 1956 мс на окно 1.2 с, 4 — 58 мс,
/// 1 — 99 мс). Cap задевает только элементные операции: matmul считает
/// через собственный трединг matrixmultiply и не меняется.
pub fn init_tuning() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let threads = cores.min(4).max(1).to_string();
        // RAYON_NUM_THREADS читается rayon'ом при создании глобального пула —
        // до первого par_iter. set_var небезопасен в edition 2024: других
        // записей окружения в процессе нет, гонок нет.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", threads) };
    });
}

pub mod agent;
pub mod applescript;
pub mod asr;
pub mod audio;
pub mod commands;
pub mod engine;
pub mod events;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod model_fetch;
pub mod nn;
pub mod omograph;
pub mod paths;
pub mod platform;
pub mod prompt_budget;
pub mod router;
pub mod schedule;
pub mod settings;
pub mod site_cache;
pub mod stress;
pub mod telegram;
pub mod speaker;
pub mod stt;
pub mod tasks;
pub mod text;
pub mod timers;
pub mod tool_guard;
pub mod tools;
pub mod topic;
pub mod tts;
pub mod tts_normalizer;
pub mod tts_silero;
pub mod auto;
pub mod wake;
pub mod wake_ctc;
pub mod wake_phrase;
pub mod wake_train;
pub mod weather;
pub mod web;
