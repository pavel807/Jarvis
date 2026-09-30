//! Шина событий движка: движок публикует, UI (Tauri) и CLI подписываются.

use serde::Serialize;
use std::sync::mpsc::Sender;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JarvisEvent {
    /// Смена состояния движка: wake_word | listening | speaking | paused
    StateChanged { state: String },
    WakeWordDetected { name: String, score: f32 },
    Transcript { text: String },
    CommandMatched { name: String },
    CommandNotFound { text: String },
    Reply { text: String },
    /// Потоковый чанк ответа LLM (накопленный текст)
    LlmChunk { text: String },
    LlmDone { text: String },
    /// Статус загрузки LLM: loading | ready | failed
    LlmStatus { status: String },
    /// Прогресс тренировки wake word: 0..=100
    TrainProgress { epoch: usize, total: usize, loss: f64 },
    /// Тренировка завершена, модель готова к применению.
    TrainDone { path: String },
    /// Сэмпл wake word записан.
    RecordSaved { path: String },
    Error { message: String },
    Log { message: String },
}

pub type EventTx = Sender<JarvisEvent>;
