//! jarvis-core: движок голосового ассистента (wake word, STT, TTS, LLM, команды).

pub mod audio;
pub mod commands;
pub mod engine;
pub mod events;
pub mod kokoro_g2p;
pub mod kokoro_vocab;
pub mod llm;
pub mod paths;
pub mod settings;
pub mod stt;
pub mod text;
pub mod tts;
pub mod tts_kokoro;
pub mod wake;
pub mod wake_train;
pub mod web;
