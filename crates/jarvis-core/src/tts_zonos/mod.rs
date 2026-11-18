//! ZONOS3-Mini — голосовая модель Джарвиса: клон голоса и речь.
//!
//! Пайплайн целиком на candle:
//!
//! 1. `speaker` — энкодер MECT-B2 считает 192-мерный эмбеддинг голоса по
//!    опорной записи (или берётся готовый эмбеддинг из ассетов);
//! 2. `tokenizer` — SuperBPE превращает текст в текстовые id;
//! 3. `prompt` — собирает промпт: строки голоса/условий/текста/тишины;
//! 4. `backbone` — авторегрессия по 10 аудиокодбукам (GDN + глобальное
//!    внимание, Full AttnRes, MoE);
//! 5. `codec` — ZAC-декодер превращает коды в 44.1 кГц PCM.
//!
//! `engine` связывает шаги и отдаёт готовый WAV в `tts`.

pub mod backbone;
pub mod codec;
pub mod config;
pub mod engine;
pub mod gguf;
pub mod prompt;
pub mod sampler;
pub mod speaker;
pub mod tokenizer;
pub mod voice;

pub use backbone::Backbone;
pub use codec::ZacDecoder;
pub use engine::{AudioDecoder, Synthesis, ZonosEngine};
pub use config::ZonosConfig;
pub use gguf::{GgufFile, Value};
pub use prompt::{Prompt, PromptSpec, build_prompt};
pub use sampler::{Sampling, Sampler};
pub use speaker::SpeakerEncoder;
pub use tokenizer::TextTokenizer;
pub use voice::{SpeechEngine, ZonosVoice};
