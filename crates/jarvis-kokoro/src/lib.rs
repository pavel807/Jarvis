//! # jarvis-kokoro — kokoro-ru на чистом Rust
//!
//! Русский TTS на базе [zaakirio/kokoro-ru] (82M, StyleTTS2-lite): ONNX Runtime
//! (CPU, RTF ~0.1) + espeak-ng, вкомпилированный через `espeak-rs`, без
//! внешних бинарников и docker. Подход повторяет [lucasjinreal/Kokoros],
//! русские G2P-правила портированы из `ru_g2p.py` исходного репозитория.
//!
//! [zaakirio/kokoro-ru]: https://huggingface.co/zaakirio/kokoro-ru
//! [lucasjinreal/Kokoros]: https://github.com/lucasjinreal/Kokoros
//!
//! ```no_run
//! use jarvis_kokoro::KokoroEngine;
//! use std::path::Path;
//!
//! # fn main() -> anyhow::Result<()> {
//! let engine = KokoroEngine::new(
//!     Path::new("models/kokoro-ru/onnx/model_dima.onnx"),
//!     Path::new("models/kokoro-ru/voices/dima.bin"),
//! )?;
//! let audio = engine.synthesize("Привет! Как дела?", 1.0)?;
//! # assert!(!audio.is_empty());
//! # Ok(())
//! # }
//! ```

pub mod engine;
pub mod g2p;
pub mod vocab;

pub use engine::{KokoroEngine, SAMPLE_RATE};
