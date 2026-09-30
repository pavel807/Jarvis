//! Audio utilities for Qwen3-TTS: WAV I/O and the [`AudioBuffer`] container.

mod io;

pub use io::{load_wav, save_wav, AudioBuffer};

/// Standard sample rate used by Qwen3-TTS
pub const SAMPLE_RATE: u32 = 24000;
