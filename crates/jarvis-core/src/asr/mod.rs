//! Собственная ASR-реализация на candle: NVIDIA FastConformer-Hybrid Large RU.
//!
//! Модель: `nvidia/stt_ru_fastconformer_hybrid_large_pc` (~115М параметров,
//! гибрид CTC + RNN-T, словарь 1024 SentencePiece Unigram), лицензия CC-BY-4.0.
//! Веса конвертируются из оригинального `.nemo` чекпоинта в safetensors
//! одноразовым скриптом `tools/convert_fastconformer.py`; рантайм Jarvis
//! читает только safetensors + config.json + файлы токенизатора.
//!
//! Состав:
//! - [`frontend`] — mel-фронтенд (preemph, STFT, mel, per-feature нормализация);
//! - [`subsampling`] — глубоко-свёрточное сжатие 8× перед энкодером;
//! - [`conformer`] — 17 Conformer-блоков (FFN+GLU... SiLU, rel-pos attention,
//!   свёрточный модуль, все residual-коэффициенты и порядок норм — из графа
//!   ONNX-эталона istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx);
//! - [`attention`] — MultiHeadAttention с относительной позиционной кодировкой;
//! - [`ctc`] — CTC-голова: greedy и beam search (с опциональным словарём Jarvis);
//! - [`rnnt`] — Prediction Network (LSTM 640) + Joint Network + greedy-декодер;
//! - [`tokenizer`] — SentencePiece Unigram: декодирование id→текст и
//!   Viterbi-кодирование текста→id (для словаря и beam search);
//! - [`model`] — сборка всего пайплайна, выбор декодера, точки входа.

pub mod attention;
pub mod ctc;
pub mod conformer;
pub mod config;
pub mod frontend;
pub mod model;
pub mod rnnt;
pub mod subsampling;
pub mod tokenizer;