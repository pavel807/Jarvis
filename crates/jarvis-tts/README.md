# jarvis-tts

Вендоренный движок TTS для Jarvis: [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
на чистом Rust поверх [candle](https://github.com/huggingface/candle) (Metal на Apple Silicon), без Python/ONNX.

Форк [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) (MIT), урезанный до сценария
ассистента: только preset-голоса CustomVoice (9 голосов, русский поддерживается), стриминговая
генерация чанками. Вырезаны клонирование голоса, VoiceDesign, Base-модель, CUDA, hub/CLI.

Используется только крейтом `jarvis-core` (`src/tts.rs`): загрузка модели, прогрев,
стриминговый синтез в аудиоплеер. Вся интеграционная логика — там.
