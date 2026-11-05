#!/usr/bin/env python3
"""Генерация сэмпла обученным голосом Джарвиса (Style-Bert-VITS2).

Запуск: /tmp/sbv2/bin/python scripts/voice-train/gen_sample.py [файл_модели.safetensors]
Без аргумента берётся последний safetensors из model_assets/jarvis_ru/.
Текст — транслитерация (та же схема, что в prepare_dataset.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, "/tmp/sbvits2")

import numpy as np
import soundfile as sf
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

ASSETS = Path("/tmp/sbvits2/model_assets/jarvis_ru")
OUT = Path(__file__).resolve().parents[2] / "resources" / "voice_jarvis_ru" / "sample.wav"

SAMPLE_TEXT = (
    "Da, ser. Vse sistemy rabotayut v shtatnom rezhime. "
    "Zaryad batarei — dyevyanosto vosyem protsentov. "
    "Vsegda k vashim uslugam, ser."
)


def main() -> None:
    if len(sys.argv) > 1:
        model_file = Path(sys.argv[1])
    else:
        candidates = sorted(ASSETS.glob("G_*.safetensors"))
        if not candidates:
            sys.exit(f"нет safetensors в {ASSETS}")
        model_file = candidates[-1]

    model = TTSModel(
        model_path=model_file,
        config_path=ASSETS / "config.json",
        style_vec_path=ASSETS / "style_vectors.npy",
        device="cpu",
    )
    try:
        sr, audio = model.infer(SAMPLE_TEXT, language=Languages.EN, style="Neutral")
    except Exception:
        # Имя стиля зависит от style_vectors — fallback на дефолтный стиль.
        sr, audio = model.infer(SAMPLE_TEXT, language=Languages.EN)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUT, np.asarray(audio), sr)
    print(f"сэмпл: {OUT}")


if __name__ == "__main__":
    main()
