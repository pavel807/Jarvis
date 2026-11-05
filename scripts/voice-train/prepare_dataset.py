#!/usr/bin/env python3
"""Собирает датасет Style-Bert-VITS2 из корпуса голоса Джарвиса.

Вход:  resources/voice_jarvis_ru/transcripts.csv (см. transcribe_corpus.py)
Выход: /tmp/sbvits2/Data/jarvis_ru/
         wavs/NNNN.wav  — 44.1 кГц mono (формат модели)
         esd.list       — строки "wavs/NNNN.wav|jarvis_ru|EN|текст"

Русский текст транслитерируется в латиницу: у базовой модели в алфавите
нет кириллицы, и консистентная транслитерация — стандартный способ
обучать такую модель русскому (модель выучит маппинг при файнтюне).
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "resources" / "voice_jarvis_ru"
SBV2 = Path("/tmp/sbvits2")
DATASET = SBV2 / "Data" / "jarvis_ru"
WAVS = DATASET / "wavs"

MIN_DURATION_SEC = 0.7

RU2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def translit(text: str) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        if low in RU2LAT:
            t = RU2LAT[low]
            out.append(t.capitalize() if ch.isupper() else t)
        else:
            out.append(ch)
    return "".join(out)


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ar", "44100", "-ac", "1",
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def main() -> None:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    WAVS.mkdir(parents=True)

    rows = list(csv.DictReader((CORPUS / "transcripts.csv").open(encoding="utf-8")))
    kept, skipped = 0, 0
    with (DATASET / "esd.list").open("w", encoding="utf-8") as esd:
        for i, r in enumerate(rows, 1):
            text = (r["text"] or "").strip()
            dur = float(r["duration_sec"] or 0)
            src = CORPUS / r["file"]
            if not text or text.startswith("<ERROR") or dur < MIN_DURATION_SEC or not src.exists():
                skipped += 1
                continue
            out = WAVS / f"{i:04d}.wav"
            to_wav(src, out)
            esd.write(f"wavs/{out.name}|jarvis_ru|EN|{translit(text)}\n")
            kept += 1

    total_sec = sum(
        float(r["duration_sec"] or 0)
        for r, _ in zip(rows, range(kept))
    )
    print(f"готово: {kept} записей (~{total_sec:.0f} с речи), пропущено {skipped}")
    print(f"датасет: {DATASET}")


if __name__ == "__main__":
    sys.exit(main())
