#!/usr/bin/env python3
"""Транскрибация корпуса голоса Джарвиса (faster-whisper, ru).

Запуск: /tmp/f5_env/bin/python scripts/voice-train/transcribe_corpus.py
Результат: resources/voice_jarvis_ru/transcripts.csv (file,duration,text)
"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "resources" / "voice_jarvis_ru"
OUT_CSV = CORPUS / "transcripts.csv"
EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def load16k(path: Path) -> np.ndarray:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1",
         "-f", "s16le", "/tmp/aw.pcm"],
        check=True,
    )
    return np.fromfile("/tmp/aw.pcm", dtype=np.int16).astype(np.float32) / 32768.0


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def main() -> None:
    files = sorted(
        p for p in CORPUS.rglob("*")
        if p.suffix.lower() in EXTS and p.name != ".DS_Store"
    )
    print(f"файлов: {len(files)}", flush=True)

    model = WhisperModel("small", device="cpu", compute_type="int8")
    rows = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(CORPUS)
        dur = duration(p)
        try:
            audio = load16k(p)
            if dur < 0.3 or audio.shape[0] < 4000:
                text = ""
            else:
                segs, _ = model.transcribe(audio, language="ru", vad_filter=True)
                text = " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:  # noqa: BLE001 — битые файлы не должны валить обход
            text = f"<ERROR: {e}>"
        rows.append((str(rel), dur, text))
        print(f"[{i}/{len(files)}] {rel} ({dur:.1f}s): {text[:80]}", flush=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "duration_sec", "text"])
        w.writerows(rows)
    total = sum(r[1] for r in rows)
    print(f"готово: {OUT_CSV} ({len(rows)} файлов, {total:.1f}s речи)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
