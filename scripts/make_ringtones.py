#!/usr/bin/env python3
"""Генерирует собственные рингтоны Джарвиса для таймера и будильника.

Пишет моно 48 кГц / 16 бит PCM в resources/audio/. Файлы кладутся в payload
DMG (scripts/build_macos.sh) и играются штатным плеером (AudioMessage::PlayWav).

Запуск: python3 scripts/make_ringtones.py
"""

import math
import os
import struct
import wave

SAMPLE_RATE = 48_000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "audio")


def tone(freq, seconds, volume=0.5, fade=0.03):
    """Синус с плавным входом/выходом, чтобы не было щелчков на границах."""
    n = int(SAMPLE_RATE * seconds)
    fade_n = max(1, int(SAMPLE_RATE * fade))
    out = []
    for i in range(n):
        env = 1.0
        if i < fade_n:
            env = i / fade_n
        elif i > n - fade_n:
            env = (n - i) / fade_n
        # Два обертона — звук «богаче», чем чистый синус.
        s = math.sin(2 * math.pi * freq * i / SAMPLE_RATE)
        s += 0.35 * math.sin(2 * math.pi * freq * 2 * i / SAMPLE_RATE)
        out.append(volume * env * s / 1.35)
    return out


def silence(seconds):
    return [0.0] * int(SAMPLE_RATE * seconds)


def normalize(samples, peak=0.6):
    top = max((abs(s) for s in samples), default=0.0)
    if top < 1e-6:
        return samples
    k = peak / top
    return [s * k for s in samples]


def write_wav(name, samples):
    path = os.path.normpath(os.path.join(OUT_DIR, name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
            for s in samples
        )
        w.writeframes(frames)
    print(f"✅ {path} ({len(samples) / SAMPLE_RATE:.2f} с)")


def make_timer():
    """Мягкий двухнотный чип: A4 → C#5, пауза. Зацикливается без стыка."""
    phrase = tone(440.0, 0.55) + tone(554.37, 0.75) + silence(0.9)
    return normalize(phrase * 1, peak=0.5)


def make_alarm():
    """Настойчивый будильник: три коротких высоких сигнала + пауза."""
    beep = tone(988.0, 0.18, volume=0.6) + silence(0.12)
    phrase = beep * 3 + silence(0.55)
    return normalize(phrase, peak=0.7)


if __name__ == "__main__":
    write_wav("ringtone_timer.wav", make_timer())
    write_wav("ringtone_alarm.wav", make_alarm())
