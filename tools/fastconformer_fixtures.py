#!/usr/bin/env python3
"""Эталонные фикстуры FastConformer для верификации candle-реализации.

Готовит для каждого wav-файла:
1. mel-фичи (реализация препроцессора onnx-asr istupakov, NumPy);
2. выход ONNX-энкодера (encoder-model.onnx, onnxruntime);
3. CTC-логиты (NumPy из весов safetensors — та же голова, что в Rust);
4. жадный RNNT-транскрипт (encoder + decoder_joint-model.onnx);
5. жадный CTC-транскрипт (по логитам из п. 3).

Выход: build/fastconformer/fixtures/<name>.json — читает пример `fc_verify`.

Запуск (torch не нужен):
  uv run --with numpy --with onnx --with onnxruntime --with safetensors \
      python tools/fastconformer_fixtures.py
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as rt
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "build/fastconformer/onnx-ref"
WAVS = ROOT / "build/fastconformer/wavs"
OUT = ROOT / "build/fastconformer/fixtures"
ST = ROOT / "models/fastconformer"

SAMPLE_RATE = 16_000
N_FFT = 512
WIN_LENGTH = 400
HOP = 160
PREEMPH = 0.97
LOG_GUARD = float(2**-24)
BLANK = 1024


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, path
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm="slaney", mel_scale="slaney"):
    # Из onnx-asr/preprocessors/fbanks.py (slaney/slaney — как в эталоне).
    if f_max <= 0.0:
        f_max += sample_rate / 2

    def hz_to_mel(freq, scale):
        if scale == "htk":
            return 2595 * np.log10(1.0 + freq / 700.0)
        return np.where(
            freq < 1000,
            3 * freq / 200.0,
            15 + 27 * np.log(freq / 1000.0 + np.finfo(np.float32).eps) / np.log(6.4),
        )

    def mel_to_hz(mels):
        if mel_scale == "htk":
            return 700 * (np.pow(10.0, mels / 2595.0) - 1.0)
        return np.where(mels < 15, 200 * mels / 3.0, 1000 * np.pow(6.4, ((mels - 15) / 27.0)))

    all_freqs = np.linspace(0, sample_rate // 2, n_freqs)
    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    mel = all_freqs
    m_pts = mel_to_hz(m_pts)
    up = (mel[:, None] - m_pts[:-2]) / (m_pts[1:-1] - m_pts[:-2])
    down = (m_pts[2:] - mel[:, None]) / (m_pts[2:] - m_pts[1:-1])
    fb = np.maximum(0.0, np.minimum(up, down))
    if norm == "slaney":
        fb *= 2.0 / (m_pts[2:] - m_pts[:-2])
    return fb.astype(np.float32)


def nemo_preprocess(waveform: np.ndarray) -> tuple[np.ndarray, int]:
    """Mel-фичи как в препроцессоре onnx-asr для NeMo-моделей.

    Возвращает [80, T] и число валидных кадров (wave_len / hop).
    """
    wave_len = waveform.shape[0]
    # preemph: первый семпл не меняется
    w = np.concatenate([waveform[:1], waveform[1:] - PREEMPH * waveform[:-1]])
    # zero-pad по 256 с обеих сторон
    w = np.pad(w, (N_FFT // 2, N_FFT // 2))
    frames = wave_len // HOP + 1
    # окно Hann 400 -> в центр 512
    window = np.pad(np.hanning(WIN_LENGTH), (N_FFT // 2 - WIN_LENGTH // 2, N_FFT // 2 - WIN_LENGTH // 2))
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    framed = w[idx] * window  # [frames, 512]
    spec = np.abs(np.fft.rfft(framed, n=N_FFT, axis=-1)) ** 2  # [frames, 257]
    fb = melscale_fbanks(N_FFT // 2 + 1, 0, SAMPLE_RATE // 2, 80, SAMPLE_RATE)
    mel = spec @ fb
    log_mel = np.log(mel + LOG_GUARD).T  # [80, frames]

    valid = wave_len // HOP
    # per-feature нормализация по валидным кадрам
    x = log_mel[:, :valid]
    mean = x.sum(axis=-1, keepdims=True) / valid
    var = ((x - mean) ** 2).sum(axis=-1, keepdims=True) / (valid - 1)
    log_mel[:, :valid] = (x - mean) / (np.sqrt(var) + 1e-5)
    log_mel[:, valid:] = 0.0
    return log_mel.astype(np.float32), valid


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_file(str(ST / "model.safetensors"))

    encoder = rt.InferenceSession(str(REF / "encoder-model.onnx"))
    decoder_joint = rt.InferenceSession(str(REF / "decoder_joint-model.onnx"))

    # CTC-голова из safetensors: weight [1025,512,1] -> [1025,512]
    ctc_w = st["ctc_decoder.decoder_layers.0.weight"].squeeze(-1)
    ctc_b = st["ctc_decoder.decoder_layers.0.bias"]

    # Словарь: istupakov vocab.txt "piece id"
    vocab = {}
    for line in (REF / "vocab.txt").read_text(encoding="utf-8").splitlines():
        piece, idx = line.rsplit(" ", 1)
        vocab[int(idx)] = piece.replace("\u2581", " ")

    def ids_to_text(ids):
        import re

        text = "".join(vocab[i] for i in ids)
        return re.sub(r"\A\s|\s\B|(\s)\b", lambda x: " " if x.group(1) else "", text)

    state = None
    wavs = sorted(WAVS.glob("*.wav"))
    if not wavs:
        print("нет wav-файлов в", WAVS)
        return 1
    for wav in wavs:
        audio = load_wav(wav)
        mel, valid = nemo_preprocess(audio)

        enc_out, enc_len = encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": mel[None], "length": np.array([valid], dtype=np.int64)},
        )
        enc = enc_out[0].transpose(1, 0)  # [T', 512]
        t_valid = int(enc_len[0])

        # CTC-логиты: enc @ W^T + b
        ctc_logits = enc @ ctc_w.T + ctc_b  # [T', 1025]
        ctc_ids = []
        prev = -1
        for row in ctc_logits:
            best = int(np.argmax(row))
            if best != BLANK and best != prev:
                ctc_ids.append(best)
            prev = best

        # RNNT greedy: encoder_outputs — сырой выход энкодера [1, 512, T].
        tokens = []
        h = np.zeros((1, 1, 640), np.float32)
        c = np.zeros((1, 1, 640), np.float32)
        prev_token = BLANK

        def joint_call(frame3, prev_tok, h_state, c_state):
            return decoder_joint.run(
                ["outputs", "output_states_1", "output_states_2"],
                {
                    "encoder_outputs": frame3,
                    "targets": np.array([[prev_tok]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32),
                    "input_states_1": h_state,
                    "input_states_2": c_state,
                },
            )

        for t in range(t_valid):
            frame = enc_out[:, :, t : t + 1]  # [1, 512, 1]
            # Семантика onnx-asr: каждый вызов встраивает ТЕКУЩИЙ контекст
            # (prev_token) из текущего состояния. Состояние фиксируется
            # только при эмиссии (в него вошло встраивание старого контекста);
            # при blank — откатывается, чтобы тот же токен не вошёл в LSTM
            # дважды через границу кадров.
            outs, h_next, c_next = joint_call(frame, prev_token, h, c)
            best = int(np.argmax(outs.reshape(-1)))
            emitted = 0
            while best != BLANK and emitted < 10:
                tokens.append(best)
                prev_token = best
                h, c = h_next, c_next
                outs, h_next, c_next = joint_call(frame, prev_token, h, c)
                best = int(np.argmax(outs.reshape(-1)))
                emitted += 1

        fixture = {
            "wav": wav.name,
            "audio": audio.tolist(),
            "mel": mel.tolist(),
            "mel_valid": valid,
            "enc_out": enc[None].tolist(),
            "enc_valid": t_valid,
            "ctc_ids": ctc_ids,
            "ctc_text": ids_to_text(ctc_ids),
            "rnnt_ids": tokens,
            "rnnt_text": ids_to_text(tokens),
        }
        out_path = OUT / f"{wav.stem}.json"
        out_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
        print(f"✅ {wav.name}: enc_len={t_valid}, RNNT: «{fixture['rnnt_text']}»")
        print(f"                     CTC:  «{fixture['ctc_text']}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
