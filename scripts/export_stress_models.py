#!/usr/bin/env python3
"""Конвертирует ONNX-модели ударителя в safetensors для candle-реализаций.

ONNX-модели ruaccent динамически квантованы (int8-веса + динамическая
квантизация активаций). Здесь веса де-квантуются в f32 и раскладываются
в PyTorch-порядке ([out, in] для линейных слоёв), чтобы их читал
`candle_nn::linear`.

Вход:  models/stress-ru/accent.onnx, models/stress-ru/omograph/model.onnx
Выход: models/stress-ru/accent.safetensors,
       models/stress-ru/omograph/model.safetensors

Запуск: uv run scripts/export_stress_models.py
Зависимости берутся из заголовка ниже: uv сам поставит их в изолированное
окружение (torch — CPU-сборка, втрое меньше скачивание). Без uv можно
запустить python3, если все пакеты уже стоят в системе.

Куда писать: JARVIS_MODELS_DIR, иначе models/ репозитория — так же, как
определяет папку моделей само приложение (paths.rs).
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnx", "torch", "safetensors"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# ///

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("JARVIS_MODELS_DIR", ROOT / "models"))

# Откуда берём исходные ONNX (в репозитории они не хранятся).
SOURCES = {
    "accent.onnx": "https://huggingface.co/TigreGotico/stressonnx-models/resolve/main/ru_ruaccent/nn_accent/model.onnx",
    "omograph/model.onnx": "https://huggingface.co/ruaccent/accentuator/resolve/main/nn/nn_omograph/tiny2/model.onnx",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 {url} -> {dest.relative_to(ROOT)}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def export(onnx_path: Path, out_path: Path) -> int:
    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: i for i in graph.initializer}

    def array(name):
        return numpy_helper.to_array(inits[name]).astype(np.float32)

    def dequantize(name):
        q = array(name)
        scale = float(array(name.replace("_quantized", "_scale")).reshape(-1)[0])
        zp = array(name.replace("_quantized", "_zero_point"))
        return (q - zp) * scale

    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    tensors = {}
    for name in inits:
        if name.endswith("_scale") or name.endswith("_zero_point"):
            continue
        quantized = name.endswith("_quantized")
        value = dequantize(name) if quantized else array(name)
        key = None
        for node in consumers.get(name, []):
            # Линейные веса: имя берём из пути узла (MatMul/Gemm), f32 — из имени тензора.
            if node.op_type == "MatMulInteger" and node.input[1] == name:
                parts = [p for p in node.name.split("/") if p and "MatMul" not in p and "Gemm" not in p]
                key = ".".join(parts) + ".weight"
        if key is None:
            key = name[: -len("_quantized")] if quantized else name
        else:
            value = value.T  # [in, out] -> [out, in]
        tensors[key] = torch.from_numpy(np.ascontiguousarray(value))

    save_file(tensors, str(out_path), metadata={"format": "pt"})
    return len(tensors)


def main() -> int:
    jobs = [
        (MODELS / "stress-ru/accent.onnx", MODELS / "stress-ru/accent.safetensors"),
        (
            MODELS / "stress-ru/omograph/model.onnx",
            MODELS / "stress-ru/omograph/model.safetensors",
        ),
    ]
    for src, dst in jobs:
        if not src.exists():
            url = SOURCES.get(src.relative_to(ROOT / "models/stress-ru").as_posix())
            if url is None:
                print(f"⚠️  нет {src} — пропускаю")
                continue
            download(url, src)
        count = export(src, dst)
        try:
            shown = dst.relative_to(MODELS)
        except ValueError:
            shown = dst
        print(f"✅ {shown}: {count} тензоров")
    return 0


if __name__ == "__main__":
    sys.exit(main())
