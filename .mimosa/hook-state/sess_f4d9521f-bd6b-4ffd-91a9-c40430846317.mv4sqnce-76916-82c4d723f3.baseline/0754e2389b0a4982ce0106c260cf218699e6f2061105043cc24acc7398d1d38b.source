#!/usr/bin/env python3
"""Скачивает Bonsai-8B-unpacked (safetensors) в models/bonsai-8b.
Работает напрямую через huggingface_hub, минуя сломанный hf CLI.
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "prism-ml/Bonsai-8B-unpacked"
OUTPUT_DIR = Path("models/bonsai-8b")
LEGACY_DIRS = [
    Path("models/farabi-1.7b-agent-rag"),
    Path("models/minicpm5-1b-ru"),
    Path("models/qwen3-1.7b"),
    Path("models/qwen3-1.7b-gguf"),
    Path("models/qwen3-1.7b-xlam"),
]


def remove_legacy() -> None:
    for d in LEGACY_DIRS:
        if not d.exists():
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d)
        print(f"🗑️  Удалена старая модель {d} ({size / 1e9:.2f} ГБ)")


def main() -> int:
    remove_legacy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Загрузка {REPO_ID} → {OUTPUT_DIR}…")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=OUTPUT_DIR,
        ignore_patterns=["*.gguf", "*.onnx", "*.bin"],
    )

    required = ["config.json", "tokenizer.json"]
    missing = [f for f in required if not (OUTPUT_DIR / f).exists()]
    has_weights = any(OUTPUT_DIR.glob("*.safetensors"))
    if missing or not has_weights:
        print(f"❌ missing={missing}, weights={has_weights}", file=sys.stderr)
        return 1

    total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"✅ Готово: {total / 1e9:.2f} ГБ в {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
