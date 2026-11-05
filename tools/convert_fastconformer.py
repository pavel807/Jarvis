#!/usr/bin/env python3
"""Конвертер NVIDIA FastConformer-Hybrid Large RU (.nemo) -> safetensors для candle.

Оригинальный чекпоинт: nvidia/stt_ru_fastconformer_hybrid_large_pc (CC-BY-4.0).
Файл .nemo — это tar-архив с model_config.yaml, model_weights.ckpt (PyTorch
state_dict) и файлами токенизатора. Torch нужен только для одноразовой
подготовки чекпоинта; в рантайме Jarvis его нет — только Rust + candle.

Делает:
1. Раскладывает model_config.yaml -> models/fastconformer/config.json
   (весь конфиг целиком: приоритет всегда у конфигурации чекпоинта).
2. Раскладывает веса -> models/fastconformer/model.safetensors.
   Имена тензоров сохраняются КАК В NeMo (без переименования): маппинг
   на слои Rust-реализации делает weights.rs, чтобы отчёт о соответствии
   не расходился с кодом.
3. Копирует файлы токенизатора (tokenizer.json/tokenizer.vocab) как есть.
4. Печатает отчёт: число тензоров, суммарные параметры, список слоёв
   с формами (для ревизии отсутствующих/лишних весов).

Запуск:
  uv run --with torch,safetensors,pyyaml python tools/convert_fastconformer.py

Вход (при отсутствии скачивается):
  build/fastconformer/nemo/stt_ru_fastconformer_hybrid_large_pc.nemo
Выход:
  models/fastconformer/{config.json,model.safetensors,tokenizer.json,tokenizer.vocab}
"""

import io
import json
import sys
import tarfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
NEMO_URL = (
    "https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc/"
    "resolve/main/stt_ru_fastconformer_hybrid_large_pc.nemo"
)
NEMO_PATH = ROOT / "build/fastconformer/nemo/stt_ru_fastconformer_hybrid_large_pc.nemo"
OUT_DIR = ROOT / "models/fastconformer"


def download_if_needed() -> None:
    if NEMO_PATH.exists() and NEMO_PATH.stat().st_size > 100_000_000:
        return
    import urllib.request

    NEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 {NEMO_URL} -> {NEMO_PATH.relative_to(ROOT)}")
    with urllib.request.urlopen(NEMO_URL) as response, open(NEMO_PATH, "wb") as out:
        while chunk := response.read(1 << 21):
            out.write(chunk)


def main() -> int:
    download_if_needed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config_yaml: dict | None = None
    tokenizer_files: dict[str, bytes] = {}
    weights_bytes: bytes | None = None

    with tarfile.open(NEMO_PATH) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name  # плоские имена внутри архива
            if name == "model_config.yaml":
                config_yaml = yaml.safe_load(tar.extractfile(member))
            elif name == "model_weights.ckpt":
                weights_bytes = tar.extractfile(member).read()
            elif name.startswith("tokenizer"):
                tokenizer_files[name] = tar.extractfile(member).read()

    if config_yaml is None or weights_bytes is None:
        print(f"❌ В .nemo не найден model_config.yaml/model_weights.ckpt; содержимое: {list(tokenizer_files)}")
        return 1

    # --- Конфигурация: целиком, без усечения (источник истины для Rust).
    (OUT_DIR / "config.json").write_text(
        json.dumps(config_yaml, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Токенизатор: копируем как есть.
    for name, data in tokenizer_files.items():
        (OUT_DIR / name).write_bytes(data)
        print(f"🔑 tokenizer: {name} ({len(data)} байт)")

    # --- Веса: torch.load (pickle из архива), сохраняем под исходными именами.
    state = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    tensors: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        name = key[6:] if key.startswith("_orig_mod.") else key
        if not isinstance(value, torch.Tensor):
            skipped.append(key)
            continue
        if name in tensors:
            print(f"⚠️  дубликат имени после нормализации: {name}")
        t = value.detach().float().contiguous()
        if torch.isnan(t).any():
            print(f"❌ NaN в тензоре {name}")
            return 1
        tensors[name] = t

    safetensors_path = OUT_DIR / "model.safetensors"
    save_file(tensors, str(safetensors_path), metadata={"format": "pt"})
    size_mb = safetensors_path.stat().st_size / 1e6
    total_params = sum(t.numel() for t in tensors.values())
    print(f"✅ {safetensors_path.relative_to(ROOT)}: {len(tensors)} тензоров, "
          f"{total_params/1e6:.1f}M параметров, {size_mb:.0f} MB")
    if skipped:
        print(f"⚠️  не-тензоры пропущены: {skipped}")

    # --- Отчёт о соответствии имён и размеров.
    report_path = ROOT / "build/fastconformer/weights_report.json"
    report = {name: list(t.shape) for name, t in sorted(tensors.items())}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"📄 отчёт форм: {report_path.relative_to(ROOT)}")

    # Краткая сводка по группам слоёв (для ревизии глазами).
    groups: dict[str, int] = {}
    for name in tensors:
        group = name.split(".")[0] + ("." + name.split(".")[1] if name.count(".") else "")
        groups[group] = groups.get(group, 0) + 1
    for group, count in sorted(groups.items()):
        print(f"   {group}: {count} тензоров")

    # Ключевые значения конфига — сразу в консоль, чтобы сверить ожидания.
    preproc = config_yaml.get("preprocessor", {})
    print("\n— preprocessor:", json.dumps(
        {k: preproc.get(k) for k in (
            "sample_rate", "features", "normalize", "window", "window_size",
            "window_stride", "sample_rate_norm", "log", "log_zero_guard_type",
            "log_zero_guard_value", "dither", "preemph", "n_fft", "pad_to",
            "frame_splicing", "freq_masks", "time_masks")}, ensure_ascii=False))
    enc = config_yaml.get("encoder", {})
    print("— encoder:", json.dumps(
        {k: enc.get(k) for k in (
            "_target_", "d_model", "n_layers", "n_heads", "feat_in", "feat_out",
            "subsampling", "subsampling_factor", "ff_expansion_factor",
            "conv_kernel_size", "pos_emb_max_len", "self_attention_model",
            "att_context_size")}, ensure_ascii=False))
    print("— decoder (CTC):", json.dumps(config_yaml.get("decoder", {}), ensure_ascii=False))
    print("— decoder (RNNT):", json.dumps(config_yaml.get("decoder2", config_yaml.get("rnnt_decoder", {})), ensure_ascii=False))
    print("— tokenizer:", json.dumps(config_yaml.get("tokenizer", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
