#!/usr/bin/env python3
"""Ставит иконку приложения в iOS-каталог (gen/apple/Assets.xcassets).

Зачем свой скрипт, а не `tauri icon`: та команда раскладывает по размерам
и мастер со скруглёнными углами и прозрачностью, а iOS такие иконки не
принимает — углы скругляет сама система, прозрачность запрещена (actool
ругается, в магазине App Store не пропускает). Здесь размеры берутся из
Contents.json каталога, а картинка — квадратная и непрозрачная.

Мастер: resources/icons/icon-ios.png (его рисует scripts/make_icon.py).
Результат: app/src-tauri/gen/apple/Assets.xcassets/AppIcon.appiconset/*

Запуск: python3 scripts/ios_icon.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "resources" / "icons" / "icon-ios.png"
CATALOG = ROOT / "app" / "src-tauri" / "gen" / "apple" / "Assets.xcassets" / "AppIcon.appiconset"


def pixels(size: str, scale: str) -> int:
    """Размер в точках × масштаб → пиксели."""
    points = float(size.split("x")[0])
    factor = int(scale.rstrip("x"))
    return round(points * factor)


def main() -> int:
    if not MASTER.exists():
        print(f"❌ Нет мастера иконки: {MASTER}", file=sys.stderr)
        print("   Сгенерируй: python3 scripts/make_icon.py", file=sys.stderr)
        return 1
    if not CATALOG.exists():
        print(f"❌ Нет iOS-каталога: {CATALOG}", file=sys.stderr)
        print("   Сначала проект: cd app && bun tauri ios init", file=sys.stderr)
        return 1
    if not shutil.which("sips"):
        print("❌ Нужен sips (macOS) для изменения размеров иконки", file=sys.stderr)
        return 1

    manifest = json.loads((CATALOG / "Contents.json").read_text())
    written = 0
    for image in manifest.get("images", []):
        name = image.get("filename")
        if not name:
            continue
        px = pixels(image["size"], image["scale"])
        target = CATALOG / name
        subprocess.run(
            ["sips", "-z", str(px), str(px), str(MASTER), "--out", str(target)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        written += 1

    print(f"→ иконка iOS обновлена: {written} файлов в {CATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
