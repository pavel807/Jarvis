#!/usr/bin/env python3
"""Мастер иконки приложения: рисует реактор Mark V и перегенерирует все размеры.

Иконка повторяет геометрию 3D-сцены интерфейса (app/src/lib/reactor/geometry.ts):
РАВНОСТОРОННИЙ треугольник вершиной вниз, углы срезаны на 0.2547 радиуса —
шесть рёбер: три длинных и три скоса. Слои: корпус, внутреннее кольцо,
канал-проводник, ядро. Цвета взяты из scene.ts, чтобы иконка и приложение
выглядели одним устройством.

Запуск:  python3 scripts/make_icon.py
Результат: resources/icons/icon.svg (мастер) и все размеры в app/src-tauri/icons

Никаких SVG-фильтров: свечение набрано широкими полупрозрачными штрихами,
потому что растеризатор (librsvg) трактует фильтры хуже, чем браузеры, а
иконка должна выглядеть одинаково и там и там.
"""

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
# Готовые размеры, которые читает Tauri.
ICONS = APP / "src-tauri" / "icons"
# Векторный мастер — исходник, его правим руками.
MASTER = ROOT / "resources" / "icons" / "icon.svg"
SIZE = 1024
CX, CY = SIZE / 2, SIZE / 2
# Доля радиуса, на которую срезан каждый угол — как CUT/S в геометрии сцены.
CHAMFER = 0.2547

# Палитра из app/src/lib/reactor/scene.ts
CYAN = "#4bcbff"
ENERGY = "#bdf2ff"
DEEP = "#0b4a68"
INK = "#070d12"
TITANIUM = "#7f95a6"
TITANIUM_DARK = "#2b3a46"


def triangle_points(radius: float) -> list[tuple[float, float]]:
    """Вершины равностороннего треугольника вершиной ВНИЗ.

    В сцене (Three.js, ось Y вверх) это углы −90°, 30°, 150°; в SVG ось Y
    обратная, поэтому те же вершины — 90°, 210°, 330°.
    """
    return [
        (CX + radius * math.cos(a), CY + radius * math.sin(a))
        for a in (math.pi / 2, math.pi * 7 / 6, -math.pi / 6)
    ]


def truncated(radius: float) -> list[tuple[float, float]]:
    """Шесть точек контура: вершины, срезанные вдоль смежных сторон."""
    cut = CHAMFER * radius
    verts = triangle_points(radius)
    out: list[tuple[float, float]] = []
    for i in range(3):
        prev, v, nxt = verts[(i + 2) % 3], verts[i], verts[(i + 1) % 3]
        for target in (prev, nxt):
            dx, dy = target[0] - v[0], target[1] - v[1]
            norm = math.hypot(dx, dy)
            out.append((v[0] + dx / norm * cut, v[1] + dy / norm * cut))
    return out


def path(points: list[tuple[float, float]]) -> str:
    body = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"M{body} Z"


def ring(outer: float, inner: float) -> str:
    """Рама: внешний контур и дыра обратного обхода."""
    hole = " ".join(f"{x:.1f},{y:.1f}" for x, y in reversed(truncated(inner)))
    return f"{path(truncated(outer))} M{hole} Z"


def midpoints(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Середины трёх длинных рёбер (2i+1 -> 2i+2) — туда смотрят излучатели."""
    n = len(points)
    return [
        (
            (points[(i * 2 + 1) % n][0] + points[(i * 2 + 2) % n][0]) / 2,
            (points[(i * 2 + 1) % n][1] + points[(i * 2 + 2) % n][1]) / 2,
        )
        for i in range(3)
    ]


def glow_stroke(points: list[tuple[float, float]], color: str, width: float) -> str:
    """Свечение: три вложенных штриха от широкого тусклого к узкому яркому."""
    d = path(points)
    layers = [
        (width * 3.4, 0.10),
        (width * 1.9, 0.22),
        (width, 0.85),
    ]
    return "".join(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w:.1f}" '
        f'stroke-opacity="{o}" stroke-linejoin="round"/>'
        for w, o in layers
    )


def build_svg() -> str:
    r = SIZE * 0.385  # внешний радиус корпуса
    outline = truncated(r)
    chassis = ring(r, r * 0.79)
    inner_ring = ring(r * 0.57, r * 0.44)
    channel = truncated(r * 0.505)
    well = truncated(r * 0.42)
    floor = path(truncated(r * 0.755))
    corners = midpoints(outline)

    # Ядро: гексагон (огранка кристалла) по центру.
    core_r = r * 0.235
    core = [(CX + core_r * math.cos(a), CY + core_r * math.sin(a)) for a in
            [math.radians(d) for d in range(90, 450, 60)]]
    core_d = path(core)

    emitters = ""
    for (mx, my) in corners:
        ang = math.atan2(my - CY, mx - CX)
        # Дуло смотрит в центр: короткий стержень от середины ребра внутрь.
        x1, y1 = mx, my
        x2 = mx - math.cos(ang) * r * 0.16
        y2 = my - math.sin(ang) * r * 0.16
        emitters += (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{TITANIUM}" stroke-width="{r * 0.055:.1f}" stroke-linecap="round"/>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{ENERGY}" stroke-width="{r * 0.018:.1f}" stroke-linecap="round" '
            f'stroke-opacity="0.9"/>'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{r * 0.035:.1f}" fill="{CYAN}" '
            f'fill-opacity="0.85"/>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">
  <defs>
    <radialGradient id="bg" cx="50%" cy="42%" r="72%">
      <stop offset="0%" stop-color="#10202c"/>
      <stop offset="60%" stop-color="{INK}"/>
      <stop offset="100%" stop-color="#03070a"/>
    </radialGradient>
    <linearGradient id="metal" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{TITANIUM}"/>
      <stop offset="45%" stop-color="{TITANIUM_DARK}"/>
      <stop offset="100%" stop-color="#425665"/>
    </linearGradient>
    <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{ENERGY}"/>
      <stop offset="55%" stop-color="{CYAN}"/>
      <stop offset="100%" stop-color="{DEEP}"/>
    </radialGradient>
  </defs>

  <rect width="{SIZE}" height="{SIZE}" rx="{SIZE * 0.2237:.0f}" fill="url(#bg)"/>

  <!-- Колодец под ядром -->
  <path d="{path(well)}" fill="{DEEP}" fill-opacity="0.35"/>
  <path d="{floor}" fill="none" stroke="{CYAN}" stroke-width="{r * 0.012:.1f}"
        stroke-opacity="0.28"/>

  <!-- Корпус -->
  <path d="{chassis}" fill="url(#metal)" stroke="#0d161d" stroke-width="{r * 0.02:.1f}"/>
  <path d="{chassis}" fill="none" stroke="{CYAN}" stroke-width="{r * 0.016:.1f}"
        stroke-opacity="0.5"/>

  <!-- Внутреннее кольцо и канал -->
  <path d="{inner_ring}" fill="{INK}" fill-opacity="0.72" stroke="{TITANIUM_DARK}"
        stroke-width="{r * 0.018:.1f}"/>
  {glow_stroke(channel, CYAN, r * 0.02)}

  <!-- Излучатели по серединам длинных рёбер -->
  {emitters}

  <!-- Ядро -->
  <circle cx="{CX}" cy="{CY}" r="{core_r * 2.1:.1f}" fill="{CYAN}" fill-opacity="0.10"/>
  <circle cx="{CX}" cy="{CY}" r="{core_r * 1.45:.1f}" fill="{CYAN}" fill-opacity="0.16"/>
  <path d="{core_d}" fill="url(#coreGlow)" stroke="{ENERGY}" stroke-width="{r * 0.022:.1f}"
        stroke-opacity="0.9" stroke-linejoin="round"/>
  <circle cx="{CX}" cy="{CY}" r="{core_r * 0.42:.1f}" fill="{ENERGY}"/>
</svg>
'''


def main() -> int:
    svg = build_svg()
    MASTER.parent.mkdir(parents=True, exist_ok=True)
    MASTER.write_text(svg, encoding="utf-8")
    print(f"→ мастер иконки: {MASTER}")

    ICONS.mkdir(parents=True, exist_ok=True)
    png = ICONS / "icon.png"
    subprocess.run(
        ["rsvg-convert", "-w", "1024", "-h", "1024", "-o", str(png), str(MASTER)],
        check=True,
    )
    print(f"→ растер: {png} (1024×1024)")

    subprocess.run(["bun", "tauri", "icon", "src-tauri/icons/icon.png"], cwd=APP, check=True)
    print("→ все размеры пересобраны")
    return 0


if __name__ == "__main__":
    sys.exit(main())
