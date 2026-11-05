#!/usr/bin/env python3
"""Рисует фон для окна DMG-установщика Jarvis (тёмный градиент + заголовок)."""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 600, 400
BG = (8, 12, 24)
GLOW = (56, 189, 248)


def gradient() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    top, bottom = (13, 20, 38), (4, 6, 14)
    for y in range(H):
        t = y / (H - 1)
        c = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        d.line([(0, y), (W, y)], fill=c)
    return img


def glow() -> Image.Image:
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy, r = W // 2, 120, 220
    for i in range(60, 0, -1):
        rr = int(r * i / 60)
        a = int(38 * (i / 60) ** 2)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(a, a, a))
    layer = layer.filter(ImageFilter.GaussianBlur(40))
    return layer


def font(size: int, bold: bool = False):
    names = [
        "/System/Library/Fonts/Supplemental/Futura.ttc" if bold else None,
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for n in names:
        if not n:
            continue
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "bg.png"
    img = gradient()
    img = Image.blend(img, glow(), 0.6)

    d = ImageDraw.Draw(img)
    title = font(46, bold=True)
    sub = font(16)
    hint = font(13)

    def center(text, y, f, fill, tracking=0):
        if tracking == 0:
            box = d.textbbox((0, 0), text, font=f)
            d.text(((W - (box[2] - box[0])) / 2, y), text, font=f, fill=fill)
            return
        widths = [d.textlength(ch, font=f) for ch in text]
        total = sum(widths) + tracking * (len(text) - 1)
        x = (W - total) / 2
        for ch, w in zip(text, widths):
            d.text((x, y), ch, font=f, fill=fill)
            x += w + tracking

    center("J A R V I S", 66, title, (226, 240, 255), tracking=1)
    center("локальный голосовой ассистент", 128, sub, (148, 176, 208))

    # Стрелка-шеврон между иконками (позиции 150 и 450 по X, Y=190).
    for k in range(3):
        x0 = 262 + k * 22
        d.line([(x0, 178), (x0 + 16, 190), (x0, 202)], fill=(56, 189, 248), width=3)

    center("перетащите в папку «Программы»", 250, hint, (120, 148, 178))
    center("при первом запуске модели распакуются сами", 270, hint, (96, 122, 150))

    img.save(path)


if __name__ == "__main__":
    main()
