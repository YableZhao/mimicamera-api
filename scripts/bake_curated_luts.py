"""Bake the six curated photographer-look LUTs + thumbnails + manifest.

Reads a short palette definition for each look, fits a 33³ LUT via IDT, and
writes `.cube` + `.jpg` thumbnail + `manifest.json` into the iOS bundle
directory `mimicamera-ios/Mimicamera/Resources/CuratedLUTs`.

Run from the mimicamera-api repo root with the .venv activated:

    python scripts/bake_curated_luts.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from app.fitting.cube_io import write_cube
from app.fitting.idt import fit_lut_idt

OUT_DIR = Path(
    os.environ.get(
        "MIMICAMERA_CURATED_DIR",
        Path(__file__).resolve().parents[2]
        / "mimicamera-ios"
        / "Mimicamera"
        / "Resources"
        / "CuratedLUTs",
    )
)

LOOKS = [
    (
        "golden-hour",
        "Golden Hour",
        "Warm highlights, honey mids, cocoa shadows",
        [(230, 180, 100), (255, 220, 170), (120, 80, 50)],
    ),
    (
        "overcast",
        "Overcast",
        "Cool highlights, desaturated mids, slate shadows",
        [(190, 200, 210), (130, 150, 170), (70, 85, 100)],
    ),
    (
        "noir",
        "Noir",
        "Crushed blacks, low contrast, whisper of blue",
        [(30, 35, 45), (95, 100, 110), (160, 162, 170)],
    ),
    (
        "pastel",
        "Pastel",
        "Lifted blacks, soft saturation, airy",
        [(220, 205, 235), (240, 225, 240), (200, 195, 220)],
    ),
    (
        "film",
        "Film",
        "Faded cyan shadows, warm highlights, teal-orange",
        [(65, 90, 75), (210, 180, 140), (235, 210, 170)],
    ),
    (
        "portrait",
        "Portrait",
        "Skin-tone forward, clean highlights, green shadows",
        [(180, 140, 115), (235, 210, 185), (60, 80, 60)],
    ),
]


def make_reference(palette: list[tuple[int, int, int]], size: int = 256) -> np.ndarray:
    """Render a 3-colour triangular-blend gradient as a placeholder reference."""
    h = w = size
    y, x = np.mgrid[0:h, 0:w]
    a = (y / (h - 1)).astype(np.float32)
    b = (x / (w - 1)).astype(np.float32)
    pal = np.array(palette, dtype=np.float32)
    w0 = 1 - a
    w1 = a * (1 - b)
    w2 = a * b
    rgb = w0[..., None] * pal[0] + w1[..., None] * pal[1] + w2[..., None] * pal[2]
    img = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(3)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for idx, (slug, name, description, palette) in enumerate(LOOKS):
        ref = make_reference(palette)
        lut = fit_lut_idt(ref, n_iter=25, seed=idx)
        write_cube(lut, OUT_DIR / f"{slug}.cube", title=name)
        Image.fromarray(ref).resize((96, 96), Image.LANCZOS).save(
            OUT_DIR / f"{slug}.jpg", quality=88
        )
        manifest.append({"id": slug, "name": name, "description": description})
        print(f"baked {slug}: {name}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"looks": manifest}, indent=2) + "\n"
    )
    print(f"wrote manifest.json with {len(manifest)} looks → {OUT_DIR}")


if __name__ == "__main__":
    main()
