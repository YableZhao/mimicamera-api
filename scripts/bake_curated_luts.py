"""Bake the six curated photographer-look LUTs + thumbnails + manifest.

These are hand-designed grades (lift / gamma / gain + saturation +
optional hue push), applied directly to the identity LUT. No IDT, no
reference photos — the goal is six *visibly* distinct looks that
transform any scene strongly, the way a real colourist's preset does.

IDT is still the right algorithm for fitting a LUT from a *reference
photo* (Photos picker, Unsplash, URL paste, Share Extension); it is
the wrong algorithm for packaged looks where we want dramatic,
scene-agnostic transformation.

Run from the mimicamera-api repo root with the .venv activated:

    python scripts/bake_curated_luts.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this script directly without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fitting.cube_io import identity_lut, write_cube  # noqa: E402

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


@dataclass
class Grade:
    id: str
    name: str
    description: str
    # Per-channel lift (added to the shadows), gain (multiplies highlights), gamma (midtone curve).
    # The standard colourist model: out = (in + lift) * gain, then out ** (1/gamma), then saturation.
    lift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    saturation: float = 1.0
    # Optional global tonal S-curve strength (0 = off, 1 = heavy).
    contrast: float = 0.0


LOOKS: list[Grade] = [
    Grade(
        id="golden-hour",
        name="Golden Hour",
        description="Warm highlights, honey mids, cocoa shadows",
        lift=(0.03, 0.015, -0.02),
        gain=(1.18, 1.05, 0.80),
        gamma=(1.0, 0.95, 0.9),
        saturation=1.15,
        contrast=0.2,
    ),
    Grade(
        id="teal-orange",
        name="Teal & Orange",
        description="Cinematic — blue shadows, amber highlights",
        lift=(-0.04, -0.02, 0.05),
        gain=(1.15, 0.92, 0.85),
        gamma=(1.0, 1.0, 1.05),
        saturation=1.30,
        contrast=0.3,
    ),
    Grade(
        id="noir",
        name="Noir",
        description="Crushed blacks, high contrast, barely any colour",
        lift=(-0.10, -0.10, -0.08),
        gain=(1.12, 1.12, 1.12),
        gamma=(1.15, 1.15, 1.15),
        saturation=0.20,
        contrast=0.55,
    ),
    Grade(
        id="pastel",
        name="Pastel",
        description="Lifted blacks, soft saturation, airy",
        lift=(0.08, 0.09, 0.10),
        gain=(0.95, 0.95, 0.98),
        gamma=(0.9, 0.9, 0.9),
        saturation=0.70,
        contrast=-0.25,
    ),
    Grade(
        id="film",
        name="Film",
        description="Faded cyan shadows, warm highlights, low-contrast",
        lift=(0.02, 0.04, 0.07),
        gain=(1.05, 1.0, 0.92),
        gamma=(0.95, 0.95, 1.05),
        saturation=0.85,
        contrast=-0.15,
    ),
    Grade(
        id="bleach-bypass",
        name="Bleach Bypass",
        description="Silver-screen — desaturated, punchy, slight cyan cast",
        lift=(-0.03, -0.03, 0.0),
        gain=(1.10, 1.10, 1.15),
        gamma=(1.1, 1.1, 1.0),
        saturation=0.35,
        contrast=0.5,
    ),
]


def _s_curve(x: np.ndarray, strength: float) -> np.ndarray:
    """Gentle S-curve around 0.5 — negative strength flattens contrast,
    positive steepens it. Bounded so saturation at the extremes stays sane."""
    if strength == 0.0:
        return x
    # Map [0, 1] → [-1, 1], apply sigmoid-ish curve, map back.
    y = (x - 0.5) * 2.0
    shaped = np.tanh(y * (1.0 + strength * 1.5)) / np.tanh(1.0 + strength * 1.5)
    return (shaped * 0.5 + 0.5).clip(0.0, 1.0)


def apply_grade(lut: np.ndarray, grade: Grade) -> np.ndarray:
    """lift → gain → gamma → saturation → contrast, each channel-aware."""
    out = lut.astype(np.float32).copy()

    lift = np.asarray(grade.lift, dtype=np.float32)
    gain = np.asarray(grade.gain, dtype=np.float32)
    gamma = np.asarray(grade.gamma, dtype=np.float32)

    # Lift + gain: shadows shifted, highlights scaled.
    out = (out + lift) * gain
    out = out.clip(0.0, 1.0)

    # Per-channel gamma (1/gamma in the exponent matches DaVinci's convention —
    # larger gamma = brighter midtones).
    out = np.power(out.clip(1e-6, 1.0), 1.0 / gamma)

    # Saturation around the luminance proxy.
    if grade.saturation != 1.0:
        luma = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., None]
        out = luma + (out - luma) * grade.saturation
        out = out.clip(0.0, 1.0)

    # Global tonal S-curve (last — runs on the whole thing).
    if grade.contrast != 0.0:
        out = _s_curve(out, grade.contrast)

    return out.astype(np.float32)


def make_grade_preview(grade: Grade, size: int = 96) -> np.ndarray:
    """Render a small preview image by applying the grade to a reference gradient.
    Used as the thumbnail next to each look in the iOS ReferenceStrip."""
    # A natural-looking test scene with sky, horizon, shadow, skin-tone patch.
    h = w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ty = y / (h - 1)

    sky = np.array([0.55, 0.68, 0.82])
    horizon = np.array([0.95, 0.85, 0.72])
    ground = np.array([0.45, 0.42, 0.28])
    colour = np.where(
        ty[..., None] < 0.5,
        sky + (horizon - sky) * (ty[..., None] * 2.0),
        horizon + (ground - horizon) * ((ty[..., None] - 0.5) * 2.0),
    )
    # Skin-tone oval in the lower half.
    cy, cx = int(h * 0.58), int(w * 0.5)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - cy) ** 2 / (0.25 * h) ** 2 + (xx - cx) ** 2 / (0.15 * w) ** 2) < 1
    skin = np.array([0.82, 0.65, 0.55])
    colour[mask] = 0.7 * colour[mask] + 0.3 * skin

    graded = apply_grade(colour, grade)
    return (graded * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    base = identity_lut(33)

    # Drop stale files from prior generations (different slugs).
    for stale in list(OUT_DIR.glob("*.cube")) + list(OUT_DIR.glob("*.jpg")):
        stale.unlink(missing_ok=True)

    for grade in LOOKS:
        lut = apply_grade(base, grade)
        write_cube(lut, OUT_DIR / f"{grade.id}.cube", title=grade.name)

        thumb = make_grade_preview(grade)
        Image.fromarray(thumb).save(OUT_DIR / f"{grade.id}.jpg", quality=88)

        manifest.append({"id": grade.id, "name": grade.name, "description": grade.description})
        print(f"baked {grade.id:14s} → {grade.name}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"looks": manifest}, indent=2) + "\n"
    )
    print(f"\nwrote manifest.json with {len(manifest)} looks → {OUT_DIR}")


if __name__ == "__main__":
    main()
