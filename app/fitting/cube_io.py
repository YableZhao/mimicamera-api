from __future__ import annotations

from pathlib import Path

import numpy as np

LUT_SIZE = 33


def identity_lut(n: int = LUT_SIZE) -> np.ndarray:
    """Return an identity 3D LUT of shape (n, n, n, 3) with sRGB values in [0, 1].

    Indexed as `lut[r, g, b]` — the first axis is the R input, second G, third B.
    """
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def write_cube(lut: np.ndarray, path: str | Path, *, title: str = "Mimicamera") -> None:
    """Write a 3D LUT to a `.cube` file.

    `lut` must have shape (n, n, n, 3) with sRGB values in [0, 1], indexed as
    `lut[r, g, b]`. File layout matches LUTor and Adobe's conventions:
    R varies fastest, then G, then B.
    """
    if lut.ndim != 4 or lut.shape[-1] != 3 or lut.shape[0] != lut.shape[1] != lut.shape[2]:
        raise ValueError(f"Expected (n, n, n, 3) LUT, got {lut.shape}")

    n = lut.shape[0]
    lines: list[str] = [
        "# Mimicamera generated LUT",
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {n}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    for b in range(n):
        for g in range(n):
            for r in range(n):
                rgb = lut[r, g, b]
                lines.append(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}")

    Path(path).write_text("\n".join(lines) + "\n")


def read_cube(path: str | Path) -> np.ndarray:
    """Read a `.cube` file and return a (n, n, n, 3) float32 array indexed as `lut[r, g, b]`."""
    size: int | None = None
    values: list[list[float]] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head = line.split(maxsplit=1)[0].upper()
        if head == "LUT_3D_SIZE":
            size = int(line.split()[1])
            continue
        if head in {"TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE"}:
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                values.append([float(p) for p in parts])
            except ValueError:
                continue

    if size is None:
        raise ValueError(f"No LUT_3D_SIZE header found in {path}")
    if len(values) != size ** 3:
        raise ValueError(
            f"Expected {size ** 3} LUT entries, got {len(values)} in {path}"
        )

    arr = np.array(values, dtype=np.float32)
    bgr_indexed = arr.reshape(size, size, size, 3)
    return np.transpose(bgr_indexed, (2, 1, 0, 3))
