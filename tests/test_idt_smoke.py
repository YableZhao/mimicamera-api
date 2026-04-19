from __future__ import annotations

import numpy as np
import pytest

from app.fitting.cube_io import LUT_SIZE, identity_lut, read_cube, write_cube
from app.fitting.idt import (
    _idt,
    _idt_2d,
    _smooth_lut,
    fit_lut_chroma,
    fit_lut_from_reference,
    fit_lut_histmatch,
    fit_lut_idt,
)


def test_identity_lut_shape_and_range() -> None:
    lut = identity_lut(33)
    assert lut.shape == (33, 33, 33, 3)
    assert lut.min() == pytest.approx(0.0)
    assert lut.max() == pytest.approx(1.0)
    assert lut[0, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert lut[-1, -1, -1].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_cube_roundtrip(tmp_path) -> None:
    lut = identity_lut(LUT_SIZE)
    path = tmp_path / "id.cube"
    write_cube(lut, path, title="Identity")
    loaded = read_cube(path)
    assert loaded.shape == lut.shape
    np.testing.assert_allclose(loaded, lut, atol=1e-5)


def _synthetic_warm_reference() -> np.ndarray:
    """A 128×128 warm-tinted patch with visible structure. Uint8 sRGB."""
    h = w = 128
    y, x = np.mgrid[0:h, 0:w]
    r = (180 + 0.3 * x).clip(0, 255)
    g = (140 + 0.1 * y).clip(0, 255)
    b = (90 - 0.1 * x).clip(0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def test_idt_transfers_distribution_mean() -> None:
    """Pure IDT: source pixels should be warped so their mean approaches target's mean."""
    rng = np.random.default_rng(0)
    source = rng.normal(0.5, 0.1, size=(4000, 3)).astype(np.float32)
    target = rng.normal(0.85, 0.05, size=(4000, 3)).astype(np.float32)

    warped = _idt(source, target, n_iter=25, seed=0)

    src_dist = float(np.linalg.norm(source.mean(axis=0) - target.mean(axis=0)))
    warp_dist = float(np.linalg.norm(warped.mean(axis=0) - target.mean(axis=0)))
    assert warp_dist < src_dist * 0.2, (
        f"IDT did not pull source mean toward target; "
        f"before={src_dist:.3f}, after={warp_dist:.3f}"
    )


def test_fit_lut_differs_from_identity_in_reference_zone() -> None:
    """The fitted LUT should differ from identity in the color region the reference covers."""
    ref = _synthetic_warm_reference()
    lut = fit_lut_from_reference(ref, n_iter=15, seed=0)
    id_lut = identity_lut(LUT_SIZE)

    assert lut.shape == (LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    assert 0.0 <= lut.min() and lut.max() <= 1.0

    diff = np.linalg.norm(lut - id_lut, axis=-1)
    assert diff.max() > 0.02, (
        f"LUT never deviates from identity (max diff {diff.max():.4f})"
    )

    # Reference RGB ranges: R in ~[180, 218], G in ~[140, 152], B in ~[77, 90] (u8).
    # Corresponding LUT cell ranges at N=33: R≈22–27, G≈18–19, B≈10–11.
    zone = diff[22:28, 18:20, 10:12]
    assert zone.mean() > 0.005, (
        f"LUT barely deviates inside reference coverage zone (mean {zone.mean():.4f})"
    )


def test_fit_lut_deterministic_with_seed() -> None:
    ref = _synthetic_warm_reference()
    a = fit_lut_from_reference(ref, n_iter=5, seed=42)
    b = fit_lut_from_reference(ref, n_iter=5, seed=42)
    np.testing.assert_allclose(a, b, atol=0.0)


def test_smoothing_reduces_local_variation() -> None:
    """A smoothed LUT should have less cell-to-cell variation than its unsmoothed source."""
    rng = np.random.default_rng(0)
    noisy = rng.uniform(0, 1, size=(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)).astype(np.float32)
    smoothed = _smooth_lut(noisy, sigma=1.0)

    def total_variation(lut: np.ndarray) -> float:
        dx = np.abs(np.diff(lut, axis=0)).sum()
        dy = np.abs(np.diff(lut, axis=1)).sum()
        dz = np.abs(np.diff(lut, axis=2)).sum()
        return float(dx + dy + dz)

    assert total_variation(smoothed) < 0.5 * total_variation(noisy)


def test_histmatch_fallback_also_shifts() -> None:
    """The L2 fallback (per-channel LAB histogram matching) should also deviate from identity."""
    ref = _synthetic_warm_reference()
    lut = fit_lut_histmatch(ref)
    id_lut = identity_lut(LUT_SIZE)
    assert lut.shape == id_lut.shape
    diff = np.linalg.norm(lut - id_lut, axis=-1)
    assert diff.max() > 0.02, f"histmatch LUT flat (max diff {diff.max():.4f})"


def _correlated_reference() -> np.ndarray:
    """A 128×128 reference with strong cross-channel correlation (R high ↔ B low, G mid-varies)."""
    rng = np.random.default_rng(7)
    h = w = 128
    noise = rng.normal(0.0, 10.0, size=(h, w)).astype(np.float32)
    r = np.clip(80 + 0.9 * np.arange(w)[None, :] + noise, 0, 255)
    b = np.clip(220 - 0.9 * np.arange(w)[None, :] + noise, 0, 255)
    g = np.clip(150 + 0.5 * np.arange(h)[:, None] - 0.4 * np.arange(w)[None, :], 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def test_idt_and_histmatch_differ() -> None:
    """On a reference with cross-channel correlation IDT must capture structure the
    per-channel fallback cannot — the LUTs should diverge."""
    ref = _correlated_reference()
    idt = fit_lut_idt(ref, n_iter=20, seed=0)
    hist = fit_lut_histmatch(ref)
    diff = np.linalg.norm(idt - hist, axis=-1)
    assert diff.max() > 0.01, (
        f"IDT and histmatch nearly identical (max diff {diff.max():.4f}) — "
        "expected IDT to diverge on cross-channel-correlated reference"
    )


def test_idt_2d_transfers_distribution_mean() -> None:
    """_idt_2d should pull a 2-D source toward the target distribution."""
    rng = np.random.default_rng(0)
    source = rng.normal(0.5, 0.1, size=(3000, 2)).astype(np.float32)
    target = rng.normal(0.85, 0.05, size=(3000, 2)).astype(np.float32)
    warped = _idt_2d(source, target, n_iter=25, seed=0)
    src_dist = float(np.linalg.norm(source.mean(axis=0) - target.mean(axis=0)))
    warp_dist = float(np.linalg.norm(warped.mean(axis=0) - target.mean(axis=0)))
    assert warp_dist < src_dist * 0.3


def test_fit_lut_chroma_differs_from_full_idt() -> None:
    """Chroma-only fitting should produce a LUT distinct from full 3-D IDT."""
    ref = _correlated_reference()
    full = fit_lut_idt(ref, n_iter=15, seed=0)
    chroma = fit_lut_chroma(ref, n_iter=15, seed=0)
    diff = np.linalg.norm(full - chroma, axis=-1)
    assert diff.max() > 0.01, (
        f"chroma and full-IDT LUTs nearly identical (max diff {diff.max():.4f})"
    )
