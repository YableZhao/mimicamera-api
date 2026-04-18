from __future__ import annotations

import numpy as np
import pytest

from app.fitting.cube_io import LUT_SIZE, identity_lut, read_cube, write_cube
from app.fitting.idt import _idt, fit_lut_from_reference


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
