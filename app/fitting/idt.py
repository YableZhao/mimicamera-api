from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.fitting.cube_io import LUT_SIZE, identity_lut


def _rgb01_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """Convert an (..., 3) sRGB array in [0, 1] to OpenCV-scale CIELAB."""
    flat = rgb01.reshape(-1, 1, 3)
    rgb_u8 = np.clip(flat * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(rgb01.shape)


def _lab_to_rgb01(lab: np.ndarray) -> np.ndarray:
    """Convert OpenCV-scale CIELAB back to sRGB in [0, 1]."""
    flat = lab.reshape(-1, 1, 3)
    lab_u8 = np.clip(flat, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return rgb.reshape(lab.shape)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a 3x3 rotation matrix uniformly from SO(3) via QR of a Gaussian."""
    g = rng.standard_normal((3, 3)).astype(np.float32)
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def _match_hist_1d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map 1-D `source` values to have the same CDF as `target` via linear interpolation."""
    s_sorted = np.sort(source)
    t_sorted = np.sort(target)
    n_s = s_sorted.size
    n_t = t_sorted.size
    s_rank = np.searchsorted(s_sorted, source, side="left").astype(np.float32)
    s_rank /= max(n_s - 1, 1)
    t_axis = np.linspace(0.0, 1.0, n_t, dtype=np.float32)
    return np.interp(s_rank, t_axis, t_sorted).astype(np.float32)


def _idt(
    source: np.ndarray,
    target: np.ndarray,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    """Pitié–Kokaram Iterative Distribution Transfer.

    `source` (N, 3) and `target` (M, 3) live in the same color space. Repeatedly:
    draw a random 3-D rotation, match 1-D marginals along each rotated axis, rotate
    back. Preserves cross-channel correlations that per-channel matching destroys.
    """
    rng = np.random.default_rng(seed)
    s = source.astype(np.float32, copy=True)
    t = target.astype(np.float32)
    for _ in range(n_iter):
        rot = _random_rotation(rng)
        s_rot = s @ rot
        t_rot = t @ rot
        for d in range(3):
            s_rot[:, d] = _match_hist_1d(s_rot[:, d], t_rot[:, d])
        s = s_rot @ rot.T
    return s


def _downsample(img: Image.Image, max_edge: int) -> Image.Image:
    if max(img.size) <= max_edge:
        return img
    img = img.copy()
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return img


def fit_lut_from_reference(
    reference_rgb_u8: np.ndarray,
    *,
    lut_size: int = LUT_SIZE,
    n_iter: int = 20,
    seed: int = 0,
    downsample_edge: int = 256,
    confidence_floor: float = 0.02,
) -> np.ndarray:
    """Fit a 3-D LUT that maps neutral sRGB cells → the reference's color distribution.

    Uses Pitié–Kokaram IDT in CIELAB with a uniform identity grid as the source. The
    fitted LUT is blended toward identity in cells where the reference's coverage
    (measured as a coarse RGB 3-D histogram) is below `confidence_floor`. This is the
    "confidence-weighted LUT" that keeps the live viewfinder from posterizing when
    it ventures into colors the reference never showed.

    Args:
        reference_rgb_u8: (H, W, 3) uint8 sRGB reference image.
        lut_size: grid resolution. Default 33 (CoreImage CIColorCube sweet spot).
        n_iter: IDT iterations. 15–25 is typical; higher = more faithful, slower.
        seed: RNG seed for IDT rotations (reproducible fits).
        downsample_edge: reference is resized so the long edge is at most this many pixels.
        confidence_floor: cells with normalized reference density below this value
            are fully blended toward identity.

    Returns:
        (lut_size, lut_size, lut_size, 3) float32 in [0, 1], indexed as `lut[r, g, b]`.
    """
    if reference_rgb_u8.ndim != 3 or reference_rgb_u8.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8, got {reference_rgb_u8.shape}")

    ref_img = Image.fromarray(reference_rgb_u8).convert("RGB")
    ref_img = _downsample(ref_img, downsample_edge)
    ref_rgb01 = np.asarray(ref_img, dtype=np.float32) / 255.0
    ref_rgb01_flat = ref_rgb01.reshape(-1, 3)
    ref_lab = _rgb01_to_lab(ref_rgb01_flat)

    id_lut_rgb = identity_lut(lut_size)
    id_flat_rgb = id_lut_rgb.reshape(-1, 3)
    id_flat_lab = _rgb01_to_lab(id_flat_rgb)

    warped_lab = _idt(id_flat_lab, ref_lab, n_iter=n_iter, seed=seed)
    warped_rgb = _lab_to_rgb01(warped_lab).reshape(lut_size, lut_size, lut_size, 3)

    rgb_hist, _ = np.histogramdd(
        ref_rgb01_flat,
        bins=lut_size,
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    peak = float(rgb_hist.max())
    density = rgb_hist / peak if peak > 0 else rgb_hist
    confidence = np.clip((density - confidence_floor) / max(1.0 - confidence_floor, 1e-6), 0.0, 1.0)
    confidence = confidence[..., None].astype(np.float32)

    blended = confidence * warped_rgb + (1.0 - confidence) * id_lut_rgb
    return np.clip(blended, 0.0, 1.0).astype(np.float32)
