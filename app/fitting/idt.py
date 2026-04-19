from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian as skimage_gaussian

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


def _idt_2d(
    source: np.ndarray,
    target: np.ndarray,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    """2-D variant of IDT for when we want to operate on a subset of channels
    (e.g., `a*` and `b*` only). Each iteration draws a random 2-D rotation,
    matches 1-D marginals along each rotated axis, rotates back.
    """
    rng = np.random.default_rng(seed)
    s = source.astype(np.float32, copy=True)
    t = target.astype(np.float32)
    for _ in range(n_iter):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        c, sn = float(np.cos(theta)), float(np.sin(theta))
        rot = np.array([[c, -sn], [sn, c]], dtype=np.float32)
        s_rot = s @ rot
        t_rot = t @ rot
        for d in range(2):
            s_rot[:, d] = _match_hist_1d(s_rot[:, d], t_rot[:, d])
        s = s_rot @ rot.T
    return s


def _downsample(img: Image.Image, max_edge: int) -> Image.Image:
    if max(img.size) <= max_edge:
        return img
    img = img.copy()
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return img


def _smooth_lut(lut: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable 3-D Gaussian smoothing along the grid axes only (channels untouched)."""
    if sigma <= 0:
        return lut
    return skimage_gaussian(lut, sigma=sigma, channel_axis=-1).astype(np.float32)


def _prepare_reference(
    reference_rgb_u8: np.ndarray,
    downsample_edge: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, load and downsample a reference image; return flat (rgb01, lab) pixel arrays."""
    if reference_rgb_u8.ndim != 3 or reference_rgb_u8.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8, got {reference_rgb_u8.shape}")
    ref_img = Image.fromarray(reference_rgb_u8).convert("RGB")
    ref_img = _downsample(ref_img, downsample_edge)
    ref_rgb01 = np.asarray(ref_img, dtype=np.float32) / 255.0
    ref_rgb01_flat = ref_rgb01.reshape(-1, 3)
    ref_lab = _rgb01_to_lab(ref_rgb01_flat)
    return ref_rgb01_flat, ref_lab


def _compose_lut(
    warped_lab_flat: np.ndarray,
    lut_size: int,
    ref_rgb01_flat: np.ndarray,
    confidence_floor: float,
    smoothing_sigma: float,
    confidence_kde_sigma: float = 2.5,
) -> np.ndarray:
    """Shared post-processing: LAB→sRGB → confidence-weighted blend toward identity → Gaussian smooth.

    `confidence_kde_sigma` smooths the reference's 3D RGB histogram before normalisation,
    producing a kernel-density-like coverage map. Without this, a sparse synthetic reference
    only influences the handful of cells it directly touches, leaving the rest of the LUT
    as identity — the fitted style barely affects real scenes. A ~2–3 lattice-unit KDE
    bandwidth lets each reference pixel contribute to its neighbourhood while still
    falling off in genuinely unseen regions.
    """
    id_lut_rgb = identity_lut(lut_size)
    warped_rgb = _lab_to_rgb01(warped_lab_flat).reshape(lut_size, lut_size, lut_size, 3)

    rgb_hist, _ = np.histogramdd(
        ref_rgb01_flat,
        bins=lut_size,
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    if confidence_kde_sigma > 0:
        rgb_hist = skimage_gaussian(rgb_hist.astype(np.float32), sigma=confidence_kde_sigma)
    peak = float(rgb_hist.max())
    density = rgb_hist / peak if peak > 0 else rgb_hist
    confidence = np.clip(
        (density - confidence_floor) / max(1.0 - confidence_floor, 1e-6),
        0.0,
        1.0,
    )
    confidence = confidence[..., None].astype(np.float32)

    blended = confidence * warped_rgb + (1.0 - confidence) * id_lut_rgb
    smoothed = _smooth_lut(blended, smoothing_sigma)
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def fit_lut_idt(
    reference_rgb_u8: np.ndarray,
    *,
    lut_size: int = LUT_SIZE,
    n_iter: int = 20,
    seed: int = 0,
    downsample_edge: int = 256,
    confidence_floor: float = 0.02,
    smoothing_sigma: float = 0.5,
) -> np.ndarray:
    """L0 — Pitié–Kokaram IDT in CIELAB, confidence-weighted, Gaussian-smoothed.

    This is the plan's target algorithm. See the plan file for the framing:
    the confidence weighting is a low-rank-style extension of LoR-LUT.
    """
    ref_rgb01_flat, ref_lab = _prepare_reference(reference_rgb_u8, downsample_edge)
    id_flat_lab = _rgb01_to_lab(identity_lut(lut_size).reshape(-1, 3))
    warped_lab = _idt(id_flat_lab, ref_lab, n_iter=n_iter, seed=seed)
    return _compose_lut(warped_lab, lut_size, ref_rgb01_flat, confidence_floor, smoothing_sigma)


def fit_lut_histmatch(
    reference_rgb_u8: np.ndarray,
    *,
    lut_size: int = LUT_SIZE,
    seed: int = 0,  # accepted for API parity; histogram matching is deterministic
    downsample_edge: int = 256,
    confidence_floor: float = 0.02,
    smoothing_sigma: float = 0.5,
) -> np.ndarray:
    """L2 fallback — per-channel histogram matching in CIELAB.

    The algorithm Yable flagged as "很生硬" (harsh). Kept as a safety net: if IDT
    is misbehaving, this still produces a recognizable style transfer, even if
    cross-channel correlations are lost.
    """
    del seed  # deterministic given the reference
    ref_rgb01_flat, ref_lab = _prepare_reference(reference_rgb_u8, downsample_edge)
    id_flat_lab = _rgb01_to_lab(identity_lut(lut_size).reshape(-1, 3))
    warped_lab = np.empty_like(id_flat_lab)
    for d in range(3):
        warped_lab[:, d] = _match_hist_1d(id_flat_lab[:, d], ref_lab[:, d])
    return _compose_lut(warped_lab, lut_size, ref_rgb01_flat, confidence_floor, smoothing_sigma)


def fit_lut_chroma(
    reference_rgb_u8: np.ndarray,
    *,
    lut_size: int = LUT_SIZE,
    n_iter: int = 20,
    seed: int = 0,
    downsample_edge: int = 256,
    confidence_floor: float = 0.02,
    smoothing_sigma: float = 0.5,
) -> np.ndarray:
    """Chromaticity-only transfer: match the reference's (a*, b*) distribution
    via 2-D IDT while matching L* through a 1-D CDF.

    Produces a lighter-touch style than full 3-D IDT — scene brightness structure
    is preserved by the 1-D L* curve rather than replaced. Useful when a
    photographer's look is chromatic (e.g., "teal & orange") and you don't want
    to redistribute the scene's luminances as well.

    The plan calls this the "luminance pass" — in a LUT-baking context the
    spatially-aware guided filter becomes a 1-D tone curve on L*.
    """
    ref_rgb01_flat, ref_lab = _prepare_reference(reference_rgb_u8, downsample_edge)
    id_flat_lab = _rgb01_to_lab(identity_lut(lut_size).reshape(-1, 3))

    warped_L = _match_hist_1d(id_flat_lab[:, 0], ref_lab[:, 0])
    warped_ab = _idt_2d(id_flat_lab[:, 1:], ref_lab[:, 1:], n_iter=n_iter, seed=seed)
    warped_lab = np.concatenate([warped_L[:, None], warped_ab], axis=-1).astype(np.float32)

    return _compose_lut(warped_lab, lut_size, ref_rgb01_flat, confidence_floor, smoothing_sigma)


# Backward-compatible alias — callers using the pre-fallback-ladder name still work.
fit_lut_from_reference = fit_lut_idt
