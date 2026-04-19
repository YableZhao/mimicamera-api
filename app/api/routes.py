from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from app.curation.claude_curator import curate_references
from app.fitting.cube_io import write_cube
from app.fitting.idt import fit_lut_chroma, fit_lut_histmatch, fit_lut_idt
from app.integrations import unsplash

_CACHE_DIR = Path(tempfile.gettempdir()) / "mimicamera-fit-cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(raw_jpegs: list[bytes], mode: str) -> str:
    """SHA-256 over mode + sorted-by-hash list of reference JPEGs. Order-invariant so
    a photographer supplying the same bag in a different order still hits the cache.
    """
    digests = sorted(hashlib.sha256(j).hexdigest() for j in raw_jpegs)
    acc = hashlib.sha256()
    acc.update(mode.encode())
    for d in digests:
        acc.update(d.encode())
    return acc.hexdigest()

router = APIRouter()

MODES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "idt": fit_lut_idt,
    "hist": fit_lut_histmatch,
    "chroma": fit_lut_chroma,
}


@router.post("/fit_lut")
async def fit_lut(
    references: list[UploadFile] = File(...),
    mode: str = Query("idt", pattern="^(idt|hist|chroma)$"),
) -> dict:
    """Fit a 33³ LUT from one or more reference JPEGs.

    `mode=idt`    — Pitié–Kokaram IDT in CIELAB (default, target quality).
    `mode=hist`   — Per-channel histogram matching in CIELAB (L2 fallback).
    `mode=chroma` — IDT on (a*, b*) only with 1-D L* CDF (light-touch chromaticity).
    """
    if not references:
        raise HTTPException(status_code=400, detail="at least one reference image required")
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}")

    raw_jpegs: list[bytes] = []
    pixels: list[np.ndarray] = []
    for up in references:
        raw = await up.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode {up.filename}: {exc}")
        raw_jpegs.append(raw)
        pixels.append(np.asarray(img, dtype=np.uint8))

    cache_key = _cache_key(raw_jpegs, mode)
    cached_path = _CACHE_DIR / f"{cache_key}.cube"
    t0 = time.perf_counter()
    if cached_path.exists():
        cube_bytes = cached_path.read_bytes()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "mode": mode,
            "cube_b64": base64.b64encode(cube_bytes).decode("ascii"),
            "style_name": "Untitled",
            "style_description": "Cached fit — Claude curation wires style metadata separately.",
            "timing_ms": elapsed_ms,
            "cached": True,
        }

    combined = np.concatenate([p.reshape(-1, 3) for p in pixels], axis=0)
    side = int(np.ceil(np.sqrt(combined.shape[0])))
    pad = side * side - combined.shape[0]
    if pad > 0:
        combined = np.concatenate([combined, combined[:pad]], axis=0)
    reference_block = combined.reshape(side, side, 3)

    lut = MODES[mode](reference_block)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    write_cube(lut, cached_path, title=f"Mimicamera ({mode})")
    cube_b64 = base64.b64encode(cached_path.read_bytes()).decode("ascii")

    return {
        "mode": mode,
        "cube_b64": cube_b64,
        "style_name": "Untitled",
        "style_description": "Style naming is wired in D10 with Claude curation.",
        "timing_ms": elapsed_ms,
        "cached": False,
    }


@router.post("/curate")
async def curate(references: list[UploadFile] = File(...)) -> dict:
    """Given a bag of reference JPEGs, ask Claude vision to pick the 3-5 most
    stylistically representative ones and name the look. Falls back to a
    deterministic selection when `ANTHROPIC_API_KEY` is unset.
    """
    if not references:
        raise HTTPException(status_code=400, detail="at least one reference image required")

    jpegs: list[bytes] = []
    for up in references:
        raw = await up.read()
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode {up.filename}: {exc}")
        jpegs.append(raw)

    t0 = time.perf_counter()
    result = curate_references(jpegs)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "selected_indices": result.selected_indices,
        "style_name": result.style_name,
        "style_description": result.style_description,
        "timing_ms": elapsed_ms,
    }


@router.get("/luts/curated/{lut_id}.cube")
async def get_curated_lut(lut_id: str) -> None:
    raise HTTPException(status_code=501, detail="curated LUTs not yet available")


@router.get("/unsplash/search")
async def unsplash_search(q: str, per_page: int = 12, page: int = 1) -> dict:
    """Proxy Unsplash Search API — keeps the access key server-side and
    normalises the response so the iOS client sees a stable shape."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="query 'q' is required")
    photos = await unsplash.search(q, per_page=min(per_page, 30), page=max(page, 1))
    return {
        "query": q,
        "results": [photo.__dict__ for photo in photos],
        "keyed": bool(os.environ.get("UNSPLASH_ACCESS_KEY")),
    }
