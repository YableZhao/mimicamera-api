from __future__ import annotations

import base64
import io
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

    pixels: list[np.ndarray] = []
    for up in references:
        raw = await up.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode {up.filename}: {exc}")
        pixels.append(np.asarray(img, dtype=np.uint8))

    combined = np.concatenate([p.reshape(-1, 3) for p in pixels], axis=0)
    side = int(np.ceil(np.sqrt(combined.shape[0])))
    pad = side * side - combined.shape[0]
    if pad > 0:
        combined = np.concatenate([combined, combined[:pad]], axis=0)
    reference_block = combined.reshape(side, side, 3)

    t0 = time.perf_counter()
    lut = MODES[mode](reference_block)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_cube(lut, tmp_path, title=f"Mimicamera ({mode})")
        cube_b64 = base64.b64encode(tmp_path.read_bytes()).decode("ascii")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "mode": mode,
        "cube_b64": cube_b64,
        "style_name": "Untitled",
        "style_description": "Style naming is wired in D10 with Claude curation.",
        "timing_ms": elapsed_ms,
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
