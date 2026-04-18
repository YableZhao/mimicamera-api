from __future__ import annotations

import base64
import io
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.fitting.cube_io import write_cube
from app.fitting.idt import fit_lut_from_reference

router = APIRouter()


@router.post("/fit_lut")
async def fit_lut(references: list[UploadFile] = File(...)) -> dict:
    """Fit a 33³ LUT from one or more reference JPEGs via IDT in LAB.

    Returns the `.cube` file as base64 plus placeholder style metadata.
    """
    if not references:
        raise HTTPException(status_code=400, detail="at least one reference image required")

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
    lut = fit_lut_from_reference(reference_block)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_cube(lut, tmp_path)
        cube_b64 = base64.b64encode(tmp_path.read_bytes()).decode("ascii")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "cube_b64": cube_b64,
        "style_name": "Untitled",
        "style_description": "Style naming is wired in D10 with Claude curation.",
        "timing_ms": elapsed_ms,
    }


@router.post("/curate")
async def curate() -> None:
    raise HTTPException(status_code=501, detail="curate not yet implemented")


@router.get("/luts/curated/{lut_id}.cube")
async def get_curated_lut(lut_id: str) -> None:
    raise HTTPException(status_code=501, detail="curated LUTs not yet available")
