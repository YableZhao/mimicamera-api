from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import _CACHE_DIR
from app.main import app


@pytest.fixture(autouse=True)
def clean_cache_dir() -> None:
    for item in _CACHE_DIR.iterdir() if _CACHE_DIR.exists() else []:
        item.unlink(missing_ok=True)
    yield


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_fit_lut_returns_valid_cube_and_is_not_cached_first_time() -> None:
    client = TestClient(app)
    ref = _jpeg(1)
    response = client.post(
        "/fit_lut?mode=idt",
        files={"references": ("warm.jpg", ref, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "idt"
    assert data["cached"] is False
    assert data["timing_ms"] > 0
    # cube_b64 decodes to a file starting with "# Mimicamera"
    import base64
    cube = base64.b64decode(data["cube_b64"]).decode()
    assert "LUT_3D_SIZE 33" in cube
    assert cube.count("\n") > 33 ** 3


def test_fit_lut_second_call_hits_cache() -> None:
    client = TestClient(app)
    ref = _jpeg(2)
    first = client.post(
        "/fit_lut?mode=idt",
        files={"references": ("same.jpg", ref, "image/jpeg")},
    ).json()
    second = client.post(
        "/fit_lut?mode=idt",
        files={"references": ("same.jpg", ref, "image/jpeg")},
    ).json()
    assert first["cached"] is False
    assert second["cached"] is True
    # Cached hit should be materially faster than the fresh fit.
    assert second["timing_ms"] <= first["timing_ms"]
    # And the cube bytes are identical.
    assert first["cube_b64"] == second["cube_b64"]


def test_fit_lut_different_modes_cache_independently() -> None:
    client = TestClient(app)
    ref = _jpeg(3)
    idt = client.post(
        "/fit_lut?mode=idt",
        files={"references": ("same.jpg", ref, "image/jpeg")},
    ).json()
    hist = client.post(
        "/fit_lut?mode=hist",
        files={"references": ("same.jpg", ref, "image/jpeg")},
    ).json()
    assert idt["cached"] is False
    assert hist["cached"] is False
    # Different cache entries so the second call was not served from IDT's.
    assert idt["cube_b64"] != hist["cube_b64"]


def test_fit_lut_rejects_invalid_mode() -> None:
    client = TestClient(app)
    response = client.post(
        "/fit_lut?mode=bogus",
        files={"references": ("x.jpg", _jpeg(4), "image/jpeg")},
    )
    assert response.status_code == 422  # FastAPI query-pattern validation
