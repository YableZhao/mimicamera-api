from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.curation.claude_curator import (
    CurationResult,
    _extract_json_object,
    _fallback,
    curate_references,
)


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_extract_json_object_finds_embedded_payload() -> None:
    raw = 'sure. here is my picks: {"selected_indices": [0, 2], "style_name": "Test"} cheers'
    obj = _extract_json_object(raw)
    assert obj["selected_indices"] == [0, 2]
    assert obj["style_name"] == "Test"


def test_extract_json_object_handles_malformed_text() -> None:
    assert _extract_json_object("no json here") == {}
    assert _extract_json_object("") == {}
    assert _extract_json_object("{not: valid}") == {}


@pytest.mark.parametrize("count, expected_picks", [(0, 0), (2, 2), (3, 3), (5, 5), (10, 3)])
def test_fallback_picks_reasonable_subset(count: int, expected_picks: int) -> None:
    result = _fallback(count)
    assert len(result.selected_indices) == expected_picks


def test_curate_references_without_api_key_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = curate_references([_jpeg(i) for i in range(4)])
    assert isinstance(result, CurationResult)
    assert result.selected_indices == [0, 1, 2, 3]
    assert result.style_name == "Your Style"
    assert "curation disabled" in result.style_description.lower()


def test_curate_references_empty_input() -> None:
    result = curate_references([])
    assert result.selected_indices == []
    assert result.style_name == "Untitled"
