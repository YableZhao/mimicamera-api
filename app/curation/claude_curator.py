"""Claude vision-based reference curation.

Given a bag of reference JPEGs (e.g. a photographer's folder), pick the 3-5
most stylistically representative, scene-diverse images and name the look.
Naive averaging of a folder that is 80 % sunsets gives a sunset LUT, not a
photographer LUT — this step guards against that failure mode and produces
the "Moody Editorial" style chip the iOS app displays.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Sequence

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


@dataclass
class CurationResult:
    selected_indices: list[int]
    style_name: str
    style_description: str


_SYSTEM_PROMPT = (
    "You are a photography-style analyst. You will see a bag of images from a single "
    "photographer's portfolio. Your job is to pick the 3-5 most stylistically representative, "
    "scene-diverse images, then name the underlying style. Name should be 1-3 words, evocative "
    "(e.g. 'Moody Editorial', 'Sun-bleached', 'Low-key Noir'). Description should be one short "
    "sentence naming the tonal signature (e.g. 'warm highlights, crushed blacks, cyan mids').\n"
    "\n"
    "Respond as JSON with keys `selected_indices` (array of ints, 0-based), `style_name` "
    "(string), `style_description` (string). No other keys, no commentary."
)


def curate_references(
    images_jpeg: Sequence[bytes],
    *,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> CurationResult:
    """Pick a scene-diverse, style-representative subset and name the style.

    Falls back to a deterministic first-N-images selection + generic naming when
    the Anthropic SDK is unavailable or `api_key` is unset; the iOS app still
    gets something sensible to display. The real curation quality only kicks in
    once `ANTHROPIC_API_KEY` is present in the backend environment.
    """
    if not images_jpeg:
        return CurationResult(selected_indices=[], style_name="Untitled", style_description="")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not _ANTHROPIC_AVAILABLE or not key:
        return _fallback(len(images_jpeg))

    client = anthropic.Anthropic(api_key=key)
    content: list[dict] = []
    for idx, jpeg in enumerate(images_jpeg[:20]):  # hard cap for latency + cost
        b64 = base64.b64encode(jpeg).decode("ascii")
        content.append({"type": "text", "text": f"Image {idx}:"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    payload = _extract_json_object(response.content[0].text if response.content else "")
    return CurationResult(
        selected_indices=list(payload.get("selected_indices", []))[:5] or list(range(min(3, len(images_jpeg)))),
        style_name=str(payload.get("style_name", "Your Style")),
        style_description=str(payload.get("style_description", "")),
    )


def _fallback(count: int) -> CurationResult:
    take = min(count, 5) if count <= 5 else 3
    return CurationResult(
        selected_indices=list(range(take)),
        style_name="Your Style",
        style_description="Claude curation disabled — using first references directly.",
    )


def _extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of Claude's response, ignoring prose around it."""
    import json
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
