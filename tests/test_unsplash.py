from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.integrations import unsplash
from app.main import app


def test_demo_results_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    photos = asyncio.run(unsplash.search("anything"))
    assert len(photos) == 6
    for p in photos:
        assert p.id
        assert p.thumbnail_url.startswith("https://")
        assert p.full_url.startswith("https://")
        assert p.photographer_name


def test_unsplash_search_route_uses_fallback_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    client = TestClient(app)
    response = client.get("/unsplash/search", params={"q": "golden hour"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "golden hour"
    assert data["keyed"] is False
    assert len(data["results"]) == 6
    assert data["results"][0]["photographer_name"]


def test_unsplash_search_rejects_empty_query() -> None:
    client = TestClient(app)
    response = client.get("/unsplash/search", params={"q": "   "})
    assert response.status_code == 400
