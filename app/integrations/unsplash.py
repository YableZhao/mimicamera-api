"""Thin Unsplash Search API proxy.

The iOS client never sees our Unsplash access key — calls go through the
backend, which also normalises the response shape so the app doesn't
care what Unsplash's internal schema looks like next quarter.

No-key fallback mirrors the Claude curator: without `UNSPLASH_ACCESS_KEY`
set in the server environment, `search()` returns a small deterministic
"demo" payload so the iOS flow can still be driven end-to-end in dev.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass
class UnsplashPhoto:
    id: str
    description: str | None
    thumbnail_url: str
    full_url: str
    download_url: str
    photographer_name: str
    photographer_username: str | None
    profile_url: str | None


async def search(
    query: str,
    *,
    per_page: int = 12,
    page: int = 1,
    access_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[UnsplashPhoto]:
    key = access_key or os.environ.get("UNSPLASH_ACCESS_KEY")
    if not key:
        return _demo_results(query)

    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=10)
    try:
        response = await client.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": per_page, "page": page, "orientation": "portrait"},
            headers={"Authorization": f"Client-ID {key}", "Accept-Version": "v1"},
        )
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned:
            await client.aclose()

    return [_from_api(item) for item in payload.get("results", [])]


def _from_api(item: dict) -> UnsplashPhoto:
    user = item.get("user", {}) or {}
    urls = item.get("urls", {}) or {}
    return UnsplashPhoto(
        id=str(item.get("id", "")),
        description=item.get("description") or item.get("alt_description"),
        thumbnail_url=urls.get("small", ""),
        full_url=urls.get("regular", ""),
        download_url=urls.get("regular", ""),
        photographer_name=user.get("name", "Unknown"),
        photographer_username=user.get("username"),
        profile_url=(user.get("links") or {}).get("html"),
    )


def _demo_results(query: str) -> list[UnsplashPhoto]:
    """Hand-picked Unsplash-hosted URLs so the flow works without a key during dev.

    These are real Unsplash CDN URLs (public, hot-linkable per their terms) picked to
    have broad colour coverage so the LUT fitting produces visible style transfer.
    """
    seeds = [
        ("cFPPYV6Dzqs", "golden hour portrait", "Jake Nackos", "jnackos"),
        ("C9EoPpe8NBM", "moody street", "Alexander Andrews", "alexandrebrondino"),
        ("_Wu37E_GbiA", "pastel interior", "Toa Heftiba", "heftiba"),
        ("R_bfN6YZOj4", "low-key noir", "Kazi Mizan", "kazi_mizan"),
        ("LvB8K7h04Zc", "teal and orange", "Vince Veras", "vinceveras"),
        ("ZHfXp12Vh6c", "desaturated overcast", "Alex Block", "alexblock"),
    ]
    photos: list[UnsplashPhoto] = []
    for photo_id, desc, name, username in seeds:
        photos.append(
            UnsplashPhoto(
                id=photo_id,
                description=f"{desc} (demo — {query!r})",
                thumbnail_url=f"https://images.unsplash.com/photo-{photo_id}?w=400&q=80",
                full_url=f"https://images.unsplash.com/photo-{photo_id}?w=1080&q=85",
                download_url=f"https://images.unsplash.com/photo-{photo_id}?w=1080&q=85",
                photographer_name=name,
                photographer_username=username,
                profile_url=f"https://unsplash.com/@{username}",
            )
        )
    return photos
