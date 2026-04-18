# mimicamera-api — Claude Code Conventions

## Project

FastAPI backend for the [Mimicamera](https://github.com/YableZhao/mimicamera-ios) iOS app. Fits a 3D LUT from reference photos using classical color-transfer algorithms. Uses Claude vision for reference curation and style naming.

## Code style

- Python 3.11+.
- Numpy-vectorized; no Python loops in hot paths.
- No PyTorch, no CUDA. CPU-only inference.
- Type hints on every public function.
- English identifiers and comments.
- Comments: only for non-obvious WHY, not WHAT.

## Dependencies (hard cap — do not add without approval)

- `fastapi`, `uvicorn`, `pydantic`, `python-multipart`
- `numpy`, `pillow`, `opencv-python-headless`, `scikit-image`
- `anthropic`

Nothing else.

## File organization

```
app/
  main.py                    FastAPI entrypoint
  api/
    routes.py                HTTP handlers only, no business logic
  fitting/
    idt.py                   Pitié–Kokaram IDT + confidence-weighted baking
    luma_pass.py             Guided-filter luminance refinement
    cube_io.py               .cube read/write (ported from LUTor)
  curation/
    claude_curator.py        Claude vision: pick refs + name style
  curated/
    manifest.json
    *.cube
```

## Algorithm invariants

- LUT grid: **33³**. Never 17³.
- Color space: CIELAB for IDT; convert to sRGB on bake.
- Luminance pass: guided filter, radius 8–16 px at 512-long-edge, ε = 0.01².
- Confidence weighting: blend toward identity LUT in cells where the reference KDE density is below a threshold (document the threshold inline).
- Every new fitting algorithm ships with a visual regression test (golden `.cube` snapshot).

## Forbidden

- Persisting user images beyond a hashed `/tmp` request-scope cache.
- Auth, sessions, users, or any stateful feature in v1.
- Synchronous `requests.get` — use `httpx.AsyncClient` if external HTTP is ever needed.

## Testing

- `pytest` + golden-image fixtures.
- Run `pytest -q` before every push.
