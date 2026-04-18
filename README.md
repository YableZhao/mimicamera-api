# mimicamera-api

Backend for [Mimicamera](https://github.com/YableZhao/mimicamera-ios) — fits a 3D color LUT from one or more reference photos and returns a `.cube` file ready to apply in CoreImage.

**Algorithm:** Pitié–Kokaram Iterative Distribution Transfer in CIELAB with confidence-weighted regularization, followed by a guided-filter luminance refinement and a 3D Gaussian pre-smooth. Baked to a 33³ `.cube`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/fit_lut` | Multipart reference JPEG(s) → `{ cube_b64, style_name, style_description, timing_ms }` |
| `POST` | `/curate` | Folder of JPEGs → Claude picks 3–5 scene-diverse refs and names the style |
| `GET`  | `/luts/curated/{id}.cube` | One of six pre-baked photographer looks |
| `GET`  | `/health` | Liveness probe |

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/health.

## Deploy to Fly.io

```bash
fly launch --copy-config --no-deploy
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

## Research

This service extends two of the author's LUT-based image processing projects:

- [LUTor](https://github.com/YableZhao/LUTor) — web tool for histogram-matching style transfer (its per-channel algorithm is kept as the L2 fallback here).
- **LoR-LUT** (ECCV 2026 under review) — the low-rank LUT framing that motivates the confidence-weighted regularization in `app/fitting/idt.py`.

## Built with

- [Claude Code](https://claude.com/claude-code)

## License

MIT © 2026 Ziqi Zhao (Yable)
