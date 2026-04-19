# mimicamera-api

Backend for [Mimicamera](https://github.com/YableZhao/mimicamera-ios) — fits a 3D color LUT from one or more reference photos and returns a `.cube` file ready to apply in `CoreImage.CIColorCube` on-device.

**Algorithm:** Pitié–Kokaram Iterative Distribution Transfer in CIELAB with KDE-smoothed confidence weighting, baked to a 33³ `.cube`. Also ships a per-channel histogram-matching fallback and an optional Claude vision curation pass that picks the 3-5 most scene-diverse images from a photographer's folder and names the style.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/fit_lut?mode=idt\|hist` | Multipart reference JPEG(s) → `{ cube_b64, style_name, style_description, timing_ms, mode }`. IDT is the default; `hist` is the L2 fallback (per-channel LAB, ~30× faster). |
| `POST` | `/curate` | Bag of reference JPEGs → Claude picks the 3-5 most stylistically representative images + names the style. Graceful no-key fallback returns the first N images. |
| `GET`  | `/luts/curated/{id}.cube` | One of six pre-baked photographer looks (stub — the iOS app ships these in-bundle). |
| `GET`  | `/health` | Liveness probe. |

## Algorithm overview

Full IDT pipeline (`mode=idt`):

1. **Prepare**: downsample the reference to 256 px long edge, convert to CIELAB.
2. **Fit**: Pitié–Kokaram IDT — 15-25 iterations of random 3-D rotation + 1-D histogram matching along each rotated axis. Preserves cross-channel correlations that per-channel matching destroys.
3. **Compose**: warp the uniform identity grid, convert back to sRGB.
4. **Confidence**: 3-D histogram of the reference in sRGB space, smoothed with a 2.5-lattice-unit Gaussian to produce a KDE-like coverage map. Blend toward identity where coverage is low.
5. **Smooth**: 3-D Gaussian (σ=0.5) over the LUT grid to kill outlier cells that would flicker under handheld motion.
6. **Bake**: 33³ `.cube` (R varies fastest, matching both Adobe and `CIColorCube` conventions).

Fallback ladder (from the plan):

- **L0 target**: full IDT + confidence + smooth.
- **L1**: toggle `mode=hist` for per-channel LAB histogram matching — visibly flatter but 30× faster and deterministic.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in ANTHROPIC_API_KEY if you want real curation
uvicorn app.main:app --reload
```

Hit <http://127.0.0.1:8000/health>; `/docs` for the auto-generated OpenAPI UI.

## Deploy to Fly.io

```bash
fly launch --copy-config --no-deploy
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

## Tests

```bash
pytest -q
```

17 tests currently: IDT shape/determinism/distribution-mean, `.cube` round-trip, Gaussian smoothing total-variation, IDT-vs-histmatch divergence on a cross-channel-correlated reference, Claude-curation fallback path, JSON-extractor resilience.

## Baking the curated iOS looks

```bash
source .venv/bin/activate
python scripts/bake_curated_luts.py
```

Writes six `.cube` + thumbnail + `manifest.json` into `../mimicamera-ios/Mimicamera/Resources/CuratedLUTs/`.

## Research

This service extends two of my LUT-based image processing projects:

- [LUTor](https://github.com/YableZhao/LUTor) — web tool for histogram-matching style transfer (its per-channel algorithm is the L2 fallback here).
- **LoR-LUT** (ECCV 2026 under review) — the low-rank LUT framing that motivates the confidence-weighted regularization in `app/fitting/idt.py`.

## Built with

- [Claude Code](https://claude.com/claude-code)

## License

MIT © 2026 Ziqi Zhao (Yable)
