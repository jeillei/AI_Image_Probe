# REPRODUCIBILITY

Two reproduction paths exist. **Start with the minimal path** — it reproduces every headline table and figure
from committed, compact feature tables in minutes, on an ordinary CPU, with no model downloads. The full path
regenerates those feature tables from raw images and is heavier: it runs on CPU, CUDA, or Apple Silicon MPS
(auto-detected), needs ~15GB of model weights, and is noticeably faster with a GPU/MPS than CPU-only, but a GPU
is not conceptually required to understand or reproduce the underlying method.

## Environment setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (`pyproject.toml` + `uv.lock`).

```bash
git clone <this-repository-url>
cd SynthImage
uv sync
```

Python 3.12 is required (pinned in `pyproject.toml`). `uv sync` creates a `.venv/` and installs exact pinned
versions from `uv.lock` — the same discipline that avoided several real version-compatibility failures
encountered during development (see `docs/research_history/` for the environment-compatibility notes this pin
discipline came out of).

## Minimal reproduction path (no GPU, no model downloads)

The final, compact per-image feature tables that every headline number in `docs/FINAL_RESULTS.md` is computed
from are committed to this repository under `results/` (see `results/summary/` for the specific files each
analysis reads). To reproduce the headline tables and figures:

```bash
uv run python scripts/dit_stage_decomposition_analysis.py
uv run python scripts/vae_curvature_redundancy_analysis.py
uv run python scripts/path_length_mechanism_analysis.py
uv run python scripts/dit_stage_decomposition_plots.py
uv run python scripts/vae_curvature_redundancy_plots.py
uv run python scripts/path_length_mechanism_plots.py
uv run python scripts/final_validation_analysis.py
uv run python scripts/final_validation_plots.py
```

Each of these reads a cached JSON/CSV feature table already in `results/`, runs the frozen CV/bootstrap analysis
(`scripts/stage_decomposition_analysis.py`), and writes tables and PNG figures back to `results/`. None of them
touch a GPU, download a model, or require the image files themselves — only the already-extracted scalar
features.

## What is not stored in Git

Per `.gitignore` and the data policy below, the following are **not** committed:
- `data/` in full (generated/downloaded images, manifests with local paths) — see `data/README.md` for how to
  regenerate.
- Model weights of any kind (`.safetensors`, `.ckpt`, `.gguf`, Hugging Face / PyTorch Hub caches).
- Large per-run raw intermediate dumps superseded by the frozen v2 panel (kept in git history from the v1-era
  exploration, not re-tracked going forward).
- Job/batch logs, `.out`/`.err` files, local absolute paths.

## Where models come from

| model | source | used for |
|---|---|---|
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` (Hugging Face) | the frozen forensic probe — every feature in the v2 panel |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | second generator (UNet, larger, differently-trained VAE) |
| PixArt-Sigma | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (official transformer+VAE) + `city96/t5-v1_1-xxl-encoder-gguf` (GGUF-quantized T5 text encoder, a disclosed adaptation — see `docs/research_history/PREREGISTRATION_DIT_GENERATOR_V1.md`) | third generator (genuine Diffusion Transformer, no UNet) |
| aMUSEd | `amused/amused-512` | fourth generator (masked-token model, no diffusion process at all — the architectural negative control) |
| LPIPS (VGG16 backbone) | `lpips` PyPI package | the `lpips_ae`/`lpips_roundtrip` features (AEROBLADE-style) |

All are fetched automatically via `huggingface_hub`/`diffusers`/`lpips` on first use; `uv sync` installs the
client libraries, not the weights themselves.

## What datasets are required

Real images: 60 COCO (`coco2014_val2014`) photographs, referenced by image id in
`data/content_matched/manifest.csv` (regenerate via `scripts/build_content_matched.py`, which downloads only
the specific 60 COCO images needed, not the full dataset). Generated counterparts are produced locally by this
project's own generation scripts (`scripts/generate_sdxl.py`, `scripts/generate_pixart_dit.py`, etc.) from the
same 60 human COCO captions — no external "AI image" dataset is used, by design (see `docs/FINAL_RESULTS.md` §2
for why naive external corpora were rejected as confounded).

## How manifests are constructed

A manifest is a CSV with (at minimum) `path, label, generator, content_id, caption` columns — `label=0` for real,
`label=1` for generated, `content_id` shared between a real image and every generator's counterpart for the same
underlying photo/caption. `scripts/build_*_manifest.py` scripts construct these from the base
`data/content_matched/manifest.csv`; see each script's docstring for its specific extension (e.g.
`build_final_validation_manifest.py` adds transform/transform_value columns for the robustness benchmark,
applied on-the-fly at extraction time rather than by generating new image files).

## How to extract the frozen features (full path, heavier — runs on CPU, CUDA, or MPS)

```bash
# 1. Build the matched-content dataset (downloads 60 COCO images, ~25MB)
uv run python scripts/build_content_matched.py

# 2. Generate each additional generator's counterparts (each is its own script; GPU strongly
#    recommended -- these are full diffusion sampling runs, not cheap)
uv run python scripts/generate_sdxl.py
uv run python scripts/generate_pixart_dit.py
# aMUSEd generation: see scripts/build_content_matched.py (amused is generated inline)

# 3. Extract the frozen v2 panel (two passes -- see extract_stage_panel.py's docstring for why
#    LPIPS and the SD1.5 UNet must run in separate processes on MPS)
uv run python scripts/extract_stage_panel.py --output results/stage_decomposition/panel_features_pass1.json
uv run python scripts/compute_lpips_panel.py --pass1 results/stage_decomposition/panel_features_pass1.json \
    --output results/stage_decomposition/panel_features.json
```

This full path was developed and runs successfully on ordinary hardware (a MacBook-class machine, Apple Silicon
MPS) — a GPU is not conceptually required, only helpful for larger runs. It requires: `cuda`/`mps`/`cpu`
(auto-detected by the probe), ~15GB of disk for model weights, and runtime proportional to device speed (the
6-step SD1.5 inversion pass is the dominant cost, roughly 3-5 seconds/image on Apple Silicon MPS, faster on a
discrete GPU, slower on CPU-only). For bulk extraction (many transform conditions × many generators), any
machine with a GPU will finish faster, but this is a convenience, not a requirement.

## Regenerating figures

Each `*_plots.py` script under `scripts/` regenerates one analysis phase's figures from that phase's already-
extracted feature tables (see the minimal reproduction path above) — no GPU needed for plotting itself.

## Tests

```bash
uv run pytest
```

Tests cover the scientifically load-bearing behaviors: frozen feature definitions, content-grouping correctness
(no train/test leakage across a real/fake pair), deterministic transform seeding, and manifest schema integrity.
See `tests/` and `docs/METHODS.md` §5 for what each test is protecting.
