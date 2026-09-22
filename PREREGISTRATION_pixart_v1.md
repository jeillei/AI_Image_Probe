# PREREGISTRATION_pixart_v1 — third-generator decision experiment

Written and committed **before** any third-generator image was generated, per this phase's own requirement.

## Blocker: PixArt-Sigma is infeasible on this machine (documented, not silently bypassed)

**Checked and ruled out, in order:**

1. **`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`** (preferred model). Its text encoder is T5-XXL, shipped as two
   safetensors shards totaling **19.05 GB**, plus a 2.44 GB transformer and 0.34 GB VAE — **~21.8 GB total**.
   This machine has **19 GB free disk** (`df -h .` → `19Gi` on `/dev/disk3s1`, 91% used). The download alone
   does not fit, before any consideration of loading it into 17 GB of unified memory.
2. **`PixArt-alpha/PixArt-XL-2-1024-MS`** (original PixArt-alpha) and **`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`**
   (lower-resolution Sigma checkpoint) — checked via the HF API's file listing; both use the **same T5-XXL text
   encoder** (same ~19 GB), so neither resolves the blocker.
3. **8-bit/4-bit quantized T5** (the documented diffusers low-memory path, `load_in_8bit=True` via
   `bitsandbytes`) — `bitsandbytes` is not installed in this environment, and its low-bit kernels are CUDA-only
   with no Apple-Silicon/MPS backend, so it would not help on this machine even if installed. It also does not
   reduce the *download* size (the full-precision weights are fetched first, then quantized on load), so it does
   not resolve the disk-space blocker regardless.
4. **`stabilityai/stable-diffusion-3-medium-diffusers`** (MMDiT, a genuine Diffusion Transformer, the same
   family PixArt-Sigma belongs to) was considered as a closer architectural match than a UNet model, and its
   footprint is small enough (~6 GB) if its own T5 encoder is dropped (`text_encoder_3=None`, a standard,
   commonly used low-memory SD3 configuration). **Blocked separately**: the repository is gated
   (`curl` returns HTTP 401; the HF API reports `"gated": "auto"`), requiring an authenticated, license-accepted
   HF token this environment does not have and cannot obtain non-interactively.

**Closest scientifically useful alternative, chosen instead: Stable Cascade (lite/bf16 variant).**

* `stabilityai/stable-cascade-prior` (Stage C prior, `prior_lite` subfolder) +
  `stabilityai/stable-cascade` (Stage B decoder, `decoder_lite` subfolder, + VQGAN Stage A) — **not gated**,
  fully open. Total download for the lite/bf16 configuration ≈ **5 GB** (`prior_lite` bf16 2.06 GB +
  `decoder_lite` bf16 1.40 GB + CLIP-bigG text encoder bf16 1.39 GB + VQGAN 0.07 GB), comfortably inside the
  19 GB free disk with headroom to spare.
* **Why it still distinguishes the competing hypotheses.** Stable Cascade is architecturally very different from
  SD1.5: rather than one VAE (compression factor 8) plus one UNet operating in that latent space, it uses a
  **two-stage cascade** in a far more compressed 24×24 latent (compression factor ~42) — a semantic-compressor
  encoder, a Stage-C prior diffusion model operating on that tiny latent, and a *separate* Stage-B decoder
  diffusion model that expands it before a final VQGAN decode. It is CLIP-conditioned (no T5), trained by a
  different lab on a different schedule from SD1.5, and neither of its two diffusion stages is the SD1.5 UNet or
  a scaled variant of it. It is **not** a Diffusion Transformer (PixArt-Sigma's specific architectural class),
  so it is a weaker test of the *DiT-specific* sub-hypothesis than PixArt-Sigma would have been — but it is a
  strong test of the actual hypothesis under evaluation this phase: *does trajectory curvature reflect something
  about "being a diffusion-generated image" broadly, or is it specific to SD1.5's own architecture/training?*
  A structurally unrelated cascaded diffusion model answers that question almost as informatively as a DiT would.
* **Critically, this substitution requires zero change to the frozen evaluation protocol.** The 10-feature panel
  is computed by running the **frozen SD1.5 probe's own inversion** on whatever pixels a generator produces —
  it was already applied unchanged to aMUSEd (a masked-token/VQGAN model with no diffusion process at all). The
  generator's own internal architecture is irrelevant to feature extraction; only the resulting RGB image
  matters. Stable Cascade requires no adaptation of `diffpath_curvature`, `lare_t200`, or any other feature
  formula, exactly as PixArt-Sigma would not have.

**Naming discipline.** To avoid ever implying this is PixArt-Sigma, all data/manifests/feature records use the
generator label `cascade` (never `pixart`). The two deliverable filenames requested by this phase's task
(`PREREGISTRATION_pixart_v1.md`, `PIXART_STAGE_DECOMPOSITION.md`) are kept as named, since renaming deliverables
was not requested — but their content states the substitution prominently, as here.

## Primary hypothesis (unchanged from the task's original framing, generator substituted)

> Adding the frozen trajectory stage (`diffpath_curvature`, `path_length`) to the frozen VAE + score stage
> improves real-vs-Cascade discrimination on held-out content.

**Primary comparison:** `AUROC(VAE+score+trajectory) − AUROC(VAE+score)`, content-grouped, same procedure
already used for SD1.5 and aMUSEd (`scripts/stage_decomposition_analysis.py`, unmodified).
**Primary uncertainty:** paired content-bootstrap 95% CI on that difference (1000 resamples, same code path).
**This is the one primary test. It will not be changed after seeing results.**

## Exact frozen configuration (all decided before generation)

| parameter | value |
|---|---|
| prior model | `stabilityai/stable-cascade-prior`, subfolder `prior_lite`, dtype bf16 |
| decoder model | `stabilityai/stable-cascade`, subfolder `decoder_lite`, dtype bf16 |
| generation resolution | 512×512 (Stable Cascade's `resolution_multiple`-driven latent sizing at this height/width; downstream canonicalization to 256px is identical for every generator regardless of native output size, per the existing frozen `load()` function) |
| prior steps | 20 (diffusers' own documented example value; not the library default of 60, chosen for feasible runtime on this machine — decided now, not tuned after seeing outputs) |
| prior guidance scale | 4.0 (diffusers default) |
| decoder steps | 10 (diffusers default) |
| decoder guidance scale | 0.0 (diffusers default — Stable Cascade's decoder is not CFG-guided by design) |
| scheduler | `DDPMWuerstchenScheduler`, as shipped with each pipeline, unmodified |
| seed policy | `seed = int(sha256(f"cascade:{content_id}")[:8], 16)`, one `torch.Generator("cpu")` per image — identical convention to `scripts/build_content_matched.py`'s SD1.5/aMUSEd generation |
| captions | the human COCO caption already in `data/content_matched/manifest.csv` for that `content_id` (the same column used for real/SD1.5/aMUSEd) — **not** a BLIP caption |
| negative prompt | `""` (empty), matching the SD1.5/aMUSEd generation convention already in this project |
| image format | PNG, RGB |
| preprocessing / canonicalization | identical to every other generator: `scripts/extract_detector_features.py::load()`, squash-resize to 256px, no crop |
| feature definitions | the frozen 10-feature v2 panel, `src/features/panel_v2.py`, **unchanged** |
| classifier | `StandardScaler + LogisticRegression(C=0.1, class_weight="balanced")`, **unchanged** |
| grouping strategy | content-grouped 5×10 CV, identical code path (`scripts/stage_decomposition_analysis.py`) |
| bootstrap procedure | content-level resampling with replacement, 1000 draws, identical code path |

## Dataset

The same 60 COCO content identities already used for real/SD1.5/aMUSEd. One Cascade image per content id, same
caption. No hand-selection, no re-generation on appearance, no prompt engineering.

## Decision scenarios (fixed in advance)

* **A — behaves like SD1.5**: trajectory ΔAUROC 95% CI excludes 0 and is positive, comparable in size to SD1.5's
  +0.125 [0.056, 0.204]. Supports a general diffusion-model property.
* **B — behaves like aMUSEd**: trajectory ΔAUROC CI includes 0 (or is negative), and VAE alone is already
  strong. Supports an SD1.5/probe-specific effect; curvature is not rescued by redefinition.
* **C — intermediate**: some positive trend, weaker or qualitatively different from SD1.5 (e.g. significant but
  much smaller, or significant only in a targeted comparison, not the primary cumulative one). Treat generator
  architecture as a plausible explanatory variable; characterize before adding a fourth generator.

## What will NOT be done

No new features, no curvature redefinition, no timestep search, no feature selection on Cascade data, no
classifier hyperparameter tuning for Cascade, no concatenation with the old 652-feature representation, no
learned detector, no dataset expansion after seeing the first result, no full transformation benchmark, no
fourth generator, no reinterpretation from standalone AUROC alone.
