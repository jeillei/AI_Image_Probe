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

**Second blocker found during implementation (Stable Cascade, documented before any generation succeeded).**
Stable Cascade (lite/bf16, below) downloaded successfully (~10 GB across both repos) but **failed to load**:
`StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", ...)` raises
`ValueError: vqgan/wuerstchen.py as defined in model_index.json does not exist in stabilityai/stable-cascade and
is not a module in 'diffusers/pipelines'` — a reproducible incompatibility between this environment's pinned
`diffusers==0.39.0` and the current hub repo's `model_index.json` (which references a custom VQGAN pipeline
module diffusers 0.39.0 does not ship), not fixed by `trust_remote_code=True`. Upgrading the project's core
`diffusers` dependency mid-project was rejected as a fix: it risks destabilizing every other frozen script in
this repository (the SD1.5 probe, the aMUSEd/SD1.5 generation scripts, the entire v1/v2 feature-extraction
pipeline), which is a materially worse risk than picking a different third generator. The partial download was
removed to reclaim disk space (`rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-cascade*`).

**Final substitute, actually used: SDXL (`stabilityai/stable-diffusion-xl-base-1.0`).** Not gated, ~7 GB fp16
(UNet + 2 CLIP text encoders + VAE), uses only diffusers' standard, extremely mature `StableDiffusionXLPipeline`
class with no custom components — verified to load cleanly with the installed `diffusers==0.39.0` before any
generation was attempted. **This is explicitly the weaker of the two rejected-then-blocked alternatives from
the task's own reasoning** (same broad UNet + CLIP-conditioning family as SD1.5, not a Diffusion Transformer or
a structurally distinct cascade) — chosen only because the two better-motivated candidates (PixArt-Sigma: disk;
SD3-medium: gating; Stable Cascade: library/repo incompatibility) were each concretely, independently blocked
in this exact environment. SDXL still differs from SD1.5 in real ways worth naming honestly, not overselling:
~2.6B-parameter UNet (vs SD1.5's 860M) with a different channel/attention configuration, two text encoders
(CLIP-L + OpenCLIP-bigG, vs SD1.5's single CLIP-L), a separately-trained, refined VAE, native 1024px training
with size/crop micro-conditioning SD1.5 never had, and a later, different training run/dataset. It is a
same-family, larger/refined-generation test, not an architecture-family test — this materially weakens Scenario
A/B's interpretation (see below) and is stated plainly in the results, not glossed over.

For the historical record, the Stable Cascade candidate that was implemented, downloaded, and only then found
to fail to load (see above) was chosen for the following reasons, which no longer apply since it was not used:

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
generator label `sdxl` (never `pixart`, never `cascade`). The two deliverable filenames requested by this
phase's task (`PREREGISTRATION_pixart_v1.md`, `PIXART_STAGE_DECOMPOSITION.md`) are kept as named, since renaming
deliverables was not requested — but their content states the substitution prominently, as here.

## Primary hypothesis (unchanged from the task's original framing, generator substituted)

> Adding the frozen trajectory stage (`diffpath_curvature`, `path_length`) to the frozen VAE + score stage
> improves real-vs-SDXL discrimination on held-out content.

**Primary comparison:** `AUROC(VAE+score+trajectory) − AUROC(VAE+score)`, content-grouped, same procedure
already used for SD1.5 and aMUSEd (`scripts/stage_decomposition_analysis.py`, unmodified).
**Primary uncertainty:** paired content-bootstrap 95% CI on that difference (1000 resamples, same code path).
**This is the one primary test. It will not be changed after seeing results.**

## Exact frozen configuration (all decided before generation)

| parameter | value |
|---|---|
| model | `stabilityai/stable-diffusion-xl-base-1.0`, dtype fp16 |
| generation resolution | **512×512** (amended before the bulk of generation — see "Amendment" below; downstream canonicalization to 256px is identical for every generator regardless of native output size, per the existing frozen `load()` function) |
| steps | 25 (matches this project's existing SD1.5 img2img/generation convention, `scripts/build_content_matched.py`; not tuned for SDXL specifically, decided before generation) |
| guidance scale | 7.5 (diffusers/SDXL standard default, matches this project's existing SD1.5 generation convention) |
| scheduler | pipeline default (`EulerDiscreteScheduler`, as shipped), unmodified |
| seed policy | `seed = int(sha256(f"sdxl:{content_id}")[:8], 16)`, one `torch.Generator("cpu")` per image — identical convention to `scripts/build_content_matched.py`'s SD1.5/aMUSEd generation |
| captions | the human COCO caption already in `data/content_matched/manifest.csv` for that `content_id` (the same column used for real/SD1.5/aMUSEd) — **not** a BLIP caption |
| negative prompt | `""` (empty), matching the SD1.5/aMUSEd generation convention already in this project |
| image format | PNG, RGB |
| preprocessing / canonicalization | identical to every other generator: `scripts/extract_detector_features.py::load()`, squash-resize to 256px, no crop |
| feature definitions | the frozen 10-feature v2 panel, `src/features/panel_v2.py`, **unchanged** |
| classifier | `StandardScaler + LogisticRegression(C=0.1, class_weight="balanced")`, **unchanged** |
| grouping strategy | content-grouped 5×10 CV, identical code path (`scripts/stage_decomposition_analysis.py`) |
| bootstrap procedure | content-level resampling with replacement, 1000 draws, identical code path |

### Amendment (made after 4/60 images at 1024×1024, before any further generation, during a live compute-load
concern raised by the machine's owner mid-run)

At 1024×1024/25 steps, SDXL took ~3 minutes/image on this machine's MPS backend — a ~3-hour unattended run that
was placing an unacceptable sustained load on the user's own computer. The run was stopped at 4/60 images and
those 4 were **discarded** (not reused) to keep the dataset at one consistent native resolution. Generation
resolution is amended to **512×512**, unchanged otherwise (steps=25, guidance=7.5, same seed policy, same
captions). This is a machine-stewardship change, not a result-driven one — made before any feature was
extracted from any SDXL image, let alone any of the primary comparison's numbers seen. 512×512 also brings SDXL
into line with this project's own existing convention for SD1.5/aMUSEd generation (`scripts/
build_content_matched.py` already generates at 512×512, not 1024), so this amendment arguably makes the
three-generator comparison *more* consistent, not less. Generation is additionally run in small paced batches
(not one continuous unattended block) so load can be monitored between batches.

## Dataset

The same 60 COCO content identities already used for real/SD1.5/aMUSEd. One SDXL image per content id, same
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

No new features, no curvature redefinition, no timestep search, no feature selection on SDXL data, no
classifier hyperparameter tuning for SDXL, no concatenation with the old 652-feature representation, no
learned detector, no dataset expansion after seeing the first result, no full transformation benchmark, no
fourth generator, no reinterpretation from standalone AUROC alone.
