# PREREGISTRATION_DIT_GENERATOR_V1 — architecture-disambiguation decision experiment

Written and committed **before** any of the 60 full-dataset PixArt-Sigma images were generated (a small number of
purely technical feasibility/loading checks preceded this document — see "Engineering validation" below; none of
them were used to tune generation settings for appearance or expected detector performance).

## Generator: PixArt-Sigma — feasible this time, with one disclosed adaptation

**PixArt-Sigma is being used, not substituted.** The disk-space blocker recorded in
`PREREGISTRATION_pixart_v1.md` (T5-XXL's official weights: ~19 GB, fp32 only, no fp16 variant, exceeding this
machine's free disk at the time) is resolved by using a **community GGUF-quantized T5-XXL encoder**
(`city96/t5-v1_1-xxl-encoder-gguf`, `Q5_K_M` quantization, 3.39 GB) in place of the official fp32 T5 weights,
combined with PixArt-Sigma's own official transformer and VAE (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`,
transformer 2.44 GB + VAE 0.34 GB). Total footprint ≈ 6.2 GB, comfortably inside the ~25 GB now free on this
machine (disk usage has fluctuated across sessions; re-checked immediately before this attempt).

**Why this is a valid lineage-independent diffusion test.** The denoiser being tested is PixArt-Sigma's own
**Transformer2DModel** (a genuine Diffusion Transformer, `diffusion_pytorch_model.safetensors`, official
weights, unmodified) — this is the architectural component the whole experiment is about, and it is untouched.
The VAE is PixArt-Sigma's own official VAE, also unmodified. Only the **text encoder** (which produces
conditioning vectors, not denoising dynamics) is a quantized version of the same T5-v1.1-XXL architecture
PixArt-Sigma was trained with — same architecture, same 4096-dim output space, same tokenizer, reduced numeric
precision on its weights. `city96`'s GGUF quantizations are widely used in the open-source diffusion community
specifically for this purpose (running T5-conditioned models on constrained hardware) and Q5_K_M is above the
quantizer's own stated quality floor for acceptable text-conditioning fidelity ("Q5_K_M or larger for acceptable
results" per the quantization's documentation). **Disclosed limitation**: this is not bit-identical to official
PixArt-Sigma generations; a lower-precision text encoder could in principle shift the exact image distribution
somewhat (verified in engineering validation below to still produce coherent, correctly-captioned images).
Verified before finalizing: `T5EncoderModel.from_pretrained(..., gguf_file=...)` loads without error and produces
the expected `(batch, seq_len, 4096)` embedding shape with no NaNs; `PixArtSigmaPipeline` accepts this encoder via
its standard `text_encoder=`/`tokenizer=` constructor arguments with no other code changes; a full end-to-end
generation completes and produces a valid, correctly-captioned, non-degenerate image (see below).

**Why PixArt-Sigma (with this adaptation) resolves the architectural confound better than SDXL did.** SDXL is a
UNet denoiser in the Stable Diffusion lineage. PixArt-Sigma's denoiser is a **Diffusion Transformer** (DiT
family, closely related to `facebook/DiT-XL-2-256` already used elsewhere in this project) with no UNet and no
Stable-Diffusion training lineage. Its VAE (also used by SD1.5/SDXL's broader ecosystem via `sdxl-vae`-style
compatible checkpoints in some PixArt variants, but *not* SD1.5's own VAE) and text-conditioning mechanism
(T5-XXL cross-attention, not CLIP) are likewise architecturally distinct from SD1.5's setup. This is the
lineage-independent test the prior phase's SDXL result could not provide.

## Primary hypothesis

> Adding the frozen trajectory stage (`diffpath_curvature`, `path_length`) to the frozen VAE+score stage
> improves held-content real-vs-PixArt-Sigma discrimination.

**Primary comparison:** `AUROC(VAE+score+trajectory) − AUROC(VAE+score)`, content-grouped, identical code path
to every prior generator (`scripts/stage_decomposition_analysis.py`, unmodified).
**Primary success criterion:** paired bootstrap 95% CI for ΔAUROC excludes zero and the point estimate is
positive.
**This is the only primary test. It will not be changed after seeing results.**

## Exact frozen configuration

| parameter | value |
|---|---|
| model / checkpoint | transformer+VAE: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (official); text encoder: `city96/t5-v1_1-xxl-encoder-gguf`, file `t5-v1_1-xxl-encoder-Q5_K_M.gguf`; tokenizer: `google/t5-v1_1-xxl` (slow/`T5Tokenizer`, not the fast tokenizer — the fast tokenizer class fails `PixArtSigmaPipeline`'s internal type check) |
| architecture | Diffusion Transformer (DiT), T5-XXL cross-attention conditioning, dedicated VAE |
| dtype | float16 throughout (transformer, VAE, text encoder) |
| generation resolution | 512×512 (matches this project's existing SD1.5/aMUSEd convention; validated with no artifacts, unlike SDXL's documented sub-native-resolution failure at 512px — PixArt-Sigma is explicitly designed/trained to support multiple resolutions including 512, unlike SDXL's harder 1024px anchor) |
| sampling steps | 20 (PixArt-Sigma's own commonly-used default step count; not reduced further for speed — reducing steps is a quality-affecting change and 20 is already a standard, non-excessive value, not chosen to inflate quality) |
| scheduler | pipeline default (`DPMSolverMultistepScheduler`, as shipped with `PixArtSigmaPipeline`), unmodified |
| guidance scale | 4.5 (PixArt-Sigma's own commonly-recommended default, lower than SD1.5/SDXL's 7.5 because of its different conditioning setup — not tuned by this project) |
| seed policy | `seed = int(sha256(f"pixart_dit:{content_id}")[:8], 16)`, one `torch.Generator("cpu")` per image — identical convention to every other generator in this project |
| captions | the human COCO caption already in `data/content_matched/manifest.csv` for that `content_id` (same column used for real/SD1.5/aMUSEd/SDXL) — **not** a BLIP caption |
| negative prompt | `""` (empty), matching this project's convention for every prior generator |
| image format | PNG, RGB |
| preprocessing / canonicalization | identical to every other generator: `scripts/extract_detector_features.py::load()`, squash-resize to 256px, no crop |
| feature definitions | the frozen 10-feature v2 panel, `src/features/panel_v2.py`, **unchanged** |
| classifier | `StandardScaler + LogisticRegression(C=0.1, class_weight="balanced")`, **unchanged** |
| grouping strategy | content-grouped 5×10 CV, identical code path (`scripts/stage_decomposition_analysis.py`) |
| bootstrap procedure | content-level resampling with replacement, 1000 draws, identical code path |

**Timing measured during engineering validation (informational, not a scientific parameter):** ~62s/image for
text encoding (fixed cost per unique caption, resolution-independent) + ~158s/image for the 20-step
512×512 denoising+decode ≈ **~220s/image**. For 60 images this is a substantial, multi-hour compute commitment
(~3.5–4 hours). Generation will be run in small paced batches with load checks between batches (the same
discipline used for the SDXL run), not one continuous unattended block.

## Engineering validation performed (technical only, not tuned for appearance)

1. `T5EncoderModel.from_pretrained(gguf_file=...)` loads, produces correctly-shaped, non-NaN embeddings — pass.
2. `PixArtSigmaPipeline` accepts the GGUF text encoder via standard constructor arguments (after switching from
   `AutoTokenizer` to the explicit slow `T5Tokenizer`, required because the pipeline's internal type check
   rejects the fast tokenizer class — a technical fix, not a generation-quality change) — pass.
3. Pipeline moves to MPS and runs a full forward generation without error — pass.
4. One validation image (`"a photo of a cat standing on top of an open laptop computer"`, seed 42, 512×512, 20
   steps, guidance 4.5) produced a coherent, correctly-captioned, non-degenerate photograph with no tiling or
   resolution artifacts (unlike SDXL's 512px failure mode) — pass. This single image was inspected **only** for
   the checklist above (loads/completes/correct resolution/no corruption/correct caption) — its visual realism
   was not used to adjust steps, guidance, resolution, or scheduler.
5. Deterministic seeding, output filename/manifest linkage, and resumability follow the exact same pattern as
   `scripts/generate_sdxl.py`, already validated in the prior phase.

**No amendment to generation-quality parameters was required this time** — the frozen configuration above is
exactly what was validated.

### Amendment (execution only, made after 15/60 images, during full-dataset generation)

At 15/60 images, the machine's swap usage reached ~10GB/11GB (near-exhausted) and generation slowed
catastrophically (one image's first denoising step took 917s vs. the validated ~8s/step) — the classic signature
of swap thrashing, not a hang. Investigation found this was **not solely caused by this generation job**: a
separate, unrelated 6+ hour Jupyter session for a different project (`Aptamer-Motifs`) was independently
consuming significant memory on the same machine. Generation was stopped (safely — the script only writes a
completed PNG after each image finishes, so no partial/corrupt output was produced) and `pipe.
enable_model_cpu_offload(device=dev)` was added before resuming. This is a **standard, diffusers-documented
low-memory execution technique** (moves pipeline components between CPU and the accelerator per-call rather than
keeping all of them resident on MPS simultaneously) — it changes memory management only. **It does not alter
steps, guidance scale, resolution, scheduler, seed policy, or captions**, so it does not affect the science under
test; "OOM" is explicitly listed in this document's own decision rules as a valid, non-outcome-driven reason to
amend execution.

**This fix was tried and reverted.** `enable_model_cpu_offload()` assumes a discrete GPU with its own VRAM
separate from host RAM (the standard CUDA memory model it was designed for) — on this machine's Apple Silicon
unified memory architecture, "CPU" and "MPS" share the same physical pool, so there is nothing to offload *to*
that isn't already competing for the same memory. Validating it on one image made swap usage measurably worse
(19GB used, vs. a ~5GB baseline with the fix removed), not better. Reverted to the original `.to(device)`
placement (the configuration that successfully produced the first 15/60 images before the swap incident).
Generation resumed in smaller paced batches with memory checks between batches, coexisting with the user's own
separate, unrelated workload on the same machine, rather than via a change to the pipeline's memory management.

## Dataset

The same 60 COCO content identities already used for real/SD1.5/aMUSEd/SDXL. One PixArt-Sigma image per content
id, same caption, no hand-selection, no regeneration for realism (only for an objective validity failure, none
of which occurred in engineering validation).

## Decision scenarios (fixed in advance, per this phase's task)

* **Outcome A — behaves like SD1.5/SDXL**: positive trajectory ΔAUROC, CI excludes zero, curvature direction
  consistent with SD1.5/SDXL, preferably meaningful transfer. Interpretation: the trajectory-curvature signal is
  no longer plausibly explained only by Stable-Diffusion lineage. **After this result, stop adding generators;
  next phase is robustness + AI-edit scaling around curvature.**
* **Outcome B — behaves like aMUSEd**: ΔAUROC CI crosses zero or is negative, VAE/other stages dominate,
  curvature weak/inconsistent, poor transfer to/from SD1.5/SDXL. Interpretation: the SD1.5/SDXL result is more
  likely a Stable-Diffusion-lineage affinity effect. **Do not rescue with new trajectory features; investigate
  why SD-family images specifically show the effect.**
* **Outcome C — intermediate**: e.g. direction matches but CI crosses zero, or modest ΔAUROC with weak transfer,
  or trajectory adds only in one targeted comparison. Interpretation: architecture modulates the effect.
  **Characterize the discrepancy with one targeted mechanism test, not another generator.**

## What will NOT be done

No new features, no curvature redefinition, no timestep search, no feature selection on PixArt data, no
classifier tuning for PixArt, no concatenation with the old 652-feature representation, no learned detector, no
dataset expansion after seeing the first result, no robustness/AI-edit scaling before this result is in, no
fourth generator, no reinterpretation from standalone AUROC alone, no use of PixArt's own internal
latents/scheduler/denoiser as features (only RGB pixels leave the generator; all forensic measurement comes from
the frozen SD1.5 probe, exactly as for every prior generator).
