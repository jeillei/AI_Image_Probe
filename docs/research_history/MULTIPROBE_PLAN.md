# MULTIPROBE_PLAN

Goal: test whether any provenance-related structure survives across **independently trained generative priors**, without forcing the 608 SD1.5-specific
features onto models whose dynamics differ.  Status labels: **[implemented]**, **[measured]** (run here), **[estimate]** (not run), **[literature]**.

## 1. Design rules

* **Class A (probe-specific)**: any feature defined on a probe's own tensors (SD1.5 v1 rich 608).  Never concatenated across probes.
* **Class B (cross-probe quantities)**: 30 columns defined by identical code (`src/probes/base.py::cross_probe_features`) on quantities every diffusion-type
  probe has: predicted-noise (eps) RMS curve summaries (18), latent-path geometry (4), endpoint typicality moments (6), round trip (2).
  They are the *legacy* SD1.5 families minus guidance (classifier-free guidance is undefined for an unconditional probe).  For SD1.5 the Class-B columns
  are simply the cached legacy columns -- **no re-extraction is needed to compare SD1.5 with any new probe**.
* Non-eps probes (rectified flow) would replace "eps RMS" with "velocity RMS"; the mapping must be argued per probe, not assumed.
* Every probe uses the **same canonical pixels** as SD1.5 v1 (squash-resize 256) unless a normalized pipeline is being tested, so a probe sees a deterministic function of the same image.
* Same 6-step DDIM-inverse / DDIM-reverse protocol, `DDIMInverseScheduler`, no partial round trips.

## 2. Abstraction  [implemented]

`src/probes/base.py` defines the thin `Probe` protocol (preprocess / measure) and `cross_probe_features`.  Probes return the same result dict as SD1.5
(`forward, reverse, reconstruction, eps_norm, cond_gap[, cond_scores, uncond_scores]`), so `row_features`, `compatibility_features`, `rich_features` run unchanged.
`SD15Probe` is untouched (v1 reproducibility); it is not wrapped because its cached Class-B columns already exist.  `scripts/extract_crossprobe.py`
is the resumable driver (`--probe cifar32|dit|church256|bedroom256|celebahq256`).

## 3. Candidate probes

| probe | family / novelty vs SD1.5 | conditioning | space | params | status | est. cost (16 GB Apple MPS, 256px, 6-step) | expected Class-B quantities | main caveat |
|---|---|---|---|---|---|---|---|---|
| **SD1.5** | LDM UNet, CLIP-L, LAION | BLIP text | KL-f8 latent | 860M UNet | v1 frozen [implemented] | ~11 s/img batch-2 [measured] | all (Class A + B) | probe family = 4/10 dataset generators; SD15 dataset images are self-generated |
| **CIFAR-10 DDPM-32** | pixel UNet, no VAE, no text, CIFAR-10 | none | pixel 32x32 | 36M | **[implemented, measured]** | 0.5 s/img CPU (~1-5 s under contention) | Class B only | sees only a 32px-downsampled image => measures low-frequency/low-res statistics; very different domain.  Value: a *control probe* |
| **DiT-XL/2-256** | latent **transformer**, ImageNet-1k, class-cond (null class 1000 used => no caption) | null class | SD-VAE(ft-EMA) latent 32x32x4 | 675M | **[implemented; smoke-tested on 2 images CPU]** | 74 s/img CPU fp32 under load [measured]; ~2-6 s/img MPS fp16 [estimate] | Class B (eps head = first 4 of 8 channels) | **shares the SD-family VAE** => VAE-compatibility is not independent; ImageNet-only prior |
| DDPM-256 church / bedroom / celebahq | pixel UNet, no VAE, unconditional, narrow domains | none | pixel 256 | ~114M | wired via `--probe` [implemented, not run] | ~1-3 s/img [estimate] | Class B | narrow-domain prior => mostly a resolution/texture compatibility probe; content mismatch |
| ADM / guided-diffusion ImageNet-256 | pixel UNet (the *actual* ADM generator's family) | none/class | pixel 256 | 550M | not implemented (non-diffusers checkpoint, ~2 GB) [literature] | ~5-10 s/img [estimate] | Class B | ADM images then have a *same-weights* prior: useful positive control for affinity, useless as independent evidence for ADM |
| SDXL base | LDM UNet (larger), 2 text enc., new VAE | text (BLIP) | latent | 2.6B UNet | not implemented [literature] | ~30-60 s/img at 512 [estimate]; weights ~7 GB fp16 -> tight on 16 GB | Class B + guidance | same lineage as SD1.5: tests *within-family* transfer only |
| PixArt-alpha/sigma | DiT + T5-XXL | text | SD-VAE latent | 0.6B + 4.7B T5 | not implemented [literature] | heavy T5 => >10 GB | Class B + guidance | shares SD VAE; T5 memory |
| SD3 / FLUX.1-schnell | MMDiT **rectified flow** | text | 16-ch VAE | 2B / 12B | not implemented [literature] | needs >=24 GB GPU or quantization | velocity-RMS analogue | gated weights / memory; needs ODE (not DDIM eps) inversion; different Class-B semantics |

## 4. Recommended first additional probe

**DiT-XL/2-256 (null class).**  It is the only cached candidate that differs from SD1.5 in denoiser architecture (transformer vs UNet), training
set (ImageNet vs LAION), and conditioning mechanism (label-null vs text), it fits in 16 GB, and it is already implemented and smoke-tested.
Its main weakness -- the shared SD-family VAE -- is testable: run the **CIFAR-32** and **DDPM-256 (church)** VAE-free probes on the same images; if a
cross-probe direction appears under DiT only, it is VAE-compatibility.

Do **not** add SDXL first (same lineage; adds cost, not independence).

## 5. Cross-probe experiments (design; A-D from the task)

* **A. Directional replication.**  For each Class-B quantity, compute real-vs-fake standardized effect sizes per probe and per (real source, generator) cell;
  report the sign-agreement rate and rank correlation of effect vectors between probes, against a permutation null.
* **B. Probe agreement.**  Per image, the standardized residual of SD1.5 Class-B quantity q regressed on probe-2's q (regression fit **only on training reals** in each crossed fold);
  agreement score = mean |residual|; test whether fakes differ from reals (one scalar => cannot hide a classifier).
* **C. Generator-probe affinity.**  Matrix rows = source (10 generators + 2 reals) x columns = probes, cell = mean of a probe-normalized compatibility scalar
  (relative round-trip error, standardized against the *real* reference within that probe).  Positive controls: SD15 images under the SD1.5 probe at
  its native 512 resolution (not 256!); ADM images under an ADM-family probe.
* **D. Universal component.**  Fit the 30-feature logistic on probe P (training folds excluding the held real source and held generator), apply its weight vector to probe Q's
  independently standardized features (features are analogous by construction).  AUROC>0.5 in *both* directions and across both real sources = a shared direction.

Implemented for SD1.5 x CIFAR-32 in `scripts/crossprobe_analysis.py` (E73+).  DiT x SD1.5 requires the GPU run below.

## 6. Commands to continue

```bash
# CIFAR-32 (CPU, resumable; ~0.5 s/img alone)
uv run python scripts/extract_crossprobe.py --probe cifar32
# VAE-free 256px pixel probes (downloads ~450 MB each; CPU or MPS)
uv run python scripts/extract_crossprobe.py --probe church256 --device mps --batch 4
# DiT (needs MPS free; ~550 imgs; resumable)
uv run python scripts/extract_crossprobe.py --probe dit --device mps --batch 2
```

## 7. Results so far (measured in this phase)

| probe | real-source AUROC (30 Class-B) | crossed real/fake (confounded corpus) | matched SD1.5-fake vs real (n=60 pairs) |
|---|---|---|---|
| SD1.5 (Class-B legacy columns) | 0.752 [0.691, 0.814] (550) / 0.811 (core-200) | 0.585 (550) / 0.528 (core-200) | 0.697 [0.619, 0.783] |
| CIFAR-10 DDPM-32 | **0.832 [0.784, 0.879]** | 0.507 | **0.816 [0.753, 0.890]** |
| DiT-XL/2-256 null class | 0.784 [0.695, 0.880] | 0.558 (core-200) | 0.741 [0.663, 0.822] |
| church-256 (VAE-free, 256 px) | not run (interrupted; see commands) | - | - |

Recommendation stands (DiT first) but the CIFAR-32 result changes its purpose: the low-resolution VAE-free probe is the **essential baseline** any new probe must beat.  Next probe to add is the VAE-free 256-px pixel DDPM, not SDXL.
