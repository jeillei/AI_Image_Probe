# LITERATURE_FEATURE_PANEL — SynthImage v2 stage-wise feature panel

Every feature below is either (a) directly reproduced from a cited paper's exact published formula, or (b) a
small, explicit, pre-declared adaptation of one, labeled as such. No feature was added because a tensor was
available or because a generator "looked separable." Verification method: WebSearch + WebFetch against the
arXiv abstract/HTML pages listed; formulas below are transcribed from those fetches, not from memory or from
this project's own prior notes. Where a fetch could not retrieve a precise numeric detail (marked below), the
feature is scoped down to what could be verified rather than guessed.

## Literature reviewed

| method | arXiv | what it establishes | reproduced or adapted here |
|---|---|---|---|
| **AEROBLADE** (Ricker et al., CVPR 2024) | [2401.17879](https://arxiv.org/abs/2401.17879) | training-free LDM-image detection via AE reconstruction error | Stage 1, S1.1 — reproduced |
| **DIRE** (Wang et al., ICCV 2023) | [2303.09295](https://arxiv.org/abs/2303.09295) | pixel-space diffusion-inversion round-trip residual, fed to a trained ResNet-50 | Stage 4, S4.1 — adapted |
| **LaRE²** (Luo et al., CVPR 2024) | [2403.17465](https://arxiv.org/pdf/2403.17465) | single-step latent noise-prediction error at fixed t, feeding a trained CNN (EGRE) | Stage 2, S2.1 — the LaRE scalar itself reproduced; EGRE/CNN not reproduced |
| **FakeInversion** (Cazenavette et al., CVPR 2024) | [2406.08603](https://arxiv.org/abs/2406.08603) | trains a ResNet-50 on [image, decoded noise, decoded reconstruction] from text-conditioned SD1.5 DDIM inversion | supporting citation for Stage 4's choice of signal (decoded reconstruction); the trained CNN itself is **not** reproduced (Phase 8 restriction: no learned representation this pass) |
| **DiffPath** (from "OOD Detection with a Single Unconditional Diffusion Model", Heng et al.) | [2405.11881](https://arxiv.org/abs/2405.11881) | rate-of-change / curvature of the score trajectory as an OOD statistic | Stage 3, S3.1 — reproduced (DiffPath-1D), conditioning adapted |
| **Graham et al. 2023**, "Denoising diffusion models for out-of-distribution detection" | [2211.07740](https://arxiv.org/abs/2211.07740) | multi-noise-level reconstruction error as a general diffusion-OOD signal | supporting/contextual citation for Stage 2's general framing; exact formula not independently re-verified (fetch could not retrieve numeric noise-level/metric detail), so **not** claimed as a reproduction of any specific number |

## Exact definitions verified (Phase 0)

### AEROBLADE (Stage 1)
- Score: `ΔAE_i(x) := d(x, D_i(E_i(x)))`, `d` = **LPIPS**, VGG16 backbone.
- The paper evaluates individual LPIPS layers and finds **layer 2 (LPIPS₂) captures the most meaningful
  difference**; that is the layer used here.
- With multiple autoencoders: `ΔMin(x) := min_i ΔAE_i(x)`. We have exactly one autoencoder (SD1.5's), so this
  reduces to a single score — no `min` combination is needed or implemented.
- Fully training-free; no parameter is fit.
- Original resolution: 512×512 (1024×1024 for Midjourney). **Adaptation**: this project's frozen canonicalization
  is 256×256 (established before this phase, `PREREGISTRATION_content_matched_v1.md`); AEROBLADE's own ablations
  show the method is resolution-robust in direction (not verified numerically here), but the absolute score is
  not claimed to match the paper's reported AP at 256px — only the *formula* is reproduced exactly.

### DIRE (Stage 4)
- Inversion: DDIM inversion, `S=20` steps by default, exact recurrence: `x_{t+1}/√α_{t+1} = x_t/√α_t + (√(1/α_{t+1}−1) − √(1/α_t−1)) ε_θ(x_t,t)`.
- Reverse: standard deterministic DDIM step.
- **`DIRE(x_0) := |x_0 − R(I(x_0))|`** — an L1 pixel-space residual **image** (not a scalar), fed to a trained
  ResNet-50 classifier via binary cross-entropy.
- Base reconstructor in the original paper: **ADM (guided-diffusion), pretrained on LSUN-Bedroom**, pixel-space,
  unconditional.
- **Adaptations made here (both required, both documented)**: (1) reconstructor is SD1.5's **latent-space**
  6-step DDIM inversion (this project's frozen probe), not ADM/pixel-space/20-step — a different base model and
  a different number of steps; (2) the residual is summarized as a **scalar mean** (`mean|x_0 − recon|` and
  `mean(x_0−recon)²`), not fed to a trained CNN — consistent with the "no learned representation this pass" rule.
  This is therefore an explicit, small **adaptation** of DIRE's residual definition, not a reproduction of DIRE's
  detector.

### LaRE² (Stage 2)
- `L_ε = ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)` (single-step; x_0 here is the **latent** code).
- `LaRE = (1/e) Σ_i (L_{ε_i} ⊙ L_{ε_i})`, `e=4` noise-ensemble samples, **fixed `t=200`** (on the standard 1000-
  step DDPM schedule).
- This is a **single-step approximation** — it does *not* require the multi-step DDIM inversion the rest of this
  project's protocol uses; it needs only one UNet call per noise sample, starting from the already-encoded clean
  latent `z_0`.
- The paper's downstream EGRE module (spatial/channel attention refinement) and final FC-layer classifier are
  **not reproduced** — only the LaRE quantity itself, reduced from EGRE's spatial map to a single scalar
  (`mean` over channels/spatial locations of `L_ε ⊙ L_ε`) is used here, per the "no CNN, keep the classifier
  simple" project rule.
- **Adaptation**: the fetch could not confirm LaRE²'s original text-conditioning setup for Stable-Diffusion-
  generated inputs. This project computes `ε_θ` conditioned on the same frozen human caption used everywhere
  else here (`PREREGISTRATION_content_matched_v1.md`), not verified to match the original paper's conditioning
  choice — labeled an adaptation rather than assumed identical.

### DiffPath (Stage 3)
- **DiffPath-1D** (the "curvature" statistic): `Σ_t ‖∂_t ε_θ(x_t,t)‖²₂`, with
  `∂ε_θ(x_t,t)/∂t ≈ [ε_θ(x_{t+Δt},t+Δt) − ε_θ(x_t,t)] / Δt`, evaluated on the pairs produced by standard DDIM
  integration. Default `10` NFEs (DDIM steps) in the paper.
- DiffPath-6D (signed power sums of score and its derivative) is **not** reproduced — it requires fitting a GMM
  density on training statistics, which is a learned component out of scope for this confirmatory pass.
- **Adaptations made here**: (1) this project's frozen protocol uses **6** DDIM-inverse steps, not 10 NFEs —
  documented, not changed to match the paper; (2) the original method targets an **unconditional** diffusion
  model for natural-image OOD; this project's `ε_θ` is **text-conditioned** (the same frozen human-caption
  conditioning used throughout). Both are small, explicit, pre-declared deviations from the source paper, kept
  because changing the frozen SD1.5 protocol itself is out of scope for this phase.

## Frozen v2 panel

**Stage 0 (image-space control, reused, not new).** The existing 12-statistic 32×32 colour/contrast thumbnail
baseline (`scripts/thumbnail_baseline_multigen.py`, already used in the aMUSEd decision experiment) is reused
verbatim as the pre-generative-probe control. It is **not** counted against the "10–15 core features" budget
below — it is an existing, already-validated baseline, not a new engineered feature, and its only role here is
to show how much signal exists before any generative probe is invoked.

**Stages 1–4 (10 core scalar features, all literature-anchored):**

| id | stage | name | formula (as implemented) | source | status |
|---|---|---|---|---|---|
| S1.1 | VAE | `lpips_ae` | LPIPS₂(x, D(E(x))), VGG16 | AEROBLADE | **reproduced** |
| S1.2 | VAE | `pixel_mse_ae` | mean((x − D(E(x)))²) | transparent control | control |
| S1.3 | VAE | `latent_mse_ae` | mean((z₀ − E(D(z₀)))²) | motivated extension (latent-space companion to S1.1/S1.2; not itself a cited paper's exact formula) | **adapted/motivated** |
| S2.1 | score | `lare_t200` | mean(LaRE at t=200, e=4) | LaRE² | **reproduced** (conditioning adapted, see above) |
| S2.2 | score | `score_norm_step0` | RMS(ε_θ(z₀, t₀)) at the first (least-noised) inversion step, conditional score | general diffusion score-magnitude framing (Graham et al. 2023 and this project's own prior use); pre-declared single fixed step, not scanned | control / motivated |
| S3.1 | trajectory | `diffpath_curvature` | Σ_t‖ε_θ(z_t,t) − ε_θ(z_{t+1},t+1)‖² over the 6 frozen inversion steps (conditional score) | DiffPath-1D | **reproduced** (NFE count + conditioning adapted, see above) |
| S3.2 | trajectory | `path_length` | Σ_t‖z_{t+1} − z_t‖ over the 6 steps | simple documented control (this project's own prior legacy feature, kept only as a transparent geometry control, not claimed as a literature reproduction) | control |
| S4.1 | round-trip | `pixel_l1_roundtrip` | mean(\|x − R(I(x))\|), SD1.5 6-step latent inversion→reverse | DIRE (adapted) | **adapted** |
| S4.2 | round-trip | `lpips_roundtrip` | LPIPS₂(x, R(I(x))) | same metric as S1.1, applied to the full round trip instead of AE-only — direct apples-to-apples comparison to S1.1 | control (parallels AEROBLADE's metric, not itself from a paper) |
| S4.3 | round-trip | `latent_mse_roundtrip` | mean((z₀ − z_reverse_final)²) | this project's own prior `latent_roundtrip_mse` (v1 legacy), reused as the latent-space DIRE analogue | control |

Pre-declared timesteps/steps (fixed **before** any results were seen, and unchanged from this project's existing
frozen protocol wherever the panel reuses it): t=200 for LaRE (from the LaRE² paper itself); step index 0 (the
first, least-noised inversion step) for `score_norm_step0`; all 6 frozen DDIM-inverse steps for the trajectory
statistics (this project's existing protocol, not re-chosen for this panel).

## Computational cost

S1.1–S1.3 and S4.1–S4.3 reuse the existing frozen 256px/6-step SD1.5 inversion pass (already the project's
standard cost, ~3.2–4.2s/image at batch 2) plus one extra VAE-only encode/decode/re-encode per image (negligible,
no UNet calls). S2.1 requires 4 extra single UNet forward passes per image at a fixed external timestep (t=200,
independent of the 6-step schedule) — no multi-step inversion needed for this feature alone. S2.2 and S3.1–S3.2
require no extra compute: they are read directly from tensors the frozen inversion pass already produces.
