# STAGE_DECOMPOSITION_RESULTS — where forensic information enters the SD1.5 pipeline

Scope: the frozen v2 literature-anchored panel (`LITERATURE_FEATURE_PANEL.md`, 10 core features across 4 stages
+ a reused Stage-0 thumbnail control), evaluated on the existing 60 content-matched triplets (real COCO / SD1.5
/ aMUSEd), plus a small robustness pilot (Phase 6) and an exploratory AI-edit-strength pilot (Phase 7). Content
is the grouping unit throughout — 180 images are never treated as 180 independent observations. Classifier:
`StandardScaler + LogisticRegression(C=0.1, class_weight=balanced)` everywhere, content-grouped 5×10 CV, content
bootstrap CIs. Code: `scripts/stage_decomposition_analysis.py` (Phases 3–5), `scripts/robustness_pilot_analysis.py`
(Phase 6), `scripts/ai_edit_pilot_analysis.py` (Phase 7). Data: `results/stage_decomposition/`.

## Phase 3 — individual feature and stage-level results

Full table: `results/stage_decomposition/univariate_feature_results.csv`. Plot: `plots/01_effect_size_by_stage.png`.

| feature | stage | d (SD1.5) | AUROC (SD1.5) | d (aMUSEd) | AUROC (aMUSEd) |
|---|---|---:|---:|---:|---:|
| lpips_ae | vae | −0.38 [−0.69,−0.12] | 0.65 | **−2.20** [−2.85,−1.81] | **0.98** |
| pixel_mse_ae | vae | −0.27 | 0.59 | −0.98 | 0.84 |
| latent_mse_ae | vae | +0.19 (n.s.) | 0.56 | −0.47 | 0.75 |
| lare_t200 | score | −0.33 | 0.64 | −1.09 | 0.83 |
| score_norm_step0 | score | +0.41 | 0.63 | +1.14 | 0.86 |
| diffpath_curvature | trajectory | **−0.54** | **0.71** | −0.18 (n.s.) | 0.57 |
| path_length | trajectory | −0.33 | 0.58 | **+1.39** | **0.90** |
| pixel_l1_roundtrip | roundtrip | −0.16 (n.s.) | 0.59 | −0.66 | 0.71 |
| lpips_roundtrip | roundtrip | **−0.60** | **0.71** | −1.61 | 0.93 |
| latent_mse_roundtrip | roundtrip | −0.10 (n.s.) | 0.58 | +0.23 (n.s.) | 0.59 |

Stage-level (all features of a stage together, `stage_level_results.csv`, plot `02_stage_auroc.png`):

| stage | SD1.5 AUROC [95% CI] | aMUSEd AUROC [95% CI] |
|---|---:|---:|
| vae | 0.639 [0.543, 0.736] | **0.981 [0.959, 0.997]** |
| score | 0.621 [0.529, 0.710] | 0.845 [0.775, 0.911] |
| **trajectory** | **0.753 [0.666, 0.837]** | 0.892 [0.833, 0.946] |
| roundtrip | 0.696 [0.598, 0.782] | 0.955 [0.914, 0.988] |

For SD1.5, **trajectory is the single strongest stage** — stronger than VAE, score, or round-trip alone. For
aMUSEd, **VAE alone is already close to ceiling** and no other stage beats it.

## Phase 4 — incremental information (the central experiment)

Full table: `results/stage_decomposition/incremental_information.csv`. Plot: `plots/03_incremental_gain.png`.

**SD1.5** (cumulative stage sets, paired-bootstrap CI on the AUROC difference, same folds shared by both models):

| comparison | ΔAUROC | 95% CI | adds information? |
|---|---:|---|---|
| VAE → VAE+score | −0.015 | [−0.053, 0.019] | **no** |
| VAE+score → VAE+score+trajectory | **+0.125** | **[0.056, 0.204]** | **yes** |
| VAE+score+trajectory → +round-trip | +0.040 | [0.003, 0.080] | yes (small) |
| targeted: VAE vs VAE+trajectory | +0.116 | [0.034, 0.205] | **yes** |
| targeted: VAE vs VAE+round-trip | +0.046 | [−0.006, 0.102] | no |
| targeted: score vs score+trajectory | +0.123 | [0.053, 0.197] | **yes** |

**aMUSEd** (same design):

| comparison | ΔAUROC | 95% CI | adds information? |
|---|---:|---|---|
| VAE → VAE+score | −0.007 | [−0.020, 0.006] | no |
| VAE+score → VAE+score+trajectory | +0.004 | [−0.006, 0.015] | no |
| VAE+score+trajectory → +round-trip | +0.000 | [−0.009, 0.009] | no |
| targeted: VAE vs VAE+trajectory | −0.008 | [−0.021, 0.002] | no |
| targeted: VAE vs VAE+round-trip | +0.001 | [−0.015, 0.017] | no |
| targeted: score vs score+trajectory | +0.071 | [0.024, 0.126] | yes (but score alone is already the *weakest* stage here — this adds over a stage that was never competitive with VAE) |

**Does trajectory add anything beyond the VAE? Yes for SD1.5 (a genuine, statistically robust +0.125 AUROC gain,
entirely attributable to `diffpath_curvature`); no for aMUSEd (every comparison against the VAE-alone baseline
has a CI crossing zero).**

**Does round-trip reconstruction add anything beyond the VAE? Marginally for SD1.5 (targeted VAE-vs-+round-trip
CI [−0.006, 0.102] does *not* clear zero on its own, though the sequential VAE+score+trajectory→+round-trip step
does, barely); no for aMUSEd.**

## Phase 5 — cross-generator direction

Full table: `results/stage_decomposition/cross_generator_direction.csv`, `cross_generator_summary.json`. Plot:
`plots/04_cross_generator_direction.png`.

**Sign agreement: 7/10 features (70%).** The 6 large, robust features (`lpips_ae`, `pixel_mse_ae`, `lare_t200`,
`score_norm_step0`, `lpips_roundtrip`, `pixel_l1_roundtrip`) all agree in sign and are the ones driving most of
the stage-level AUROC in both generators. Three reverse: `latent_mse_ae` and `latent_mse_roundtrip` (both weak,
non-significant for SD1.5 — marginal features flipping is unremarkable) and **`path_length`** (SD1.5 d=−0.33 vs
aMUSEd d=+1.39 — large in *both* directions). Per the task's own rule, `path_length` is treated as
**generator-specific**, not as evidence of a shared mechanism; no attempt was made to "fix" its direction.

**Transfer**: a classifier trained on the full 10-feature panel for one generator, applied content-grouped to the
other, reaches AUROC **0.581** (SD1.5→aMUSEd) and **0.597** (aMUSEd→SD1.5) — barely above chance, despite 70%
feature-level sign agreement. The qualitative direction is often shared, but the *magnitude* differs so much
(aMUSEd's stages are near-ceiling, SD1.5's are moderate) that a quantitative decision boundary learned on one
does not transfer to the other.

## Phase 6 — small robustness pilot (10 content ids × 3 generators × 5 conditions: clean, JPEG-50,
0.5× resize round-trip, blur, center-crop-0.8)

Full tables: `results/stage_decomposition/robustness_pilot_*.csv`. Plot: `plots/05_robustness_pilot.png`.
n=10/generator/condition — small pilot, CIs are wide; reported as such, not smoothed over.

* **aMUSEd's VAE-only signal is essentially transformation-invariant**: stage-level AUROC stays at 0.97–1.00
  across every one of the four transformations (vs. 1.00 clean on this 10-content subset) — JPEG, blur, resize
  and cropping do not meaningfully erode it.
* **SD1.5's full-panel signal (which needs trajectory to be competitive) survives transformation qualitatively**:
  full-panel AUROC on this 10-content pilot ranges 0.70–0.78 across all four transformed conditions versus 0.77
  clean — no collapse, and consistently well above VAE-alone at every condition (0.43–0.66).
* **Every one of the 10 features' image-level *ranking* is highly stable under transformation** — before/after
  Spearman correlation (clean value vs. transformed value, same image) ranges **0.66–1.00** across all
  feature/condition pairs (`robustness_pilot_before_after_correlation.csv`), lowest for `center_crop` (which
  changes framing, so some rank shuffling is expected) and highest for `resize`/`jpeg`.
* No feature or stage collapsed to chance under any tested condition, at this small n.

## Phase 7 — AI-edit strength pilot (exploratory; n=8 content ids)

`results/stage_decomposition/ai_edit_pilot_monotonicity.csv`. Plot: `plots/06_ai_edit_strength.png`, QC strip
`plots/ai_edit_qc.png`.

**Feasibility note (required by the task):** the ideal localized-edit continuum (object edit / background
replacement / masked inpainting) was **not** implemented — it requires an inpainting-specific SD checkpoint not
already present among this project's validated, locally-cached weights, and downloading + validating a new
~4–5GB model was judged out of proportion to a feasibility pilot. The **feasible substitute** used instead is an
**img2img strength continuum** (partial forward-noise to an intermediate timestep, then denoise), built entirely
from the SD1.5 checkpoint already in use everywhere else in this project. It gives a genuine, continuous
"degree of generative intervention" knob (strength ∈ {0 = real, 0.3, 0.6, 0.9}) but is a **global**, not
localized, edit — disclosed explicitly as a scope limitation, not hidden.

6 of 10 features show a statistically significant monotonic relationship with edit strength (Spearman, pooled
across the 32 points, n=8 content ids × 4 strengths):

| feature | pooled Spearman r | p | monotonic in how many of 8 content ids |
|---|---:|---:|---:|
| diffpath_curvature | −0.72 | <0.001 | 8/8 (all negative) |
| lpips_roundtrip | −0.71 | <0.001 | 8/8 |
| score_norm_step0 | +0.62 | <0.001 | 8/8 |
| lare_t200 | −0.55 | 0.001 | 8/8 |
| lpips_ae | −0.53 | 0.002 | 8/8 |
| path_length | +0.36 | 0.045 | 8/8 |
| pixel_mse_ae | −0.34 | 0.055 (borderline) | — |
| pixel_l1_roundtrip, latent_mse_ae, latent_mse_roundtrip | \|r\|<0.22 | n.s. | — |

**Every one of the 6 significant trends moves in the direction consistent with (and continuously extending) that
same feature's established binary real-vs-fake direction from Phase 3** — e.g. `diffpath_curvature` is lower for
fakes than reals (both generators) and gets monotonically lower still as edit strength increases;
`score_norm_step0` is higher for fakes and increases monotonically with strength. `path_length` increases with
strength, consistent with its *aMUSEd* direction specifically (not its SD1.5 direction) — expected, given Phase
5 already flagged `path_length` as the one feature with a genuine generator-specific sign reversal; img2img
editing is itself closer to a full-regeneration process than to SD1.5's own DDIM-inversion trajectory. This is
**exploratory, convergent-validity evidence, not an "AI percentage" claim** — n=8, no classifier fit, no held-out
test, global (not localized) edits only.

## Limitations

* Stage-0 image-space control was reused from the prior phase's 12-statistic thumbnail baseline, not recomputed;
  it establishes only that *some* real/fake signal exists before any generative probe (already known from prior
  phases) and was not re-run through the full Phase 3–5 machinery here to keep this pass's scope to the
  literature-anchored 10-feature panel.
* n=60 triplets (Phase 3–5), n=10 (Phase 6), n=8 (Phase 7) — all confidence intervals reflect this; none of these
  numbers should be read as population-level detector performance.
* The panel deliberately omits several published methods' most powerful components (LaRE²'s EGRE+CNN,
  FakeInversion's trained ResNet-50, DIRE's trained CNN on the full residual map) because this phase's rule is
  "no learned representation" — the *scalar reductions* used here are necessarily weaker than what those papers
  report, and the numbers above should not be compared directly to those papers' published AUROCs.
* `path_length`'s sign reversal and the AI-edit pilot's global-only edits are both real, disclosed limitations of
  generalizability, not resolved here.
* Phase 6/7 pilots reuse the *same* frozen protocol version and the *same* 60-triplet-derived captions; they are
  not independent held-out data in the sense of a fresh content draw.
