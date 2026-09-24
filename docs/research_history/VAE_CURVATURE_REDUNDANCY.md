# VAE_CURVATURE_REDUNDANCY — does diffpath_curvature carry information beyond static VAE reconstruction?

**Status: exploratory/mechanistic follow-up, not an independent preregistered confirmation.** This analysis was
designed after seeing `DIT_STAGE_DECOMPOSITION.md`'s primary result (PixArt's incremental trajectory test failed
while `diffpath_curvature`'s raw effect size stayed large) specifically to explain that discrepancy. Plan frozen
in `ANALYSIS_PLAN_VAE_CURVATURE_REDUNDANCY.md` before any output below was inspected. No new generation, no new
feature extraction — all 300 already-cached rows (`results/stage_decomposition/panel_features_dit.json`) were
sufficient; everything ran as cached features → local CPU statistics → plots/tables, no HPC needed. Code:
`scripts/vae_curvature_redundancy_analysis.py`, `scripts/vae_curvature_redundancy_plots.py`. Data:
`results/vae_curvature_redundancy/`.

## Raw curvature pattern

Restated from the prior phase, unchanged (paired Cohen's d, real vs that generator):

| generator | d (raw curvature) | 95% CI | direction-free AUROC |
|---|---:|---|---:|
| SD1.5 | −0.542 | [−0.871, −0.273] | 0.710 |
| SDXL | −0.665 | [−0.889, −0.440] | 0.750 |
| PixArt-Sigma DiT | −0.525 | [−0.841, −0.269] | 0.679 |
| aMUSEd | −0.175 (n.s.) | [−0.437, 0.079] | 0.571 |

All three diffusion generators show a similar-magnitude effect; aMUSEd's is weak and not significant.

## VAE-curvature dependence

**Correlations (Part 1, `correlation_tables.csv`)**: pooled Pearson(lpips_ae, curvature) is 0.638 (SD1.5), 0.626
(SDXL), 0.555 (PixArt), 0.319 (aMUSEd). Because "real" is the same 60 images in every comparison, the *real-only*
correlation is necessarily identical across generators (r=0.534) — the informative comparison is
**generated-only**: 0.669 (SD1.5) → 0.471 (SDXL) → 0.400 (PixArt) → 0.119, n.s. (aMUSEd), a monotonically
decreasing gradient. Pooled correlation alone is not interpreted as within-class dependence (per the plan) — this
generated-only gradient already hints that curvature's coupling to the VAE feature *within the fake class*
weakens from SD1.5 through PixArt to aMUSEd, but pooled correlation mixes real/fake separation with true
dependence, which is exactly why cross-fitted residualization (below) is the primary test, not this table.

**Cross-fitted R² (VAE features → curvature), pooled (Part 6, `vae_to_curvature_regression_performance.csv`)**:
SD1.5 0.397, SDXL 0.409, PixArt 0.299, aMUSEd 0.095. VAE features explain roughly 30–41% of curvature's pooled
variance for the three diffusion generators and only ~10% for aMUSEd — but this R² mixes between-class and
within-class variance and, on its own, does **not** distinguish "VAE and curvature separate along the same axis"
from "curvature adds nothing once VAE is known." That distinction requires the residualized effect-size test.

## Residual curvature result

**This is the primary result of this analysis (Part 4/5, `raw_vs_residual_effect_table.csv`,
`plots/02_raw_vs_residual_effect.png`)**: label-blind, content-grouped cross-fitted ridge residualization
(`C ~ Ridge(VAE features)`, alpha=1.0, fit on training folds only), then the same paired Cohen's d test applied
to the held-out residual.

| generator | raw d | residual d (vs VAE) | residual 95% CI | effect retention | residual d (vs S_vae) |
|---|---:|---:|---|---:|---:|
| SD1.5 | −0.542 | **−0.337** | **[−0.629, −0.091]** | **62.2%** | −0.434 |
| SDXL | −0.665 | −0.070 | [−0.306, 0.192] | 10.5% | −0.068 |
| PixArt-Sigma DiT | −0.525 | −0.060 | [−0.334, 0.186] | 11.4% | +0.008 |
| aMUSEd | −0.175 (n.s.) | +0.166 (n.s.) | [−0.094, 0.455] | 94.9%* | +0.143 (n.s.) |

*aMUSEd's retention ratio is not meaningful (raw effect was already non-significant and the residual even flips
sign) — reported per the plan's requirement but not interpreted as a real retained effect.

**The result does not match the working hypothesis as stated.** SD1.5 clearly retains a residual effect (CI
excludes zero, 62% of the raw magnitude survives). But **SDXL's residual curvature collapses just as much as
PixArt's** (10.5% vs 11.4% retention, CIs both crossing zero) — the working hypothesis predicted SD1.5 *and*
SDXL would both retain meaningful residual curvature while only PixArt collapsed; instead only SD1.5 does.
Residualizing against the learned VAE discriminant (`S_vae`, Part 7, label-informed through the training-fold
classifier) gives essentially the same pattern, ruling out "the three raw VAE features happen to be a poor
predictor set" as an alternative explanation.

## Does curvature contain information beyond VAE?

- **SD1.5: yes.** Residual effect remains substantial and statistically distinguishable from zero (d=−0.337, CI
  excludes zero). `diffpath_curvature` carries real information not already present in `lpips_ae`/`pixel_mse_ae`/
  `latent_mse_ae` for this generator.
- **SDXL: no, evidence does not support it.** Residual collapses to d=−0.070, CI crosses zero, only 10.5%
  retention — essentially indistinguishable from PixArt's collapse.
- **PixArt-Sigma DiT: no.** Residual collapses to d=−0.060, CI crosses zero, 11.4% retention.
- **aMUSEd: inconclusive/no.** Raw effect was already weak and non-significant; residualization does not clarify
  further (point estimate flips sign, stays non-significant throughout).

## Proper-scoring-rule result

`incremental_proper_scoring.csv`, `plots/04_incremental_proper_scoring.png`. Curvature **alone** (not the full
trajectory pair — `path_length` deliberately excluded per the plan, since it reverses sign across architectures).
`Δ = extended − baseline`; negative Δ improves log loss/Brier, positive Δ improves AUROC.

**VAE vs VAE+curvature:**

| generator | ΔAUROC | ΔLogLoss | ΔBrier | net read |
|---|---|---|---|---|
| SD1.5 | +0.044 [−0.015, 0.098] n.s. | −0.025 [−0.055, 0.006] n.s. | −0.013 [−0.027, 0.001] n.s. | no significant change on its own (though all three point estimates favor improvement) |
| SDXL | −0.001 [−0.021, 0.018] n.s. | −0.003 [−0.020, 0.014] n.s. | −0.001 [−0.008, 0.007] n.s. | no change |
| **PixArt** | **−0.021 [−0.042, −0.004]** | −0.001 [−0.014, 0.014] n.s. | +0.001 [−0.005, 0.008] n.s. | **AUROC significantly *worse*** with curvature added to VAE alone; log loss/Brier unaffected |
| aMUSEd | ~0.000 n.s. | +0.001 n.s. | ~0.000 n.s. | no change (already near-ceiling) |

**VAE+score vs VAE+score+curvature (closest to the original primary comparison, curvature only):**

| generator | ΔAUROC | ΔLogLoss | ΔBrier | net read |
|---|---|---|---|---|
| **SD1.5** | **+0.050 [0.009, 0.091]** | −0.022 [−0.044, 0.001] n.s. | **−0.011 [−0.021, −0.001]** | curvature adds real information beyond VAE+score, confirmed on 2 of 3 proper scoring rules |
| SDXL | +0.006 [−0.013, 0.026] n.s. | −0.007 [−0.021, 0.007] n.s. | −0.004 [−0.009, 0.003] n.s. | no significant change on any metric |
| **PixArt** | +0.003 [−0.025, 0.030] n.s. | −0.018 [−0.036, 0.004] n.s. | −0.007 [−0.016, 0.003] n.s. | **no significant change on any metric** — the null replicates under log loss/Brier, not just AUROC |
| **aMUSEd** | **+0.007 [0.001, 0.015]** | **−0.013 [−0.023, −0.004]** | **−0.006 [−0.011, −0.002]** | small but statistically real improvement on **all three** metrics, despite a weak/n.s. raw marginal effect |

## Why PixArt's primary incremental test failed

**Not adequately explained by a simple ceiling effect.** SDXL's VAE-alone baseline (AUROC 0.855) is comparable to
PixArt's (0.876) and SDXL shows the *same* magnitude of VAE-redundancy in its curvature signal (10.5% vs 11.4%
retention) — yet SDXL's originally-reported primary test (the full trajectory stage, curvature+path_length
together, from `PIXART_STAGE_DECOMPOSITION.md`) still passed. If ceiling height alone explained PixArt's null
result, SDXL should have failed too; it did not. Ceiling height is not the deciding variable.

The evidence instead points to two compounding factors specific to PixArt: (1) curvature's own real/fake signal
is substantially redundant with VAE-level information for PixArt — matching SDXL's redundancy almost exactly,
so this alone is not PixArt-specific — and, critically, (2) `path_length` (the trajectory stage's other feature,
deliberately excluded from this curvature-only analysis) is the one already flagged in
`DIT_STAGE_DECOMPOSITION.md` as reversing sign for PixArt specifically (d=+0.38, agreeing with aMUSEd's
reversal, disagreeing with both SD1.5 and SDXL). For SDXL, `path_length` keeps the same sign as SD1.5 and can
plausibly supply the complementary information that curvature alone (per this analysis) does not. For PixArt,
`path_length` cannot play that role — it points the wrong way. **The most evidence-supported explanation:
PixArt's trajectory stage fails to add incremental information not because curvature uniquely lost its signal
(SDXL's curvature is equally redundant with VAE), but because PixArt's other trajectory feature does not
compensate the way it does for SDXL.**

## PixArt → aMUSEd transfer diagnostic

`pixart_amused_effect_vector_diagnostic.csv/.json`, `plots/05_pixart_amused_effect_vectors.png`. Across the 10
feature-level paired Cohen's d values: **Pearson r=0.766 (p=0.010), cosine similarity=0.797** between the
pixart_dit and amused effect vectors — a genuinely strong alignment, not an artifact of one dominant feature.
Top-3 contributors to that cosine similarity: `lpips_ae`, `lpips_roundtrip`, `path_length` (the latter
contributes *positively* here specifically because pixart_dit and amused are the two generators where
`path_length` shares the same, positive sign — the same reversal flagged above). Refitting the real-vs-pixart_dit
10-feature classifier and inspecting its standardized coefficients (`pixart_dit_classifier_weights.csv`) shows
the top-3 |weight| features are **`lpips_roundtrip`, `lpips_ae`, `lare_t200`** — VAE-stage and round-trip-stage
features, not trajectory features, dominate the classifier that transfers so well to aMUSEd. **This explains the
0.949 transfer**: the pixart_dit-trained classifier's decision boundary is driven mostly by VAE/round-trip-level
features that behave similarly (large, consistently signed) for both pixart_dit and amused, not by
diffusion-trajectory-specific information that would be expected to differentiate a diffusion generator from a
non-diffusion one.

## Mechanism outcome

**R4 — mixed/nonlinear**, but with a specific, identifiable failure mode rather than an unexplained mess. The
pre-specified R1 (PixArt uniquely collapses, SD1.5/SDXL both retain) does not hold — SDXL collapses just as much
as PixArt. R2 (all three diffusion generators retain independent curvature) does not hold — SDXL and PixArt both
collapse. R3 (all diffusion generators collapse) does not hold either — SD1.5 clearly retains a significant
residual effect. **The linear residualization model is internally consistent and well-behaved (R² 0.30–0.41,
stable across VAE-feature and S_vae residualization) — what fails is the *hypothesized grouping* of generators.**
The data instead groups SD1.5 alone (low VAE-redundancy, high independent curvature) against {SDXL, PixArt}
(both high VAE-redundancy, low independent curvature) — an axis that does not track the UNet-vs-Diffusion-
Transformer distinction the working hypothesis proposed. The more plausible shared property of SDXL and PixArt,
neither tested directly here, is that both are later, higher-capacity models with measurably stronger VAE stages
than SD1.5 (VAE-alone AUROC 0.855/0.876 vs SD1.5's 0.639) — this is a real correlation in the data, not yet a
proven causal account of why VAE-redundancy is higher for both.

## Strongest current scientific claim

The degree to which `diffpath_curvature`'s real-vs-generated signal is redundant with static VAE-reconstruction
information varies substantially across generators, but this variation does not track the UNet-vs-Diffusion-
Transformer architecture distinction the working hypothesis proposed: SD1.5 retains a statistically robust
curvature effect independent of VAE-level features (38% reduction in magnitude, CI still excludes zero), while
both SDXL (UNet) and PixArt-Sigma (DiT) show a nearly identical, much larger collapse (~89–90% loss of effect
magnitude, CI crossing zero for both) despite having different denoiser architectures. PixArt's null primary
incremental-AUROC result is best explained not by a ceiling effect (SDXL shares PixArt's VAE-redundancy pattern
and comparable VAE baseline, yet still passed its own primary test) but by the combination of curvature's
VAE-redundancy — which PixArt shares with SDXL, not a PixArt-specific property — and `path_length`'s
already-documented sign reversal specifically for PixArt, which removes the complementary contribution that
appears to rescue SDXL's combined trajectory-stage result.

## Exactly one next experiment

Not another generator. The single most informative next step, using entirely existing cached data and the
residualization machinery already built and validated here: **apply the identical Part 4/5/7 residualization
pipeline to `path_length` instead of `diffpath_curvature`, across all four generators.** This directly tests
whether SDXL's positive combined-trajectory-stage result is attributable to `path_length` carrying VAE-independent
information that curvature alone does not supply, and whether PixArt's failure to benefit from the trajectory
stage is specifically because `path_length`'s VAE-redundancy (or its reversed real/fake direction) differs from
SDXL's — the open question this analysis raises but does not yet resolve. Zero new generation, zero new feature
extraction required.
