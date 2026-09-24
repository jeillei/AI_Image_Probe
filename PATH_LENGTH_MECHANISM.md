# PATH_LENGTH_MECHANISM — does path_length explain the SDXL-vs-PixArt trajectory-stage discrepancy?

**Status: exploratory/mechanistic stopping-point experiment, not an independent preregistered confirmation.**
Motivated directly by `VAE_CURVATURE_REDUNDANCY.md`'s open question: SDXL passed its original trajectory-stage
incremental test while PixArt did not, yet `diffpath_curvature` is equally VAE-redundant for both. This is the
project's final planned feature-level mechanism decomposition. Plan frozen in
`ANALYSIS_PLAN_PATH_LENGTH_MECHANISM.md` before any result below was inspected. No new generation, no new
feature extraction, no HPC — pure CPU statistics on the already-cached 300-row v2 panel; `C_resid` reused
verbatim from the prior phase, not recomputed. Code: `scripts/path_length_mechanism_analysis.py`,
`scripts/path_length_mechanism_plots.py`. Data: `results/path_length_mechanism/`.

## Raw path-length pattern

`raw_path_length_effect.csv` — confirms the existing sign pattern exactly before any further analysis:

| generator | d | 95% CI | direction-free AUROC | sign |
|---|---:|---|---:|---|
| SD1.5 | −0.333 | [−0.551, −0.111] | 0.577 | negative |
| SDXL | −0.358 | [−0.607, −0.105] | 0.570 | negative |
| PixArt-Sigma DiT | **+0.380** | [0.155, 0.674] | 0.660 | **positive** |
| aMUSEd | **+1.390** | [1.144, 1.793] | 0.898 | **positive** |

## VAE–path-length dependence

`path_length_correlation_tables.csv`, `vae_to_path_length_regression_performance.csv`. Cross-fitted R²
(VAE features → path_length), pooled: SD1.5 0.158, SDXL 0.044, PixArt 0.340, aMUSEd 0.564 — a completely
different pattern from curvature's R² (0.30–0.41 for the three diffusion generators, 0.10 for aMUSEd). Here
**SDXL's path_length is the *least* predictable from VAE features (R²=0.044)** and aMUSEd's is the *most*
(R²=0.564) — already a hint that path_length's redundancy structure inverts curvature's.

## Residual path-length result

`raw_vs_residual_path_length_effect.csv`, `plots/01_raw_vs_residual_path_length.png`:

| generator | raw d | residual d (vs VAE) | residual 95% CI | effect retention |
|---|---:|---:|---|---:|
| SD1.5 | −0.333 | **−0.576** | [−0.799, −0.369] | 173% (strengthened) |
| SDXL | −0.358 | **−0.663** | [−0.923, −0.425] | 186% (strengthened) |
| PixArt-Sigma DiT | +0.380 | **−0.170** (n.s., sign flips) | [−0.426, 0.088] | not meaningful (sign reversal) |
| aMUSEd | +1.390 | +0.074 (n.s.) | [−0.162, 0.324] | 5.3% (collapses) |

**This is the inverse of curvature's pattern.** SD1.5 and SDXL's path_length effect is *not* VAE-redundant — it
gets *stronger*, not weaker, once VAE-predictable variance is removed. PixArt's raw positive effect reverses
sign and becomes non-significant. aMUSEd's very large raw effect almost entirely disappears — confirming
aMUSEd's path_length signal is overwhelmingly a restatement of its VAE-level separation, not independent
information.

## Residual trajectory information

`residual_trajectory_model_results.csv`. AUROC of a classifier fit on residualized trajectory features alone:

| generator | C_resid alone | P_resid alone | C_resid + P_resid |
|---|---:|---:|---:|
| SD1.5 | 0.622 | 0.663 | **0.708** |
| **SDXL** | 0.496 (chance) | **0.695** | 0.682 |
| PixArt | 0.491 (chance) | 0.517 (~chance) | 0.500 (chance) |
| aMUSEd | 0.534 | 0.526 | 0.523 |

For SDXL, `C_resid_alone` is indistinguishable from chance while `P_resid_alone` reaches 0.695 — **essentially
all of SDXL's residual trajectory-stage discriminative power comes from `path_length`, not `diffpath_curvature`**,
and adding curvature on top of path_length does not improve (0.682 vs 0.695, within noise). For PixArt, neither
residual feature works, alone or combined (all ≈ chance). For SD1.5, both residual features contribute and
combine to something significantly better than curvature alone (Δ=+0.084, CI [0.007, 0.164]), though not
significantly better than path_length alone (Δ=+0.044, CI crosses zero) — a genuinely more complementary picture
for SD1.5 specifically.

## Why SDXL passed while PixArt failed

**`plots/03_sdxl_pixart_incremental_contributions.png` is the decisive plot.** The central diagnostic
(`sdxl_pixart_feature_addition_comparison.csv`), on top of VAE alone:

| comparison | SDXL ΔAUROC | SDXL 95% CI | PixArt ΔAUROC | PixArt 95% CI |
|---|---:|---|---:|---|
| +curvature | +0.001 | [−0.021, 0.018] n.s. | **−0.021** | **[−0.042, −0.004]** sig. worse |
| **+path_length** | **+0.061** | **[0.026, 0.099]** sig. better | **−0.012** | **[−0.021, −0.004]** sig. worse |
| +both | +0.060 | [0.020, 0.101] sig. better | **−0.025** | **[−0.048, −0.007]** sig. worse |

...and on top of VAE+score (closest to the original primary comparison):

| comparison | SDXL ΔAUROC | SDXL 95% CI | PixArt ΔAUROC | PixArt 95% CI |
|---|---:|---|---:|---|
| +curvature | +0.006 | [−0.013, 0.026] n.s. | +0.003 | [−0.025, 0.030] n.s. |
| **+path_length** | **+0.052** | **[0.019, 0.090]** sig. better | −0.009 | [−0.024, 0.003] n.s. |
| +both | **+0.053** | **[0.016, 0.092]** sig. better (matches the original preregistered SDXL result exactly) | −0.005 | [−0.029, 0.018] n.s. (matches the original preregistered PixArt null exactly) |

**Both original preregistered numbers are exactly reproduced by this decomposition** (SDXL +0.053
[0.016,0.092]; PixArt −0.005 [−0.029,0.018]), confirming the decomposition is consistent with, not a
reinterpretation of, the established results. `path_length` alone accounts for essentially the entire SDXL gain
(+0.052 of the +0.053 total); curvature's own marginal contribution is not distinguishable from zero at any
point in this table, for either generator. For PixArt, both features individually *hurt* when added to VAE
alone, and neither helps on top of VAE+score.

**Strongest evidence-supported explanation: SDXL's trajectory-stage gain is explained almost entirely by
`path_length`, not by `diffpath_curvature` or by any interaction between them. PixArt shows no comparable
`path_length` benefit — its residual path_length effect is null (after reversing sign from raw), and adding
`path_length` measurably hurts PixArt's ranking on top of VAE alone.** This is not a ceiling effect (both
generators have comparable VAE baselines) and is not attributable to curvature (curvature's own conditional
contribution is null or negative for both generators in every configuration tested).

## Curvature × path-length interaction

`interaction_analysis.csv`. The single preregistered `curvature × path_length` interaction term was added to
both `VAE+curvature+path_length` and `VAE+score+curvature+path_length` for all four generators. **No confidence
interval excludes zero for any generator, on any metric (AUROC, log loss, or Brier).** The interaction adds
nothing measurable anywhere, including for SDXL — its gain is fully accounted for by `path_length`'s own
additive (not interactive) contribution. This directly rules out Outcome P2 (joint/interaction geometry) and P3
(score-conditioned interaction): the story does not change qualitatively whether score features are present or
absent, and no nonlinear combination of the two trajectory features outperforms `path_length` alone.

## PixArt → aMUSEd diagnostic

`pixart_amused_path_length_diagnostic.json`. In the full 10-feature PixArt-trained classifier, `path_length`'s
standardized coefficient is small and not highly stable (mean 0.061, SD 0.051 across 50 fold-fits — roughly
1.2 SDs from zero, not a dominant weight). Controlled ablations of the frozen classifier's feature set, using
the exact same content-grouped transfer procedure that produced the original 0.949 result:

| feature set | PixArt→aMUSEd transfer AUROC |
|---|---:|
| full 10-feature panel | 0.9486 |
| panel minus `path_length` | 0.9495 (essentially unchanged) |
| panel minus entire trajectory stage (curvature + path_length) | **0.9765** (increases) |

**`path_length` does not materially contribute to the PixArt→aMUSEd transfer** — removing it changes almost
nothing, and removing the whole trajectory stage *improves* transfer further. This confirms and sharpens
`VAE_CURVATURE_REDUNDANCY.md`'s earlier finding that the 0.949 transfer is driven by VAE/round-trip features,
not trajectory features; the trajectory stage is not merely uninvolved in this transfer, it is mildly diluting.

## Mechanism outcome

**P1 — path length explains SDXL vs PixArt.** SDXL retains a substantial, VAE-independent, statistically robust
`path_length` effect (residual d=−0.663, CI excludes zero, AUROC 0.695 alone) that fully accounts for its
original trajectory-stage incremental gain. PixArt's `path_length` effect reverses sign under residualization
and contributes nothing (or measurably hurts) in every incremental configuration tested. The interaction term
(P2) and score-conditioning (P3) were both explicitly tested and both ruled out — neither changes the picture.
This is as clean a P1 result as the preregistered decision rule anticipated.

## Strongest current scientific claim

The SD1.5/SDXL trajectory-stage advantage over PixArt-Sigma is attributable specifically to `path_length`, not
to `diffpath_curvature` or to any interaction between the two trajectory features. `path_length`'s real/fake
signal is *not* redundant with static VAE-reconstruction information for SD1.5 or SDXL (it survives — and even
strengthens under — VAE residualization) but is either VAE-redundant or genuinely absent for PixArt-Sigma
(its raw positive effect reverses sign and becomes non-significant once VAE-predictable structure is removed).
`diffpath_curvature`'s own conditional contribution, once path_length and VAE features are accounted for, is not
distinguishable from zero for any of the three diffusion generators in this analysis — its earlier standalone
effect sizes (SD1.5 −0.54, SDXL −0.67, PixArt −0.53) are real univariate effects but do not translate into
additional incremental classifier information beyond what VAE and path_length already provide.

## Should mechanism exploration stop?

**YES.** This experiment produced a clean, decisive, evidence-supported mechanism (P1) that exactly reproduces
both prior generators' preregistered primary-test numbers under decomposition, rules out both alternative
explanations (interaction, score-conditioning) with confidence intervals that uniformly include zero, and
resolves the PixArt→aMUSEd transfer diagnostic as a VAE/round-trip artifact uninvolved with trajectory features.
There is no remaining open mechanistic question at the feature level that further decomposition would resolve;
continuing to search within the existing 10-feature panel would be feature-level fishing, which the task
explicitly prohibits.

## Exactly one next experiment

Not another generator, not further feature-level decomposition. The trajectory mechanism is now sufficiently
resolved. The single next experiment should be the **final robustness / controlled-AI-edit validation phase**:
test whether `path_length`'s newly-established role (the primary carrier of SD1.5/SDXL's trajectory-stage
advantage) survives the same realistic-transformation and AI-edit-strength pilots already used for the broader
panel (`STAGE_DECOMPOSITION_RESULTS.md` Phases 6–7), specifically checking whether `path_length`'s
discriminative power and sign are stable under JPEG/blur/resize/crop and under partial-AI-edit strength, for
SD1.5 and SDXL specifically (PixArt/aMUSEd included as the established negative comparisons, not as further
mechanism search).
