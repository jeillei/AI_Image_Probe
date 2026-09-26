# REVIEWER_VALIDATION_RESULTS

Reports both reviewer-validation experiments exactly as preregistered in
`REVIEWER_VALIDATION_PLAN.md`, using its decision rules and reporting negative results plainly if they occur.
This document is updated incrementally: the Experiment 1 section below was written and committed before
Experiment 2 was run.

## Experiment 1 — conditioning ablation

**Result: C1 — conditioning-independent.** The `path_length` real-vs-generated effect for SD1.5 and SDXL is
essentially unchanged whether the SD1.5 probe is conditioned on the image's own correct caption, an empty
("null") caption, or a caption belonging to a different, deterministically deranged content id.

| generator | condition | paired Cohen's d | 95% CI | retention vs. correct |
|---|---|---:|---:|---:|
| sd15 | correct | −0.333 | [−0.551, −0.111] | 1.00 |
| sd15 | null | −0.324 | [−0.549, −0.103] | 0.975 |
| sd15 | shuffled | −0.327 | [−0.553, −0.101] | 0.982 |
| sdxl | correct | −0.358 | [−0.607, −0.105] | 1.00 |
| sdxl | null | −0.367 | [−0.622, −0.117] | 1.028 |
| sdxl | shuffled | −0.376 | [−0.628, −0.121] | 1.052 |
| pixart_dit (secondary) | correct | +0.380 | [0.155, 0.674] | 1.00 |
| pixart_dit (secondary) | null | +0.380 | [0.155, 0.670] | 0.998 |
| pixart_dit (secondary) | shuffled | +0.370 | [0.145, 0.661] | 0.973 |
| amused (secondary) | correct | +1.390 | [1.144, 1.793] | 1.00 |
| amused (secondary) | null | +1.393 | [1.144, 1.797] | 1.002 |
| amused (secondary) | shuffled | +1.389 | [1.142, 1.787] | 0.999 |

Retention ratios for every generator, under both null and shuffled conditioning, land within ±5% of 1.0 — far
inside the ±30% band the plan set as the threshold for "conditioning-independent." Full table (including
univariate AUROC and both incremental comparisons):
`results/reviewer_validation/conditioning_summary.csv`. Figure: `results/reviewer_validation/plots/conditioning_ablation.png`
(three conditioning bars per generator, visually indistinguishable).

**Incremental value.** `VAE+score` vs `VAE+score+path_length` (the more powered of the two preregistered
incremental comparisons) stays positive with a 95% CI excluding zero for SD1.5 and SDXL under **all three**
conditioning modes:

| generator | condition | Δ AUROC (VAE+score → +path_length) | 95% CI |
|---|---|---:|---:|
| sd15 | correct | +0.088 | [0.016, 0.156] |
| sd15 | null | +0.085 | [0.013, 0.154] |
| sd15 | shuffled | +0.082 | [0.011, 0.149] |
| sdxl | correct | +0.060 | [0.026, 0.100] |
| sdxl | null | +0.060 | [0.025, 0.100] |
| sdxl | shuffled | +0.060 | [0.025, 0.100] |

**One honest caveat.** The narrower `VAE` vs `VAE+path_length` comparison (no score features) for **SD1.5
specifically** has a 95% CI that includes zero under null (−0.003, 0.160) and shuffled (−0.004, 0.159)
conditioning, where it excluded zero under correct conditioning (0.001, 0.163) — barely, in both directions. The
point estimate itself barely moves (0.080 → 0.076 → 0.075), so this reads as a comparison sitting right at the
edge of what a bootstrap CI can resolve at n=60 rather than a real effect change; it is reported here rather
than smoothed over. SDXL's equivalent comparison stays clearly positive in all three conditions.

PixArt-Sigma's pattern (small positive `path_length` effect but a **negative**, CI-excluding-zero incremental
contribution — i.e. adding `path_length` on top of VAE+score measurably *hurts* AUROC for this generator) is
unchanged across conditioning modes, consistent with the v1.0 finding that trajectory information does not help
for PixArt-Sigma. aMUSEd's incremental comparison crosses zero in all three conditions (VAE alone already
separates aMUSEd almost perfectly), also unchanged.

**Interpretation.** The `path_length` effect for SD1.5- and SDXL-generated images is not primarily a
caption–image semantic-compatibility artifact: removing the correct caption entirely, or replacing it with a
caption that describes a different photograph, leaves the effect size and (with one narrow, borderline exception
for SD1.5) the incremental predictive value essentially intact. This directly addresses reviewer concern 1 and
strengthens the provenance interpretation of the v1.0 result.

## Experiment 2 — probe swap

Not yet run. Conditioning mode is fixed in advance to mode A (correct caption) per the preregistered plan,
independent of the C1 result above.
