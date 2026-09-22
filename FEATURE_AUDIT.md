# FEATURE_AUDIT — which columns of v1 drive the matched-content results

Scope: a feature-level audit of the frozen 652-column SynthImage Representation v1 on the two content-matched
tasks (real vs SD1.5, real vs aMUSEd; 60 pairs each). No new features were created and no new model family was
introduced — every number here comes from the same `StandardScaler + LogisticRegression(C=0.1, class_weight=
balanced)` pipeline already used throughout the project, plus closed-form univariate statistics. Full tables are
in `results/feature_audit/`; the generating code is `scripts/feature_audit_matched.py` and
`scripts/incremental_information_test_spatialfft.py`. This document is a direct follow-up to `SCIENTIFIC_AUDIT.md`
§6–8 and does not change that document's frozen "unsupported" verdict — it explains *why* the aMUSEd number
(0.994) is so high, and that explanation reinforces rather than reopens the verdict (see §8 below).

## 1. Complete feature list, by family

652 columns fall into 14 families (assignment is by name pattern only, in `src/analysis/families.py`, fixed before
any of this audit ran). Full list: `results/feature_audit/feature_list_by_family.csv`.

| family | n cols | what it measures |
|---|---:|---|
| rich_alignment | 120 | cosine alignment between conditional/unconditional/guidance score and the latent step direction |
| rich_latent_typicality | 90 | per-timestep latent distribution moments + extreme-value/participation-ratio stats |
| rich_score_dynamics | 86 | predicted-score norm curve, consecutive-step change/angle, cross-time cosine, cumulative angle |
| rich_score_moments | 78 | per-timestep predicted (conditional) score distribution moments/quantiles |
| rich_guidance_moments | 78 | per-timestep classifier-free-guidance gap (conditional − unconditional score) distribution moments |
| rich_latent_geometry | 60 | latent-step size, turn angle, acceleration, path straightness |
| rich_score_fft | 36 | 2-D FFT low/mid/high band energy, entropy, spectral centroid of the predicted score map |
| rich_score_spatial | 30 | channel coefficient-of-variation/participation-ratio, horizontal/vertical spatial autocorrelation of the score map |
| rich_crosstime | 30 | pairwise cosine similarity of the score (and latent step) between all timestep pairs |
| legacy_noise | 18 | original 6-step predicted-noise (eps) RMS curve summary |
| legacy_guidance | 14 | original 6-step guidance-gap RMS curve summary |
| endpoint | 6 | final-latent norm/variance/skew/kurtosis/extreme-fraction |
| legacy_geometry | 4 | path length, mean/max speed, mean |acceleration| |
| roundtrip | 2 | pixel and latent round-trip reconstruction MSE |

## 2–3. Per-task coefficients and cross-task comparison

Full per-feature tables (coefficient, rank, univariate AUROC, paired Cohen's d, 500-bootstrap sign stability and
coefficient CI): `results/feature_audit/coef_table_sd15.csv`, `coef_table_amused.csv`.

**Headline cross-task numbers** (`results/feature_audit/cross_task_summary.json`):

| quantity | value |
|---|---|
| Pearson r (652 standardized coefficients) | **0.117** (p = 0.0027) |
| Spearman r | 0.145 (p = 0.0002) |
| coefficient-shuffle null (1000 shuffles) | mean r ≈ 0.00, sd ≈ 0.038 |
| sign agreement (all 652) | **55.5%** |
| sign-agreement null (1000 shuffles) | mean 48.5%, sd 2.0% |
| top-10 overlap | **0 / 10** |
| top-25 overlap | **0 / 25** |
| top-50 overlap | 7 / 50 (6 same sign) |
| features "strong" (top-100 by \|coef\|) in **both** tasks | 19 |
| — of those, same sign | 11 |
| — of those, **opposite sign** | 8 |

Reading: the two coefficient vectors are statistically distinguishable from independent/shuffled vectors (r is ~3
SD above the shuffle-null mean, sign agreement ~3.6 SD above 50%) — there is a small, real, non-zero shared
component. But it is **practically tiny and does not touch the features that matter**: the single strongest
features in each task (top-10, top-25 by |coefficient|) share **zero** members. Even among the 19 features that
are independently "strong" in both tasks, nearly half (8/19) point in **opposite** directions.

## 4. Family attribution of the top features

| | SD1.5 top-30 | aMUSEd top-30 |
|---|---|---|
| dominant families | rich_guidance_moments (11), rich_latent_geometry (7), rich_score_moments (4), rich_latent_typicality (3) | **rich_score_fft (13)**, **rich_score_spatial (7)**, rich_latent_typicality (4), rich_score_dynamics (3) |
| mean univariate AUROC (direction-free) | 0.564 (barely above chance, individually) | **0.824** (strong, individually) |
| mean \|paired Cohen's d\| | 0.193 (small) | **1.066** (very large — conventional threshold for "large" is 0.8) |
| mean bootstrap sign stability | 0.960 | **1.000** |

SD1.5's top features are the classifier-free-guidance gap statistics and latent-path geometry — a weak, diffuse
signal (E75's guidance family, already flagged as the one surviving the confound audit). aMUSEd's top features are
almost entirely the **spatial autocorrelation and 2-D frequency-spectrum structure of the predicted score map** —
a completely different family, and one with individually huge, highly stable effect sizes.

## 5. Clean-residual comparison (confound families excluded)

Using the real-source-identity AUROC already measured per family in `results/audit/real_source_auroc_by_family.csv`
(E60), the "confounded" families (AUROC for AIGC-real-vs-RR-real > 0.60, i.e. everything except guidance and
round-trip) are excluded, leaving 94 columns (`legacy_guidance`, `rich_guidance_moments`, `roundtrip`):

| quantity | full 652 | clean residual (94 cols) |
|---|---:|---:|
| Pearson r | 0.117 | **0.490** |
| Spearman r | 0.145 | 0.406 |
| sign agreement | 55.5% | **59.6%** |
| top-10 overlap | 0/10 | 3/10 |
| top-25 overlap | 0/25 | **11/25 (44%)** |

Restricted to the one part of the representation *not* shown to encode real-source identity, cross-generator
agreement is substantially higher. This is the most defensible candidate for a genuinely shared direction — but
§6/§8 below show it is also exactly the part of the representation that E86 (incremental-information test) found
adds **zero** information beyond a generic VAE+CIFAR-32 baseline on aMUSEd (paired-diff CI [-0.039, 0.067]). A
higher-agreement direction that carries no unique information is not evidence of a working mechanism.

## 6. Is the aMUSEd 0.994 result distributed, concentrated, unstable, or reproducible?

Top-*k*-by-|coefficient| truncation, re-evaluated with the same held-out content-grouped 5×10 CV used everywhere
else in the project (`results/feature_audit/topk_truncation_curve.csv`):

| k | SD1.5 held-out AUROC | aMUSEd held-out AUROC |
|---:|---:|---:|
| 1 | 0.513 | **0.857** |
| 2 | 0.598 | **0.978** |
| 5 | 0.609 | 0.978 |
| 10 | 0.649 | 0.988 |
| 30 | 0.724 | 0.992 |
| 50 | 0.905 | 0.999 |
| 75–150 | 0.915–0.923 (peak) | **1.000** |
| 250 | 0.891 | 1.000 |
| 400 | 0.852 | 0.999 |
| 652 (all) | 0.800 | 0.994 |

**aMUSEd: concentrated and reproducible, not unstable.** A single feature already reaches held-out AUROC 0.857;
two features reach 0.978. Performance is flat near-ceiling from k=50 to k=652 — adding hundreds of extra columns
neither helps nor meaningfully hurts. Combined with §4's near-perfect bootstrap sign stability (1.000 for the top
30) and §2's targeted follow-up (below), this is **not** the signature of high-dimensional overfitting: a
pre-specified 66-column subset (`rich_score_spatial` + `rich_score_fft`, chosen *because* this audit located them,
not tuned to the number) reproduces almost the entire incremental gain over VAE+CIFAR-32 that the full 652-column
fit found in E86:

| generator | baseline (VAE+CIFAR-32) | +spatial+FFT (66 cols) | paired diff | 95% CI | adds info? |
|---|---:|---:|---:|---:|---|
| SD1.5 | 0.916 | 0.876 | −0.037 | [−0.076, −0.004] | **no (hurts)** |
| aMUSEd | 0.910 | **0.976** | **+0.064** | **[0.016, 0.120]** | **yes** |

**SD1.5: diffuse and weaker.** A single feature barely beats chance (0.513); it takes 50 features to reach 0.905,
performance *peaks* around 75–150 features (0.92) and then **degrades** as more (noisier) columns are added,
falling to 0.800 at the full 652 — the classic pattern of a weak, diffuse signal being progressively diluted by
uninformative columns under L2 regularization, not a small number of strong, located features.

## 7. Top-30 feature tables

Full tables: `results/feature_audit/top30_sd15.csv`, `top30_amused.csv`. Abbreviated (5 of 30 shown per task; full
tables have every column specified in the task):

**SD1.5** (feature | family | coef | univariate AUROC | paired d | sign-stability | rank in aMUSEd)
```
rich_guidance_t2_median   rich_guidance_moments  -0.239  0.536  -0.113  0.996  rank 206 (opp. sign)
rich_guidance_t4_mean     rich_guidance_moments   0.212  0.623   0.236  0.980  rank 174 (same sign)
rich_guidance_t0_skew     rich_guidance_moments   0.203  0.518   0.098  0.950  rank 601 (opp. sign)
rich_score_t5_nearzero    rich_score_moments     -0.193  0.520  -0.036  0.976  rank 257 (same sign)
rich_guidance_t5_median   rich_guidance_moments   0.189  0.596   0.187  0.980  rank 315 (same sign)
```

**aMUSEd** (feature | family | coef | univariate AUROC | paired d | sign-stability | rank in SD1.5)
```
rich_score_t1_v_autocorr    rich_score_spatial   0.107  0.863  1.259  1.000  rank 61  (same sign)
rich_score_relchange_middle rich_score_dynamics -0.105  0.872 -1.191  1.000  rank 501 (opp. sign)
rich_score_t3_v_autocorr    rich_score_spatial   0.097  0.891  1.331  1.000  rank 83  (same sign)
rich_score_t2_v_autocorr    rich_score_spatial   0.096  0.871  1.284  1.000  rank 40  (same sign)
rich_score_relchange_t3     rich_score_dynamics -0.096  0.766 -0.731  1.000  rank 481 (opp. sign)
```

None of aMUSEd's top-5 appear in SD1.5's own top-30 (best is rank 40), and vice versa — consistent with §3's 0%
top-10/25 overlap.

## 8. Mechanistic interpretation

**A vs B vs C.** The evidence does not support a single clean answer, but strongly favours **A (genuinely
different mechanisms)** for the dominant, decision-relevant part of each signal, with a small, real, but
information-free **B-like** residual, and **rules out C (unstable overfitting)** as the explanation for aMUSEd:

* **Not C.** aMUSEd's separation is carried by 1–2 features with univariate AUROC 0.86–0.89, paired Cohen's d up
  to 1.3–1.6, and bootstrap sign stability of 1.000. Held-out (content-grouped CV) performance is flat from 50 to
  652 columns. A pre-specified 66-column subset — located by this audit, not tuned to a target number — reproduces
  the incremental-information gain almost exactly. This is a strong, stable, reproducible effect, not high-
  dimensional noise-fitting.
* **Mostly A.** The families responsible are different (guidance-gap statistics and latent-path geometry for
  SD1.5 vs score-map spatial autocorrelation/frequency spectrum for aMUSEd), the strongest individual features are
  entirely disjoint (0% top-10/top-25 overlap), and where features are independently strong in both tasks, 42%
  (8/19) reverse sign. Combined with §Baseline-comparison evidence from the prior phase (VAE-only 0.869, DiT
  Class-B 0.940, CIFAR-32 0.748 — but a colour-only thumbnail baseline flat at 0.675 for both generators),
  aMUSEd's mechanism is best read as a **generic decoder/texture artifact** (VQGAN quantization/texture visible in
  the *spatial and frequency structure* of the SD1.5 UNet's predicted score map — something any sufficiently
  texture-sensitive probe would likely pick up, not something specific to diffusion trajectories or "generative
  compatibility"), while SD1.5's own separation runs on the weaker, diffuse guidance-family response documented
  in E75/E86.
* **A small B-like residual exists but carries no information.** Restricted to the guidance+round-trip columns
  (the only families not shown to encode real-source identity), cross-task agreement is substantially higher
  (r = 0.49, 44% top-25 overlap) — a plausible shared low-level direction. But this is exactly the subset E86
  showed adds nothing beyond VAE+CIFAR-32 on aMUSEd. A direction that generalizes across generators but carries no
  information beyond a generic baseline is not evidence of a *SynthImage-specific* mechanism; it may just be a
  compressed reflection of whatever VAE/CIFAR-32 already measure.

## Summary

* **Strongest shared features:** `rich_score_t1/t2/t3_v_autocorr` (rich_score_spatial), `rich_latent_t0/t1_skew`
  (rich_latent_typicality), `rich_score_consecutive_cos_*`/`rich_score_crosscos_4_5` (score-trajectory
  consistency) — all same-sign in both tasks, but each individually near-chance in SD1.5 (univariate AUROC ≈
  0.5–0.6) and only becomes decisive in aMUSEd.
* **Strongest generator-specific features:** SD1.5 — `rich_guidance_t2_median/t4_mean/t0_skew/t4_skew/t5_median`
  (guidance-gap distribution). aMUSEd — `rich_score_t1/t2/t3_v_autocorr`, `rich_score_relchange_*`,
  `rich_score_t*_fft_high/low/centroid` (score-map spatial/frequency structure).
* **Strongest suspicious/confounded features:** the entire `rich_score_spatial` + `rich_score_fft` block for
  aMUSEd (huge, decoder-artifact-shaped effect sizes, not plausibly a "provenance" signal) and, more generally,
  every family shown in E60 to carry real-source identity (all except guidance/round-trip) — none of those is
  specifically implicated in the top-30 tables here, but their presence in the full 652-column fit is the likely
  reason the full-representation held-out curve *degrades* past k≈150 for SD1.5.
* **Reversed features:** `rich_score_t0_mean/kurtosis`, `rich_score_t1_maxabs`, `rich_score_t3_nearzero` (raw
  score-distribution moments) and `rich_latent_step_auc/middle/t2`, `rich_latent_accel_min` (latent-step-size
  dynamics) — all flip sign between SD1.5 and aMUSEd despite being independently "strong" in both.
* **Status of the hypothesis: remains "unsupported."** This audit does not reopen §7 of `SCIENTIFIC_AUDIT.md` —
  if anything it sharpens the negative conclusion. It replaces an open question ("is the aMUSEd number a fragile
  artifact of fitting 652 columns to 60 pairs?") with a located, mechanistic answer ("no — it is a small,
  extremely stable, reproducible set of score-map texture/frequency features that plausibly detect the VQGAN
  decoder rather than anything about diffusion-trajectory provenance, and they are almost entirely disjoint from
  what separates SD1.5"). A generator-specific decoder-artifact detector is not the generalizing provenance
  mechanism the project set out to test.
