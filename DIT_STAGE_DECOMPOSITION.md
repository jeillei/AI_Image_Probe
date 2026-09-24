# DIT_STAGE_DECOMPOSITION — architecture-disambiguation decision experiment (genuine PixArt-Sigma DiT)

**Read `PREREGISTRATION_DIT_GENERATOR_V1.md` first.** This is the fourth generator in the SynthImage stage-panel
series, and the first one outside the Stable Diffusion UNet lineage: a genuine PixArt-Sigma **Diffusion
Transformer** (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`, official transformer + VAE, unmodified), conditioned by
a GGUF-quantized T5-XXL text encoder (disclosed adaptation, not a substitution of the denoiser under test). All
60 images were generated with the exact frozen configuration in the preregistration (512×512, 20 steps, guidance
4.5, `DPMSolverMultistepScheduler`, seed=`hash("pixart_dit:"+content_id)`, same human COCO captions used by every
prior generator). Generation ran partly on this machine (15/60 images, validated) and partly on an HPC GPU
cluster (remaining 45/60, after fixing eight execution-environment bugs unrelated to the frozen protocol —
missing system packages, `HF_HUB_OFFLINE` cache-resolution quirks, and diffusers-version compatibility issues;
none of these touched steps/guidance/resolution/scheduler/seed/captions). Generator label used everywhere:
`pixart_dit`, never conflated with the earlier SDXL substitute run (`sdxl`).

## Dataset status

60/60 `pixart_dit` images present, all 512×512, none corrupted (PIL `verify()` on every file). QC: a random
9-image contact sheet (`results/stage_decomposition/pixart_dit_qc.png`) shows coherent, correctly-captioned
photographs with no tiling or resolution artifacts. Features extracted with the **frozen, unmodified** v2
10-feature panel (`src/features/panel_v2.py`) via the same two-pass extractor already used for every prior
generator — 0 NaN across all 300 rows (real/sd15/amused/sdxl/pixart_dit, 60 each), all assertions passed.

## Primary result (preregistered)

> **AUROC(VAE+score+trajectory) − AUROC(VAE+score) = −0.005, 95% paired content-bootstrap CI [−0.029, 0.018].**
> **The CI includes zero (and the point estimate is slightly negative). Adding the frozen trajectory stage does
> NOT improve real-vs-pixart_dit discrimination beyond VAE+score.**

| stage set | AUROC |
|---|---:|
| VAE | 0.876 |
| VAE + score | 0.894 |
| **VAE + score + trajectory** | **0.891** |
| VAE + score + trajectory + round-trip | 0.913 |

Full incremental table (`results/stage_decomposition/dit_generator/incremental_information_dit.csv`):

| comparison | ΔAUROC | 95% CI | adds information? |
|---|---:|---|---|
| VAE → VAE+score | +0.017 | [−0.003, 0.040] | no (CI crosses zero) |
| **VAE+score → VAE+score+trajectory (primary)** | **−0.005** | **[−0.029, 0.018]** | **no** |
| VAE+score+trajectory → +round-trip | +0.022 | [−0.007, 0.053] | no |
| targeted: VAE vs VAE+trajectory | **−0.025** | **[−0.048, −0.007]** | **no — significantly negative** |
| targeted: VAE vs VAE+round-trip | +0.013 | [−0.018, 0.046] | no |
| targeted: score vs score+trajectory | **+0.132** | **[0.032, 0.232]** | **yes** |

This is a genuinely mixed pattern, not a clean pass or fail: trajectory adds real, CI-excludes-zero information on
top of the score stage alone, but **not** on top of VAE+score (the primary test), and actively **hurts** when
added directly on top of VAE alone (CI excludes zero on the negative side). No result was dropped, re-run, or
reinterpreted after seeing this — all six comparisons in the preregistered incremental-information table are
reported above exactly as computed.

## Curvature result

`diffpath_curvature`: paired Cohen's d = **−0.525** [−0.841, −0.269], univariate AUROC 0.679 (direction-free).
This is **closely in line with both SD1.5 (d=−0.542) and SDXL (d=−0.665)** — the curvature feature itself remains
a strong, consistently-signed effect for the genuine DiT generator, unlike aMUSEd's non-significant d=−0.175
(`results/stage_decomposition/dit_generator/plots/03_diffpath_curvature_four_generators.png`). Sign agrees with
SD1.5 and SDXL both.

`path_length` (the other trajectory feature): d=**+0.380** [0.155, 0.674] — **positive**, disagreeing with SD1.5
(−0.333) and SDXL (−0.358), and instead **agreeing in sign with aMUSEd's large reversal (+1.390)**. This is the
third generator (after aMUSEd) to show this specific sign flip on `path_length`, and the first UNet-vs-DiT case
where the DiT result patterns with the non-diffusion negative control rather than with the other diffusion
models.

## VAE result

VAE alone reaches AUROC **0.876** — close to SDXL's 0.855 and well above SD1.5's 0.639, though still below
aMUSEd's 0.981. Like SDXL (and unlike aMUSEd), this does not fully saturate the available signal — the
+round-trip stage still adds numerically (+0.022 cumulative from VAE+score+trajectory, though CI crosses zero) —
but unlike SDXL, the +trajectory stage does not clear the ceiling; it adds nothing on top of VAE+score and
actively subtracts on top of VAE alone.

## Feature-level table (all 10 features, four generators)

`results/stage_decomposition/dit_generator/four_generator_feature_table.csv`:

| feature | stage | d (SD1.5) | d (aMUSEd) | d (SDXL) | d (pixart_dit) | sign SD1.5↔DiT | sign SDXL↔DiT | sign aMUSEd↔DiT |
|---|---|---:|---:|---:|---:|---|---|---|
| lpips_ae | vae | −0.38 | −2.20 | −1.15 | −1.13 | agree | agree | agree |
| pixel_mse_ae | vae | −0.27 | −0.98 | −0.65 | +0.05 (n.s.) | **disagree** | **disagree** | **disagree** |
| latent_mse_ae | vae | +0.19 (n.s.) | −0.47 | +0.34 | −0.04 (n.s.) | **disagree** | **disagree** | agree |
| lare_t200 | score | −0.33 | −1.09 | −0.55 | +0.09 (n.s.) | **disagree** | **disagree** | **disagree** |
| score_norm_step0 | score | +0.41 | +1.14 | +0.76 | +0.30 | agree | agree | agree |
| **diffpath_curvature** | trajectory | **−0.54** | −0.18 (n.s.) | **−0.67** | **−0.53** | agree | agree | agree |
| **path_length** | trajectory | −0.33 | +1.39 | −0.36 | **+0.38** | **disagree** | **disagree** | agree |
| pixel_l1_roundtrip | roundtrip | −0.16 (n.s.) | −0.66 | +0.09 (n.s.) | +0.04 (n.s.) | disagree(noise) | agree(noise) | disagree(noise) |
| lpips_roundtrip | roundtrip | −0.60 | −1.61 | −0.61 | −1.05 | agree | agree | agree |
| latent_mse_roundtrip | roundtrip | −0.10 (n.s.) | +0.23 (n.s.) | +0.24 (n.s.) | +0.01 (n.s.) | disagree(noise) | agree(noise) | agree(noise) |

**4/10 features sign-agree SD1.5↔DiT; 6/10 sign-agree SDXL↔DiT** (down from 9/10 for SD1.5↔SDXL) — computed
directly from `four_generator_feature_table.csv`'s `sign_*_dit_agree` columns. `lare_t200`, `latent_mse_ae`,
`path_length`, and `pixel_mse_ae` reverse sign against **both** SD1.5 and SDXL; `latent_mse_roundtrip` and
`pixel_l1_roundtrip` reverse only against SD1.5 (both are noise-level, CI-crosses-zero features for every
generator, so these two are not read as meaningful reversals). `diffpath_curvature`, `score_norm_step0`, and
`lpips_ae`/`lpips_roundtrip` are the four features that stay large and sign-consistent across all four
generators. This is a substantially weaker sign-agreement pattern than the SD1.5↔SDXL comparison (9/10), though
still well above the ~50% agreement rate chance would predict if the effects were unrelated.

## Cross-generator transfer

`results/stage_decomposition/dit_generator/transfer_results.json`, full pairwise matrix, content-grouped, same
procedure as every prior transfer test (`results/stage_decomposition/dit_generator/plots/04_transfer_matrix_four_generators.png`):

| direction | AUROC |
|---|---:|
| SD1.5 → pixart_dit | 0.729 |
| pixart_dit → SD1.5 | 0.666 |
| SDXL → pixart_dit | 0.768 |
| pixart_dit → SDXL | 0.804 |
| aMUSEd → pixart_dit | 0.735 |
| pixart_dit → aMUSEd | **0.949** |
| SD1.5 → SDXL (reference) | 0.898 |
| SD1.5 → aMUSEd (reference) | 0.581 |

Transfer into and out of `pixart_dit` (0.67–0.80 for the three diffusion-family directions) is **weaker than the
SD1.5↔SDXL transfer (0.77–0.90)** but **stronger than the SD1.5↔aMUSEd transfer (0.58–0.60)** — DiT sits between
the two prior results, consistent with a real but partial architectural shift. The one striking outlier is
**pixart_dit → aMUSEd = 0.949**, the strongest transfer direction in the entire four-generator matrix, including
directions between two UNet models — a classifier trained on real-vs-DiT transfers almost perfectly to
real-vs-aMUSEd, despite aMUSEd being the non-diffusion negative control. This was not predicted by any prior
phase and is flagged as-is, not explained away.

## Which scenario occurred

**Scenario C — intermediate**, per the preregistered decision rule's own example: *"trajectory adds only in one
targeted comparison."* The primary test (VAE+score+trajectory vs VAE+score) fails — CI includes zero, point
estimate slightly negative — which on its own would suggest Scenario B. But `diffpath_curvature` itself remains
strong and consistently signed with both SD1.5 and SDXL (ruling out "curvature weak/inconsistent," Scenario B's
other criterion), and trajectory *does* add significant information in the `score+trajectory vs score` targeted
comparison. Scenario B's full criteria are not met (curvature is not weak or inconsistent) and Scenario A's are
not met (the primary CI does not exclude zero). This is Scenario C as written in the preregistration, not a
forced middle ground.

## Status of the active hypothesis: **modulated by architecture, not falsified**

The hypothesis under test — "diffusion-generated images may contain trajectory-level forensic information
beyond static VAE reconstruction" — is **not cleanly confirmed or refuted** by this generator. The trajectory
signal that replicated cleanly across two UNet-based diffusion models (SD1.5, SDXL) does **not** replicate as
cleanly on a genuine Diffusion Transformer: the specific *incremental* claim (trajectory adds beyond VAE+score)
fails here, even though the underlying `diffpath_curvature` feature's raw effect size is essentially unchanged.
The most parsimonious reading: `diffpath_curvature`'s standalone signal is not SD-lineage-specific (it survives
the architecture change with almost the same effect size), but **how much it adds on top of the other stages is
lineage- or ceiling-dependent** — VAE+score alone already reaches 0.894 for pixart_dit, leaving little headroom,
and unlike SDXL's case this headroom is not recovered by trajectory (it *is* partially recovered by round-trip,
+0.022, though that CI also crosses zero).

## Biggest remaining confound

**Ceiling effects across generators are not uniform, and this experiment's four generators now span three
distinct VAE-alone baselines (0.639 / 0.855 / 0.876 / 0.981) that are not otherwise matched for image quality,
compression artifacts, or realism.** The trajectory stage's marginal contribution appears to shrink as the VAE
baseline rises, but pixart_dit's failure to show a clean incremental gain despite a VAE baseline similar to
SDXL's (0.876 vs 0.855) — where SDXL *did* show a clean gain — suggests architecture is not simply proxying for
ceiling height. Disentangling "ceiling effect" from "genuine architecture-dependent trajectory information" is
the open question this result raises and the prior three-generator phase could not.

## Exactly one recommended next experiment

Per the preregistration's own Scenario C instruction: **"Characterize the discrepancy with one targeted
mechanism test, not another generator."** The most informative single test is a **direct ceiling-controlled
comparison**: select or construct a subset of real/pixart_dit pairs where VAE-alone AUROC is deliberately matched
to SD1.5's or SDXL's operating range (e.g. stratifying on VAE-only prediction confidence), and re-run the primary
incremental test only within that matched subset. If trajectory recovers its incremental contribution once
ceiling is controlled for, the ceiling-effect explanation is confirmed and no further generators are needed. If
it still fails to add information in the ceiling-matched subset, that is evidence the DiT architecture itself —
not merely its stronger VAE — changes what the trajectory stage captures, which would be a materially different
and more architecture-specific finding worth its own follow-up.
