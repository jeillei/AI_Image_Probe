# ANALYSIS_PLAN_VAE_CURVATURE_REDUNDANCY — frozen before results are seen

**Status: this is a targeted exploratory/mechanistic follow-up motivated by an already-observed discrepancy
(DIT_STAGE_DECOMPOSITION.md's primary result). It is NOT an independent preregistered confirmation** — it was
designed after seeing E113's result and exists specifically to explain it. Every analysis below is recorded
before its output is inspected; any analysis added after seeing a result will be labeled post hoc in
`VAE_CURVATURE_REDUNDANCY.md`, not silently folded in here.

## Question

Does `diffpath_curvature` encode a diffusion-associated signal across generator architectures, but overlap with
VAE-level information to different degrees depending on architecture? This tests **information redundancy**, not
detector performance.

## Working hypothesis (not assumed correct)

`diffpath_curvature` has a similarly directed generated-image effect for SD1.5, SDXL and PixArt-Sigma DiT, but
the component of curvature that is independent of VAE information is substantially larger for SD1.5/SDXL than
for PixArt. aMUSEd is a non-diffusion comparison. A null result (no architecture-dependent redundancy pattern) is
a useful, reportable outcome.

## Data

Reuse the existing 60 matched content identities, all four generators already extracted with the frozen v2 panel
(`results/stage_decomposition/panel_features_dit.json`, 300 rows: real=60, sd15=60, sdxl=60, pixart_dit=60,
amused=60, 10 features each, 0 NaN — verified before writing this plan). **No new feature extraction, no new
generation.** Compute-sufficiency check performed before writing this plan: all required columns
(`lpips_ae`, `pixel_mse_ae`, `latent_mse_ae`, `diffpath_curvature`, `lare_t200`, `score_norm_step0`, plus the
label/generator/content_id columns) are present for every generator — the entire analysis below runs as
**cached features → local CPU statistics → plots/tables**, no HPC needed. This will be re-confirmed once at
script start (assertion on row counts/NaN) before any numeric result is produced.

`C := diffpath_curvature`. `V := [lpips_ae, pixel_mse_ae, latent_mse_ae]` (the three VAE-stage features).
`GENS := [sd15, sdxl, pixart_dit, amused]`, each compared against `real` using `sub(d, gen)` (existing helper,
`scripts/stage_decomposition_analysis.py`) — 120 rows per generator (60 real + 60 fake), grouped by content id.

## Methods frozen in advance

**Classifier**: unchanged — `StandardScaler() -> LogisticRegression(C=0.1, class_weight="balanced",
random_state=17)`, from `scripts/stage_decomposition_analysis.py::make()`. No new model family, no tuning.

**CV structure**: content-grouped 5-fold, 10 repeats, seed=0, identical fold-generation code to the existing
`grouped_cv_auc`/`paired_diff_ci` (`perm = default_rng(seed+r).permutation(ids); fold = {c: i%5 for i,c in
enumerate(perm)}`) — reused verbatim so every new CV loop in this analysis partitions content ids identically to
the already-published incremental-information results.

**Bootstrap**: content-level resampling with replacement, 1000 draws, seed=0, matching
`content_boot_ci`/`paired_diff_ci`.

**Ridge residualization (Part 4)**: `Ridge(alpha=1.0)`, fixed, not tuned against class separation. Within each
training fold only: fit `StandardScaler` on that fold's VAE features, transform train and held-out fold, fit
`C ~ scaled(V)` by ridge on the training fold, predict on the held-out fold, `C_resid = C_observed -
C_predicted`. The real/fake label is never used in this regression. Repeated for all 10 reps × 5 folds; each
image's final residual is the **mean of its held-out residual across the 10 reps** (each image is held out
exactly once per rep). Per-repeat residuals are also retained for the repeat-level record required by Part 3's
analogous request for `S_vae`.

**S_vae (Part 3)**: same CV structure, `LogisticRegression` on `V` only (the frozen `make()` pipeline, unchanged),
strictly out-of-fold. Two records kept: the repeat-level OOF matrix (10 × 120) and the per-image mean OOF score
(average over the 10 reps) for descriptive/plotting use.

**Residualizing against S_vae (Part 7)**: within the same fold structure, fit `Ridge(alpha=1.0)` on
`C ~ scaled(S_vae)` using that rep's own OOF `S_vae` values for the training rows (themselves already
out-of-fold with respect to their own label, since `S_vae` for any row was produced by a classifier that never
saw that row) and predict for the held-out fold. This is explicitly **label-informed through the training-fold
VAE classifier** and is described as such, distinct from Part 4's fully label-blind residualization against the
raw VAE features.

**Effect-size statistics**: unchanged — `cohen_paired` (paired real/fake Cohen's d over per-content differences)
and its existing bootstrap-CI machinery, applied to both raw `C` and to `C_resid` (and the `S_vae`-residualized
variant). Univariate AUROC is direction-free (`max(auc, 1-auc)`). "Established curvature direction" is fixed as
the **sign of the raw (unresidualized) paired Cohen's d for that generator** — computed once per generator before
any residual statistic is examined, then applied unchanged when scoring the fraction of matched pairs for both
raw and residual curvature.

**Proper scoring rules (Part 8)**: AUROC, log loss, Brier score, all from strictly out-of-fold predicted
probabilities (same CV structure as everywhere else in this plan). Four feature sets per generator: `VAE`,
`VAE+curvature` (curvature alone, not the full trajectory pair — `path_length` is excluded from this part
specifically because its sign reverses across architectures and would confound the redundancy question), `VAE+
score`, `VAE+score+curvature`. Paired content-bootstrap CI (1000 draws) on the metric difference,
`Δ = extended − baseline`, so negative Δ is an improvement for log loss/Brier and positive Δ is an improvement
for AUROC — stated explicitly in every results table, not left implicit.

**Conditional curvature coefficient (Part 9)**: fit `label ~ V + C` with the unchanged standardized
logistic pipeline across all 50 fold-fits (10 reps × 5 folds); record the curvature feature's standardized
coefficient at each fold-fit. Report mean, SD, and fraction of fold-fits with the same sign across the 50
fits, for all four generators. Supportive/descriptive only — coefficients are not interpreted causally.

**PixArt→aMUSEd diagnostic (Part 11)**: reuse the already-computed 10-feature paired Cohen's d vectors for
`pixart_dit` and `amused` (`results/stage_decomposition/dit_generator/four_generator_feature_table.csv`, no
recomputation). Report Pearson correlation across the 10 effect sizes, cosine similarity of the two 10-vectors,
and each feature's per-dimension contribution to that cosine similarity (`a_i b_i / (|a||b|)`, which sums exactly
to the cosine value — a transparent, non-invented decomposition). Separately, refit the real-vs-pixart_dit
10-feature classifier with the same CV structure, additionally recording each fold's standardized coefficient
vector, to inspect whether VAE/round-trip features dominate the weights driving the observed 0.949 transfer AUROC
to aMUSEd (already measured in `results/stage_decomposition/dit_generator/transfer_results.json`, not
recomputed here — only the coefficient vector is new).

**No confidence matching, no subset selection, no new features, no neural networks, no timestep search, no
tuning of ridge alpha against class performance, no second generator/generation.** If a confidence-matched subset
check is added later, it will be clearly labeled a secondary robustness check, not the primary result.

## Outputs to be produced (before inspection, this list is exhaustive)

`results/vae_curvature_redundancy/`:
- `correlation_tables.csv` (Part 1: Pearson/Spearman, C vs each of 3 VAE features, per generator × {pooled, real-only, generated-only})
- `s_vae_oof.csv` (Part 3: per-image mean OOF VAE-only score, all 4 generators + shared real rows)
- `s_vae_oof_repeats.npz` (Part 3: repeat-level OOF matrix)
- `curvature_predicted_vs_observed.csv` (Part 4/6: per-image mean predicted C, observed C, residual C, from VAE-feature ridge)
- `curvature_residual_repeats.npz` (Part 4: repeat-level residuals)
- `svae_residual_curvature.csv` (Part 7: per-image mean predicted/residual C from S_vae ridge)
- `raw_vs_residual_effect_table.csv` (Part 5: paired d / CI / AUROC / frac-matched-direction / effect-retention, raw vs both residual variants, all 4 generators)
- `vae_to_curvature_regression_performance.csv` (Part 6: pooled + within-class cross-fitted R², MAE, Pearson)
- `incremental_proper_scoring.csv` (Part 8: AUROC/logloss/Brier, VAE vs VAE+curvature and VAE+score vs VAE+score+curvature, all 4 generators)
- `conditional_curvature_coefficients.csv` (Part 9: fold-level standardized coefficients)
- `pixart_amused_effect_vector_diagnostic.csv` + `.json` (Part 11)

`results/vae_curvature_redundancy/plots/`:
1. `01_vae_vs_curvature_scatter.png` — VAE feature vs curvature, all 4 generators, real/generated distinguished
2. `02_raw_vs_residual_effect.png` — raw vs VAE-residualized Cohen's d, all 4 generators
3. `03_predicted_vs_observed_curvature.png` — cross-fitted predicted vs observed C, all 4 generators
4. `04_incremental_proper_scoring.png` — VAE vs VAE+curvature, AUROC/logloss/Brier
5. `05_pixart_amused_effect_vectors.png` — only if it clearly explains the 0.949 transfer (produced conditionally)

## Decision rule (fixed in advance)

- **R1 (architecture-dependent redundancy)**: PixArt raw curvature d≈−0.53 collapses toward zero after VAE
  residualization while SD1.5/SDXL retain a meaningful residual effect.
- **R2 (independently informative)**: residual curvature remains meaningfully class-separating for SD1.5, SDXL
  *and* PixArt — the null incremental AUROC is not explained by VAE redundancy; look to Part 8's proper scoring
  rules / model interaction instead of calling it a ceiling effect.
- **R3 (mostly VAE-correlated everywhere)**: residualizing collapses curvature separation for all three diffusion
  generators, not just PixArt.
- **R4 (mixed/nonlinear)**: the four-generator pattern does not fit cleanly into R1–R3 under a linear
  residualization; report exactly where the linear model fails rather than immediately fitting a nonlinear one.

This rule will be applied mechanically to whatever Part 5 produces — it will not be adjusted after seeing the
numbers.
