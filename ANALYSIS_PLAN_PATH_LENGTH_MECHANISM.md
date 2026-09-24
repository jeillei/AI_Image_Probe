# ANALYSIS_PLAN_PATH_LENGTH_MECHANISM — frozen before results are seen

**Status: exploratory/mechanistic stopping-point experiment, not an independent preregistered confirmation.**
Motivated directly by `VAE_CURVATURE_REDUNDANCY.md`'s open question: SDXL passed its original trajectory-stage
incremental test while PixArt did not, yet `diffpath_curvature` is equally VAE-redundant for both (10.5% vs
11.4% residual retention). `path_length` is the trajectory stage's only other feature and has a suggestive sign
pattern (SD1.5 −0.33, SDXL −0.36, PixArt +0.38, aMUSEd +1.39) not yet tested for VAE-redundancy. This is the
project's final planned feature-level mechanism decomposition — the task explicitly requires establishing a
stopping point, not opening a new feature search.

## Question

Does `path_length`, alone or jointly with `diffpath_curvature`, explain why the trajectory stage added
information for SDXL but not for PixArt?

## Data and reuse discipline

Same 60 matched content identities, same 4 generators (sd15/sdxl/pixart_dit/amused vs real), same frozen v2 panel
(`results/stage_decomposition/panel_features_dit.json`, already verified 300/300 rows, 0 NaN). **No new
generation, no new feature extraction.** `C_resid` (VAE-residualized curvature) is **reused verbatim** from
`results/vae_curvature_redundancy/curvature_predicted_vs_observed.csv` (`curvature_residual_vs_vae` column,
already rep-averaged per image) — it is not recomputed by a different method, per the task's explicit
instruction. `P_resid` is computed fresh in this experiment using the identical residualization procedure.
Compute-sufficiency confirmed before writing this plan (same data source as the prior phase); runs entirely as
cached features → local CPU statistics → plots/tables. HPC used only if a bootstrap loop proves unexpectedly
slow locally — not expected given the prior phase's identical-scale computation ran in seconds.

`P := path_length`, `C := diffpath_curvature`, `V := [lpips_ae, pixel_mse_ae, latent_mse_ae]`,
`SCORE := [lare_t200, score_norm_step0]`. No new forensic feature is introduced.

## Methods frozen in advance (identical machinery to `VAE_CURVATURE_REDUNDANCY.md`, reused verbatim)

- **Classifier**: `StandardScaler() -> LogisticRegression(C=0.1, class_weight="balanced", random_state=17)`
  (`scripts/stage_decomposition_analysis.py::make()`). No new model family.
- **CV structure**: content-grouped 5-fold × 10 repeats, seed=0, identical fold-generation code reused from the
  curvature-redundancy script (`fold_assignment`), so every new CV loop partitions content ids identically to
  both the original incremental-information results and the curvature-redundancy results.
- **Bootstrap**: content-level resampling with replacement, 1000 draws, seed=0.
- **Ridge residualization**: `Ridge(alpha=1.0)`, fixed, label-blind, train-fold-only `StandardScaler` + fit,
  exactly mirroring the curvature pipeline, applied to `P` in place of `C`.
- **Effect-size statistics**: paired Cohen's d with bootstrap CI, direction-free univariate AUROC, matched-pair
  sign consistency against the established raw-effect sign — identical machinery, applied to `P` and `P_resid`.
- **Proper scoring rules**: AUROC, log loss, Brier, strictly out-of-fold, paired content-bootstrap CI on
  `Δ = extended − baseline` (negative Δ improves log loss/Brier, positive Δ improves AUROC) — stated explicitly
  in every table.
- **Interaction term (Part 7)**: exactly one, `curvature × path_length` (raw, standardized within the pipeline
  like any other feature), added to `VAE+curvature+path_length` and separately to
  `VAE+score+curvature+path_length`. This is the **only** allowed nonlinear extension — no polynomial terms, no
  further interactions.
- **Coefficient stability (Part 6)**: standardized logistic coefficients for `curvature` and `path_length`
  recorded at every one of the 50 fold-fits (10 reps × 5 folds); mean, SD, fraction-same-sign reported.
- **Transfer ablations (Part 10)**: the existing PixArt-trained 10-feature classifier's `path_length` weight is
  inspected directly (not retrained/feature-selected against destination performance); transfer AUROC is
  recomputed with `path_length` removed and with the full trajectory stage removed, using the identical
  content-grouped transfer procedure already used for the full-panel PixArt→aMUSEd result (0.949).

**No new feature, no timestep search, no FFT/spatial summaries, no boosted trees, no neural networks, no
interaction beyond `curvature×path_length`, no ridge-alpha tuning, no fifth generator.**

## Outputs (exhaustive list, before inspection)

`results/path_length_mechanism/`:
- `raw_path_length_effect.csv` (Part 1)
- `path_length_correlation_tables.csv`, `vae_to_path_length_regression_performance.csv` (Part 2)
- `raw_vs_residual_path_length_effect.csv`, `path_length_predicted_vs_observed.csv`,
  `path_length_residual_repeats_{gen}.npz` (Part 3)
- `residual_trajectory_model_results.csv` (Part 4: C_resid alone / P_resid alone / both, proper scoring rules)
- `sdxl_pixart_feature_addition_comparison.csv` (Part 5: the central stage-level diagnostic)
- `coefficient_stability.csv` (Part 6)
- `interaction_analysis.csv` (Part 7)
- `pixart_amused_path_length_diagnostic.csv/.json` (Parts 9–10)

`results/path_length_mechanism/plots/`:
1. `01_raw_vs_residual_path_length.png`
2. `02_residual_trajectory_geometry.png` (C_resid vs P_resid, 4 generators, real/generated distinguished)
3. `03_sdxl_pixart_incremental_contributions.png`
4. `04_coefficients.png` (optional, produced if it clarifies the discrepancy)

## Decision rule (fixed in advance)

- **P1**: SDXL retains significant VAE-independent path-length information; PixArt does not (or reverses).
- **P2**: neither residual feature alone explains SDXL's gain, but their combination or the
  `curvature×path_length` interaction does.
- **P3**: trajectory features add information for SDXL only when conditioned on the score stage.
- **P4**: no clean path-length or interaction mechanism explains the SDXL/PixArt split — treat the original SDXL
  gain as generator-specific empirical performance, not a resolved mechanism, and stop feature-level mechanism
  exploration.

This rule is applied mechanically to whatever Part 5 (the central diagnostic) and Part 7 (interaction) produce.
Per the task's explicit framing, the default expectation on the final "should mechanism exploration stop?"
question is **yes**, unless one exceptionally clear, pre-specified follow-up is produced — not license to keep
searching.
