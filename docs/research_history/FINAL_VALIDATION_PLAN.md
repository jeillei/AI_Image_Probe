# FINAL_VALIDATION_PLAN — frozen before final validation results are seen

**Status: this is the project's final experimental phase.** Tests whether the frozen stage-level signals
identified during mechanism analysis (`docs/research_history/PATH_LENGTH_MECHANISM.md`, `docs/research_history/VAE_CURVATURE_REDUNDANCY.md`,
`docs/research_history/DIT_STAGE_DECOMPOSITION.md`) survive realistic image circulation (Track A) and respond coherently to increasing
generative intervention (Track B). No new feature development. After this phase, active experimentation ends.

## Datasets / content ids

Same 60 matched content identities used throughout the project (`data/content_matched/manifest.csv`, all 60,
no subset — no concrete technical blocker exists to justify a smaller set for Track A). Primary generators:
**real, SD1.5, SDXL** — these are the two generators for which `path_length` was established as carrying
VAE-independent trajectory information (`docs/research_history/PATH_LENGTH_MECHANISM.md`). aMUSEd and PixArt-Sigma DiT are included as
**secondary comparisons** where already cheap (their clean-condition features are already fully cached); they are
not the primary robustness claim and their inclusion must not delay this phase.

## Track A — realistic transformation benchmark

**Transform families and parameters** (all already implemented and frozen in
`src/corruption/robustness_suite.py::CONDITIONS`, unchanged — chosen before any result in this phase was seen):

| family | conditions |
|---|---|
| JPEG | quality 90, 70, 50, 30 |
| Gaussian blur | sigma 0.5, 1.0, 2.0 |
| Resize round-trip | 0.5×, 0.25× (LANCZOS down then up) |
| Gaussian noise | sigma 0.02, 0.05, 0.10 (on [0,1]-normalized pixels, clipped) |
| Colour jitter | one seeded ±20% brightness/contrast/saturation condition |
| Center crop | 80% linear size, resized back to canonical dimensions |

14 non-clean conditions total; "clean" (untransformed) is already fully extracted and cached for every
generator, so it is reused, not recomputed. Every image (real and generated alike) passes through the identical
`apply_condition()` function and the identical canonicalization path
(`scripts/extract_detector_features.py::load()`, 256px squash-resize) before any feature is computed —
transforms are applied symmetrically, with no metadata, filename, path, or transform label ever entering the
classifier as a feature.

**Scale**: real + SD1.5 + SDXL × 60 content ids × 14 conditions = 2,520 new transformed images, feature-extracted
through the frozen two-pass SD1.5-probe extractor (`scripts/extract_stage_panel.py` + `compute_lpips_panel.py`,
byte-identical protocol to every prior phase — only a device-selection (`cuda`/`mps`/`cpu`) execution-only patch
was added, no scientific parameter changed). Executed on a remote GPU machine since this is bulk feature
extraction, not statistical analysis — reusing the existing offline-execution workflow (local `.sif` container, pre-built venv,
pre-downloaded SD1.5/LPIPS-VGG16 weights, `local_files_only=True`) rather than rebuilding it.

**Frozen feature set for this phase** (unchanged from the v2 panel, `src/features/panel_v2.py`), with emphasis
on the mechanism-resolved quantities per the task's framing:
- static/VAE: `lpips_ae`, `pixel_mse_ae`, `latent_mse_ae`
- score controls: `lare_t200`, `score_norm_step0`
- trajectory (primary focus): `diffpath_curvature`, `path_length`
- round-trip controls: `pixel_l1_roundtrip`, `lpips_roundtrip`, `latent_mse_roundtrip`

**Robustness question 1 (does the feature itself survive?)**: for `lpips_ae`, `diffpath_curvature`,
`path_length`, per transform and generator: paired Cohen's d, bootstrap CI, direction-free univariate AUROC,
matched-pair sign consistency, clean→transformed Spearman rank correlation, and `effect_retention =
d_transformed / d_clean` (sign preserved, never hidden).

**Robustness question 2 (does incremental information survive?)**: repeat the stage-addition comparison under
every transform — `VAE` vs `VAE+path_length`, `VAE+score` vs `VAE+score+path_length`, with curvature as a
secondary comparison — using the identical content-grouped CV (5-fold × 10 reps, seed=0) and classifier
(`StandardScaler`+`LogisticRegression(C=0.1)`) as every prior phase. ΔAUROC, Δlog loss, ΔBrier, paired
content-bootstrap CIs (1000 draws). No per-transform hyperparameter tuning.

**Robustness question 3 (clean-trained transfer, kept explicitly separate from Q1/Q2)**: fit models once on
**clean images only**, freeze, apply unchanged to every transformed condition. Report AUROC, log loss, Brier
(AUPRC/fixed-FPR-TPR only if trivially available from existing infrastructure — not built new). No threshold
recalibration on transformed data. This is explicitly **deployment-style robustness**, reported separately from
the **signal-retention** evaluation (Q1/Q2, where models/effect sizes are computed within each condition).

## Track B — controlled AI-edit continuum

Scales the existing n=8 img2img-strength pilot (`scripts/generate_ai_edit_pilot.py`) to all 60 content ids
(`scripts/generate_ai_edit_scaled.py`) — the identical mechanism, strengths (0.0/0.3/0.6/0.9), steps (25),
guidance (7.5), caption policy, and seed policy (`hash(content_id:strength)`), unchanged. Not a new editing
model. 180 new generated images (60 × 3 non-zero strengths), feature-extracted through the same frozen
extractor. Executed on a remote GPU machine given the scale (GPU-bound generation + extraction).

**AI-edit analysis**: for the static/VAE and trajectory features specifically, per-content feature curves,
population mean curve with bootstrap CI at each strength, within-content Spearman correlation with strength,
fraction of contents with a monotonic trend. A repeated-measures trend test is reported only if it is a direct,
simple extension of already-available infrastructure (e.g. a paired Wilcoxon or linear-mixed-model slope test);
no new modeling framework is introduced solely for this. **No claim of "percent AI" is made** — this measures
whether a signal changes systematically with degree of intervention, not a calibrated proportion.

**Static-vs-dynamical comparison figure**: normalized (per-feature, clean→heaviest-edit scale) response curves
for VAE signal, `path_length`, and `diffpath_curvature`, side by side. Normalization is for visualization only,
never used in any inferential statistic. The figure reports whatever pattern is actually present — no narrative
is forced if the curves do not support one.

## Success / failure interpretation

There is no single preregistered pass/fail threshold — this phase reports retention/response as measured. A
signal is treated as "surviving" a transform when its effect-size CI excludes zero and its sign matches the
clean condition; "degrading" when the CI still excludes zero but sign/magnitude shrinks; "collapsing" when the CI
crosses zero or the sign reverses. These labels are applied consistently and reported for every
feature/transform/generator combination in the output tables — no combination is dropped or summarized away
because it is inconvenient.

## Stopping rule (restated from the task, binding)

After Track A and Track B are analyzed, active experimentation stops. No new generators, no new feature
families, no timestep search, no curvature/path_length redefinition, no neural classifiers, no additional
benchmark added merely to improve the narrative. Negative or partial robustness results are final results, not
a reason to keep iterating.

## Amendments

**Analysis-script bug, found and fixed before any result was inspected (2026-09-25):** the first draft of
`scripts/final_validation_analysis.py` grouped transform conditions by family name only (e.g. all four JPEG
quality levels pooled as "jpeg"), collapsing the intended 14 distinct conditions to 6. This was an aggregation
bug in the analysis script, not in the extraction protocol or the underlying data (`src/corruption/
robustness_suite.py::CONDITIONS` and the extracted feature table were always correctly per-condition). Caught
immediately on first inspection of the output (an implausibly small `Q1 done: 84 rows` instead of the expected
180) and fixed by grouping on the full condition id (transform name + parameter value) before any table or
figure was produced. No frozen protocol parameter was touched.

Status: **complete**. Results reported in `docs/FINAL_RESULTS.md` §8–9.
