# METHODS

This document describes the frozen methodology behind SynthImage's final results. It complements
`docs/FINAL_RESULTS.md` (what was found) with how it was measured. For the full chronological research ledger,
including abandoned approaches, see `docs/research_history/RESEARCH_REPORT.md`.

## 1. The probe

Everything in the final result set is computed with one frozen instrument: **Stable Diffusion 1.5**
(`stable-diffusion-v1-5/stable-diffusion-v1-5`), used as a measurement device, not a generator, via
`src/synthimage/probes/sd15.py`. For a given image, the probe:

1. Encodes the image through SD1.5's VAE to a latent `z0`.
2. Runs a fixed, short (6-step) DDIM inversion — conditioned on the image's own human caption — producing a
   trajectory of latents and the UNet's predicted noise (score) at each step.
3. Reconstructs the image from that trajectory (round-trip) and separately from the VAE alone (no diffusion).

The candidate generator under test never needs to be diffusion-based, differentiable, or even locally hosted —
only its output pixels are needed. This is why the same frozen probe was reused unmodified across five different
generators (real photographs, SD1.5, SDXL, PixArt-Sigma DiT, aMUSEd) without any protocol change.

## 2. The frozen v2 feature panel

Ten scalar features, each either a direct reproduction of a published forensic method or a small, explicitly
labeled adaptation. Full formulas, citations, and reproduced-vs-adapted status for every feature:
`docs/research_history/LITERATURE_FEATURE_PANEL.md`. Implementation: `src/synthimage/features/panel_v2.py`.

| stage | features | literature anchor |
|---|---|---|
| VAE (static reconstruction) | `lpips_ae`, `pixel_mse_ae`, `latent_mse_ae` | AEROBLADE (Ricker et al.) |
| score response | `lare_t200`, `score_norm_step0` | LaRE² (Luo et al.), general diffusion-OOD framing |
| trajectory (dynamic) | `diffpath_curvature`, `path_length` | DiffPath (Heng et al.) |
| round-trip | `pixel_l1_roundtrip`, `lpips_roundtrip`, `latent_mse_roundtrip` | DIRE (Wang et al.) |

No feature was added because a tensor happened to be available. The panel has been frozen since it was first
defined and has not changed across any subsequent generator, robustness, or mechanism experiment.

## 3. Canonicalization

Every image — real or generated, any generator — passes through the identical preprocessing path before any
feature is computed: EXIF-transpose, RGB conversion, an optional transform (Section 5), then a 256×256
squash-resize (no crop). No filename, path, file size, native resolution, or transform label is ever available
to the feature extractor or classifier. This discipline exists because the project's *first* large-scale result
(`docs/research_history/SCIENTIFIC_AUDIT.md`) was falsified by exactly this class of shortcut — see
`docs/FINAL_RESULTS.md` §2.

## 4. Caption-matched experimental design

The dataset is 60 real COCO photographs, each paired with a counterpart from every generator under test that was
*generated from that photograph's human caption* — same caption, same canonicalization. **This is caption
matching, not content matching**: the generated image does not reproduce the real photograph's actual scene, it
is an independent generation conditioned on the caption describing it. This still removes the semantic and
framing shortcuts a naive "scrape real images from one place, generated images from another" benchmark would
leave in. Every statistical test in this project groups by **content id** (the shared caption/pairing key),
never by individual image: a train/test split, a cross-validation fold, or a bootstrap resample never separates
a real image from its caption-matched counterpart. Construction: `scripts/build_caption_matched.py`; manifest:
`data/caption_matched/manifest.csv` (regenerated locally, not redistributed — see `docs/REPRODUCIBILITY.md`).

## 5. Classifier, cross-validation, and bootstrap (used identically everywhere)

- **Classifier**: `StandardScaler() -> LogisticRegression(C=0.1, class_weight="balanced", random_state=17)`.
  Deliberately simple and fixed — no hyperparameter search, no model-family change across any experiment in this
  project's final phase.
- **Cross-validation**: content-grouped 5-fold, repeated 10 times with different random fold partitions
  (seed=0), out-of-fold predictions averaged across repeats. No image's own fold ever contributes to its own
  prediction.
- **Bootstrap**: content-level resampling with replacement, 1000 draws, 95% percentile CIs. Never image-level
  resampling (which would treat a real/fake pair as independent when they are not).
- Effect sizes are always **paired Cohen's d** over per-content real-minus-fake differences, not an unpaired
  comparison of two distributions.

Implementation: `scripts/stage_decomposition_analysis.py` (the core CV/bootstrap library, imported by every
later analysis script rather than reimplemented).

## 6. Stage decomposition and the incremental-information question

The central recurring question is not "how well does stage X classify alone" but **"does stage X add
information beyond every earlier stage, on the same held-out folds, with a paired bootstrap CI on the AUROC
difference."** This is what separates SynthImage's late-stage results from a standard leaderboard comparison —
a stage can classify well in isolation and still add nothing once earlier stages are already known, or vice
versa. See `docs/FINAL_RESULTS.md` §4 for the results this produced.

## 7. Cross-fitted residualization (the mechanism-resolution tool)

To ask whether one feature's real/fake signal is redundant with another (e.g. whether `diffpath_curvature`'s
effect is just a restatement of VAE-level information), the project uses **label-blind, cross-fitted ridge
residualization**: within each training fold of the same content-grouped CV structure, standardize the
predictor features on the training fold only, fit `Ridge(alpha=1.0)` regressing the feature of interest on the
predictors — never using the real/fake label — and predict on the held-out fold. The held-out residual (observed
minus predicted) is then tested for a real/fake effect exactly as any raw feature would be. This is how
`docs/FINAL_RESULTS.md`'s `path_length` result was established: implementation in
`scripts/vae_curvature_redundancy_analysis.py` and `scripts/path_length_mechanism_analysis.py`.

## 8. Robustness and AI-edit validation protocol

Full frozen protocol: `docs/research_history/FINAL_VALIDATION_PLAN.md`. In brief: 14 realistic-transformation conditions (JPEG, blur,
resize round-trip, Gaussian noise, color jitter, center crop — all parameters fixed in advance,
`src/synthimage/corruption/robustness_suite.py`) applied symmetrically to real and generated images at the same
canonicalization stage; and a controlled img2img edit-strength continuum (0.0/0.3/0.6/0.9) using the same SD1.5
checkpoint, scaled from an 8-content pilot to the full 60-content set.

## 9. What is deliberately not done

- No feature selection on test data, ever.
- No model family beyond the one fixed logistic-regression pipeline for any confirmatory result.
- No hyperparameter tuning per generator, per transform, or per mechanism test.
- No claim that a feature's magnitude represents a calibrated "percent AI" quantity.
- No revival of the earlier 652-feature representation (`docs/research_history/FEATURE_AUDIT.md` documents why it
  was abandoned).
