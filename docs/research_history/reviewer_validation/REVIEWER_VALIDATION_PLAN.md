# REVIEWER_VALIDATION_PLAN — conditioning ablation and cross-probe stress test

Written and committed **before** either experiment's results were inspected. A small number of purely technical
engineering-validation checks preceded this document (see "Engineering validation performed" below); none of
them were used to tune any scientific parameter.

**This document is not part of SynthImage's original preregistration chain
(`PREREGISTRATION_content_matched_v1.md`, `PREREGISTRATION_pixart_v1.md`, `PREREGISTRATION_DIT_GENERATOR_V1.md`,
`FINAL_VALIDATION_PLAN.md`).** It is a separate, later validation phase motivated by two reviewer-level
alternative explanations identified *after* that original study was completed and tagged as `v1.0`. Nothing here
retroactively changes what was preregistered for the v1.0 result; it either survives independent stress-testing
or it does not, and both outcomes will be reported.

## Motivation: two reviewer-level alternative explanations

The v1.0 headline result is, in brief: under a frozen SD1.5 diffusion probe, trajectory `path_length` carries
predictive signal beyond the frozen VAE features for SD1.5- and SDXL-generated images, but not for
PixArt-Sigma, and this holds up under 14 realistic image transformations. Two structural properties of the study
design mean this is compatible with alternative explanations that have not yet been directly tested:

1. **Caption-conditioning confound.** Every generated image was produced from the same human COCO caption that
   is later supplied to the SD1.5 probe during inversion. The paired real photograph is only *described* by that
   caption, never generated from it. `path_length` could therefore partly measure prompt–image semantic
   compatibility rather than image provenance — a generated image is, almost by construction, unusually
   compatible with the prompt used to make it.
2. **Probe dependence.** Every measurement in the v1.0 study passes through one frozen SD1.5 probe. The observed
   generator-dependent differences could reflect compatibility between an image and *that specific probe's*
   learned manifold rather than an image-intrinsic property that would show up under a different measuring
   instrument.

Both are directly testable without any new generation, new feature, or new dataset. This document preregisters
exactly two experiments in response.

## What is explicitly out of scope for this phase

No new forensic features, no fifth generator, no new robustness/corruption suite, no new learned-classifier
family, no neural trajectory embeddings, no timestep search, no hyperparameter search, no feature-subset search,
no dataset expansion, no reinterpretation from a single metric in isolation, no rescue attempt if either
experiment weakens the v1.0 result. The objective is not a higher AUROC; it is determining what the existing
`path_length` finding actually means.

---

## Experiment 1 — conditioning ablation

### Hypothesis

> The `path_length` real-vs-generated effect and its incremental predictive value (beyond the frozen VAE
> features) for SD1.5- and SDXL-generated images depend substantially on supplying the image's own correct
> generation caption to the SD1.5 probe.

**Evidence against the current (provenance) interpretation:** the effect collapsing or reversing under null or
shuffled conditioning, for SD1.5 and/or SDXL.
**Evidence for the current interpretation surviving:** the effect and incremental value remaining broadly
comparable across correct, null, and shuffled conditioning.

### Data reused (no new images, no new generation)

`data/content_matched/{real,sd15,sdxl,pixart_dit,amused}/*.png` — the same 60 matched content identities and
five image sets (300 images) already used throughout the v1.0 study. Captions from
`data/content_matched/manifest.csv` (one human COCO caption per `content_id`, shared by all five generator rows
for that id — verified: 0 of 60 content ids have more than one distinct caption in the manifest).

Correct-caption results are **not recomputed** — they are read directly from the already-committed
`results/stage_decomposition/panel_features_dit.json` (300 rows, one per image, produced under exactly this
protocol: 256px squash-resize canonicalization, `SD15Probe(steps=6)`, `guidance=1.0`, `mode="caption"`, the
manifest's own caption).

### Frozen conditioning modes

For every one of the 300 images, run the unmodified `SD15Probe(steps=6)` (identical model, steps, guidance,
canonicalization to the v1.0 protocol — **only the text conditioning changes**) under two *new* modes:

- **B. null conditioning** — `probe.invert_reconstruct(image, "", mode="null", guidance=1.0)`. `SD15Probe.embeds`
  already implements this mode (forces the prompt to `""` before encoding), so both the conditional and
  unconditional branches of classifier-free guidance are derived from the empty string — no semantic
  conditioning enters the trajectory at all. No probe code change required; verified in engineering validation
  below.
- **C. shuffled caption** — `probe.invert_reconstruct(image, shuffled_caption, mode="caption", guidance=1.0)`,
  where `shuffled_caption` is the *other* content id's caption assigned by a single fixed derangement (defined
  next). No probe code change required — this is the existing `mode="caption"` path with a different string.

**A. correct caption** is the existing v1.0 protocol, reused from `panel_features_dit.json` as stated above.

### Deterministic derangement (fixed before any feature extraction, not resampled)

Let the 60 unique `content_id` values be sorted lexicographically (ascending string sort) and indexed
`0..59`. Define `shuffled_caption(content_id at index i) = caption(content_id at index (i + 1) mod 60)`. This is
a single 60-cycle: a bijection over content ids, with no fixed point (shift amount 1 ≠ 0 mod 60 guarantees
`(i+1) mod 60 ≠ i` for every `i`), fully deterministic, and requires no random-number generator. All five
generator rows for a given `content_id` receive the *same* shuffled caption in mode C, preserving the paired
real/generated structure used throughout this project's analysis. This mapping is fixed by construction before
any inversion is run and will not be changed after seeing results.

### Features computed per (image, mode)

Primary: `path_length`, `diffpath_curvature` (`src/synthimage/features/panel_v2.py`, unmodified — both are
computed purely from the `forward` latent trajectory / `cond_scores` that `invert_reconstruct` already returns,
so the same functions apply unchanged to every conditioning mode).

Secondary (score stage): `score_norm_step0` only — it is read directly from the already-captured `cond_scores`
array at zero additional UNet cost. **`lare_t200` is deliberately excluded** from this ablation: it costs 4 extra
UNet forward passes per image (an ensemble, per `LARE_ENSEMBLE = 4`) and the central question here concerns the
trajectory stage, not the score stage — the task's own instructions allow including score features "only if
already inexpensive," and `lare_t200` is not free.

VAE features (`lpips_ae`, `pixel_mse_ae`, `latent_mse_ae`) are **caption-independent by construction**
(`vae_only()` never touches the text encoder or the UNet's conditional branch) and are reused verbatim from
`panel_features_dit.json` for all three conditioning modes rather than recomputed.

### Analysis (reusing existing, unmodified statistical code)

`cohen_paired`, `boot_ci`, `grouped_cv_auc`, `content_boot_ci`, `paired_grouped_cv`, `paired_diff_ci`, `make`,
`sub` — moved from `scripts/stage_decomposition_analysis.py` into `src/synthimage/analysis/cv.py` as part of
this phase (see "Code reuse" below) and imported, not reimplemented, by the new script. Same
`StandardScaler + LogisticRegression(C=0.1, class_weight="balanced")` classifier, same content-grouped 5×10 CV,
same 1000-draw content-level bootstrap, same no-leakage discipline as every prior phase.

For each generator (`sd15`, `sdxl`, `pixart_dit` primary; `amused` secondary — included because it is free given
the shared cached image set) × conditioning mode (`correct`, `null`, `shuffled`):

- paired Cohen's d for `path_length`, bootstrap 95% CI, univariate AUROC (direction-free), fraction of matched
  pairs with the mode-A (correct-caption) established direction.
- Incremental comparison `VAE` vs `VAE+path_length` (content-grouped CV AUROC, paired bootstrap CI).
- Incremental comparison `VAE+score` vs `VAE+score+path_length` (`score` = `score_norm_step0` only, per above).
- Retention ratios `d_null / d_correct` and `d_shuffled / d_correct` (descriptive; not a formal test).

No comparison is tuned per conditioning mode; the classifier, CV scheme, and bootstrap procedure are identical
across all three modes and all four generators.

### Decision rule (fixed in advance; exactly one outcome will be reported)

- **C1 — conditioning-independent.** SD1.5/SDXL effect and incremental AUROC remain broadly comparable
  (retention ratios roughly within [0.7, 1.3] and incremental-AUROC CIs tell the same story) across correct,
  null, and shuffled conditioning. → the signal is unlikely to be primarily prompt–image compatibility; this
  strengthens the provenance interpretation.
- **C2 — conditioning-sensitive but persistent.** Effect weakens substantially (retention below ~0.7) under
  null/shuffled but a CI-excluding-zero effect and positive incremental AUROC remain. → the trajectory signal
  contains both image-origin and prompt-compatibility components.
- **C3 — conditioning-driven.** Effect collapses (CI crosses zero, or incremental AUROC CI crosses zero) under
  null and/or shuffled conditioning. → the v1.0 `path_length` result is substantially explained by
  caption–image compatibility rather than provenance alone. This will be reported plainly if it happens.
- **C4 — generator-specific mixture.** Conditioning sensitivity differs qualitatively between SD1.5 and SDXL
  (e.g. C1 for one, C3 for the other). → semantic compatibility interacts with generator family; the simple
  provenance account is incomplete.

---

## Experiment 2 — probe swap

Begins only after Experiment 1 is complete and its results are committed. **The conditioning mode used here is
fixed in advance as mode A (correct caption, the standard v1.0 protocol)** — this choice is made now, before
Experiment 1's results are known, specifically so that Experiment 1's outcome cannot influence it.

### Hypothesis

> The generator-dependent `path_length` effect is stable across different measuring instruments, rather than
> being an artifact of SD1.5-probe/generator compatibility.

**Evidence against the current interpretation:** the effect pattern changing so that each probe preferentially
separates images from its own generator family (probe–generator affinity), or the SD1.5-observed effect
disappearing entirely under the SDXL probe.
**Evidence for the current interpretation surviving:** qualitatively consistent generator-specific effects under
both probes.

### New probe: `src/synthimage/probes/sdxl.py`

A second frozen measurement instrument, structurally parallel to `SD15Probe` (same public interface:
`encode_image`, `decode_latent`, `embeds`, `invert_reconstruct`, `.device`, `.dtype` — so every function in
`synthimage/features/panel_v2.py` applies to it completely unchanged) so that no feature-computation code is
duplicated between probes.

| parameter | SD1.5 probe (unchanged) | SDXL probe (new) |
|---|---|---|
| model / checkpoint | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `stabilityai/stable-diffusion-xl-base-1.0` |
| UNet / text-encoder dtype | float16 (MPS) | float16 (MPS) |
| VAE dtype | float16 (shares UNet dtype) | **float32**, loaded as a separate `AutoencoderKL` component — SDXL's own VAE is documented to be numerically unstable in float16; verified empirically below that float32 VAE + float16 UNet produces no NaNs |
| scheduler (forward/inverse) | `DDIMScheduler` / `DDIMInverseScheduler`, derived from the pipeline's own config, `clip_sample=False` | identical scheduler classes and `clip_sample=False`, derived from the SDXL pipeline's own config — kept identical on purpose so the *inversion algorithm* is matched and only the network/VAE/text-encoders differ |
| inversion steps | 6 | 6 (matched, not tuned for SDXL) |
| guidance | 1.0 | 1.0 (matched) |
| canonicalization | 256×256 squash-resize, `synthimage.data.loading.load`, unchanged | identical — the *same* 256×256 canonicalized images are fed to both probes; this is deliberate: the comparison is "does a different instrument measuring the *same* input see a different pattern," not "how does each generator look at its own native resolution" |
| text conditioning | single CLIP encoder, prompt only | SDXL's native dual-encoder conditioning (`prompt_embeds` + `pooled_prompt_embeds`), `prompt_2` left unset (defaults to reusing `prompt`) — not simplified to single-encoder, since that would not be genuine SDXL conditioning |
| micro-conditioning (`add_time_ids`) | n/a | `(original_size=(256,256), crop_top_left=(0,0), target_size=(256,256))` — tells the model the input is a full, uncropped 256×256 image; the standard recipe for using SDXL off its native 1024px training resolution |
| latent scaling factor | `pipe.vae.config.scaling_factor` (SD1.5's own, read at runtime, not hardcoded) | `pipe.vae.config.scaling_factor` (SDXL's own — empirically 0.13025, read at runtime) |
| latent shape at 256px | `(1, 4, 32, 32)` | `(1, 4, 32, 32)` — confirmed identical in engineering validation, purely coincidental (same 8× VAE downsample, same 4 latent channels), not relied upon for any claim beyond "the two probes' path_length values live in comparable-dimensionality spaces" |

**Explicitly not done:** no attempt to numerically match SD1.5 and SDXL latents, no rescaling of one probe's
`path_length` to match the other's magnitude, no per-probe hyperparameter search, no swap of the SDXL probe to
its native 1024px resolution (which would confound "different probe" with "different input"), no secret tuning
of the SDXL probe to reproduce the SD1.5 result. Per the task's explicit instruction, **raw absolute
`path_length` magnitudes are not compared across probes** — only within-probe, standardized quantities (paired
Cohen's d, AUROC, incremental AUROC) are compared across the probe dimension.

### Dataset and conditioning

The same 300 images (60 content ids × {real, sd15, sdxl, pixart_dit, amused}), same manifest captions, mode A
(correct caption) only, both probes. No new generation, no new content ids.

SD1.5-probe values are **reused verbatim** from `results/stage_decomposition/panel_features_dit.json` (no
recomputation — that file already is "SD1.5 probe, correct caption, all five generators"). SDXL-probe values are
newly computed by `scripts/reviewer_validation/probe_swap.py`.

### Features computed under the SDXL probe

`path_length`, `diffpath_curvature`, `score_norm_step0` (same rationale as Experiment 1 for excluding
`lare_t200`), and `lpips_ae` / `pixel_mse_ae` / `latent_mse_ae` via `vae_only()` (unchanged function, applied to
the SDXL probe — this *is* recomputed, not reused, because the whole point of this experiment is a different
VAE).

### Analysis

For each probe (`sd15`, `sdxl`) × generator (`sd15`, `sdxl`, `pixart_dit` primary; `amused` secondary) pair:
paired Cohen's d for `path_length`, bootstrap 95% CI, univariate AUROC. Assembled into the 2-probe × 4-generator
matrix specified below. Within each probe: `VAE` vs `VAE+path_length` (content-grouped CV, paired bootstrap CI),
and `VAE+score` vs `VAE+score+path_length` using `score_norm_step0` (included — it is a like-for-like functional
definition applied to each probe's own `cond_scores`, satisfying "the score-stage implementation is truly
comparable").

|              | SD1.5 probe | SDXL probe |
|--------------|------------:|-----------:|
| SD1.5        |             |            |
| SDXL         |             |            |
| PixArt-Sigma |             |            |
| aMUSEd       |             |            |

### Decision rule (fixed in advance)

- **P1 — probe-stable.** Generator-specific `path_length` effects are qualitatively consistent (same sign,
  comparable magnitude class) under both probes. → not an SD1.5-probe idiosyncrasy; strengthens the project.
- **P2 — probe-family affinity.** Each probe preferentially separates images from its own or a closely related
  generator family (e.g. SDXL probe shows a markedly stronger SDXL-vs-real effect than SD1.5-vs-real, while the
  SD1.5 probe shows the reverse pattern). → `path_length` is better described as a probe–generator compatibility
  measure than an intrinsic provenance signature — a reframing, not a refutation.
- **P3 — SD1.5-specific.** The SD1.5/SDXL-generator effect that the SD1.5 probe finds materially weakens or
  disappears under the SDXL probe, with no offsetting pattern. → the original result should be narrowed to "an
  SD1.5-probe-specific phenomenon."
- **P4 — mixed.** Effects change across probes in a pattern not explained by simple probe-family affinity. → the
  trajectory statistic is instrument-dependent in a more complex way; characterized but not resolved further in
  this phase (no third probe will be added to adjudicate).

### Optional sanity check (only if free)

If already-cached CV partitions make it trivial, report whether the SD1.5 probe's own `path_length` result is
stable across repeated CV folds (a stability check on existing data, not a new experiment). Skipped entirely if
it requires any new computation beyond what Experiment 2 already produces.

---

## Code reuse (repository hygiene)

`src/synthimage/analysis/cv.py` (new module) receives the CV/bootstrap/residualization utilities currently
defined once in `scripts/stage_decomposition_analysis.py` and `scripts/vae_curvature_redundancy_analysis.py` but
needed by both the pre-existing analysis scripts and the two new reviewer-validation scripts. Those two existing
scripts are updated to import from the new module (re-exporting the same names) rather than defining the
functions locally, so every existing `from stage_decomposition_analysis import ...` /
`from vae_curvature_redundancy_analysis import ...` elsewhere in `scripts/` keeps working unchanged. This is the
only refactor performed in this phase; it is required because the new scripts live in a different directory
(`scripts/reviewer_validation/`) and copy-pasting the statistics code was explicitly ruled out.

## Repository footprint for this phase

```
docs/research_history/reviewer_validation/
├── REVIEWER_VALIDATION_PLAN.md      (this file)
└── REVIEWER_VALIDATION_RESULTS.md   (written after each experiment; not yet created)

scripts/reviewer_validation/
├── conditioning_ablation.py
└── probe_swap.py

results/reviewer_validation/
├── conditioning_summary.csv
├── probe_swap_summary.csv
├── cache/                            (gitignored: per-image raw feature tables, regenerable)
└── plots/
    ├── conditioning_ablation.png
    └── probe_swap_matrix.png

src/synthimage/
├── analysis/cv.py                    (new: moved, not duplicated, CV/bootstrap utilities)
└── probes/sdxl.py                    (new: second frozen measurement instrument)
```

No other files are added. Per-image raw feature tables produced while running either script are written to
`results/reviewer_validation/cache/` (gitignored) and are not committed.

## Engineering validation performed (technical only, not used to tune any scientific parameter)

1. `SD15Probe.invert_reconstruct(image, "", mode="null", ...)` and `invert_reconstruct(image, <other caption>,
   mode="caption", ...)` both run to completion with no NaNs, on one real image, using the exact frozen protocol
   (steps=6, guidance=1.0) — pass. Per-image timing ≈3.5–4.5s on this machine's Apple Silicon MPS, consistent
   with the ≈3–5s/image already documented in `docs/REPRODUCIBILITY.md`. At that rate, 300 images × 2 new
   conditioning modes ≈ 35–45 minutes of new compute for Experiment 1.
2. `StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=float16,
   local_files_only=True)` loads from the already-cached weights (13GB, no download needed) and moves to MPS —
   pass.
3. VAE loaded as a separate float32 `AutoencoderKL` component (SDXL's documented fp16-VAE instability) — encode
   of a synthetic 256×256 test image produces a finite, non-NaN `(1,4,32,32)` latent with `scaling_factor =
   0.13025` — pass.
4. `pipe.encode_prompt(...)` produces `(1,77,2048)` prompt embeddings and `(1,1280)` pooled embeddings (SDXL's
   dual-encoder output shapes) — pass. One UNet forward pass with `added_cond_kwargs={"text_embeds":...,
   "time_ids":...}` at 256×256 produces a finite, non-NaN `(2,4,32,32)` output — pass.
5. One full 6-step forward-inversion + 6-step reverse-reconstruction round trip (encode → invert → reconstruct →
   decode) on a synthetic 256×256 test image completed in ≈9.6s with no NaNs — pass. At that rate, 300 images ×
   one conditioning mode ≈ 50–60 minutes of new compute for Experiment 2 (VAE-only feature extraction and LPIPS
   add a small constant overhead per image on top of this).
6. This machine has 16GB unified memory and, at the time of this check, ~22GB free disk; SDXL float16 UNet +
   float32 VAE + both text encoders fit and ran without an out-of-memory error at 256×256 with attention/VAE
   slicing enabled (the same techniques `SD15Probe` already uses).

No amendment to any scientific parameter was required after these checks; the configuration documented above is
exactly what will be run.
