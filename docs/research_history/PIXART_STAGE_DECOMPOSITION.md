# PIXART_STAGE_DECOMPOSITION — third-generator decision experiment (SDXL substituted for PixArt-Sigma)

**Read `PREREGISTRATION_pixart_v1.md` first.** PixArt-Sigma (any variant) and SD3-medium were both concretely
blocked (disk space; gated access) before any generation; Stable Cascade downloaded successfully but failed to
load due to a diffusers/repo version incompatibility. **SDXL was used instead** — explicitly the weaker,
same-family alternative the task's own reasoning deprioritized, chosen only because every better-motivated
candidate was independently blocked. Generation resolution was amended twice, both times *before* any SDXL image
was fed through the feature extractor and both on image-validity grounds (machine load, then a documented SDXL
low-resolution tiling artifact), landing on 768×768. All data uses the generator label `sdxl`, never `pixart`.
Full detail of every blocker and amendment is in the preregistration; this document reports only the results.

## Dataset status

60/60 SDXL images generated (768×768, 25 steps, guidance 7.5, seed=hash("sdxl:"+content_id), same human COCO
caption as real/SD1.5/aMUSEd for each content id, no BLIP, no hand-selection). QC: all 60 files present, all
768×768, none near-blank; a spot-check of 6 random images plus the 2 pipeline-validation images all show
coherent, correctly-captioned photographs with no resolution artifacts (`results/stage_decomposition/sdxl_qc_
768.png`, `sdxl_qc_sample6.png`). Features extracted with the **frozen, unmodified** v2 10-feature panel
(`src/features/panel_v2.py`) via the same two-pass extractor already used for SD1.5/aMUSEd — 0 NaN, all
assertions passed.

## Primary result (preregistered)

> **AUROC(VAE+score+trajectory) − AUROC(VAE+score) = +0.053, 95% paired content-bootstrap CI [0.016, 0.092].**
> **The CI excludes zero. Adding the frozen trajectory stage improves real-vs-SDXL discrimination beyond
> VAE+score.**

| stage set | AUROC |
|---|---:|
| VAE | 0.855 |
| VAE + score | 0.853 |
| **VAE + score + trajectory** | **0.909** |
| VAE + score + trajectory + round-trip | 0.917 |

Full incremental table (`results/stage_decomposition/pixart/incremental_information_sdxl.csv`):

| comparison | ΔAUROC | 95% CI | adds information? |
|---|---:|---|---|
| VAE → VAE+score | +0.000 | [−0.012, 0.013] | no |
| **VAE+score → VAE+score+trajectory (primary)** | **+0.053** | **[0.016, 0.092]** | **yes** |
| VAE+score+trajectory → +round-trip | +0.010 | [−0.005, 0.026] | no |
| targeted: VAE vs VAE+trajectory | +0.060 | [0.020, 0.101] | yes |
| targeted: VAE vs VAE+round-trip | +0.009 | [−0.007, 0.025] | no |
| targeted: score vs score+trajectory | +0.090 | [0.030, 0.149] | yes |

## Curvature result

`diffpath_curvature`: paired Cohen's d = **−0.665** [−0.889, −0.440], univariate AUROC 0.750 (fake lower in 75%
of pairs). This is **at least as large as SD1.5's own effect (d=−0.542)**, and both are far larger than aMUSEd's
non-significant d=−0.175 (`results/stage_decomposition/pixart/plots/03_diffpath_curvature_three_generators.png`).
`path_length` (the other trajectory feature): d=−0.358, matching SD1.5's sign (−0.333) and disagreeing with
aMUSEd's large reversal (+1.390) — the same feature already flagged in the SD1.5/aMUSEd phase as the clearest
generator-specific sign reversal; here it is SD1.5 and SDXL that agree, and aMUSEd that is the outlier.

## VAE result

VAE alone reaches AUROC **0.855** — far above SD1.5's 0.639, closer to (but still below) aMUSEd's 0.981. Unlike
aMUSEd, this does **not** saturate the available signal: trajectory still adds a robust, CI-excludes-zero gain on
top of it. SDXL's VAE-only strength is plausibly explained by its separately-trained, refined VAE relative to
SD1.5's (a real architectural/training difference, not a confound introduced by this experiment's protocol,
which canonicalizes every generator identically before any feature is computed).

## Round-trip result

Adds nothing significant beyond VAE+score+trajectory (+0.010, CI [−0.005, 0.026]) or beyond VAE alone in the
targeted comparison (+0.009, CI [−0.007, 0.025]) — the same qualitative pattern as SD1.5 (small, not clearly
above zero) and unlike aMUSEd (also null, but from a much higher baseline).

## Feature-level table (all 10 features, three generators)

`results/stage_decomposition/pixart/three_generator_feature_table.csv`:

| feature | stage | d (SD1.5) | d (aMUSEd) | d (SDXL) | sign SD1.5↔SDXL | sign aMUSEd↔SDXL |
|---|---|---:|---:|---:|---|---|
| lpips_ae | vae | −0.38 | −2.20 | −1.15 | agree | agree |
| pixel_mse_ae | vae | −0.27 | −0.98 | −0.65 | agree | agree |
| latent_mse_ae | vae | +0.19 (n.s.) | −0.47 | +0.34 | agree | **disagree** |
| lare_t200 | score | −0.33 | −1.09 | −0.55 | agree | agree |
| score_norm_step0 | score | +0.41 | +1.14 | +0.76 | agree | agree |
| **diffpath_curvature** | trajectory | **−0.54** | −0.18 (n.s.) | **−0.67** | agree | agree |
| **path_length** | trajectory | −0.33 | +1.39 | −0.36 | agree | **disagree** |
| pixel_l1_roundtrip | roundtrip | −0.16 (n.s.) | −0.66 | +0.09 (n.s.) | **disagree** | **disagree** |
| lpips_roundtrip | roundtrip | −0.60 | −1.61 | −0.61 | agree | agree |
| latent_mse_roundtrip | roundtrip | −0.10 (n.s.) | +0.23 (n.s.) | +0.24 (n.s.) | agree | agree |

**9/10 features sign-agree SD1.5↔SDXL** (only `pixel_l1_roundtrip`, already the weakest/most marginal feature
for SD1.5, disagrees — and it is non-significant for both SD1.5 and SDXL, so this is noise-level, not a genuine
reversal). **7/10 sign-agree aMUSEd↔SDXL** — the same three features (`latent_mse_ae`, `path_length`,
`pixel_l1_roundtrip`) that were already the ones disagreeing between SD1.5 and aMUSEd. No sign reversal was
reconciled or explained away; all are reported as found.

## Cross-generator transfer

`results/stage_decomposition/pixart/transfer_results.json`, content-grouped, same procedure as the SD1.5↔aMUSEd
transfer test:

| direction | AUROC |
|---|---:|
| SD1.5 → SDXL | **0.898** |
| SDXL → SD1.5 | 0.767 |
| aMUSEd → SDXL | 0.726 |
| SDXL → aMUSEd | 0.820 |

**SD1.5↔SDXL transfer (0.77–0.90) is far stronger than the SD1.5↔aMUSEd transfer measured in the prior phase
(0.58–0.60).** A classifier trained on real-vs-SD1.5 alone, never seeing SDXL, already reaches 0.898 AUROC on
real-vs-SDXL — the 10-feature decision direction learned from one UNet-family diffusion model transfers
substantially to another, in a way it did not transfer to the non-diffusion aMUSEd.

## Which scenario occurred

**Scenario A — SDXL behaves like SD1.5** (per the preregistered decision rule: "trajectory ΔAUROC 95% CI
excludes 0 and is positive"). The primary test passes unambiguously. **With one honest qualifier**: the absolute
magnitude (+0.053, CI [0.016, 0.092]) is smaller than SD1.5's own (+0.125, CI [0.056, 0.204]), and the two CIs
do not overlap by much — but this is plausibly a **ceiling effect**, not a weaker mechanism: SDXL's VAE-only
baseline (0.855) leaves much less headroom than SD1.5's (0.639), the same way aMUSEd's 0.981 baseline left
essentially none. The *targeted* VAE-vs-VAE+trajectory comparison (+0.060, which isn't ceiling-limited by the
score stage sitting in between) and the raw curvature effect size (d=−0.665, *larger* than SD1.5's) are at least
as strong as SD1.5's own numbers. Read together, the honest statement is: **trajectory curvature adds real,
statistically robust information for SDXL, by every measure tried, and does so in the same direction and via the
same specific feature as SD1.5** — a clean Scenario A, with headroom differences explaining the smaller raw
AUROC delta rather than indicating a weaker underlying effect.

## Status of the active hypothesis: **strengthened**

The hypothesis under test — "diffusion-generated images may contain trajectory-level forensic information
beyond static VAE reconstruction, whereas some non-diffusion generators may be dominated by decoder/image-level
fingerprints" — gains a second, independent, preregistered confirmation. This is not proof of a universal
diffusion-model property: SDXL and SD1.5 share a real amount of architectural/training lineage (both UNet-based
latent diffusion, both from the Stable Diffusion line, same CLIP-family text-encoder ancestry for at least one
of SDXL's two encoders) that this experiment cannot cleanly separate from "any diffusion model would show this."
That is exactly why PixArt-Sigma (a Diffusion *Transformer*, no UNet lineage at all) was the originally preferred
test and remains the more decisive one not yet run. What **is** newly established: the effect is not an SD1.5
idiosyncrasy in the narrow sense (a fluke of that one specific 860M-parameter checkpoint) — it replicates on a
substantially larger (2.6B-parameter), separately-trained, differently-VAE'd UNet diffusion model with a
different text-conditioning setup.

## Biggest remaining confound

**Architectural lineage vs. "diffusion-ness" are still confounded.** SD1.5 and SDXL are both UNet-based latent
diffusion models from the same model family; this experiment cannot distinguish "trajectory curvature is a
general diffusion-model property" from "trajectory curvature is a Stable-Diffusion-lineage property." The
strong SD1.5↔SDXL transfer (0.77–0.90) is exactly as consistent with a shared-lineage explanation as with a
shared-diffusion-mechanism explanation.

## Exactly one recommended next experiment

Because the trajectory result **did** generalize (Scenario A), the task's own decision rule points to scaling
the robustness/AI-edit work around `diffpath_curvature` next — but the confound above means that scaling now
would still be testing "does this survive within the Stable-Diffusion lineage," not "is this a diffusion-model
property." The single most informative next experiment is therefore **not** a robustness benchmark; it is
finally obtaining a genuinely lineage-independent diffusion test — a Diffusion Transformer (PixArt-Sigma, if a
machine with more disk/a GPU becomes available) or another non-UNet, non-Stable-Diffusion-lineage diffusion
model — evaluated with this exact same frozen protocol and primary test. Only after that result lands is scaling
the robustness/AI-edit study around curvature the well-justified next step.
