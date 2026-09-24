# FINAL_RESULTS — SynthImage

This document is the final scientific synthesis of the SynthImage project. It replaces the need to read the
full chronological research ledger (`docs/research_history/RESEARCH_REPORT.md`, 100+ entries) to understand what
was found. Methodology detail: `docs/METHODS.md`. Reproduction: `docs/REPRODUCIBILITY.md`.

## 1. Research question

> Where does forensic information about an image's origin (real photograph vs. AI-generated) live inside a
> pretrained diffusion pipeline, how much does each computational stage add beyond the earlier ones, and which
> of those signals survive realistic image transformations, partial AI editing, and — critically — changes in
> generator architecture?

This is a narrower and more defensible question than SynthImage started with. The original hypothesis was that
running a pretrained diffusion model "in reverse" on an image would reveal a generator-general provenance
signal. That specific hypothesis was tested directly and falsified (§2) before the project moved to the
question above.

## 2. Why naive AI-image benchmarks are confounded

SynthImage's first large-scale benchmark produced strong-looking results (AUROC well above 0.8) that did not
survive scrutiny. A confounding audit found:

- **Four simple file-geometry numbers** (native resolution, file size, aspect ratio, and one compression
  statistic) beat the full 652-feature trajectory representation on the original protocol.
- **Caption text alone**, with no image information at all, separated real from synthetic images at AUROC 0.79.
- The trajectory representation could predict *which real-image dataset* an image came from — a dataset-source
  signature, not a provenance signature.

None of this means diffusion probes carry no forensic signal. It means a benchmark that lets real and generated
images differ in acquisition history, resolution, compression, or caption style cannot distinguish a genuine
provenance mechanism from these shortcuts. Full detail: `docs/research_history/SCIENTIFIC_AUDIT.md`.

## 3. Matched-content experimental design

Every result after this point uses **60 real COCO photographs**, each paired with a same-caption counterpart
from every generator under test: same content, same human caption, same canonicalization (256×256, no crop),
computed by the same frozen probe. Every classifier, cross-validation fold, and bootstrap resample is grouped by
**content id** — a real image and its generated counterpart are never split across a fold. This single design
change is what turned an apparently strong but confounded benchmark into an honest, falsifiable measurement
(`docs/METHODS.md` §4-5).

## 4. Literature-grounded stage decomposition

Rather than continue engineering new features, the project reset to a **10-feature panel**, each feature either
a direct reproduction of a published forensic method or a small, explicitly disclosed adaptation
(`docs/research_history/LITERATURE_FEATURE_PANEL.md`), organized into four pipeline stages:

`image → VAE reconstruction → score response → inverse trajectory → round-trip reconstruction`

The central question at every stage: **does this stage add information beyond every earlier stage**, tested with
paired out-of-fold AUROC differences and content-bootstrap confidence intervals, not just standalone accuracy.

## 5. Generator comparison

The same frozen protocol was applied, unmodified, to four generators against the same 60 real photographs:

| generator | architecture | VAE-alone AUROC | does trajectory add info beyond VAE+score? |
|---|---|---:|---|
| SD1.5 | UNet latent diffusion | 0.639 | **yes** (+0.125 AUROC, CI [0.056, 0.204]) |
| SDXL | UNet latent diffusion (larger, separate VAE) | 0.855 | **yes** (+0.053, CI [0.016, 0.092]) |
| PixArt-Sigma | Diffusion Transformer (no UNet) | 0.876 | **no** (−0.005, CI [−0.029, 0.018]) |
| aMUSEd | masked-token model (no diffusion process) | 0.981 | no (every stage beyond VAE has a CI crossing zero) |

Two findings stand out:
- **VAE reconstruction alone is already a strong, generator-dependent forensic signal** — it ranges from
  moderate (SD1.5) to near-ceiling (aMUSEd) depending entirely on how good that generator's own decoder is.
- **The trajectory stage's contribution is architecture-dependent, not simply diffusion-vs-not-diffusion**: it
  helps both UNet-based diffusion models (SD1.5, SDXL) but not the Diffusion Transformer (PixArt-Sigma), despite
  PixArt-Sigma also being a genuine diffusion model. This ruled out the simplest version of the original
  hypothesis ("diffusion models generically show this") and motivated the mechanism analyses below.

Full detail: `docs/research_history/STAGE_DECOMPOSITION_RESULTS.md`,
`docs/research_history/PIXART_STAGE_DECOMPOSITION.md` (the SDXL run — see the naming note in `scripts/README.md`),
`docs/research_history/DIT_STAGE_DECOMPOSITION.md` (the genuine PixArt-Sigma DiT run).

## 6. VAE / trajectory redundancy analysis

Why does the trajectory stage help SD1.5/SDXL but not PixArt-Sigma, when the trajectory's headline feature
(`diffpath_curvature`) shows an almost identical raw effect size across all three diffusion generators
(d ≈ −0.53 to −0.67)? Cross-fitted, label-blind ridge residualization (regressing each trajectory feature on the
three VAE features, fit on training folds only, tested on held-out folds) answered this directly:

- `diffpath_curvature`'s real/fake signal is **substantially redundant with VAE-level information for both
  SDXL and PixArt-Sigma** (only 10–11% of its raw effect survives residualization, CI crossing zero for both) —
  but **not for SD1.5** (62% of its effect survives, CI still excluding zero). This ruled out the working
  hypothesis that curvature redundancy specifically tracked PixArt vs. the two UNet models.

Full detail: `docs/research_history/VAE_CURVATURE_REDUNDANCY.md`.

## 7. Path-length mechanism result

The trajectory stage's *other* frozen feature, `path_length`, resolved the SDXL-vs-PixArt discrepancy cleanly:

- **`path_length`'s VAE-redundancy is the inverse of curvature's.** For SD1.5 and SDXL, its real/fake effect
  *strengthens* after VAE-residualization (173%/186% retention, CI excluding zero) — for SDXL alone, the
  VAE-independent residual reaches AUROC 0.695. For PixArt-Sigma, its raw positive effect (a sign reversal
  relative to SD1.5/SDXL to begin with) collapses to non-significance under residualization.
- Decomposing SDXL's original trajectory-stage gain feature-by-feature shows **`path_length` alone reproduces
  essentially all of it** (+0.052 of the original +0.053 total AUROC gain); `diffpath_curvature`'s own marginal
  contribution is not distinguishable from zero anywhere in this decomposition, for either generator.
- A preregistered `curvature × path_length` interaction term was tested explicitly and added nothing for any
  generator, on any metric — ruling out a "joint trajectory geometry" or "score-conditioned interaction"
  explanation.

**Headline mechanistic finding**: *`path_length` provides a VAE-independent forensic signal for SD1.5/SDXL
specifically — not for the genuine Diffusion Transformer generator tested (PixArt-Sigma), and not attributable
to `diffpath_curvature` or an interaction between the two trajectory features.* This is the most specific,
decisive result in the project and the point at which feature-level mechanism exploration was deliberately
stopped (per its own preregistered decision rule). Full detail:
`docs/research_history/PATH_LENGTH_MECHANISM.md`.

## 8. Robustness (realistic transformations)

*Protocol frozen in `FINAL_VALIDATION_PLAN.md` before this section was written.*

**Status: pending.** The bulk feature-extraction job for this section (2,520+ transformed images × the frozen
SD1.5-probe extractor, across 14 realistic-transformation conditions — JPEG, blur, resize, noise, color jitter,
crop — applied to real/SD1.5/SDXL, with aMUSEd/PixArt-Sigma as secondary comparisons) was running on an HPC GPU
cluster at the time this document was drafted. This section will be completed with the actual retention/collapse
results once that job finishes — see `results/final_validation/` for the raw tables when available. It is
deliberately left as "pending," not filled with an assumed or predicted result.

## 9. AI-edit continuum

*Protocol frozen in `FINAL_VALIDATION_PLAN.md` before this section was written.*

**Status: pending**, same HPC job as §8 (Track B: a 60-content, 4-strength img2img continuum scaling the
project's earlier 8-content pilot). This section reports whether the frozen static (VAE) and trajectory
(`path_length`, `diffpath_curvature`) signals move systematically with degree of generative intervention, and
compares their response curves — not a claim of a calibrated "percent AI" measurement. See
`docs/research_history/STAGE_DECOMPOSITION_RESULTS.md` §Phase 7 for the earlier 8-content pilot's convergent-
validity result (6/10 features showed a significant monotonic trend with edit strength), which this phase scales
and re-tests rather than assumes.

## 10. Limitations

- All generator comparisons use **one frozen probe (SD1.5)** measuring the candidate generator's *output pixels*
  — the candidate generator itself never needs to be a diffusion model, but every measurement is filtered
  through SD1.5's own VAE/UNet, which is itself one specific, dated checkpoint.
- The dataset is **60 matched content identities** — enough for the paired-bootstrap statistics used throughout
  to be meaningful, not enough to support population-level detector performance claims.
- **`path_length`'s architecture-dependence is established for exactly one Diffusion Transformer (PixArt-Sigma)
  and two UNet models (SD1.5, SDXL)** — it is a real, mechanism-level finding about these three specific
  generators, not yet shown to generalize to DiT architectures in general.
- PixArt-Sigma's text encoder is a disclosed, community GGUF-quantized adaptation of the official T5-XXL weights
  (used only because the original fp32 weights exceeded available disk space during development) — the
  denoiser and VAE under test are official, unmodified weights; see
  `docs/research_history/PREREGISTRATION_DIT_GENERATOR_V1.md` for the full disclosure.
- No claim in this project should be read as a deployable, generator-independent, or adversarially robust AI-
  image detector. See §11.

## 11. Conclusion

Pretrained diffusion probes expose **multiple, architecture-dependent forensic signals at different
computational stages** of a latent-diffusion pipeline — not one universal signature. VAE-level reconstruction
carries real, generator-dependent signal on its own. The diffusion trajectory adds further information for
UNet-based diffusion models but not for the one Diffusion Transformer tested. Within the trajectory stage
itself, that added information is attributable specifically to path geometry (`path_length`), not curvature,
for the generators where it appears at all. Whether this specific architecture-dependent signal survives
realistic image transformation and partial AI editing is reported in §8–9 once available.

**This project does not claim**: a universal AI-image detector, generator-independent deployment performance,
that raw diffusion-path curvature alone proves diffusion provenance, or that any AI-edit-strength response
represents a calibrated "percent AI" score.

## Future work

Not part of active SynthImage experimentation (see §12 of `SYNTHIMAGE_PROJECT_STATE.md` for the project's
closing status). Recorded here as possible directions for a future, separate project:
- Test whether `path_length`'s VAE-independence generalizes across a broader sample of Diffusion Transformer
  checkpoints (not just PixArt-Sigma) and UNet checkpoints (not just SD1.5/SDXL).
- A calibrated, held-out-generator study of whether any frozen feature's response to AI-edit strength could
  support a genuine "degree of intervention" estimate, with proper calibration — not attempted here.
- Extending the frozen probe itself (e.g. a more modern base checkpoint) — deliberately out of scope for this
  project, which measured what one fixed, well-understood instrument reveals.
