# SynthImage — Project State (closed)

_Last updated: 2026-09-25 — final validation phase complete; active experimentation concluded._

**This project is closed to active experimentation.** For the full scientific narrative, read
[`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md) — this file is now a short closing status note, not the primary
document; it deliberately does not repeat what `docs/FINAL_RESULTS.md` already covers in full.

## Final scientific conclusion

Pretrained diffusion probes expose multiple, **architecture-dependent** forensic signals at different
computational stages of a latent-diffusion pipeline, not one universal signature:

- VAE reconstruction error alone is a strong, generator-dependent signal.
- The diffusion trajectory adds further information for UNet-based diffusion models (SD1.5, SDXL) but not for
  the one Diffusion Transformer tested (PixArt-Sigma).
- Within the trajectory stage, that added information is attributable specifically to `path_length`, not
  `diffpath_curvature` — established via cross-fitted residualization and a direct feature-by-feature
  decomposition that exactly reproduced both generators' original incremental-AUROC numbers.
- `path_length`'s VAE-independent signal for SD1.5/SDXL survives realistic JPEG/blur/resize/noise/color-jitter/
  crop transformation almost universally (29/30 and 30/30 tested condition/generator combinations for the raw
  effect and the incremental-AUROC gain respectively) and transfers from a clean-trained model to every
  transformed condition with only modest degradation.
- Under a scaled 60-content AI-edit-strength continuum, VAE reconstruction tracks increasing intervention far
  more monotonically than either trajectory feature does — a genuine, unforced difference, not evidence against
  the trajectory-mechanism result (the two measure different things).

## Robustness outcome

**Survived.** See `docs/FINAL_RESULTS.md` §8 for full numbers.

## AI-edit outcome

**Mixed, reported as found.** The static (VAE) signal tracks edit strength consistently; the trajectory features
do not track it as cleanly, at the population level. See `docs/FINAL_RESULTS.md` §9.

## What was falsified

- The original hypothesis — that SD1.5 inverse trajectories carry a **generator-general** provenance signal —
  was falsified by a confounding audit before the project's frozen-panel phase began (see
  `docs/research_history/SCIENTIFIC_AUDIT.md`).
- The working hypothesis motivating the VAE-curvature redundancy analysis (that SD1.5 *and* SDXL would both
  retain VAE-independent curvature information while only PixArt-Sigma collapsed) was rejected by the data:
  SDXL's curvature collapsed under residualization just as much as PixArt-Sigma's did
  (`docs/research_history/VAE_CURVATURE_REDUNDANCY.md`).
- The simplest "diffusion models generically show trajectory-level forensic signal" hypothesis was rejected —
  it holds for two UNet-based diffusion models but not for the one Diffusion Transformer tested.

## What remains unknown

- Whether `path_length`'s architecture-dependence (UNet vs. Diffusion Transformer) generalizes beyond the one
  DiT checkpoint tested (PixArt-Sigma) and the two UNet checkpoints tested (SD1.5, SDXL).
- Whether any frozen feature's response to AI-edit strength could support a properly calibrated "degree of
  intervention" estimate — not attempted here, and explicitly not claimed.

Both are recorded as "Future work" in `docs/FINAL_RESULTS.md`, not as open invitations to continue active
SynthImage experimentation.

## Final public-facing documentation

- [`README.md`](README.md) — start here
- [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md) — the full scientific synthesis
- [`docs/METHODS.md`](docs/METHODS.md) — frozen methodology
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — how to reproduce every result
- [`docs/research_history/`](docs/research_history/) — the full chronological research ledger, preregistrations,
  and superseded phases (including the original 652-feature representation and why it was abandoned)
- [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) — pre-publication audit; one open item (license
  choice) remains before this repository is published
