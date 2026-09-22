# SynthImage — Living Project State

_Last updated: 2026-09-23 — v2 literature-anchored stage panel executed (§8, §15); see STAGE_DECOMPOSITION_RESULTS.md_

This file is the lightweight scientific handoff for SynthImage.  
Keep it concise and update it whenever the project materially changes.

---

## 1. Original idea

SynthImage began with a simple hypothesis:

> A pretrained diffusion model may "explain" real photographs differently from AI-generated images when run in reverse.

The first implementation used Stable Diffusion 1.5 as a probe:

`image -> SD1.5 VAE -> DDIM inversion -> trajectory measurements -> simple classifier`

The intent was **not** to build the strongest AI-image detector. The intended contribution was the representation: if the inverse trajectory contained a provenance signal, a deliberately simple classifier should be enough to expose it.

---

## 2. What v1 built

The original representation eventually grew to 652 manually engineered features across 14 families, including:

- predicted-noise / score statistics
- classifier-free-guidance statistics
- latent path geometry
- endpoint statistics
- score-map spatial and FFT structure
- cross-timestep similarities
- round-trip reconstruction quantities

This produced promising held-generator AUROC results, but later controls showed that much of the apparent performance was not clean provenance signal.

---

## 3. First major failure: the benchmark was confounded

The original benchmark contained strong shortcuts.

Important findings:

- Four simple file-geometry variables outperformed the full trajectory representation on some protocols.
- Caption text alone strongly separated real and synthetic classes.
- The trajectory representation predicted which real-image dataset an image came from.
- Native resolution, resampling history, image complexity, and other acquisition effects leaked into the representation.

Interpretation:

> Strong AUROC on the original corpus could not be interpreted as evidence for a synthetic-image trajectory mechanism.

This changed the project from performance optimization to falsification and confound analysis.

---

## 4. Matched-content experiment

A cleaner dataset was constructed around 60 content-matched triplets:

- 60 real COCO photographs
- 60 SD1.5-generated counterparts from the same human captions
- 60 aMUSEd-generated counterparts from the same captions

This removed many semantic and formatting shortcuts.

Results:

- Real vs SD1.5 remained separable.
- Real vs aMUSEd became extremely separable with the full v1 representation.
- However, simple baselines such as VAE reconstruction error and an unrelated CIFAR-32 diffusion probe were also strong.

This suggested that generated images contain real low-level statistical differences, but not necessarily a unique inverse-trajectory provenance signal.

---

## 5. Cross-generator feature audit

The v1 feature audit showed that SD1.5 and aMUSEd were being separated for very different reasons.

For SD1.5-generated images, the strongest families were mainly:

- guidance-gap statistics
- latent-path geometry

For aMUSEd, the strongest families were mainly:

- score-map spatial autocorrelation
- score-map frequency / FFT structure

Cross-task overlap was very small:

- weak coefficient correlation
- zero top-10 overlap
- zero top-25 overlap
- several strong features reversed direction

A cleaner subset excluding families already known to encode real-source identity showed more agreement, but that shared subset did not add useful information beyond generic VAE + CIFAR baselines.

Conclusion:

> The 652-feature hand-designed coordinate system did not expose a stable, generator-independent mechanism.

---

## 6. Representation reset: raw trajectory analysis

Because the handcrafted feature set might itself have been the wrong test, the project returned to the raw SD1.5 trajectory tensors.

Analyzed objects included:

- latent states `z`
- conditional score tensors
- unconditional score tensors
- guidance tensors
- latent displacements

For each matched content:

`Delta_SD = T(SD1.5 image) - T(real)`

`Delta_AM = T(aMUSEd image) - T(real)`

Main result:

- A shared population-level direction exists.
- But after mean-centering, no meaningful shared cross-generator trajectory subspace survives.
- Principal-angle and cross-projection tests were consistent with largely generator-specific residual structure.

The shared direction therefore looks like a broad mean shift rather than a shared content-dependent trajectory geometry.

---

## 7. The VAE finding

The most important simplification was that the same broad cross-generator shift already appears at the SD1.5 VAE latent `z0`, before any diffusion inversion.

That means:

`image -> SD1.5 VAE -> z0`

already contains much of the shared real-vs-generated displacement.

No UNet, DDIM inversion, guidance, or multi-step trajectory is needed for that effect.

This is consistent with existing literature showing that latent-diffusion autoencoders and their reconstruction errors contain strong forensic signal.

So the original claim:

> "generated images share a special inverse-diffusion trajectory"

is no longer a good working hypothesis.

A better interpretation is:

> Generated images interact differently with pretrained generative representations, and some of that difference appears before diffusion dynamics begin.

---

## 8. What appears established vs open

### Established / strongly supported

- The original large benchmark was heavily confounded.
- Generated images can be strongly separated from real photographs by pretrained generative representations.
- VAE-level information is already strong.
- Different generator families can produce very different forensic effects.
- The original 652-feature SynthImage representation does not expose one stable universal provenance direction.
- Raw trajectory tensors do not currently show a clear shared generator-independent covariance structure beyond a broad mean shift.
- **(v2, §8a) Whether the diffusion trajectory adds genuinely new forensic information beyond the VAE is now
  answered, and the answer is generator-dependent**: yes for SD1.5 (a literature-anchored trajectory-curvature
  feature, DiffPath, adds a robust +0.125 AUROC over VAE+score, content-bootstrap CI [0.056, 0.204]); no for
  aMUSEd (VAE alone is already AUROC 0.98, and no later stage's confidence interval clears zero against it).
- **(v2, §8a) A small pilot suggests this split is not a fragile artifact of clean data**: under JPEG/blur/
  resize/crop, aMUSEd's VAE-only signal stays at 0.97–1.00 and SD1.5's trajectory-dependent gain persists
  qualitatively (full panel 0.70–0.78 across all 4 transforms vs. 0.77 clean, still well above VAE-alone).

### Not adequately established

- Which published forensic mechanisms remain robust after realistic compression, resizing, blur, editing, and
  re-encoding **at scale** (§8a's pilot is n=10/generator/condition — a first read, not a settled answer).
- Whether any forensic signal tracks the degree of partial AI editing rather than only binary real-vs-generated
  status **on a genuine held-out sample** (§8a's img2img-strength pilot, n=8, is convergent-validity evidence
  for 6/10 panel features, not a validated continuous "AI degree" measurement, and used a global rather than
  localized edit for feasibility reasons).
- Whether the SD1.5-trajectory / aMUSEd-VAE split reflects something about generator architecture (SD1.5 and
  the probe are both diffusion UNets; aMUSEd is a masked-token/VQGAN model) or is specific to these two
  generators — untested on a third, architecturally distinct generator.

---

## 9. Literature-informed pivot

Many individual mechanisms are already established in prior work:

- VAE reconstruction fingerprints — AEROBLADE
- latent reconstruction error — LaRE²
- inversion / reconstruction discrepancy — DIRE
- learned inversion representations — FakeInversion
- diffusion-path rate / curvature — DiffPath and related work
- score / noise-prediction consistency and endpoint typicality — diffusion OOD literature

Therefore, simply inventing another trajectory statistic is unlikely to be a strong contribution.

The cleaner research question is now:

> **Where does generative forensic information live across a latent-diffusion pipeline, and which signals remain meaningful under realistic image transformations and partial AI editing?**

---

## 10. Proposed new project structure

Study a staged hierarchy:

`image -> VAE -> score response -> trajectory -> round-trip reconstruction`

Use a **small, literature-anchored feature panel** rather than a new feature zoo.

Candidate stages:

### Stage 0 — image-space controls
- simple image / frequency baselines
- optional strong published detector baseline

### Stage 1 — VAE
- AEROBLADE-style perceptual reconstruction error
- pixel MSE / PSNR as transparent controls
- optional frozen latent-level quantity

### Stage 2 — score response
- score magnitude at fixed predeclared timesteps
- known-noise prediction error at fixed noise levels

### Stage 3 — trajectory
- first-order rate / change
- curvature / second-order trajectory change
- possibly path length as a simple control

### Stage 4 — round trip
- DIRE-style inverse -> forward reconstruction discrepancy
- perceptual reconstruction error
- pixel reconstruction error
- latent round-trip error

The important question is not only which stage classifies best.

It is:

> **Does each later stage contribute information that was not already available earlier?**

---

## 11. Robustness axis

Evaluate the same frozen feature panel under realistic transformations, for example:

- JPEG compression
- resize / downsample-upsample
- blur
- sensor-like noise
- crop
- possibly neural compression
- AI editing

The aim is to compare **mechanism robustness**, not merely detector robustness.

Example question:

> Does VAE reconstruction signal disappear under JPEG while trajectory curvature survives?

or:

> Does every signal collapse under resizing, indicating dependence on fragile low-level fingerprints?

---

## 12. Partial AI-editing axis

A particularly interesting extension is to build a continuum:

`real -> light AI edit -> heavy AI edit -> fully generated`

Then ask whether any literature-defined signal changes monotonically with:

- edited area
- editing strength
- amount of generative intervention

A robust monotonic signal could become more interesting than a binary detector because it might measure **degree of generative processing**.

Do not assume such a signal exists.

---

## 13. Feature-design rule going forward

This is now a hard constraint:

> Every core feature must either come directly from prior literature or be a small, explicit, mathematically motivated extension.

For every feature, the project should be able to answer:

> Why does this feature exist?

with something better than:

> Because the tensor was available.

Maintain two clearly separated tracks if needed:

- **Panel A: literature-defined features** — frozen, interpretable, confirmatory
- **Panel B: learned representation** — exploratory, trained only on designated data and tested on untouched generator(s)

Do not mix them casually.

---

## 14. Current project identity

SynthImage started as:

> Can inverse diffusion trajectories reveal whether an image is AI-generated?

It is now better framed as:

> **How does forensic information emerge across a pretrained generative pipeline, how much new information does diffusion dynamics add beyond the VAE, and which signals survive realistic transformations or partial AI editing?**

This is a more credible and better-scoped ML research project than the original detector framing.

---

## 15. Immediate next step — UPDATE: steps 1–5 executed

Steps 1–5 below were completed in one pass (`LITERATURE_FEATURE_PANEL.md`, `STAGE_DECOMPOSITION_RESULTS.md`,
`RESEARCH_REPORT.md` E98–E106). Headline result: **trajectory (DiffPath curvature) adds robust information
beyond VAE+score for SD1.5 (+0.125 AUROC, CI [0.056,0.204]); nothing adds anything beyond VAE for aMUSEd, which
is already AUROC 0.98 at the VAE stage alone.** This is a genuine "different stages dominate for different
generators" result (not a repeat of the earlier "everything is confounded" pattern) — see `STAGE_DECOMPOSITION_
RESULTS.md` §Phase 4 for the full incremental-information table before treating this as settled.

1. ~~Freeze a small literature-derived feature panel.~~ Done — 10 features, `LITERATURE_FEATURE_PANEL.md`.
2. ~~Implement each stage cleanly and verify it on the existing 60 matched triplets.~~ Done.
3. ~~Measure incremental information across stages.~~ Done — see result above.
4. ~~Run a small robustness pilot~~ Done (10 content ids × 3 generators × 5 conditions) — the split above survives
   JPEG/blur/resize/crop qualitatively at this small n.
5. ~~Include at least one controlled AI-edit condition if feasible.~~ Done via an img2img-strength continuum
   (localized/masked inpainting was judged infeasible without a new, unvalidated model download — documented,
   not silently skipped); exploratory, n=8, convergent-validity evidence only.
6. **Not yet done, and now the natural next step**: test the SD1.5-trajectory / aMUSEd-VAE split against a
   third, architecturally distinct generator before treating it as a general two-way pattern rather than a
   property of these two specific generators.

---

## 16. Research discipline

Going forward:

- no new large handcrafted feature dictionaries
- no feature selection on test data
- keep content grouping intact
- distinguish exploratory from confirmatory work
- use trivial baselines early
- compare against the previous stage before claiming a new mechanism
- do not interpret high AUROC as mechanism by itself
- preserve negative results
- freeze protocols before scaling
- use git for all new experiment states
- update this file whenever the scientific direction changes materially

