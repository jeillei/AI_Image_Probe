# REVIEWER_VALIDATION_RESULTS

Reports both reviewer-validation experiments exactly as preregistered in
`REVIEWER_VALIDATION_PLAN.md`, using its decision rules and reporting negative results plainly. Written
incrementally: the Experiment 1 section was committed before Experiment 2 was run, and Experiment 2's conditioning
mode (correct caption) was fixed in the plan before Experiment 1's C1 result was known.

## Experiment 1 — conditioning ablation

**Result: C1 — conditioning-independent.** The `path_length` real-vs-generated effect for SD1.5 and SDXL is
essentially unchanged whether the SD1.5 probe is conditioned on the image's own correct caption, an empty
("null") caption, or a caption belonging to a different, deterministically deranged content id.

| generator | condition | paired Cohen's d | 95% CI | retention vs. correct |
|---|---|---:|---:|---:|
| sd15 | correct | −0.333 | [−0.551, −0.111] | 1.00 |
| sd15 | null | −0.324 | [−0.549, −0.103] | 0.975 |
| sd15 | shuffled | −0.327 | [−0.553, −0.101] | 0.982 |
| sdxl | correct | −0.358 | [−0.607, −0.105] | 1.00 |
| sdxl | null | −0.367 | [−0.622, −0.117] | 1.028 |
| sdxl | shuffled | −0.376 | [−0.628, −0.121] | 1.052 |
| pixart_dit (secondary) | correct | +0.380 | [0.155, 0.674] | 1.00 |
| pixart_dit (secondary) | null | +0.380 | [0.155, 0.670] | 0.998 |
| pixart_dit (secondary) | shuffled | +0.370 | [0.145, 0.661] | 0.973 |
| amused (secondary) | correct | +1.390 | [1.144, 1.793] | 1.00 |
| amused (secondary) | null | +1.393 | [1.144, 1.797] | 1.002 |
| amused (secondary) | shuffled | +1.389 | [1.142, 1.787] | 0.999 |

Retention ratios for every generator, under both null and shuffled conditioning, land within ±5% of 1.0 — far
inside the ±30% band the plan set as the threshold for "conditioning-independent." Full table (including
univariate AUROC and both incremental comparisons):
`results/reviewer_validation/conditioning_summary.csv`. Figure: `results/reviewer_validation/plots/conditioning_ablation.png`
(three conditioning bars per generator, visually indistinguishable).

**Incremental value.** `VAE+score` vs `VAE+score+path_length` (the more powered of the two preregistered
incremental comparisons) stays positive with a 95% CI excluding zero for SD1.5 and SDXL under **all three**
conditioning modes:

| generator | condition | Δ AUROC (VAE+score → +path_length) | 95% CI |
|---|---|---:|---:|
| sd15 | correct | +0.088 | [0.016, 0.156] |
| sd15 | null | +0.085 | [0.013, 0.154] |
| sd15 | shuffled | +0.082 | [0.011, 0.149] |
| sdxl | correct | +0.060 | [0.026, 0.100] |
| sdxl | null | +0.060 | [0.025, 0.100] |
| sdxl | shuffled | +0.060 | [0.025, 0.100] |

**One honest caveat.** The narrower `VAE` vs `VAE+path_length` comparison (no score features) for **SD1.5
specifically** has a 95% CI that includes zero under null (−0.003, 0.160) and shuffled (−0.004, 0.159)
conditioning, where it excluded zero under correct conditioning (0.001, 0.163) — barely, in both directions. The
point estimate itself barely moves (0.080 → 0.076 → 0.075), so this reads as a comparison sitting right at the
edge of what a bootstrap CI can resolve at n=60 rather than a real effect change; it is reported here rather
than smoothed over. SDXL's equivalent comparison stays clearly positive in all three conditions.

PixArt-Sigma's pattern (small positive `path_length` effect but a **negative**, CI-excluding-zero incremental
contribution — i.e. adding `path_length` on top of VAE+score measurably *hurts* AUROC for this generator) is
unchanged across conditioning modes, consistent with the v1.0 finding that trajectory information does not help
for PixArt-Sigma. aMUSEd's incremental comparison crosses zero in all three conditions (VAE alone already
separates aMUSEd almost perfectly), also unchanged.

**Interpretation.** The `path_length` effect for SD1.5- and SDXL-generated images is not primarily a
caption–image semantic-compatibility artifact: removing the correct caption entirely, or replacing it with a
caption that describes a different photograph, leaves the effect size and (with one narrow, borderline exception
for SD1.5) the incremental predictive value essentially intact. This directly addresses reviewer concern 1 and
strengthens the provenance interpretation of the v1.0 result.

## Experiment 2 — probe swap

**Result: P2 — probe-family affinity (partial, not symmetric).** The qualitative pattern of which generators show
incremental trajectory information is preserved across both probes, but the *raw magnitude* of the `path_length`
effect is probe-dependent, most dramatically for SDXL-generated images.

| generator | SD1.5 probe (d, 95% CI) | SDXL probe (d, 95% CI) |
|---|---:|---:|
| sd15 | −0.333 [−0.551, −0.111] | −0.296 [−0.542, −0.063] |
| sdxl | −0.358 [−0.607, −0.105] | **−0.720 [−0.999, −0.495]** |
| pixart_dit (secondary) | +0.380 [0.155, 0.674] | +0.333 [0.101, 0.654] |
| amused (secondary) | +1.390 [1.144, 1.793] | +0.649 [0.414, 0.940] |

Figure: `results/reviewer_validation/plots/probe_swap_matrix.png`. Full table (including univariate AUROC and
both incremental comparisons): `results/reviewer_validation/probe_swap_summary.csv`.

**What is stable across probes:** same sign for every generator under both probes; SD1.5-generated images show a
closely comparable effect under both probes (−0.333 vs. −0.296, overlapping CIs); PixArt-Sigma shows a
comparable, modest effect under both (+0.380 vs. +0.333, overlapping CIs) — probe-stable for this generator
specifically. Incremental value (`VAE+score` vs `VAE+score+path_length`) tells the **same qualitative story**
under both probes for every generator: positive and CI-excluding-zero for sd15 and sdxl (SD1.5 probe: +0.088,
+0.060; SDXL probe: +0.066, +0.065 — all four CIs exclude zero), negative and CI-excluding-zero for pixart_dit
under both probes (SD1.5 probe: −0.009; SDXL probe: −0.011 — `path_length` measurably *hurts* for this generator
regardless of which probe measures it), and crossing zero for amused under both probes.

**What is not stable:** the SDXL probe's separation of SDXL-generated images from real ones is **roughly double**
the SD1.5 probe's (−0.720 vs. −0.358, non-overlapping CIs) — the clearest, most statistically resolvable
probe-family-affinity signature in this experiment: a probe measuring images generated by its own architecture
family finds a substantially larger effect than a differently-sourced probe does. aMUSEd's effect also drops
substantially under the SDXL probe (1.390 → 0.649, non-overlapping CIs), though aMUSEd shares no architecture
family with either probe, so this is not a clean affinity story — more likely a generic "some generators interact
more strongly with some probes' decoders" effect that is not fully explained by architecture lineage alone.
Notably, the SD1.5 probe does *not* show a correspondingly weaker effect on SDXL-generated images relative to its
own sd15-generated images (−0.358 vs. −0.333, essentially the same) — the affinity signature found here is
asymmetric, driven by the SDXL probe's amplification, not a symmetric "each probe favors its own family" pattern
in both directions.

**Interpretation.** `path_length`'s *raw* magnitude is not fully probe-independent — the SDXL probe's much larger
effect on SDXL-generated images is real evidence of some probe-generator affinity, exactly the kind of finding
reviewer concern 2 asked to be tested for. But the qualitative claim that survives both v1.0 and this stress test
— which generators show *incremental* value beyond VAE+score, and which do not — is unchanged: positive for
SD1.5- and SDXL-generated images, negative (harmful) for PixArt-Sigma, absent for aMUSEd, under **both**
measuring instruments. This is reported as a genuine, partial refinement, not swept into either "fully
probe-stable" or "purely a probe artifact": raw effect size should be described as probe-dependent for at least
one generator/probe pairing, while the incremental, decision-relevant claim (does trajectory information help
beyond reconstruction, for this generator) is not.

## Combined interpretation and wording changes

Both stress tests leave the v1.0 result's *qualitative* claim — `path_length` carries incremental information
beyond VAE+score for SD1.5- and SDXL-generated images but not for PixArt-Sigma — intact under caption removal,
caption shuffling, and a second, architecturally different measuring instrument. Neither test found grounds to
retract or substantially narrow that claim. Experiment 2 does add one genuine refinement: `path_length`'s raw
effect *magnitude* (not its incremental value, and not its sign) is probe-dependent for at least the SDXL
generator/probe pairing, so language implying the magnitude of the effect is an intrinsic, probe-independent
property of the image should be softened. Docs updated accordingly: `docs/FINAL_RESULTS.md` §10 (Limitations)
and §11 (Conclusion), and the root `README.md` headline findings.

## Relation to recent literature

"Diffusion-trajectory forensic signal exists" is not itself a novel claim as of 2024–2026: recent work
independently confirms it using different methodology — **Denoising Trajectory Biases for Zero-Shot AI-Generated
Image Detection** (NeurIPS 2025) finds generated images converge faster than real ones under diffusion inversion
and generalizes zero-shot across 21 generators ([paper](https://openreview.net/pdf?id=2h8vXbEufN)); **LATTE**
(arXiv 2507.03054, 2025) learns an embedding over the latent trajectory across denoising steps for cross-generator
detection ([paper](https://arxiv.org/abs/2507.03054)); **DRCT** (ICML 2024) extends DIRE-style reconstruction with
contrastive training on hard reconstructed samples
([paper](https://proceedings.mlr.press/v235/chen24ay.html)); **FakeInversion** (CVPR 2024) uses inversion into a
text-to-image model's noise space as a detection feature for unseen generators
([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Cazenavette_FakeInversion_Learning_to_Detect_Images_from_Unseen_Text-to-Image_Models_by_CVPR_2024_paper.pdf)).

None of these decompose which *stage* of a frozen, off-the-shelf probe adds incremental value over a simpler
reconstruction baseline, and none stress-test that specific incremental claim against a caption-conditioning
ablation or a cross-probe swap. **SynthImage's contribution is therefore not "diffusion trajectories carry
real/fake information"** (independently established by the above) **but a confound-controlled, stage-wise
decomposition of which frozen-probe signal survives after simpler reconstruction features are accounted for,
across four generator architectures, together with the conditioning and cross-probe stress tests reported in
this document.**

