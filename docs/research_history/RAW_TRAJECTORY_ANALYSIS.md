# RAW_TRAJECTORY_ANALYSIS — testing the hypothesis at the level of the raw inverse trajectory

Methodological reset: `FEATURE_AUDIT.md` showed that the 652-dimensional hand-designed v1 representation does
not expose a stable, linearly transferable provenance direction between SD1.5 and aMUSEd. That is a negative
result about *one particular hand-designed representation*, not about the raw inverse trajectory itself. This
document tests the broader hypothesis directly on the raw tensors the frozen SD1.5 probe already computes,
using only linear/subspace methods (SVD, principal angles, projection energy) with explicit permutation nulls —
no handcrafted features, no new dynamics, no neural representation learning.

**No feature was added to v1. No protocol parameter changed.** Code: `src/analysis/raw_trajectory.py`,
`scripts/extract_raw_trajectory.py`, `scripts/raw_trajectory_{normcompare,core,paired_cosine,baselines,
feature_link,plots}.py`. Results: `results/raw_trajectory/` (compact CSV/JSON/PNG; the raw tensor cache itself,
~32 MB, is in `data/raw_trajectory_cache/`, gitignored, and is regenerable in ~10 minutes by
`scripts/extract_raw_trajectory.py`).

## Phase 0 — what raw tensors exist, and their properties

The SD1.5 probe (`src/probes/sd15.py::invert_reconstruct[_batch]`) already computes, and optionally returns via
an existing flag (`capture_predictions=True` — this is what `--rich` feature extraction already turns on
internally; the tensors were simply never written to disk before this phase):

| tensor | source | shape | timesteps | conditioning-dependent? |
|---|---|---:|---|---|
| `z` (latent states) | `forward` | (7, 4, 32, 32) | t=0 (clean VAE latent) .. t=6 (final noised state) | `z[0]` is not; `z[1:]` are (computed via the conditioned score) |
| `cond` (conditional score) | `cond_scores` | (6, 4, 32, 32) | evaluated at `z[k]`, k=0..5 | yes (uses the caption embedding) |
| `uncond` (unconditional score) | `uncond_scores` | (6, 4, 32, 32) | evaluated at `z[k]`, k=0..5 | no (empty-prompt embedding) |
| `guidance` = `cond - uncond` | derived, not separately stored | (6, 4, 32, 32) | — | yes, by construction |
| `dz` = `diff(z, axis=0)` | derived from `z` | (6, 4, 32, 32) | — | yes (the step uses `cond`) |

Answers to the Phase-0 checklist:
1. **Stored:** none, previously — `extract_detector_features.py` computes them in memory to build the 652
   scalar features and discards them. This phase persists them (per-image `.npz`, `z`/`cond`/`uncond` only;
   `reverse`/`reconstruction` are dropped — not needed by any planned analysis, kept the cache to ~180 KB/image).
2. **Recomputation required:** yes, for every one of the 180 matched-set images. Extracted once (E89 below),
   resumable, one MPS process, ~3.2–4.2 s/image at batch 2 (~11 minutes total, including one unexplained
   5.7-minute stall on a single batch — logged, not repeated, no evidence it affected the resulting tensors,
   which passed every shape/completeness assertion below).
3. **Shapes:** as above; identical for every image (same 256 px canonicalization, same 6-step schedule).
4. **Spatially aligned across samples?** Only in the trivial *grid* sense — every image goes through the
   identical pipeline so latent-pixel `(c, i, j)` always refers to "the same processing step for the same VAE
   grid cell," but *not* semantically (a fixed grid cell is not the same scene content across different
   photographs, the way center-cropped face datasets are only approximately aligned). This limitation is
   carried through the whole analysis: flattening/PCA below is a documented-order raw-pixel PCA, not a
   registered/semantic one.
5. **Timestep definitions identical across samples?** Yes — `SD15Probe.set_steps(6)` fixes one schedule at
   construction; every image uses the same six DDIM-inverse timesteps regardless of content.
6. **Conditioning-dependent tensors:** `z[1:]`, `cond`, `guidance`, `dz` (all conditioned on the matched human
   caption, identical caption text for real/SD1.5/aMUSEd at a given content id — see
   `PREREGISTRATION_content_matched_v1.md`); `uncond` and `z[0]` are not.
7. **Directly comparable without another learned preprocessing step?** Yes for paired differencing (same
   generator, same content, subtract) — no learned step in between. PCA/SVD is the only "preprocessing," and it
   is itself the object under study, not a black box applied before analysis.

Flattening order (fixed, documented, never varied): each tensor is ravelled in NumPy C order —
timestep-major, then channel, then row-major spatial (`(T, C, H, W).ravel()`). Ambient dimension `p` is 24,576
for `cond`/`uncond`/`guidance`/`dz` and 28,672 for `z`.

## Phase 1 — paired differences

For every content id `i` (60 total, one real + one SD1.5 + one aMUSEd image, asserted complete and shape-matched
by `assert_triplets_complete` — see `tests/test_raw_trajectory.py`, 10 regression tests covering cache-key
collision, missing-sample detection, shape-mismatch detection, guidance/dz index alignment, content-id row
order, and the linear-algebra primitives):

```
Delta_SD_i = T(S_i) - T(R_i)      Delta_AM_i = T(A_i) - T(R_i)
```

computed separately for `z`, `cond`, `uncond`, `guidance`, `dz` — never concatenated.

## Phase 2 — normalization (compared before choosing one)

Three normalizations, each with a distinct, pre-specified interpretation (`src/analysis/raw_trajectory.py`):
**A. raw** difference (absolute magnitude of change) — chosen as primary. **B. sample-normalized** (each raw
tensor divided by its own L2 norm before differencing — pure direction/shape, removes each image's own overall
"trajectory energy"). **C. relative** (raw difference divided by the real tensor's own RMS scale).

Per-sample norm heterogeneity is moderate, not extreme (real/fake tensor-norm coefficient of variation ≈
0.25–0.30 across content, `results/raw_trajectory/svd/normalization_scale_heterogeneity.csv`), so A vs B/C is
not forced by a scale-outlier problem. On the one tensor checked in detail (`guidance`), **A shows *more*
apparent alignment than B or C** (mean top-5 principal angle 61.8° vs 80.5°/80.6°; mean-vector cosine 0.59 vs
0.50/0.50) — this is reported, not used as the reason for choosing A: A was selected in advance because it is
the literal, most interpretable reading of "paired difference" (Phase 1's own formula) and preserves absolute
magnitude of change, which the other two deliberately discard. **The main Phase 3–6 analysis below uses A for
every tensor type; that A shows more raw alignment than B/C makes the null-test results below (§Phase 4-5, which
show that alignment is *not* above chance) a stronger negative, not a weaker one — the most favourable
normalization still fails the null test.**

## Phases 3–6 — within/cross-generator subspace structure, nulls, and the strict projection test

All 60 complete triplets used (no images dropped). `k ∈ {1,2,3,5,10}`, fixed in advance; k=5 reported as the
pre-specified headline value; curves are shown for the rest (no k was chosen post hoc).

**Within-generator structure.** Neither generator's delta set is low-rank: effective (participation-ratio) rank
is 43–58 out of a maximum of 60, and the top 5 components capture only 10–28% of within-generator variance
(`results/raw_trajectory/svd/effective_rank_raw.csv`, `singular_value_spectra_raw.csv`, plots 01–02). The raw
trajectory's own synthetic-minus-real variation is diffuse/high-dimensional even before any cross-generator
comparison.

**Cross-generator subspace alignment, k=5** (`svd/cross_generator_subspace_raw.csv`, plot 03):

| tensor | mean principal angle (deg) | proj. energy AM→SD basis | proj. energy SD→AM basis |
|---|---:|---:|---:|
| z | 59.5 | 0.091 | 0.061 |
| cond | 77.1 | 0.028 | 0.028 |
| uncond | 76.8 | 0.028 | 0.028 |
| guidance | 61.8 | 0.075 | 0.072 |
| dz | 50.6 | 0.158 | 0.106 |

At face value these look like *some* alignment (angles well under 90°, projection energies above zero). The
nulls below are what actually answers the question.

**Null 1 — pair-breaking** (300 permutations/tensor; independently shuffle which real image is subtracted
within each generator, then recompute the same centered-PCA cross-generator statistics; `nulls/
pairbreaking_null_raw.csv`, plot 05): **the observed mean principal angle at k=5 is statistically
indistinguishable from — in most cases *less* aligned than — mismatched-pairing chance**, for every tensor type:
`z` p=0.890, `cond` p=0.370, `uncond` p=0.277, `guidance` p=0.503, `dz` p=0.743 (p = fraction of null draws at
least as aligned as observed; small p would mean "more aligned than chance"). None approach conventional
significance. **The centered-PCA cross-generator subspace alignment does not depend on correct real/fake
content pairing at all** — it is a property of the two populations' aggregate shapes, not of matched content.

**Null 3 — random subspace** (up to 300 draws/tensor/k; compare against `k` uniformly random orthonormal
directions in the same ambient space; `nulls/randomsubspace_null_raw.csv`, plot 06): observed projection energy
(2.8–15.8%) vastly exceeds random-subspace chance (~0.02%, p=0.0 for every tensor). This is the expected,
largely uninformative consequence of `n=60 ≪ p≈25,000` — *any* data-derived 5-dim subspace beats a truly random
one in this regime, trajectory-specific or not (confirmed in Phase 7: every baseline representation shows the
identical p=0.0 pattern against its own random-subspace null, including a 30-dimensional one). This null mainly
rules out "the projection energies are literally noise floor"; it does not support a trajectory-specific claim.

**Phase 5/6 — strict fit-on-one-freeze-project-other, with content-grouped bootstrap CI** (k=5, 300 resamples of
the 60 content ids, applied identically to both generators' matrices; `projection/bootstrap_ci_k5.csv`, plot 04):
the point estimates above sit near or outside their own 95% bootstrap CIs for several tensors (e.g. `dz`: point
50.6°, CI [55.3°, 65.0°]) — resampling with replacement measurably shifts the PCA structure at n=60, a real
instability that is reported rather than smoothed over. Combined with Null 1, there is no tensor type and no
direction (SD→AM or AM→SD) where a basis learned on one generator, frozen, and applied to the other captures
more variance than chance content pairing would already produce.

**A different, genuinely positive result — but not from Phase 3–6's method.** A *paired* per-content statistic
(cosine of `Delta_SD_i` and `Delta_AM_i` for the *same* image, un-centered — a different question from the
centered-PCA subspace tests above) is strongly positive for every tensor type: mean paired cosine 0.47–0.57,
**100% of the 60 pairs positive** for every tensor, p < 0.0005 against a content-shuffle null (2000 permutations;
`nulls/paired_cosine_summary.csv`, `paired_cosine_per_content.csv`; Wilcoxon p ≈ 1.6×10⁻¹¹). This looked at
first like it contradicted the null result above. It does not: the *mean*-vector cosine (cosine of the two
generators' population-average delta vectors, uncentered) is 0.24–0.59 — close to the paired-cosine value for
every tensor (`svd/mean_vector_cosine_raw.csv`, plot 07). **The paired-cosine effect is overwhelmingly a shared,
roughly content-independent mean direction** ("any SD1.5 or aMUSEd image differs from its real counterpart in a
broadly similar overall direction"), which centered PCA removes by construction before Phases 3–6's subspace
tests run. There is a real, large, structural difference between "any synthetic image vs. any real image" — but
it is a near-constant offset, not a content-specific, covariance-structure-level shared mechanism.

## Phase 7 — does this exceed generic low-level image representations?

Same statistics, same 60 triplets, four lightweight baselines (no new giant zoo): 32×32 RGB pixels, the VAE
latent alone (`z[0]`, already in the cache — zero extra compute), a 32×32 grayscale FFT magnitude, and the
already-frozen CIFAR-32 Class-B 30-feature vectors (reused from `results/crossprobe/cifar32_content_matched.
json`, not recomputed). `results/raw_trajectory/baselines/`, plot 08.

* **Pair-breaking null:** every baseline is *also* statistically indistinguishable from mismatched-pairing chance
  (rgb32 p=0.647, vae_latent_z0 p=0.790, fft32 p=1.000, cifar32_classB p=0.240) — the same null-consistent
  pattern as the raw trajectory. The absence of significant content-specific subspace sharing is not something
  the raw trajectory fails at while simple representations succeed; nothing tested here clears that bar.
* **Absolute principal-angle magnitude** is *lower* (more visually "aligned") for every baseline than for the raw
  trajectory's guidance tensor (plot 08) — but this comparison is not meaningful on its own: ambient dimension
  differs by 3 orders of magnitude (`cifar32_classB` p=30 vs raw trajectory p≈25,000), and low-dimensional spaces
  produce smaller expected angles between arbitrary subspaces regardless of any real structure. The
  random-subspace null confirms this: every baseline *also* trivially beats its own random-subspace null (p=0.0),
  the same uninformative n≪p artifact seen in the raw trajectory.
* **Paired per-content cosine** is positive and significant for every baseline too (0.34–0.49, p=0.0 against a
  content-shuffle null), with the same qualitative signature as the raw trajectory (`baselines/
  baseline_paired_cosine.csv`). Notably, `rgb32` (0.485, 98% positive) and `fft32` (0.467, 90% positive) show
  this effect with a *negative* mean-vector cosine (−0.05, −0.21) — meaning their shared-direction effect is
  content-driven, not a simple population offset, which is if anything a *more* specific, harder-to-dismiss
  form of shared structure than the raw trajectory's own (largely mean-offset-driven) version. `vae_latent_z0`
  (0.477 paired, 0.243 mean-vector, 100% positive) reproduces the raw trajectory's qualitative pattern almost
  exactly using only the single VAE encoding step, no diffusion trajectory at all.

**Conclusion for Phase 7: no, the raw inverse trajectory does not show cross-generator structure that exceeds
generic low-level image representations by either test used here.** The bare VAE latent alone reproduces the
trajectory's positive (paired-cosine/mean-direction) result; raw RGB pixels and FFT magnitude show an arguably
*more* specific version of it (content-driven, not just a population offset); every representation tested,
trajectory included, fails the same content-specific subspace-sharing test (Null 1).

## Phase 8 — do the 652 handcrafted features correspond to discovered raw components?

Direction of analysis: raw tensors → PCA (Phases 3–6, already computed, not re-fit) → correlate the top-5
per-generator PCs' per-content projection scores against the per-content *difference* (fake−real) of each
generator's own E88 top-30 handcrafted features (`results/feature_audit/top30_{sd15,amused}.csv`). 1,500
Spearman correlations computed (`projection/pc_feature_correlations.csv`). **No correlation exceeds |r|=0.5;**
the strongest is |r|=0.485 (aMUSEd, `z` tensor PC2 vs `rich_latent_t3_skew`); only 11/1500 exceed |r|=0.4, 75/1500
exceed |r|=0.3. **The E88 top-30 handcrafted features are not simple linear projections onto the leading
raw-trajectory PCA directions**, for either generator. Whatever those hand-designed statistics (guidance-gap
moments for SD1.5; score-map spatial autocorrelation/FFT for aMUSEd) are capturing, it is not well summarized by
the first few linear components of the raw difference tensors — consistent with (though not proof of) those
648-plus handcrafted summaries picking up narrow, specific higher-order structure that a coarse 5-component
linear PCA misses, rather than being alternate views of one dominant shared raw mode.

## Required plots

All in `results/raw_trajectory/plots/`: `01_singular_value_spectra.png`, `02_cumulative_variance.png`,
`03_principal_angles_vs_k.png`, `04_cross_projection_energy.png`, `05_pairbreaking_null.png` (null histograms
with observed statistic marked), `06_randomsubspace_null.png`, `07_mean_vector_cosine.png`,
`08_baseline_comparison.png`, `09_illustrative_2dpca.png` (**illustration only, not evidence**, per instructions
— UMAP was not used anywhere in this analysis).

## Limitations

* Grid alignment only, not semantic alignment (Phase 0 §4) — a genuine, unresolved limitation of any
  raw-pixel/raw-latent PCA on unregistered photographs; some or all of the observed structure (shared or not)
  could in principle reflect this rather than a generative mechanism either way.
* Bootstrap CIs at n=60 are visibly unstable for some tensors (point estimate outside its own 95% CI) — reported
  in Phase 5/6, not hidden.
* Random-subspace null is dominated by the `n≪p` regime and is uninformative on its own; it is reported for
  completeness (as required) but the pair-breaking and content-shuffle nulls carry the actual evidentiary weight.
* Only one normalization (raw) received the full Phase 3–6 treatment; B and C were checked only on `guidance`
  (Phase 2) — a full B/C repeat across all five tensor types was judged out of proportion to the marginal value
  given A already gives the *most* favourable-looking (and still null) result.
* Phase 8 used only the top-5 per-generator PCs; a higher-k or nonlinear (kernel) correlation might find
  structure the linear top-5 check misses — deliberately not attempted here, consistent with "avoid neural
  networks/UMAP as primary evidence" and the decision (below) that nonlinear learning is not yet warranted.
