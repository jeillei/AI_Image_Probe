# SCIENTIFIC_AUDIT — what the SynthImage evidence establishes

Auditor's stance: adversarial.  Everything below is reproducible from cached features or from scripts in `scripts/` (commands at the end of `RESEARCH_REPORT.md`).
"Trajectory representation v1" = the frozen SD1.5/BLIP/256px/6-step feature set (44 legacy + 608 rich = 652 columns).  Confidence intervals are bootstrap over images unless stated.

## 1. Bottom line

1. **The headline numbers (Stage-B LOGO median 0.793; crossed 0.703) do not demonstrate that the trajectory measures synthetic provenance.**
   Four file-geometry numbers (min-side, aspect, is-square, is-power-of-two-sides) beat the full trajectory under the original E52-style protocol
   (**0.852 vs 0.766**, 5 real-fold seeds, sd 0.03/0.04) and under the crossed protocol (**0.774 vs 0.681**).  `is_square` *alone* gets 0.750.
2. **Two further single-mechanism shortcuts also match or exceed it:** BLIP caption text alone (no pixels): crossed macro AUROC **0.791**; bytes-per-pixel (an image-complexity proxy) in a resize/JPEG-matched subset: **0.696 vs 0.646**.
3. **The trajectory encodes acquisition history and dataset identity strongly.**  Real-vs-real (AIGC-benchmark vs RR): AUROC **0.772 [0.709, 0.831]** (permutation-null max 0.59); within the *single* AIGC-real source the features predict native image
   resolution with OOF Spearman **0.556**.  A 32-px pixel-space probe encodes real-source identity *better* than SD1.5 (**0.832 vs 0.752**) while carrying no crossed real/fake signal (0.507), i.e. source identity is a probe-agnostic coarse-statistics effect.
4. **Linear removal of five geometry columns cuts crossed macro AUROC 0.681 -> 0.550** (0.613 with two file-complexity columns added); cells above chance 95% -> 65%/70%.  Over-control is possible (geometry is a near label proxy for fakes), so read the range, not a point.
5. **One thing survives the crude controls: the guidance-gap family** (conditional-minus-unconditional score; 92 columns): crossed macro 0.671 raw, 0.644 after geometry residualization, 0.596 after also removing caption length/TF-IDF-SVD.  It is the family that least encodes source identity (0.49-0.56).  It was found post hoc among 12 groups, is partly caption-dependent, and is **a hypothesis, not a result**; the pre-registered content-matched pilot exists to test it.
6. **Cross-probe: a shared *source* axis, no shared *synthetic* axis (confounded corpus).**  SD1.5, DiT-XL/2 and CIFAR-32 all identify the real source (30 shared quantities: 0.81 / 0.78 / 0.83; source-shift effect vectors correlate 0.84 SD1.5-vs-DiT).  Real/fake effect vectors agree no better than a joint-label permutation null (p 0.32-0.66) and weight transfer is 0.49 / 0.55.
7. **Content-matched pilot (pre-registered, fresh data): a real separation exists for SD1.5-on-SD1.5.**  ALL_v1 0.800 [0.732, 0.879] (rule: SUPPORTS); eps-norm lower in 42/60 fakes (p 0.003, replicates E17); the guidance family is INCONCLUSIVE (0.631, CI-low 0.548).  But (a) this is the probe's own generator, the most favourable case; (b) **a VAE auto-encoding-error scalar at 512 px gets 0.866** and (c) **a 32-px unconditional CIFAR DDPM gets 0.816 (better than SD1.5's 0.697 and DiT's 0.741)**, while 12 thumbnail colour statistics get 0.668.  The signal is real but is *not specific to the SD1.5 probe's generative compatibility*: independent priors and trivial baselines reach or exceed it.

## 2. Evidence tiers

| tier | claim | why | source |
|---|---|---|---|
| **Trustworthy (measured, controls attached)** | trajectory features encode real-source identity and native-resolution history | permutation null, CIs, replicated by a second probe | E60, E61, E69 |
| Trustworthy | metadata/caption/complexity shortcuts rival v1 real-vs-fake performance | same protocol/code/folds as the claim they undercut | E62-E64, E67 |
| Trustworthy | corpus has structural class-geometry confound (all fakes square, single native size per generator) | direct file measurement | DATASET_AUDIT.md |
| Exploratory | linear residualization sizes (0.55/0.61) | over-control possible; non-monotone across nuisance sets | E65-E66 |
| Exploratory (post hoc) | guidance family retains signal | selected after looking at 12 groups x 2 modes | E66-E67 |
| Exploratory | normalization-pipeline and conditioning comparisons (n=200 core; 10 fakes/generator; macro SE >= 0.02) | small n; wide cell CIs (~0.16) | E71-E73 |
| **Confirmatory-eligible, pre-registered** | matched-content SD1.5 separation (0.80), eps-norm direction | fresh data, frozen code, rules fixed before data | E75 |
| Exploratory | matched-set controls (VAE 0.866 @512, CIFAR-32 0.816, thumbnail 0.668) | run after seeing E75 | E76-E78 |
| **Not established** | that v1 trajectories carry synthetic provenance | no result survives content, geometry and complexity controls simultaneously with adequate n | - |
| **Not established** | universality across generative priors | second probe only a weak control | E69 |
| Prior claims to treat as historical, not confirmatory | Stage-A/B LOGO 0.72/0.79; cross-source 0.70 | see contamination map below; confounded corpus | E28-E59 |

## 3. Experimental degrees of freedom (v1)

Choices that were made (and by what evidence), any of which could have been tuned:

| choice | value | selected using |
|---|---|---|
| probe | SD1.5 (`stable-diffusion-v1-5`), fp16 on MPS | availability; E7-E8 on **one** RR real image |
| resolution | 256 px, LANCZOS **squash** (no crop) | E7-E8 (n=1) and MPS cost |
| inversion | `DDIMInverseScheduler`, **6** steps, guidance 1.0, no clip | E7-E10 (n=1 to 20), cost |
| conditioning | BLIP-base caption (max 30 tokens, greedy), null/empty as control | E10-E13 (RR pilot, 20 images) |
| feature families | legacy 44 (E19, after E15-E17 COCO lead), rich 608 (E24) | E15-E17 (7 COCO pairs, one generator); **E23 generator-stratified curves on the 80-image Stage-A chunk informed the rich design** |
| classifier | Standardscaler + logistic, class-balanced; C grid {.001,.01,.1,1,10} chosen by inner generator-held-out validation | Stage-A/B inner folds |
| primary metric | median LOGO AUROC; later crossed | chosen after E20 (DALL-E 2 inverted) |
| generator/real sources | 8 -> 10 generators, 25 each; first-N sampling; 1-2 real sources | availability |
| `core_rich` subset | pre-specified before Stage-B (E50) and then *not* used because full rich was better | the one genuinely pre-registered choice |

## 4. Contamination map (which data influenced v1 design)

| data | used for | status now |
|---|---|---|
| RR pilot (20 imgs) + 1 RR image | steps=6, size, caption conditioning (E4-E13) | **contaminated** for anything about protocol |
| COCO controlled (7 pairs, one generator) | lead feature (eps norm), legacy family definition | contaminated; too small to be confirmatory |
| AIGC pilot (ADM, BigGAN, GLIDE, DALL-E 2; 32 imgs) | detector interface, "DALL-E 2 untouched test" | DALL-E 2 stopped being untouched at E21 |
| Stage-A 80-image chunk | rich family design after seeing generator curves (E23-E27) | **contaminated** |
| Stage-A full (400), Stage-B (450), cross-source (550) | all model comparisons, C selection, `core_rich` rejection | exploratory; repeatedly reused |
| SD1.5/Wukong 50 images | evaluated after `core_rich` freeze | only partially fresh (SD1.5 = probe's own family) |
| RR partial local archive (2,370: 1,500 AI / 870 real) | only 120 RR reals touched | large unused pool, **but** 91% PNG-AI vs 0% PNG-real; usable only through a normalization pipeline |
| COCO val2014 (this phase, 60) | content-matched pilot | fresh w.r.t. v1 design; pre-registered before generation |
| core-200 subset | this phase's normalization/conditioning ablations | drawn from Stage-B (not fresh); ablations only |

**Consequence:** no evaluation in the repository through E59 is confirmatory for v1.  The only confirmatory-eligible evaluation is the content-matched set (fresh images, frozen code, hypotheses pre-registered).

## 5. Ranked confounds (what could the signal be?)

| # | candidate | evidence for | evidence against / gap |
|---|---|---|---|
| 1 | **Image geometry / resampling history** (all fakes square, single native size) | 4-feature baseline 0.774-0.852 > trajectory; trajectory predicts native size within one real source (rho 0.56); residualization removes ~0.07-0.13 | crop-to-square did not change results (E71) -> not aspect distortion per se; band-limit to 128 px lowered ALL_v1 (0.596) and sent endpoint statistics below chance while source identity stayed 0.725 |
| 2 | **Semantic/content distribution** (ImageNet-class prompts vs scene photos) | caption TF-IDF alone 0.791; fake-indicative words are animals/objects; 32px probe (layout/colour only) separates real sources at 0.83 | matched pilot: separation persists with content matched (E75), but a 32-px prior matches it (E77) |
| 3 | **Image complexity/compressibility** | bpp scalar 0.696 in size- and JPEG-matched subset | single-scalar; sign flips across generators (bpp is 0.28 in LOGO) |
| 4 | **Real-source identity** | real-vs-real 0.77 | expected given 1-2 sources |
| 5 | JPEG/PNG state | ADM PNG vs rest; RR q-table 20 vs 36 | benchmark unified JPEG tables for all but ADM/RR |
| 6 | BLIP caption behaviour | fake captions shorter (7.7 vs 8.75 words); word-count alone 0.653 | E44 null-conditioning still gave 0.684 |
| 7 | SD-family compatibility | 4/10 generators SD-lineage | no elevated crossed AUROC for SD14/SD15/Wukong (0.69/0.68/0.45 vs AIGC-real) vs GLIDE 0.77 -> no evidence of self-compat effect at 256px |

## 6. What survives the strongest controls

* **Survives:** (i) a real, pre-registered separation of SD1.5 outputs from content-matched real photos (0.80); (ii) the predicted-noise-norm direction (fake lower; 42/60, replicating E17's 7/7); (iii) the guidance family's *relative* robustness (carries real/fake signal in every normalization, blind to source identity, needs coherent-or-neutral text) - still inconclusive at n=60.
* **Does not survive / not shown:** any statement that the signal is a property of the *probe's generative compatibility*, is *general across generators*, or exceeds simple baselines.  Every strong number in this repository has a matched-or-better trivial or independent competitor (geometry 0.85, caption text 0.79, complexity 0.70, VAE error 0.87@512, CIFAR-32 probe 0.82).
* **Untested but decisive:** a non-SD generator on the same matched contents (aMUSEd, pre-registered Addendum 1; queued), JPEG-history/upsample asymmetry on the matched set (queued), a VAE-free 256-px probe (queued).

## 7. Status of the original hypothesis: **weakened (unresolved for a narrow residual)**

*Hypothesis:* inverse trajectories of a generative prior contain a provenance-related signal that measures generative compatibility.
* Weakened because the evidence that motivated it (Stage-A/B LOGO, crossed 0.70) is beaten by file-geometry, caption-text and complexity shortcuts; ~90% of the 652 features are caption-independent statistics that strongly encode source identity and native-resolution history; three unrelated priors all encode source identity, none shares a synthetic axis; and the one clean positive (matched, SD1.5-self) is matched by a 32-px unconditional DDPM and a VAE-error scalar.
* Not falsified because a pre-registered fresh-data test came out positive (0.80), the eps-norm direction replicated, and a caption-conditioned guidance component remains unexplained by geometry and caption text.
* The remaining defensible claim: *SD1.5 outputs are separable from content-matched real photos by low-level statistics that several unrelated priors also see.*  Whether any of it is "provenance" rather than the coarse look of CFG-guided generation is unknown.

## 8. Handoff

**What we learned.** (1) Source identity, resolution history and image geometry are encoded in v1 and in every probe tried.  (2) Four geometry numbers beat the headline (0.852 vs 0.766 LOGO; 0.774 vs 0.681 crossed).  (3) Caption text alone reaches 0.791 in the confounded corpus.  (4) Removing geometry linearly collapses crossed AUROC to 0.55-0.61.  (5) Neither text alignment (null prompt: 0.642 vs 0.655) nor aspect/JPEG normalization is what drives v1.  (6) The guidance family alone needs coherent-or-neutral text (0.665 / 0.633 / 0.513 shuffled / degenerate null).  (7) Matched-content SD1.5 fakes are separable at 0.80, but a 32-px probe (0.82) and a VAE-error scalar (0.87 @512) match it.

**Signal:** matched SD1.5-self separation; eps-norm direction; possibly the guidance family.  **Confounding:** file geometry/resampling history, semantic distribution (caption text), image complexity, real-source identity (probe-agnostic), coarse global look of guided generations.

**Most important next experiment (exactly one):** finish the pre-registered non-SD generator on the same 60 matched contents (aMUSEd: `scripts/run_followup3_jobs.sh` then `scripts/evaluate_content_matched_multigen.py`), then re-score it against the CIFAR-32 and VAE-error baselines.  If SD1.5-probe features do not beat the 32-px probe on a generator that shares no decoder with SD1.5, the trajectory-provenance hypothesis is not supported.

Incomplete at hand-off (partial, not reported as results): JPEG-75 and band-128 on the matched set, aMUSEd generation, church-256 probe, HPC robustness run.  Commands are at the end of `RESEARCH_REPORT.md`.  One orphaned system-Python process (PID 20657, ppid 1, pre-dates this work) has been burning a CPU core; it is not from this repository and was left alone.
