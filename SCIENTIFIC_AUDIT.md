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
8. **The cross-generator decision test (aMUSEd, pre-registered Addendum 1, E81–E87) resolves the open question against the hypothesis.**  aMUSEd separates even more strongly than SD1.5 (ALL_v1 0.994 [0.988, 1.000]), but **a classifier trained on real-vs-SD1.5 is inverted on real-vs-aMUSEd (AUROC 0.393)**: there is no shared direction.  The eps-norm lead does not replicate (36/60 pairs, p = 0.155).  The guidance family's raw AUROC on aMUSEd (0.895) is fully explained by a VAE+CIFAR-32 baseline (incremental-information CI [-0.039, 0.067], crosses 0).  The only representation that adds information beyond VAE+CIFAR-32 on aMUSEd is the full 652-column fit (+0.081, CI [0.037, 0.135]) — and that same addition *hurts* on SD1.5 (-0.044) — a pattern consistent with overfitting a high-dimensional representation to 60 pairs rather than a located mechanism.  See §7 for the resulting hypothesis status.

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

## 6. What survives the strongest controls, including the aMUSEd decision experiment (E81–E87)

The pre-registered cross-generator test (Addendum 1) is now complete.  aMUSEd (masked-token generation, VQGAN decoder,
no shared VAE/denoiser/objective with SD1.5) on the same 60 matched contents/captions gives:

* **H_primary[aMUSEd] SUPPORTS**: ALL_v1 AUROC **0.994 [0.988, 1.000]** — the strongest single number in the project.
* **H_gen-transfer FAILS in both directions**: a classifier trained on real-vs-SD1.5 is *inverted* on real-vs-aMUSEd (AUROC **0.393**, CI [0.293, 0.496], entirely below 0.5); the reverse transfer is inconclusive (0.651, CI-low 0.5497, just under the pre-registered 0.55 support bar).
* **H_eps[aMUSEd] does NOT replicate**: 36/60 pairs (60%) fake-lower, p = 0.155 — versus SD1.5's 42/60, p = 0.003. The one mechanistically-motivated, repeatedly-observed lead (E10→E17→E75) is absent on the new generator.
* **H_guidance[aMUSEd] SUPPORTS by the naive rule** (0.895) but **adds zero information beyond a VAE+CIFAR-32 baseline** (paired-bootstrap CI [-0.039, 0.067], crosses 0) — its raw separability is redundant with generic low-level cues, not a distinct mechanism.
* **Incremental-information test (Phase 5, E86)**: for aMUSEd, only the full 652-column representation adds anything beyond VAE+CIFAR-32 (+0.081 AUROC, CI [0.037, 0.135]); every interpretable sub-family (guidance, eps-norm curve, eps-mean) adds nothing. For SD1.5, adding the full representation to the same baseline *actively hurts* (-0.044, CI entirely negative). A positive increment confined to the highest-dimensional, least interpretable configuration on 60 pairs (p/n ≈ 11) is the signature of in-fold overfitting, not a located mechanism — this is not proof of overfitting, but no result here rules it out or isolates which columns carry it.
* **Generic, unrelated probes match or exceed the trajectory on aMUSEd specifically**: VAE reconstruction error alone 0.869; DiT-XL/2's 30-quantity Class-B representation 0.940; CIFAR-32 (no VAE, 32 px, pixel-space) 0.748. A 12-statistic colour/contrast thumbnail baseline stays flat across generators (0.668 SD1.5 vs 0.675 aMUSEd), so aMUSEd's separability is *not* simply "it looks different" — it is something texture/decoder-level that several unrelated probes with VAE-like machinery pick up.

* **Survives:** (i) real, pre-registered real/synthetic separation for both SD1.5-self and aMUSEd; (ii) the guidance family's descriptive robustness (present, moderate, in every probe/normalization) though it carries no information beyond generic baselines on either generator now tested.
* **Does not survive:** the claim that a shared, transferable, SynthImage-specific direction exists (H_gen-transfer fails both ways); the eps-norm lead (fails on aMUSEd); the claim that the trajectory exceeds independent baselines in general (true only for aMUSEd's full 652-column fit, in a regime prone to overfitting).
* **Resolved this phase:** the "untested but decisive" item from the prior audit (a non-SD generator on the same matched contents) is now tested. It does not rescue the hypothesis in its general form.

## 7. Status of the original hypothesis: **unsupported**

*Hypothesis:* inverse trajectories of a generative prior contain a provenance-related signal that measures generative compatibility and generalizes across generator families.

* The decisive test — whether a direction learned from one generator transfers to an architecturally independent one — **fails**: 0.393 (inverted) one way, inconclusive (0.651) the other. The addendum's own fixed interpretation table calls this exact outcome pattern "two generator-specific signals; no shared direction."
* The one previously-replicating mechanistic lead (predicted-noise norm, fake lower) **does not replicate** on aMUSEd (p = 0.155).
* The one family that looked most promising after the SD1.5-only pass (guidance) **adds no information beyond a two-baseline VAE+CIFAR model** on either generator once tested properly (Phase 5).
* The only positive increment beyond generic baselines appears exclusively in the full, uninterpretable, 652-column fit on 60 pairs — a configuration where overfitting cannot be ruled out and no follow-up here isolates a cause.
* This is not "weakened" (a term implying residual, defensible uncertainty in favour of the hypothesis) because the specific residual carried forward from the prior audit — "does a non-SD generator show the same signal via the same mechanism" — was tested directly and came back negative on both counts that matter (transfer, and information beyond baselines in interpretable form). It is not "unresolved" because the pre-registered decision rules were followed exactly and gave a clean, non-contradictory (if unflattering) answer per the addendum's own table. It is **unsupported**: the evidence collected under the project's own rules does not support the claim that SD1.5 inverse trajectories carry a generator-general provenance signal.

## 8. Handoff

**What we learned (this phase).** (1) aMUSEd is separable from matched real photos at AUROC 0.994 — but so is a bare VAE encode/decode error (0.869) and a 30-feature DiT probe that shares SD1.5's VAE family (0.940), while a colour-only thumbnail baseline is flat (0.675). (2) A classifier trained on real-vs-SD1.5 is inverted on real-vs-aMUSEd (0.393): there is no shared "fake" direction between the two generators. (3) The eps-norm lead that survived every prior test (E10, E17, E75) does not replicate on aMUSEd (p = 0.155). (4) The guidance family's raw AUROC on aMUSEd (0.895) is fully explained by a VAE+CIFAR-32 baseline (incremental CI crosses 0); it is not adding a distinct mechanism. (5) The only representation that adds information beyond VAE+CIFAR-32 for aMUSEd is the full 652-column fit, and that same addition actively hurts on SD1.5 — a pattern more consistent with high-dimensional overfitting on 60 pairs than with a located mechanism.

**Signal:** none that is both generator-general and beyond generic low-level baselines.  SD1.5-self separation and aMUSEd separation are each real but do not share a direction and are each matched or exceeded by baselines unrelated to SD1.5's diffusion trajectory.
**Confounding:** decoder/VAE-vs-VQGAN incompatibility (visible to VAE-only and DiT probes), generic low-level image statistics, high-dimensional overfitting risk in the one place a positive increment appeared, and (from the earlier audit) file geometry, source identity, caption/semantic distribution, and image complexity in the original confounded corpus.

**Most important next experiment (exactly one):** none of the remaining exploratory items (JPEG-75/band-128 on the matched set, church-256 probe, a third generator, the robustness screen) should be run as a way to rescue this hypothesis. If this line of work continues, the next experiment should test the **specific overfitting question** left open by E86: does the full-representation increment on aMUSEd replicate on an independent aMUSEd-content split with feature selection frozen *before* seeing that split (e.g., a fresh 60-pair aMUSEd draw from different COCO contents, evaluating only the columns/families already named here)? If it does not replicate, the incremental-information result was overfitting and the hypothesis has no surviving positive evidence anywhere in the project.

Incomplete at hand-off (deliberately not run, per this phase's stability/priority instructions — not needed for the decision made above): JPEG-75 and band-128 on the matched set, church-256 probe extension, a third generator, the HPC robustness screen. Commands are at the end of `RESEARCH_REPORT.md`. One orphaned system-Python process (PID 20657, ppid 1, pre-dates this work) has continued using a CPU core throughout; it is not from this repository and was left alone.
