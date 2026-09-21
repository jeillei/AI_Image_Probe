# Pre-registration: content-matched pilot (written before any content-matched feature was extracted)

Committed to git **before** SD1.5 counterparts were generated and before any v1 feature exists for this set.
(Exploratory findings already in hand that motivate it: geometry, caption-text and file-complexity baselines each rival the v1 trajectory
in the confounded corpus; see `SCIENTIFIC_AUDIT.md`.  The guidance-family result was found *post hoc* among 12 groups and is treated as a hypothesis here.)

## Data (frozen by `scripts/build_content_matched.py`, seed 2026, count 60)
60 COCO val2014 photographs (deterministic shuffle; label/feature blind).  Real = centre-crop square -> 512 PNG.  Fake = SD1.5 (25-step DDIM, CFG 7.5, seed = hash(content_id))
prompted with the same human COCO caption -> 512 PNG.  Probe conditioning for BOTH classes = that human caption (BLIP bypassed).  Protocol = frozen v1 (SD1.5, 256px squash-resize
[identity-equivalent for square inputs], 6-step DDIM inversion, rich features).  **No v1 code or feature definition may change.**

## Why SD1.5-on-SD1.5 is the most favourable case
The fake generator and the probe are the same model family and weights.  If any generative-compatibility provenance signal exists, this is where it should be largest.

## Hypotheses and pre-specified decision rules
Unit of resampling = content pair.  Classifier: StandardScaler + LogisticRegression(C=0.1, class_weight=balanced), **content-grouped** 5-fold CV repeated 10x (a real photo and its counterpart are never split).

* **H_primary**: AUROC(ALL_v1) > 0.5.  *Supports v1* if the 95% content-bootstrap CI lower bound > 0.55; *disconfirms v1 for matched content* if the CI includes 0.5.  (n=60 pairs gives SE ~0.05.)
* **H_guidance** (post-hoc-origin hypothesis): AUROC(guidance family, 92 cols) > 0.5 with the same rule.  Because this hypothesis was selected after inspection, it is only *confirmatory* here if the rule is met.
* **H_eps** (the E17 lead): paired difference of `eps_mean` (fake - real, same content) is negative; report fraction of pairs negative and a sign-test p-value.  (E17 saw 4/4 held-content.)
* **H_transfer**: a classifier fit on the 550-image confounded corpus and applied to the matched set: report AUROC only (no threshold).  Expected under "confound-only": ~0.5.

## Interpretation table (fixed in advance)
| outcome | reading |
|---|---|
| H_primary and H_guidance both disconfirmed | v1 trajectory has no detectable SD1.5-self provenance signal beyond confounds; broad hypothesis strongly weakened for v1 |
| H_primary supported, H_guidance not | some real/synthetic separation under matched content, family not localized; probe-family effect cannot be ruled out |
| both supported | candidate signal; must next replicate with an independent generator and an independent probe before any claim |
| H_transfer > 0.6 while grouped-CV disconfirmed | confounded-corpus classifier captures something that survives content matching; investigate, do not claim |

No metric other than the above is a headline.  Any additional analysis is labelled exploratory.  Failed and inverted results are reported.
