# SynthImage: generative-compatibility pilot

This repository tests whether a *probe diffusion model* explains natural and
synthetic images differently.  It is deliberately structured around
generator-held-out, corruption-matched evaluation; clean random splits are not
reported as principal evidence.

## Current status

The completed CIFAR DDPM work is an unconditional feasibility/control experiment.
The active probe is Stable Diffusion 1.5 on Apple MPS, using VAE latents,
`DDIMInverseScheduler`, null and BLIP-caption conditioning, and a constrained
continuous text-embedding optimizer. The current mechanism question is whether
the *response* of an inverse trajectory to better conditioning is a provenance
signal—not whether a high-capacity image classifier can separate images.

The first cached RRDataset cohort is invoked with:

```bash
uv run python scripts/caption_cohort.py --data data/rr_pilot --per-class 10
uv run python scripts/sd15_sensitivity.py --per-class 10 --steps 6
uv run python scripts/analyze_sensitivity.py
```

These commands run one model process at a time and checkpoint captions and
per-image results. The first 10+10 cohort is mixed: E10's one-image effect did
not reproduce as a reliable class-level reconstruction-benefit signal; no
classifier is trained from it. See `RESEARCH_REPORT.md` for the exact result.

### Controlled-content discovery cohort (active)

The next mechanism experiment holds high-level semantic content approximately
fixed: a COCO photograph and its synthetic counterpart share the same human
caption. COCO captions are used only as SD1.5 conditioning; provenance models
only receive the resulting numerical trajectory measurements. The small first
cohort is deliberately a validation/discovery set, not a generator-held-out
benchmark: it currently contains a single external synthetic source.

```bash
uv run python scripts/build_coco_karpathy_controlled.py --count 10
uv run python scripts/controlled_depth_sweep.py --groups 3 --depths 6 12
uv run python scripts/analyze_controlled_depth.py
```

The sweep runs serially, checkpoints every image, and canonicalizes every input
to the same 256px RGB pixels. `--partial-indices 2 4 6` adds scheduler-aligned
partial noising→denoising round trips for a small diagnostic subset. On this
MPS laptop, 12-step/partial round trips are much slower than six-step inversions.
The current, deliberately narrow lead is the caption-conditioned SD1.5 UNet
predicted-noise norm: it was lower for the synthetic counterpart in 7/7 paired
COCO examples (3 discovery, 4 held content). This is a single-source result,
not an AI detector claim; the next checks are transfer to RRDataset and an
independent generator.

### Analyze an arbitrary image

Give the probe a plausible content caption and it will write the numerical
trajectory signature for one image. It does **not** yet return a trustworthy
"AI" / "real" verdict: the current signal has only been established on a tiny,
single-source controlled cohort.

```bash
UV_CACHE_DIR=/tmp/synthimage-uv-cache uv run python scripts/analyze_image.py \
  /absolute/path/to/image.jpg \
  --caption "a person flying a kite on a beach at dusk" \
  --output results/ad_hoc/my_image.json
```

Use `--partial-indices 2 4 6` only when you need round-trip diagnostics; it is
substantially slower. The runner performs one foreground MPS operation and
exits, so it does not leave a model process running.

### Experimental detector command

The detector interface is implemented, but the first external held-generator
pilot failed: noise-curve logistic regression trained on ADM+BigGAN had AUROC
0.94 on GLIDE validation and **0.19** on untouched DALL-E 2 (n=4 AI; 8 real),
with zero TPR at the validation-frozen 1%/5% FPR thresholds. Therefore do not
use it for provenance decisions.

```bash
UV_CACHE_DIR=/tmp/synthimage-uv-cache uv run python scripts/detect.py image.jpg \
  --caption "a faithful description of the image" \
  --model results/detector/aigc_noise.joblib --json \
  --output results/ad_hoc/detection.json
```

The command emits an explicitly experimental score. `--caption` avoids an
extra BLIP pass; omit it to caption automatically.

### Scaled signature dataset (in progress)

The project now uses an auditable Parquet master manifest and balanced,
resumable extraction chunks. The first Stage-A checkpoint contains 200 real
images and 25 each from eight labelled generators (ADM, BigGAN, DALL-E 2,
GLIDE, Midjourney, SD1.4, SDXL, VQDM). The first balanced 80-image signature
chunk is cached at `results/features/sd15_signature_stage_a_chunk000.parquet`.
It is a scaling checkpoint, not a final detector dataset.

The exact same 80 images have also been re-extracted with the frozen SD1.5
protocol and a richer signature at
`results/features/sd15_rich_signature_stage_a_chunk000.parquet`. It retains
the original 51-feature baseline and adds 608 summaries of score
direction/evolution, score-to-latent-step alignment, conditional geometry,
spatial/frequency score structure, latent typicality and cross-timestep
correlations—without extra UNet calls. Legacy values reproduced within a
maximum absolute difference of `2.4e-5` across the full MPS batch-2 rerun.
This remains an n=5/generator discovery checkpoint, not detector evidence;
pre-specified plots are at `results/analysis/rich_stage_a/`.

### Current Stage-A result

The completed Stage-A table now has 400 images (200 real; 25 each from ADM,
BigGAN, DALL-E 2, GLIDE, Midjourney, SD1.4, SDXL and VQDM) at
`results/features/sd15_rich_signature_stage_a_full.parquet`. In nested
leave-one-generator-out evaluation, raw rich logistic regression achieved
median AUROC **0.722**, compared with **0.650** for the legacy signature. This
is a useful representation improvement, especially for SDXL (0.690 rich vs
0.339 legacy), but it is not a dependable detector: DALL-E 2 was 0.421 and the
rich model's worst source was 0.421. Do not use `detect.py` for provenance
decisions yet.

### Prompt-ensemble pilot

Five plausible caption forms were evaluated on the balanced 80-image pilot.
Prompt averaging alone did not improve held-generator performance (median
AUROC 0.520 vs 0.540 for a single caption). Adding trajectory variability
across prompts raised the small-pilot median to 0.640, but worsened the
worst-source result and used only five synthetic test images/source. The result
is exploratory: prompt sensitivity is a possible feature, not a validated
improvement. Cached prompt features, aggregate tables and plots are under
`results/prompt_ensemble/`; see the research report for generator-level detail.

### User-prompt-free conditioning result

Normal use requires only an image: `scripts/detect.py image.jpg` automatically
infers an internal BLIP caption. No original generator prompt, user prompt,
seed, sampler or generation metadata is used. A matched full-Stage-A control
finds automatic conditioning is useful (median held-generator AUROC **0.744**)
relative to null conditioning (**0.684**). However, the conditioning-response
signature (automatic minus null features) is not generator-general (median
0.572). SynthImage is therefore user-prompt-free but automatically conditioned;
it remains research-only because the score is still uneven across generators.

### Stage-B scaling checkpoint

Stage-B expands to 450 cached signatures: 200 real images and 25 examples from
10 labelled generators (the Stage-A eight plus SD1.5 and Wukong). Nested
leave-one-generator-out full-rich logistic performance rose from Stage-A's
0.722 median AUROC to **0.793**; combined rich+legacy reaches **0.797**.
This is meaningful evidence that generator diversity helps, but it is not a
working detector: the worst held source remains 0.500–0.506 and Stage-B still
has only one real-image source. The full rich representation outperformed the
pre-specified compact `core_rich` (0.753), so the current experimental model
continues to use the full rich signature.

### Cross-source warning

Stage-B now includes 100 duplicate-audited RR real photographs as a second
real source. When both a synthetic generator and the real source are held out,
full-rich median AUROC is 0.737 against unseen RR-real and 0.688 against unseen
AIGC-real (0.703 across all crossed pairs). This exposes substantial real-source
shift: SynthImage is not yet a reliable arbitrary-image detector, despite its
stronger within-source LOGO result. See `results/stage_b_scaled/` for the
cross-source metrics and real-source score shifts.

The strongest rich families were spatial score structure (median AUROC 0.720)
and score temporal dynamics (0.706). Generator diversity matters: mean AUROC
rose from 0.514 with one training source to 0.688 with six/seven. However,
trajectory features predict generator identity at 61.5% for eight sources,
showing that remaining signal is substantially source-specific. Full metrics,
learning curves and diagnostic results are in `results/stage_a_full/`.

An **experimental** arbitrary-image interface is available. It runs BLIP caption
conditioning and the six-step SD1.5 rich extractor, then applies the Stage-A
all-data logistic model. Its score is not calibrated provenance evidence and
must not be used for consequential decisions.

```bash
UV_CACHE_DIR=/tmp/synthimage-uv-cache uv run python scripts/detect.py image.jpg \
  --caption "a faithful image description" \
  --model results/detector/sd15_rich_stage_a_experimental.joblib --json
```

```bash
uv run python scripts/build_master_manifest.py data/aigc_stage_a/manifest.csv
uv run python scripts/sample_extraction_manifest.py --per-generator 5 \
  --real-per-generator 5 --output data/extraction_chunks/chunk.csv
uv run python scripts/extract_detector_features.py --manifest data/extraction_chunks/chunk.csv \
  --output results/features/chunk.json --batch-size 2
uv run python scripts/materialize_signature_parquet.py --input results/features/chunk.json
uv run python scripts/extract_detector_features.py --manifest data/extraction_chunks/chunk.csv \
  --output results/features/rich_chunk.json --batch-size 2 --rich
uv run python scripts/analyze_rich_checkpoint.py --input results/features/rich_chunk.json
```

Batch-2 extraction is numerically equivalent to serial extraction (maximum
feature difference below `6.5e-7`) and was 1.34× faster in the local benchmark.

## Run

```bash
uv run python scripts/download_pilot_real.py --count 40
uv run python scripts/run_pilot.py --data data --output results/pilot
```

For a real benchmark, place images in `data/real/<source>/` and
`data/synthetic/<generator>/`; the runner uses parent directories as group IDs
and keeps groups disjoint.  RRDataset is public at Zenodo record 14963880 and
its code is at `ChunXiaostudy/RRDataset`; GenImage's documented layout is also
supported by arranging its `ai` and `nature` folders under those roots.

## Design constraints

* All images are decoded to RGB, EXIF-stripped, resized identically, and
  re-encoded before features.
* The same seeded corruption-chain distribution is applied independently to
  both labels.
* A metadata/size and frequency shortcut baseline is reported alongside
  trajectory-only, reconstruction-only, endpoint-only, full, and raw-pixel
  embedding baselines.
* Results under fewer than two held-out generator groups are flagged as
  insufficient rather than over-interpreted.
* The controlled cohort records content and generator IDs for splitting, but
  neither is emitted as a classifier feature.
