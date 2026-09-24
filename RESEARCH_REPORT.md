# Research report

## Running experiment ledger

| experiment | hypothesis | change | dataset | result | interpretation | next action |
|---|---|---|---|---|---|---|
| E0 environment audit | A native SDXL/FLUX probe may be feasible | inspected hardware/packages | empty workspace | CPU-only Apple Silicon, no Torch initially, 86 GB free | SDXL/FLUX full inversion is not an appropriate first run; real small diffusion probe is feasible | implement deterministic DDIM inversion with a public DDPM probe |
| E1 pilot setup | trajectory feature extraction can run end-to-end | installed PyTorch/Diffusers; add public CIFAR DDPM probe and matched chains | 3 Picsum natural photos, 1 ImageGen image | completed: inversion/reconstruction stable; image-level features saved; no classifier fitted | pipeline and matched Level-3 chain work, but n=4 and only one group/class cannot test the hypothesis | expand to true generator-held-out benchmark |
| E2 robustness sanity check | aggregate trajectories survive circulation more gracefully than superficial encoding signals | same probe, Level-3 4–6-operation chain | same n=4 smoke subset | completed; no numerical failure | robustness mechanism is executable, but sample size prohibits inference | add multiple real sources and at least 3 independent synthetic generators |
| E3 split-guard diagnostic | a held-group classifier can be reported on the tiny set | two ImageGen contents and three content/source folders under Level 3 | n=5 after cap | initial AUROC=0.0 for all variants | training had one synthetic image and testing one; value is an arbitrary ordering, not evidence | enforce minimum five train/test images per class and suppress metrics below it |
| E4 RR clean | a mismatched trained diffusion probe has a compatibility signal on real-world content | RRDataset train→validation split; 8/class; 4 DDIM steps | RRDataset partial archive | trajectory AUROC 0.484; full 0.156; thumbnail 0.406 | no useful clean trajectory signal; endpoint 0.563 is weak/exploratory | apply matched Level-3 chain |
| E5 RR Level-3 | aggregate signal survives circulation | same split, 4–6 randomized matched operations | same | trajectory 0.531; full 0.500; endpoint 0.719 but TPR@1%=0 | no low-FPR utility | increase inversion to 8 steps |
| E6 inversion-quality control | insufficient trajectory steps caused E4–E5 | same frozen Level-3 split, 8 DDIM steps | same | trajectory 0.453; full 0.391; endpoint 0.656; thumbnail 0.422 | more faithful recurrence does not rescue trajectory/full detection | bounded negative conclusion |
| E7 SD1.5 inversion repair | native conditioned latent inversion can run on MPS | SD1.5, 256px, VAE + DDIM inverse scheduler, one RR real image | n=1 diagnostic | VAE MSE 0.0098; initial manual inverse MSE 0.126; dedicated inverse scheduler MSE 0.081 | native latent inversion is stable but coarse at 6 steps | test conditioning modes before classification |
| E8 conditioning control | semantic conditioning changes trajectory quality | same image, 6 DDIM steps | n=1 diagnostic | null pixel MSE 0.172; supplied photographic prompt 0.081 | conditioning strongly changes the inverse explanation | add regularized continuous-conditioning optimization |
| E9 optimized-conditioning smoke test | a constrained continuous explanation can be optimized without leaving the text manifold | frozen SD1.5 weights; 256px; 2 DDIM steps; 1 Adam update; L2 prior 0.01 | n=1 RR real diagnostic | finite loss 0.151; embedding displacement 0.00196; pixel MSE 0.094 | autograd through conditioned inverse/reverse paths works on MPS at this small scale | compare multiple updates and caption initializations once captioner is available |
| E10 conditioned sensitivity diagnostic | meaningful conditioning changes inverse explanation differently for a real and synthetic image | SD1.5, 256px, 6 DDIM inverse steps; matched RR val examples | n=1 real + n=1 AI | real pixel MSE: null 0.172, generic 0.081, specific 0.078; AI: null 0.0334, generic 0.0337 | conditioning sensitivity is present and directionally supports the hypothesis, but n=1/class and content are confounded | cache automatic captions and run a serial multi-image pilot |
| E11 automatic-caption validation | a feasible automatic caption initializes meaningful SD conditioning | BLIP-base, cache-only after initial download; RR validation images | 10 real + 10 AI captions | 19/20 broadly relevant; one malformed caption | caption quality is sufficient for initialization but is not a detector feature | freeze captions and test null→caption response |
| E12 multi-image null→caption sensitivity | E10 reconstruction/trajectory response generalizes | SD1.5 256px, 6 inverse steps, cached BLIP captions, canonical RGB | 10 real + 10 AI RR val images | reconstruction-benefit means 0.0300 vs 0.0296; normalized benefit AUROC 0.60; trajectory AUROC 0.64; endpoint AUROC 0.60; guidance AUROC 0.61; all mean-difference bootstrap CIs cross 0 | E10 reconstruction effect does not replicate; weak response features are content-confounded and uncertain | constrained optimized-conditioning subset |
| E13 optimized-conditioning subset | optimization behavior adds a response signal beyond captions | frozen SD1.5, 2 inverse steps, 3 Adam updates, L2=0.01 | 2 real + 2 AI from E12 | losses decrease for all; normalized reconstruction gain real: +6.8%, -1.6%; AI: +2.1%, +3.1% | optimizer converges in-manifold but there is no stable class ordering | report mixed negative; do not fit detector |
| E14 controlled-content construction | content matching can remove the dominant RR confound | COCO 2014 Karpathy captions; real photo plus independently generated counterpart per caption | 10 human-captioned COCO groups constructed; first 3 paired with one external generator | manifest records content ID, caption, provenance and source generator; metadata is not a feature | semantic intervention only; one source cannot establish generator-general behavior | validate conditioned inversion and feature curves |
| E15 controlled shallow trajectory sweep | compatibility may appear under fixed caption | SD1.5 caption-conditioned DDIM inversion, 6 steps; geometry, endpoint, UNet prediction, guidance, full round trip | 3 paired COCO content groups | synthetic minus real full pixel MSE negative in 3/3 pairs (mean -0.0199); latent MSE negative in 3/3 (mean -0.0863); other families have mixed signs | a round-trip lead, not a provenance finding | validate partial round trips and independent held content |
| E16 partial round-trip validation | the endpoint lead persists at intermediate depths and reverse scheduling is valid | scheduler-aligned reverse suffixes at indices 2/4/6 | 1 controlled COCO pair | real pixel MSE: 0.0090, 0.0304, 0.0534; synthetic: 0.0107, 0.0243, 0.0266 | partial reconstruction worsens with depth as expected; tentative separation occurs at 4/6, but n=1 | validate shallow round-trip candidate before any classifier |
| E17 held-content shallow validation | the E15 lead survives novel captions/content without feature tuning | pre-specified first 3 manifest groups discovery, next 4 held; SD1.5 caption-conditioned 6-step extraction | 7 paired COCO groups, one external source | held-content: UNet predicted-noise norm lower for AI in 4/4, paired mean -0.0285 (95% bootstrap [-0.0538,-0.0125]); guidance norm lower in 4/4; pixel/latent round trip lower in 3/4 | score-prediction magnitude is the most coherent candidate; round trip is weaker but directionally recurring | transfer the frozen feature definitions to RR and add an independent source before fitting a detector |
| E18 generator-labelled pilot construction | frozen feature can be tested across distinct source generators | downloaded API subset from AIGC Detection Benchmark; ADM+BigGAN train, GLIDE validation, DALL-E 2 final test, split-disjoint real images | 32 images: 4 AI/source; 16 real | public labels and pixels downloaded, dimensions/URLs/metadata excluded from feature vectors | small but genuine generator-held-out pilot; benchmark is not content paired | serial feature extraction with cached BLIP captions |
| E19 frozen signature extraction | six-step SD1.5 numerical features remain executable beyond controlled COCO | BLIP caption → SD1.5 inversion; noise curve, guidance curve, round-trip, endpoint, geometry; one model at a time | E18 32 images | 32/32 rows checkpointed in 20.4 minutes; no simultaneous BLIP/SD process | practical cost is ~38 s/image on MPS, so scale requires stronger evidence | logistic family ablations |
| E20 first held-generator detector | predicted-noise signature supports an unseen-generator detector | regularized logistic models; fixed ADM+BigGAN train, GLIDE threshold-validation, DALL-E 2 untouched test | E18 features | noise: GLIDE AUROC 0.938, DALL-E 2 AUROC 0.188 (CI [0.00,0.50]), TPR@1/5%=0. Full: 0.875→0.188; endpoint: 1.00→0.344; guidance: 0.625→0.344; round-trip: 0.125→0.563, all TPR@1%=0 | the controlled-source mechanism does not yield a working generator-general detector in this pilot; DALL-E 2 distribution reverses the learned score | do not run circulation or publish detector claims; retain interface as explicitly experimental and expand only if new multi-source evidence warrants it |
| E21 scalable corpus manifest | a multi-generator cached corpus can be acquired and audited without feature leakage | persistent Parquet master manifest; generator family, source, audit-only native metadata, content/split/status fields | public AIGC benchmark Stage-A first acquisition chunk | 200 real + 25 each from ADM, BigGAN, DALL-E 2, GLIDE, Midjourney, SD1.4, SDXL, VQDM; 400 local images | generator diversity now exists, but 25/generator is below the 100/generator Stage-A target and real source is still one benchmark source | extract a balanced signature chunk and assess throughput/curves |
| E22 batch extraction validation | batch-2 improves throughput without changing the frozen signature | batched VAE/UNet six-step inversion versus serial on same two images | 2-image numerical control | 29.85 s serial vs 22.22 s batch-2 (1.34×); max forward difference 5.36e-5; reconstruction difference 1.16e-4; feature difference <6.5e-7 | safe throughput improvement on MPS; batch size remains deliberately capped at 2 | use it for subsequent chunks |
| E23 first balanced multi-generator feature chunk | raw timestep curves reveal generator-specific structure at larger diversity | BLIP→SD1.5, 256px, six steps, batch-2, cached raw E/G curves | 40 real + 5 each from 8 generators (80 total) | all 80 signatures cached; late E_t means: ADM 0.914, BigGAN 0.880, DALL-E 2 0.853, GLIDE 0.821, Midjourney 0.858, SD14 0.822, SDXL 0.842, VQDM 0.882, real 0.872 | source curves differ in both directions; supports real-manifold/anomaly formulations and explains why a universal signed classifier can reverse | acquire/extract more images before fitting a new classifier |
| E24 rich extractor implementation | compressed norm curves may discard provenance-relevant structure | retain conditional/unconditional scores and latent states transiently; add score evolution/direction, score–step alignment, conditional geometry, spatial/frequency score summaries, latent typicality and cross-time correlations | frozen Stage-A protocol | 608 rich numerical features added with no additional UNet/VAE passes; scheduler transition residual intentionally excluded because it is tautological under DDIM inversion | this is a representation upgrade, not a detector result; it isolates the information retained from the same six SD1.5 steps | numerical legacy-feature equivalence control |
| E25 rich equivalence control | upgrading summaries must not alter the original 51-column baseline | re-extract two Stage-A images batch-2 with captured score tensors | 2-image numerical control | all 44 legacy numerical columns shared by JSON rows reproduced within 6.47e-7 and 3.15e-7; batch-2 score capture stable | legacy and rich models can be compared fairly | re-extract the exact original 80-image checkpoint |
| E26 rich 80-image checkpoint | the richer representation can be extracted at practical scale without changing membership/probe settings | SD1.5, BLIP caption, 256px, DDIM inverse six steps, batch-2; same 40 real + 5/source images as E23 | 80 images, 8 generators | 80/80 unique checkpointed rows; Parquet has 659 total columns: 51 audit/legacy columns plus 608 rich features; full legacy rerun max absolute difference 2.40e-5 (mean 2.11e-7) | small MPS batch arithmetic causes negligible variation only; old signature remains intact | descriptive, pre-specified rich-curve analysis; do not fit a detector at n=80 |
| E27 rich descriptive analysis | rich score direction, temporal evolution and spatial structure expose more mechanism than norm alone | generator-stratified pre-specified curves for score norm/change, conditional and guidance alignment, latent movement, score spectral high/low ratio | E26 80-image discovery checkpoint | plots/CSV written; no feature ranking or classifier selected | this cohort is suitable to detect implementation failures and broad source shifts, but n=5/source cannot support a provenance claim | complete remaining 320 cached images, then use identical leave-generator-out splits for old/rich comparison |

## Methodology decisions

The initial probe is `google/ddpm-cifar10-32`: a genuine trained diffusion model
but deliberately mismatched to natural 256px images. This is a feasibility and
mismatch-control probe, not an SDXL result. Its inversion is deterministic
DDIM-style forward propagation with model predicted noise, followed by standard
DDIM reverse reconstruction. Features include reconstruction errors, total path
length, per-quarter velocity, acceleration/direction changes, and endpoint
moments.  The posterior approximation will be added only after inversion is
validated.  The local simulator applies randomized JPEG/WebP/resampling/crop/
blur/sharpen/colour chains after RGB canonicalization and metadata stripping.

RRDataset/RRBench is public at Zenodo `14963880`; its repository is
`ChunXiaostudy/RRDataset`. The full payload has not been downloaded because the
initial purpose is falsification on a small pilot and its storage size was not
published by the repository. GenImage documents generator-separated folders,
but its legacy public test archive URL returned HTTP 404 on 2026-08-27.

### E1 diagnostic

On the clean smoke set, reconstruction MSE ranged from 0.064–0.102 and all
endpoint kurtoses were 3.07–3.38, so the DDIM recurrence did not diverge. The
one synthetic image had a lower path length (0.636) than the three natural
images (0.778–0.789), but it also differs in content and there is only one
sample. This is explicitly not treated as a signal. The first run failed before
loading due to sandboxed network access; after checkpoint caching it completed.
The probe now uses `local_files_only=True` to make later runs deterministic.

### E3 diagnostic and repair

The initial group code would have published an AUROC even when each held class
had only one image. It yielded 0.0 for reconstruction, endpoint, trajectory,
and full variants. This is a split-size failure, not evidence against the
hypothesis. The evaluator now returns `insufficient_group_samples` unless both
classes have at least five train and five held-out images. The two synthetic
assets are both from ImageGen, so even a larger result from them would not be a
generator-held-out evaluation.

## Result: bounded negative

The first genuine held split was RRDataset's published train/validation
partition. Images were decoded to canonical 32px RGB; metadata and names were
excluded; and the local circulation chain was independently seeded for every
image regardless of label. This is a real-vs-AI generalization test, not a
generator-held-out test: the RRDataset archive does not expose generator IDs.

| features | clean, 4 steps | Level-3, 4 steps | Level-3, 8 steps |
|---|---:|---:|---:|
| reconstruction AUROC | 0.234 | 0.312 | 0.297 |
| endpoint AUROC | 0.563 | 0.719 | 0.656 |
| trajectory geometry AUROC | 0.484 | 0.531 | 0.453 |
| full trajectory AUROC | 0.156 | 0.500 | 0.391 |
| RGB-thumbnail visual baseline AUROC | 0.406 | 0.422 | 0.422 |

At Level 3 the 8-step trajectory and full models have zero TPR at both 1% and
5% FPR. Endpoint features rank above chance on this small validation sample,
but have no low-FPR operating point. The evidence is therefore negative for
this formulation: an unconditional CIFAR-10 DDPM, strongly mismatched to
256px natural images, does not yield a usable robust dynamics detector.

This does not test the broader SDXL/FLUX conditioned, posterior-ensemble
hypothesis. A credible continuation needs a native latent inversion probe,
multiple conditioning explanations, and a generator-labelled corpus.

## Conditioned-latent sensitivity result

E11–E13 are the first modern conditional-probe test. BLIP captions are cached
only to form SD text embeddings; their text and filenames never enter any
classifier feature vector. The 10+10 RR validation cohort used identical 256px
RGB decoding and six-step inverse schedules for both labels.

| signal | real mean | AI mean | orientation-free AUROC | bootstrap 95% CI for real − AI mean |
|---|---:|---:|---:|---:|
| caption reconstruction benefit | 0.0300 | 0.0296 | 0.57 | [-0.0337, 0.0349] |
| normalized caption benefit | 0.099 | 0.268 | 0.60 | [-0.918, 0.302] |
| null→caption trajectory displacement | 0.0214 | 0.0234 | 0.64 | [-0.0072, 0.0035] |
| endpoint displacement | 0.0435 | 0.0467 | 0.60 | [-0.0136, 0.0078] |
| guidance response | 0.0378 | 0.0426 | 0.61 | [-0.0148, 0.0053] |

The preliminary E10 pattern is **not reproduced** as a reliable
reconstruction-benefit difference. The small response-feature AUROCs are not
statistically stable in this cohort. Captions show a major content imbalance:
the real selection is mostly people while synthetic images include disasters,
animals, objects, and architecture. Consequently, no detector was fit and no
claim is made that conditioning sensitivity is a provenance signal. This is a
mixed/negative replication, with valid SD inversion, meaningful automatic
captions, and bounded continuous-conditioning optimization established.

## Controlled-content trajectory discovery (E14–E16, preliminary)

E14 constructs `data/controlled_coco/manifest.csv` from COCO 2014 Karpathy
validation examples: each content ID has a real photograph and an independently
generated image supplied the same human caption. Inputs are EXIF-transposed,
converted to RGB and resized identically before the SD probe. Content and source
generator identifiers are retained only for paired/held-out evaluation.

This Stage-A slice has only three content groups and one external synthetic
source. It is **not** a detector result or evidence of unseen-generator
generalization. It does, however, establish that conditioned SD1.5 curves and
scheduler-aligned partial DDIM round trips run consistently. The shallow lead
is full round-trip compatibility:

| six-step paired statistic (AI − real) | mean | signs across 3 pairs |
|---|---:|---:|
| pixel full-round-trip MSE | -0.0199 | 3/3 lower for AI |
| latent full-round-trip MSE | -0.0863 | 3/3 lower for AI |
| path length | -0.0268 | mixed (1/3 higher) |
| UNet predicted-noise norm | -0.0204 | 3/3 lower for AI |
| guidance-response norm | +0.0005 | mixed (1/3 higher) |
| endpoint norm/variance/kurtosis | small | mixed |

For the first partial-depth pair, synthetic pixel errors at indices 2, 4 and 6
were 0.0107, 0.0243 and 0.0266, versus 0.0090, 0.0304 and 0.0534 for the real
photo. Both worsen with noising depth, as expected; only depths 4 and 6 support
the tentative difference. A single depth-12 pair was checkpointed, but 12-step
processing took several minutes per image on MPS. No depth preference is claimed
from it. The next pre-specified test is a shallow six-step validation across
held semantic content, without classifier selection or tuning against these
three discovery pairs.

### Held-content validation correction (E17)

The discovery/validation partition follows the frozen **manifest order**, not
alphabetical image IDs: the first three contents are discovery and the next four
are held semantic validation. A briefly produced alphabetical summary was
discarded before interpretation. At six steps, the conditional UNet predicted
noise norm is lower for the synthetic counterpart in all seven paired examples:
discovery mean AI−real = -0.0204 (3/3) and held-content mean = -0.0285 (4/4,
bootstrap 95% CI [-0.0538, -0.0125]). The conditional guidance-response norm
is also lower in held content (4/4; mean -0.0189), though it was mixed in the
three discovery pairs. Pixel and latent full round-trip error are lower for AI
in 6/7 pairs, but their held-content CIs include zero because one caption pair
reverses. Trajectory path length remains inconsistent.

This is the first **positive mechanism candidate** in the project: with content
held approximately fixed, an SD1.5 probe's conditional score magnitude changes
systematically for this external synthetic source. It remains strictly
single-source, tiny-n evidence; it could still be a source-specific rendering
or prompt-following effect rather than generator-independent provenance. The
feature definitions are now frozen for transfer and independent-generator
testing; no detector has been trained on this cohort.

## First detector attempt and held-generator result (E18–E20)

E18 used a compact subset of the public AIGC Detection Benchmark API: four
images each from ADM, BigGAN, GLIDE and DALL-E 2 plus split-disjoint real images.
The classifier never receives source name, path, URL, image size, metadata,
caption text or content ID. BLIP captions are used solely to form SD1.5 text
conditioning. E19 then extracted the frozen six-step signature serially: the
predicted-noise curve (mean/quantiles/early-middle-late/slope/AUC), guidance
curve, full pixel/latent round trips, endpoint moments and a small geometry
control. This cost 20.4 minutes for 32 images on MPS (~38 s/image).

The split was frozen before model fitting: ADM+BigGAN train; GLIDE selects
thresholds; DALL-E 2 is held out completely. Logistic-regression ablations:

| feature family | GLIDE validation AUROC | DALL-E 2 final AUROC | DALL-E 2 TPR @ validation-frozen 1% FPR |
|---|---:|---:|---:|
| predicted-noise curve | 0.938 | 0.188 | 0.00 |
| guidance curve | 0.625 | 0.344 | 0.00 |
| round-trip | 0.125 | 0.563 | 0.00 |
| endpoint | 1.000 | 0.344 | 0.25 |
| geometry | 0.625 | 0.188 | 0.00 |
| noise + guidance | 0.938 | 0.188 | 0.00 |
| full signature | 0.875 | 0.188 | 0.00 |

The DALL-E 2 held-test set is only four AI/ eight real images, so intervals are
wide; importantly, the noise AUROC direction is *below chance* and its 95%
bootstrap CI is [0.00, 0.50]. This is sufficient to reject the requested
success claim for the current prototype: the controlled COCO feature does not
generalize reliably to this unseen generator. Since it already fails clean,
circulation robustness and RR transfer cannot establish a practical detector
and were not used to tune around the failure. The repository nevertheless now
has a cacheable extractor, feature-only logistic/boosted training route and an
explicitly experimental `detect.py` command; these must not be used for real
provenance decisions until multi-generator validation succeeds.

## Scaling infrastructure and first multi-generator checkpoint (E21–E23)

The project now writes `data/master_manifest.parquet`, with local path, label,
generator and family, real source, source dataset, content/split IDs, native
properties for auditing, and caption/feature state. These fields are never
passed to the detector. The current public source is the AIGC Detection
Benchmark; the first retrieval checkpoint has 25 images from each of eight
generator families and 200 real images. Its row API rejects offsets after 1,000
in this environment. Direct Parquet shards are available but currently download
at only ~0.2–0.4 MB/s (each is ~500 MB), so their incomplete download was stopped
and is not part of the experiment.

E22 added a batch-2 mode after a direct numerical control showed sub-1e-6
feature differences from serial processing and a 1.34× speedup. E23 processed
a balanced 80-image checkpoint in ~20 minutes, storing 51 feature columns,
including all six predicted-noise and guidance timestep values. The mean noise
curves already differ materially by source, with some sources above the real
curve late (ADM/VQDM) and others below (DALL-E 2/GLIDE/SD14). This is not an
accuracy result at n=5/generator, but it is a mechanistic reason to compare
real-manifold distances and curve-aware models rather than assuming a single
global sign for synthetic images. No scaled detector metric is reported yet.

## Rich trajectory representation checkpoint (E24–E27)

E24 preserves the frozen SD1.5/BLIP/256px/six-step protocol and the original
signature, while retaining the score and latent tensors long enough to compute
608 additional compact summaries. They cover score temporal change and angular
evolution, latent step geometry, conditional/unconditional geometry, alignment
between score direction and the actual inverse step, score-field channel and
spatial statistics, compact FFT energy/entropy summaries, latent moments at
every step, and cross-timestep cosine structure. They require no additional
UNet or VAE calls. A requested DDIM transition residual was deliberately not
included: the scheduler constructs that transition directly from the same score
prediction, so a residual against it would be a mathematical tautology.

E25 verified the legacy control before scaling. With the same two images,
batch-2 rich extraction matched every legacy numerical feature within
6.47e-7. E26 then re-extracted exactly the E23 membership (40 real and five
from each of ADM, BigGAN, DALL-E 2, GLIDE, Midjourney, SD1.4, SDXL and VQDM).
All 80 unique rows completed and were materialized at
`results/features/sd15_rich_signature_stage_a_chunk000.parquet`. The table has
659 columns: audit fields plus the 51 legacy and 608 rich signature features.
Across the complete rerun, shared legacy values have max absolute difference
2.40e-5 and mean difference 2.11e-7—acceptable MPS batch floating-point
variation, not a change in definition.

E27 writes pre-specified generator-stratified curve summaries and plots under
`results/analysis/rich_stage_a/` for score magnitude/change, score–trajectory
alignment, guidance alignment, latent step movement and score frequency ratio.
No feature was ranked by AUROC and no detector was fit: five images/source is
only an implementation/discovery checkpoint. The next valid comparison is to
complete the remaining 320 source images with this frozen extractor, then
evaluate legacy-only, rich-only and combined signatures using identical nested
leave-one-generator-out splits.

## Stage-A rich-versus-legacy benchmark (E28–E37)

E28 completed the frozen Stage-A corpus: 400 unique images with 200 real and
25 each from ADM, BigGAN, DALL-E 2, GLIDE, Midjourney, SD1.4, SDXL and VQDM.
There were no failed rows or non-finite features. The canonical cached table is
`results/features/sd15_rich_signature_stage_a_full.parquet` (659 columns:
auditing fields plus 51 legacy and 608 rich trajectory fields). No raw pixels,
caption text, paths, source IDs, generator labels or image metadata were given
to any classifier.

E29–E31 used eight nested generator-held-out folds. Each outer fold holds one
synthetic source and a disjoint block of 25 real images; inner validation also
holds complete generators and chooses regularization (and PCA dimension where
applicable). The primary AUROCs are:

| held source | legacy logistic | rich logistic | combined logistic | rich PCA-logistic | combined boosted |
|---|---:|---:|---:|---:|---:|
| ADM | 0.626 | 0.730 | 0.738 | 0.717 | 0.624 |
| BigGAN | 0.530 | 0.555 | 0.586 | 0.448 | 0.666 |
| DALL-E 2 | 0.605 | 0.421 | 0.429 | 0.394 | 0.350 |
| GLIDE | 0.890 | 0.896 | 0.902 | 0.890 | 0.926 |
| Midjourney | 0.741 | 0.770 | 0.824 | 0.771 | 0.774 |
| SD1.4 | 0.770 | 0.717 | 0.714 | 0.746 | 0.614 |
| SDXL | 0.339 | 0.690 | 0.653 | 0.670 | 0.632 |
| VQDM | 0.675 | 0.728 | 0.725 | 0.666 | 0.758 |
| **median / worst** | **0.650 / 0.339** | **0.722 / 0.421** | **0.719 / 0.429** | **0.686 / 0.394** | **0.649 / 0.350** |

Thus rich logistic improves median AUROC by +0.072 versus legacy and improves
five held sources (ADM, BigGAN, GLIDE, SDXL and VQDM) while degrading DALL-E 2
and SD1.4; combined improves the worst case slightly but not enough to make a
practical provenance claim. PCA does not improve the rich raw logistic result,
and conservative boosting is not consistently stronger. The feature-space is
therefore more informative than the compressed signature, but no universal
direction or reliable low-FPR detector has been established.

E32 ablations identify spatial score structure (median AUROC 0.720), score
temporal dynamics (0.706), and the original noise magnitude (0.666) as the
strongest coherent families. Score–trajectory alignment itself is not a shared
mechanism here (median 0.517), while latent typicality (0.609) and round-trip /
endpoint controls (0.622) are modest. The full detailed rows, including
AUPRC, balanced accuracy, validation-frozen TPR operating points and bootstrap
AUROC intervals, are machine-readable in `results/stage_a_full/logo_metrics.csv`.

E33 learning curves show the rich model is sample-limited but not monotonically
strong at very small sample sizes: mean held-source AUROC is 0.538, 0.633,
0.639, 0.670 and 0.688 at 5, 10, 15, 20 and 25 images per training generator,
respectively (legacy: 0.590, 0.618, 0.626, 0.644, 0.647). E34 generator-count
curves rise from mean AUROC 0.514 with one training source to 0.688 with six or
seven, a strong indication that generator diversity—not merely image count—is
valuable. E35 confirms a countervailing limitation: rich signatures predict
the source generator at 61.5% five-fold CV accuracy for eight classes (12.5%
chance), so substantial information remains generator fingerprint rather than
a shared real-vs-AI signature. E36 real-only anomaly controls are comparable
but not superior (mean AUROC 0.634 diagonal distance, 0.644 shrinkage
Mahalanobis, 0.640 PCA reconstruction); they remain secondary.

**Stage-A conclusion (E37):** the richer six-step SD1.5 representation is
meaningfully less lossy than the old signature and merits further scaling with
more diverse sources. It is nevertheless not a working arbitrary-image AI
detector: DALL-E 2 remains below chance under several representations and the
worst held-generator AUROC is far from useful. The justified next experiment is
to scale source diversity and count while retaining the promising spatial-score
and score-dynamics families; only if that plateaus should the project increase
trajectory depth to 12/25 steps. No circulation result is claimed.

An all-Stage-A rich-logistic artifact was also fit solely to make the pipeline
exercisable on an arbitrary image:
`results/detector/sd15_rich_stage_a_experimental.joblib`. It uses the modal
inner-selected regularization (`C=0.1`) and all 608 rich features. The
end-to-end `scripts/detect.py` smoke test completed and emitted an explicit
experimental score/status/warning. This deployment artifact is for research
inspection only; fitting it on all Stage-A data removes the held-generator
protection and does not alter the E29–E37 evaluation conclusion.

## Prompt / approximate conditioning ensemble pilot (E38–E42)

E38 constructed a cached five-caption ensemble for the pre-existing balanced
80-image Stage-A checkpoint (40 real, five images from each of the eight
sources). Prompt 0 is the original BLIP caption; prompts 1–4 are conservative
semantic rewrites ("depicting", "scene showing", "detailed view", and
"featuring" the same caption). This is an approximate conditioning ensemble,
not a Bayesian posterior. Prompt text, length, token count and embeddings were
never classifier inputs. E39 extracted all 400 image–caption rich signatures
under the unchanged SD1.5/256px/six-step protocol and cached them by
`(image, caption_id)`.

E40 finds material but not catastrophic prompt dependence for the existing
all-Stage-A experimental rich detector: mean score SD 0.0438, median 0.0370,
mean score range 0.1078; 10/80 images (12.5%) cross the provisional 0.5 label
boundary across plausible wording variants. This confirms that a single prompt
is a measurable nuisance source, but not that prompt language itself is a
provenance shortcut.

E41–E42 aggregate trajectory features by prompt mean and prompt SD, then use
the same eight generator-held folds with disjoint five-image real test blocks.
The small-pilot results are:

| representation | median AUROC | mean AUROC | worst source |
|---|---:|---:|---:|
| single BLIP caption | 0.540 | 0.545 | 0.200 |
| mean over five prompts | 0.520 | 0.560 | 0.200 |
| prompt mean + sensitivity | 0.640 | 0.610 | 0.160 |

Mean marginalization alone does not help. Adding trajectory sensitivity across
the five prompts improves the pilot median, with gains on BigGAN, DALL-E 2,
SD1.4, SDXL and VQDM, but losses on ADM/GLIDE/Midjourney and a worse minimum.
Because the ensemble uses templated paraphrases and only five test synthetics
per source, this is a **mixed exploratory result**, not adequate evidence to
run the costly 400×5 expansion. It motivates a better diverse-caption ensemble
and larger data acquisition before optimized continuous conditioning. Outputs
including per-image score stability, feature aggregates, timestep variance and
fixed plots are in `results/prompt_ensemble/`.

## User-prompt-free self-conditioning study (E43–E48)

E43 freezes the product contract: normal inference is `detect.py image.jpg`.
BLIP infers conditioning internally; `--caption` is only a research override.
The Stage-A manifest was audited: it contains no original generation prompt,
negative prompt, seed, sampler, CFG or checkpoint data, and no such field is
used in conditioning, features, fitting or thresholds. Thus real and synthetic
images enter the same image-only conditioning pipeline.

E44 extracted matched null-conditioned rich trajectories for all 400 Stage-A
images using unchanged SD1.5, 256px and six steps. E45 builds the numerical
response signature `F_auto - F_null`; caption text and caption-quality proxies
remain excluded. The 80-image screen initially looked encouraging (response
median AUROC 0.68 versus auto 0.54), so the full expansion was justified. The
full cached result falsifies that pilot lead:

| representation | median AUROC | mean AUROC | worst generator |
|---|---:|---:|---:|
| automatic BLIP conditioning | **0.744** | **0.691** | **0.520** |
| null conditioning | 0.684 | 0.662 | 0.400 |
| null + automatic | 0.732 | 0.670 | 0.512 |
| conditioning response only | 0.572 | 0.591 | 0.272 |
| null + automatic + response | 0.640 | 0.642 | 0.384 |

Automatic conditioning is beneficial relative to no semantic explanation, but
the amount of conditioning response is not a stable generator-general
mechanism at six steps. It helps SD1.4 strongly (0.960 response AUROC) but
fails for BigGAN (0.272), demonstrating source specificity. DALL-E 2 improves
modestly under response (0.640 versus 0.560 automatic), but that does not
justify optimizing for it.

E46's prior templated rewrites are not treated as a diverse automatic
conditioning ensemble and will not be scaled. The locally available BLIP path
was deterministic, so semantic sampling or a second caption/VLM model would
be needed before a genuine `q(c|x)` experiment. E47 optimized conditioning is
deferred because the full response result did not support the sensitivity
mechanism. E48 concludes the product path is **user-prompt-free but
automatically conditioned**: BLIP is an internal hypothesis, never a supplied
or original generation prompt. Machine-readable results are in
`results/self_conditioning/stage_a_full/`.

## Stage-B generator-diversity increment (E49–E52)

E49 extends the frozen image-only SD1.5/BLIP/six-step architecture from eight
to ten labelled synthetic sources. The public AIGC benchmark supplied 25 new
SD1.5 and 25 Wukong examples; the existing 400 Stage-A feature rows were
reused, so only 50 images required new extraction. The valid Stage-B manifest
has 450 unique images: 200 real and 25 each from ADM, BigGAN, DALL-E 2, GLIDE,
Midjourney, SD1.4, SD1.5, SDXL, VQDM and Wukong. Downloaded benchmark-real
rows duplicated Stage-A real content and were deliberately excluded rather
than misrepresented as a second real source. Multi-real-source generalization
therefore remains untested.

E50 freezes `core_rich` before Stage-B evaluation: original predicted-noise
magnitude, score temporal dynamics and spatial score structure. E51 cached all
450 rich signatures at the unchanged operating point. E52 uses nested
generator-held-out validation with disjoint real-image folds. Results:

| representation/model | median LOGO AUROC | mean | worst | interpretation |
|---|---:|---:|---:|---|
| legacy logistic | 0.651 | 0.682 | 0.530 | compressed baseline |
| full rich logistic | 0.793 | 0.742 | 0.500 | strongest trajectory-only result |
| combined logistic | **0.797** | **0.747** | 0.506 | marginal median gain over rich |
| core-rich logistic | 0.753 | 0.690 | 0.340 | removing other rich families hurts |
| rich PCA-logistic | 0.745 | 0.718 | 0.450 | dimensionality reduction does not help |
| combined boosted | 0.730 | 0.703 | 0.382 | conservative nonlinear model does not help |

The additional source diversity improves full-rich median LOGO AUROC from
0.722 in Stage-A to 0.793 in Stage-B, supporting a data-limited component.
This reaches the project’s Tier-1 meaningful research-signal target, but not a
working detector: BigGAN is 0.500 rich / 0.506 combined and DALL-E 2 is 0.516
rich / 0.538 combined. The first Stage-B increment also has only 25 images per
source and one real source, so it cannot distinguish clean scaling from source
selection luck. `core_rich` failing to match full rich means weak-looking
families still contribute in combination. Full metrics, bootstrap intervals and
family ablations are in `results/stage_b/logo_metrics.csv` and
`results/stage_b/feature_family_ablation.csv`.

## Cross-source real-image stress test (E53–E59)

E53 used the already-local RR archive as an independent real-photo source.
One hundred RR validation real photos were extracted, then exact SHA-256 and
dHash duplicate-audited against all 450 existing rows. All 100 passed
(closest dHash distance 11; exclusion threshold 3). E54 extends the existing
Stage-B manifest—not a parallel dataset—to 550 rows: 10 synthetic sources,
200 AIGC-benchmark real photos and 100 RR real photos. E55 extracted only the
100 missing cached signatures under the frozen image-only BLIP/SD1.5 protocol.

E58–E59 perform the first crossed provenance test. For every pair of a held
synthetic generator and held real source, neither is used for fitting,
standardization or generator-aware regularization selection:

| held real source | median AUROC | mean | worst generator |
|---|---:|---:|---:|
| RR real (100) | 0.737 | 0.709 | 0.518 |
| AIGC benchmark real (200) | 0.688 | 0.650 | 0.407 |
| all 20 crossed pairs | 0.703 | 0.679 | 0.407 |

Low-FPR operation is weak: against RR-real, TPR@1% FPR ranges 0.00–0.20
across held generators. A direct real-source holdout diagnostic explains part
of the asymmetry: without AIGC-real in training, its mean score is 0.556 and
54.5% are above the provisional 0.5 boundary; without RR-real, its mean score
is 0.372 and 33.0% are above it. Stage-B's 0.797 within-source median therefore
does not support an arbitrary-image detector claim. Generator diversity helps
within a fixed real source, but current single-probe trajectory features remain
sensitive to real-image source. Results are in
`results/stage_b_scaled/cross_source_metrics.csv` and
`results/stage_b_scaled/real_source_holdout.csv`.

## Confounding audit, controls and multi-probe phase (E60–E76)

*Scope: adversarial re-examination of the frozen v1 representation (SD1.5 / BLIP / 256 px / 6-step, 652 features; not modified).
All classifiers are StandardScaler + LogisticRegression (C=0.1 fixed, class-balanced) unless stated.  Numbers regenerate from the scripts named in each row (see "Commands" at the end of this section).
Code/protocol changes in this phase: added `src/analysis/*`, `src/probes/{base,pixel_ddpm,dit}.py`, `src/corruption/canonical.py`; fixed one latent extractor defect (E68).  v1 feature definitions are untouched.*

| exp | hypothesis | design | data | result | interpretation | limitations | next |
|---|---|---|---|---|---|---|---|
| E60 real-source audit | trajectory features encode *which real dataset* an image came from | binary AIGC-real vs RR-real, 5x5-fold CV, per feature family; permutation null (30 label shuffles); metadata baselines | 200 AIGC-benchmark real + 100 RR real (cached v1 features) | ALL_v1 AUROC **0.772 [0.709, 0.831]** (null mean 0.512, max 0.592); legacy51 0.756; rich608 0.767. By family: latent typicality 0.744, score moments 0.740, endpoint 0.733, spatial 0.699, latent geometry 0.668, legacy noise 0.666, cross-time 0.662, FFT 0.656, score dynamics 0.648, alignment 0.642, legacy geometry 0.631, **round trip 0.569, legacy guidance 0.562, rich guidance 0.538**. File metadata alone 0.971 [0.947, 0.989]; geometry-only 0.902. Largest single-feature |d| 0.79 (none > 0.8) | source identity is broadly distributed across scale/typicality families and *absent* from guidance and round-trip families | only two real sources; RR vs AIGC differ in content, size and JPEG at once | E61, E70 |
| E61 what is the source identity? | encodes resampling/native-scale history | 4-class native-size-bucket task (aigc_256 n=83, 257-599 n=105, 600+ n=12, RR n=100); ridge regression of log native min-side within the *single* AIGC-real source | same 300 reals | balanced acc **0.604** vs chance 0.25 (legacy51 0.538, round trip 0.360, metadata 0.937). Within AIGC-real, OOF Spearman(trajectory-predicted, log native size) **0.556** (n=200) | features carry native-resolution history inside one dataset | bucket labels are metadata-derived; n=12 in the top bucket | E71 |
| E62 crossed matrix vs shortcut | the crossed 0.70 result is not explained by file geometry | held real source x held generator (20 cells); baselines: 4 geometry features (min-side, aspect, is-square, pow2), 7 metadata, legacy51; calibration via realized FPR at the OOF-1%/5% thresholds | 550-image cross-source corpus | ALL_v1 macro **0.681** (median 0.698, worst 0.385, 95% cells > 0.5, 75% with CI-low > 0.5); legacy51 0.640. **geometry-4: macro 0.774** (median 0.806, worst 0.558). metadata-7 median 0.772 (worst 0.000: held-out ADM PNG inverts the format cue). Calibration: at nominal 1% FPR the trajectory model realizes 2.4% FPR / 6.0% TPR; geometry realizes 21.7% / 35% | a four-number geometry baseline outperforms the trajectory representation on the headline metric | n=10..25 per generator; C fixed | E64, E65 |
| E63 resize/JPEG-matched subset | with no resampling and identical JPEG table, trajectory still beats a trivial scalar | keep native-256, square, non-PNG images: real = 83 AIGC reals, fake = BigGAN/DALL-E 2/GLIDE/VQDM; LOGO, real folds | 183 images | mean LOGO AUROC: ALL_v1 **0.646**, legacy51 0.663, rich 0.649; **bytes-per-pixel alone 0.696**. Per generator ALL_v1: BigGAN 0.52, DALL-E 2 0.48, GLIDE 0.88, VQDM 0.69 | with geometry and JPEG state matched, one compressibility scalar equals the 652-feature model; two of four generators are at chance | n=25/generator; one real source | E71 |
| E64 like-for-like headline protocol | E52's LOGO median 0.793 exceeds trivial baselines | LOGO, disjoint real folds, 450 images (200 real + 10x25), 5 real-fold seeds, fixed C | Stage-B | median LOGO AUROC: ALL_v1 **0.766 ± 0.036** (E52 tuned-C: 0.793), legacy51 0.672, **geometry-4 0.852 ± 0.027**, metadata-7 0.830, **is-square alone 0.750**, bpp-only 0.276 (sign-inverted across generators) | headline is not distinguishable from, and is below, a file-geometry shortcut | E52 tuned C per fold; I fixed C | E65 |
| E65 linear nuisance removal | crossed signal survives removal of acquisition metadata | ridge of each feature on nuisance columns fit on the training split only, removed from train and test; (a) 5 geometry columns; (b) + bytes-per-pixel + q-table | 550 corpus, crossed | macro AUROC **0.681 -> 0.550** (a), **0.613** (b); cells above chance 95% -> 65% / 70% | most of the crossed signal is linearly predictable from acquisition geometry; non-monotone across nuisance sets => value uncertain in [0.55, 0.61] | over-control likely (geometry is a near label proxy for fakes); linear only | E66 |
| E66 residualization by family | which family retains signal | as E65(a), per coarse group | same | **guidance 0.671 -> 0.644 (95% cells > 0.5 in both)**; crosstime 0.607 -> 0.551; alignment 0.536 -> 0.518; score moments 0.570 -> 0.528; spatial 0.662 -> 0.449; latent endpoint 0.595 -> 0.442; FFT 0.520 -> 0.399; round trip 0.555 -> 0.462 | every family except guidance collapses to <= 0.55 (several inverted) | selected among 12 groups x 2 modes after inspection: **post hoc** | E67 |
| E67 caption/content baseline | guidance signal is BLIP caption content, not image response | crossed: caption word-count; caption TF-IDF (1-2 gram) logistic; guidance after residualizing on geometry + caption length + TF-IDF-SVD(20) | BLIP captions of the 550 images | word count only **0.653**; **TF-IDF only 0.791** (100% cells > 0.5; fake-indicative words: goldfish, swan, shark, elephant, fish, animal; real-indicative: woman, group, camera, building); guidance 0.671 raw -> 0.644 (geometry) -> **0.596** (geometry + caption), 85% cells > 0.5 | caption text alone beats the whole trajectory representation: the corpus has a semantic-distribution confound (ImageNet-class prompts vs people/scenes); part of guidance signal is caption-related | ImageNet-class fakes vs scene reals is a dataset property, not a caption artifact | E72, E75 |
| E68 extractor defect (protocol audit) | robustness job is protocol-consistent | code inspection + run of the completed rows | `scripts/extract_detector_features.py` | `load()` skipped the 256 px resize whenever `--manifest` had a `transform` column; the 100 completed transformed rows are all ADM (256-native, identity) so **no completed row is affected**, but the next non-256 image would be extracted at native resolution, and mixed shapes would crash `np.stack` at batch 8 on the HPC job | would have silently broken the robustness experiment | fixed (resize after condition) + two regression tests | robustness job can now be launched safely |
| E69 second probe: CIFAR-10 DDPM-32 (control) | any real/synthetic structure replicated by an independent probe | pixel-space, unconditional, VAE-free probe on the same canonical pixels; **30 cross-probe quantities** (`CROSS_PROBE_COLUMNS`) identical-by-code for both probes; (A) effect-vector replication + joint-label permutation null; (B) agreement-residual scalar; (C) affinity matrix; (D) weight-vector transfer | 550 images | own crossed AUROC (30 features): SD1.5 0.585, **CIFAR-32 0.507**. (A) effect-vector Spearman 0.22-0.30, sign agreement 0.3-0.6, permutation p 0.42-0.53. (B) 0.562 mean. (D) SD1.5->CIFAR transfer **0.442**, CIFAR->SD1.5 0.578 (cell range 0.15-0.83) | no cross-probe directional replication; but the control probe has no signal of its own, so it neither supports nor refutes universality | 32 px probe; endpoint per-dim norm ~0.44 (far from N(0,1)); 30 features only | DiT (E74/E76) |
| E70 source identity by probe | source identity is SD-specific | AIGC-real vs RR-real using the 30 cross-probe quantities | 300 reals | **CIFAR-32 0.832 [0.784, 0.879]** vs SD1.5 0.752 [0.691, 0.814] | a 32-px unconditional DDPM identifies the real source *better* than SD1.5; source identity lives in coarse image statistics visible to any probe | 30 features | E71 |
| E71 acquisition normalization (core-200) | source separability falls and real/fake persists under a shared pipeline | core-200 frozen (120 robustness originals + 80 reals: 50 AIGC-real, 50 RR-real, 10/generator) ; pipelines: **crop256** (centre-crop square, LANCZOS 256), **crop256+JPEG75** (single controlled re-encode), **band128** (crop, down to 128, bicubic up to 256); v1 baseline reused from cache | 200 images x 3 pipelines | see table below | see interpretation below | n=10 fakes/generator (cell CI half-width ~0.16); macro SE >= 0.02 | E75 |
| E72 conditioning ablation (core-200) | signal requires semantically aligned text | same images, v1 native pixels, conditioning = BLIP / empty / fixed "a photo" / **another image's BLIP caption** | 200 images x 3 variants | see table below | see interpretation below | as above | E75 |
| E73 caption dependence by family | which families need the caption | median per-feature Spearman across images between BLIP baseline and each variant | core-200 | 44 legacy + 608 rich: endpoint/noise/typicality/FFT/spatial/score-moments/cross-time/latent-geometry/score-dynamics rank-corr **0.96-1.00** with baseline under every variant; caption-dependent: legacy guidance 0.53-0.59, rich guidance 0.49-0.55, alignment 0.39-0.42 (generic/shuffled), round trip 0.63-0.69 | ~90% of columns are effectively **caption-independent image-statistics summaries** of a 6-step UNet score trajectory | rank-corr is a coarse dependence measure | E72 |
| E74 DiT-XL/2-256 probe | an architecturally different prior (transformer, ImageNet, null class) can be measured with the same protocol | `src/probes/dit.py`; eps head = first 4 of 8 channels; null class 1000 | 2-image CPU smoke test | eps RMS 0.11 -> 0.79 (real #1), 0.08 -> 0.93 (#2); finite round trip; 74 s/img CPU under load | implemented and sane | shares the SD-family VAE | E76 |

### E71 result — acquisition normalization (core-200; crossed = held real source x held generator, 20 cells)

| pipeline | ALL_v1 crossed macro | guidance crossed | latent-endpoint crossed | real-source AUROC, ALL_v1 [95% CI] | real-source AUROC, guidance |
|---|---:|---:|---:|---:|---:|
| v1 baseline (squash-resize) | 0.655 | 0.665 | 0.539 | 0.773 [0.696, 0.875] | 0.485 |
| crop256 (no aspect distortion) | 0.660 | 0.668 | 0.549 | 0.756 [0.663, 0.853] | 0.449 |
| crop256 + JPEG q75 | 0.642 | 0.650 | 0.515 | 0.691 [0.582, 0.795] | 0.504 |
| band128 (crop, down 128, bicubic up 256) | 0.596 | **0.714** | **0.445** | 0.725 [0.621, 0.832] | 0.469 |
| *4-feature file geometry (unchanged by processing)* | 0.758 | - | - | - | - |

Cell-level bootstrap half-width ~0.16; macro SE >= 0.02 (independence lower bound, `results/core200/analysis/macro_se_lowerbound.csv`).
**Interpretation.**  (1) Aspect distortion is not the driver: crop256 changes nothing (0.655 -> 0.660; source 0.773 -> 0.756).  (2) A single controlled JPEG re-encode lowers real-source separability (0.773 -> 0.691, CIs overlap) while real/fake performance is unchanged (0.642): the "second outcome" of the design, weakly.  (3) Band-limiting to a common 128 px support *lowers* ALL_v1 real/fake performance (0.596) and drives latent-endpoint statistics to below chance (0.445) but leaves source separability at 0.725 and **raises the guidance family to 0.714** (95% of cells above chance, realized FPR at nominal 1% = 0.1%).  (4) **No pipeline removes real-source identity (0.69-0.77)**, so it is not primarily an aspect/JPEG/resampling artifact of the kinds tested: consistent with coarse content/tonal distribution (E70).  (5) Guidance stays blind to source identity (0.45-0.50) in all four pipelines while carrying real/fake signal in all four: the only family with that pattern.  Caveat: 10 fakes/generator; post hoc identification of guidance (E66).

### E72 result — conditioning ablation (same 200 images, v1 native pixels)

| conditioning | ALL_v1 crossed | legacy51 crossed | guidance crossed | real-source AUROC, ALL_v1 [CI] |
|---|---:|---:|---:|---:|
| BLIP caption (v1) | 0.655 | 0.623 | 0.665 | 0.773 [0.696, 0.875] |
| empty prompt | 0.642 | 0.509 | 0.500 (constant by construction: cond = uncond) | 0.798 [0.728, 0.889] |
| fixed "a photo" | 0.657 | 0.545 | **0.633** | 0.757 [0.679, 0.857] |
| another image's BLIP caption | 0.609 | 0.519 | **0.513** | 0.780 [0.708, 0.882] |

**Interpretation.**  (1) The bulk of v1 (ALL_v1) is caption-independent: removing text entirely changes the crossed result by -0.013 and *increases* source separability, and 90% of columns have per-feature rank-correlation >= 0.96 with the BLIP baseline under every variant (E73).  So most of the representation is *not* an image-to-text compatibility measurement.  (2) The legacy-51 signature's crossed performance (0.623) collapses without aligned text (0.51-0.55): it was carried by its 14 guidance columns.  (3) The guidance family survives a **constant** prompt ("a photo": 0.633; so it is not caption-content leakage, cf. E67) but is destroyed (0.513, 55% of cells above chance) by a *mismatched* caption: the family responds to coherent-or-neutral conditioning and is corrupted by semantically wrong conditioning (Δ = -0.15, > 5 SE lower bound).  (4) If guidance carries provenance information it is an image-to-*prior response*, not an unconditional trajectory property, and it applies to ~92 of 652 columns.

### E75–E80 — content-matched pilot and controls (60 COCO contents; pre-registered, `PREREGISTRATION_content_matched_v1.md`)

Design: real COCO photo (centre-crop, 512 PNG) vs SD1.5 counterpart generated from the **same human caption** (512 PNG); both classes probed with that human caption; frozen v1; content-grouped CV.  Pre-registration was committed before any counterpart or feature existed; Addendum 1 (aMUSEd) was committed after the SD1.5 result was seen and says so.

| exp | question | result | reading |
|---|---|---|---|
| **E75 pre-registered** | H_primary / H_guidance / H_eps / H_transfer | **ALL_v1 0.800 [0.732, 0.879] -> SUPPORTS**; guidance 0.631 [0.548, 0.734] -> **INCONCLUSIVE** (CI-low 0.548 < 0.55); legacy51 0.701 [0.618, 0.785]; H_eps: fake lower in **42/60 pairs**, sign test p = 0.0027 (mean diff -0.014); H_transfer (Stage-B-trained classifier -> matched set) **0.735 [0.644, 0.826]** (mean score real 0.38, fake 0.67). Family AUROCs (exploratory): latent geometry 0.836, cross-time 0.798, trajectory noise 0.779, latent endpoint 0.683, score moments 0.652, spatial 0.626, FFT 0.622, alignment 0.604, round trip 0.549 | with content, caption, size and container matched, SD1.5-on-SD1.5 shows genuine separation.  This is the **most favourable case** (generator = probe) so it is only weakly confirming; the eps-norm lead of E17 replicates (70% of pairs) |
| E76 VAE-only baseline (exploratory) | is it the SD-VAE decoder fixed point? | VAE encode-decode error, no diffusion: at native 512 px **0.866 [0.807, 0.921]** (fake lower in 92% of pairs); at the probe's 256 px only 0.591 [0.518, 0.662] | a single scalar at 512 px beats the whole trajectory (0.80): decoder fixed-point is a much stronger SD1.5-fake cue than anything in v1; the 256-px probe mostly does *not* see it, so v1's 0.80 is something else |
| E77 independent probes on the matched set (exploratory, Class-B 30 features) | is the signal probe-specific? | SD1.5 0.697 [0.619, 0.783]; DiT-XL/2 0.741 [0.663, 0.822]; **CIFAR-10 DDPM-32 (VAE-free, 32 px) 0.816 [0.753, 0.890]** | three unrelated priors separate SD1.5 fakes; the best is the 32-px pixel-space one => not decoder- or probe-specific; more consistent with a coarse global "look" of guided generations |
| E78 coarse-look control (exploratory) | is it just colour/contrast? | 12 global statistics of a 32-px thumbnail: 0.668 [0.582, 0.756] | explains part, not all |
| E79 SD1.5 x DiT on core-200 (Class-B) | cross-probe replication, confounded corpus | source-shift effect vectors Spearman **0.84** (sign 0.93); real-source AUROC SD1.5 0.811 [0.722, 0.897], DiT 0.784 [0.695, 0.880], CIFAR-32 0.832; real/fake effect vectors Spearman 0.61-0.79 but joint-label permutation p = 0.32-0.66 (null sd 0.19); own crossed AUROC SD1.5 0.528 / DiT 0.558; weight transfer SD->DiT 0.487, DiT->SD 0.550 | a probe-shared **source** axis exists; no probe-shared **synthetic** axis in this corpus |
| E80 pending (queued, resumable) | residual JPEG-history/upsample asymmetry; non-SD generator | JPEG-75 and band-128 pipelines on both classes; aMUSEd counterparts (Addendum 1); church-256 VAE-free probe | see `results/followup*_driver.log`; NOT reported until complete |

**Caveats that bound E75–E79.**  n = 60 pairs, one generator (probe's own family), one photo source (COCO, JPEG-decoded and up-scaled ~1.07x while fakes are pristine synthetic pixels: a residual acquisition asymmetry that E80 tests), CFG 7.5 gives generations a characteristic saturated/centred look.  No result here shows generalization to an *independent* generator.

### Commands (reproduce / continue)
```bash
# audit on cached features (no GPU)
uv run python scripts/audit_real_source.py; uv run python scripts/crossed_eval.py; uv run python scripts/matched_subset_eval.py
uv run python scripts/logo_shortcut_baseline.py; uv run python scripts/residualized_eval.py; uv run python scripts/residualized_by_family.py; uv run python scripts/caption_baseline.py
uv run python scripts/dataset_audit_tables.py
# core-200 normalization + conditioning (MPS, ~2 h, resumable)
uv run python scripts/build_core_manifests.py; scripts/run_core_jobs.sh; uv run python scripts/evaluate_core200.py
# second probes
uv run python scripts/extract_crossprobe.py --probe cifar32; uv run python scripts/crossprobe_analysis.py
uv run python scripts/extract_crossprobe.py --probe dit --device mps --manifest data/core200/core200_frozen.csv --output results/crossprobe/dit_core200.json
uv run python scripts/crossprobe_analysis.py --other results/crossprobe/dit_core200.json --name dit_core200
# content-matched (pre-registered): real -> fakes -> manifest -> features -> evaluation
uv run python scripts/build_content_matched.py --stage real --count 60; uv run python scripts/build_content_matched.py --stage fake --count 60; uv run python scripts/build_content_matched.py --stage manifest --count 60
uv run python scripts/extract_detector_features.py --manifest data/content_matched/manifest.csv --output results/content_matched/v1_humancaption.json --rich --batch-size 2
uv run python scripts/evaluate_content_matched.py; uv run python scripts/vae_recon_baseline.py; uv run python scripts/thumbnail_baseline.py
# aMUSEd cross-generator decision experiment (E81-E87) -- COMPLETE; see below. Reproduce with:
uv run python scripts/extract_crossprobe.py --probe cifar32 --manifest data/content_matched/manifest.csv --output results/crossprobe/cifar32_content_matched.json --batch 8
uv run python scripts/extract_crossprobe.py --probe dit --device mps --batch 2 --manifest data/content_matched/manifest.csv --output results/crossprobe/dit_content_matched.json
uv run python scripts/vae_recon_baseline.py; uv run python scripts/thumbnail_baseline_multigen.py
uv run python scripts/evaluate_content_matched_multigen.py; uv run python scripts/matched_baseline_comparison.py
uv run python scripts/incremental_information_test.py; uv run python scripts/matched_crossprobe_multigen.py
# left deliberately unrun (not needed for the aMUSEd decision; do not use to "rescue" a negative result -- see SCIENTIFIC_AUDIT.md #8)
scripts/run_followup2_jobs.sh    # JPEG75/band128 on matched set (exploratory, addendum labelled non-headline)
uv run python scripts/extract_crossprobe.py --probe church256 --device mps --batch 4 --manifest data/content_matched/manifest.csv --output results/crossprobe/church256_content_matched.json
# robustness screen on HPC (extractor bug fixed in E68, unrelated to the aMUSEd result): qsub hpc/robustness_gpu.pbs
```

## aMUSEd cross-generator decision experiment (E81–E87)

*Scope: the pre-registered Addendum 1 test (`PREREGISTRATION_content_matched_v1.md`) plus the Phase-5 incremental-information
extension requested for this pass.  All 60 aMUSEd counterparts (masked-token generation, VQGAN decoder, `amused/amused-512`, CLIP text
conditioning, library-default 12-step/guidance-10 sampling, seed = hash(content_id), same human COCO captions as the real/SD1.5 pairs)
were already generated and SD1.5-probed in the prior session; this phase validated that output, extended the VAE/CIFAR-32/DiT/thumbnail
baselines from real+SD1.5 to all 180 images, and ran every pre-registered and Phase-5 evaluation.  No protocol parameter (resolution, steps,
seed policy, caption source, inversion, feature definitions, CV rule, thresholds) was changed after seeing any aMUSEd number.*

| exp | question | frozen protocol | data | result | conf./expl. | interpretation | limitations |
|---|---|---|---|---|---|---|---|
| E81 pipeline validation | did aMUSEd generation run correctly and match the frozen manifest | file existence, size (512x512, non-degenerate byte range), caption-alignment cross-check (real/sd15/amused must share caption per content_id) | 60 aMUSEd PNGs + 180-row manifest | 60/60 present, 232-416 KB (no blank/degenerate files), 512x512 confirmed via `sips`; **0 caption mismatches** across all three classes; generation log shows a clean 12/12-step completion for every image, no OOM | confirmatory (process check) | generation pipeline is sound; proceed to extraction | n/a |
| E82 SD1.5 v1 extraction on aMUSEd | frozen probe measures aMUSEd images without code changes | unchanged `extract_detector_features.py`, 256px canonicalization, 6-step DDIM inversion, human-caption conditioning | 180 images (this had already completed in the prior session) | 180/180 rows cached (`results/content_matched/v1_humancaption.json`), verified generator counts real=60/sd15=60/amused=60 | confirmatory | v1 measurements exist for the pre-registered test | none found |
| **E83 H_primary[aMUSEd] / H_eps[aMUSEd] / H_guidance[aMUSEd] / H_gen-transfer** (pre-registered, Addendum 1) | does the ALL_v1 trajectory separate real-vs-aMUSEd; does the eps-norm direction replicate; does a classifier trained on one generator transfer to the other | exact estimator/CV/bootstrap/decision rules from the addendum; `scripts/evaluate_content_matched_multigen.py` | 60 real vs 60 aMUSEd matched pairs | **H_primary[aMUSEd]: AUROC 0.994 [0.988, 1.000] -> SUPPORTS** (near-ceiling). **H_guidance[aMUSEd]: 0.895 [0.832, 0.955] -> SUPPORTS** (first confirmatory test of this post-hoc hypothesis). **H_eps[aMUSEd]: 36/60 pairs fake-lower (60%), sign-test p=0.155 -> NOT significant** (SD1.5 was 42/60, p=0.003). **H_gen-transfer(sd15->amused): AUROC 0.393 [0.293, 0.496] -> DISCONFIRMS** (below chance: a classifier trained on real-vs-SD1.5 is *inverted* on real-vs-aMUSEd). **H_gen-transfer(amused->sd15): 0.651 [0.550, 0.746] -> INCONCLUSIVE** (CI-low 0.5497, just under the 0.55 support threshold) | **confirmatory** (all four were pre-registered before this run) | Per the addendum's fixed table (supported / supported / not supported): **"two generator-specific signals; no shared direction."** aMUSEd is trivially separable by ALL_v1, but not via the same direction that separates SD1.5, and the one simple mechanistic lead that *did* replicate through E17->E75 (eps-norm) does not replicate here | n=60 pairs; one real photo source; aMUSEd used library-default sampling params (not tuned) |
| E84 per-generator baseline comparison (exploratory, `scripts/matched_baseline_comparison.py`) | how does ALL_v1 compare to generic low-level probes on the identical pairs | VAE mse@512 (fake=lower), CIFAR-10-DDPM-32 Class-B (30 feat., content-grouped CV), DiT-XL/2 Class-B, SD1.5 guidance family, SD1.5 eps_mean scalar -- all with paired content bootstrap | same 60-pair sets, both generators | **SD1.5**: ALL_v1 0.800, guidance 0.631, eps_mean 0.665, **CIFAR32 0.816**, **VAE 0.866** -- baselines match or beat the trajectory. **aMUSEd**: **ALL_v1 0.994**, guidance 0.895, eps_mean 0.636, CIFAR32 0.748, VAE 0.869, DiT 0.940 -- the trajectory now exceeds every baseline, but DiT (which shares SD1.5's VAE family) and plain VAE reconstruction are close behind | exploratory | aMUSEd separates strongly under *every* probe tried, including two with no relation to SD1.5 (VAE-only, CIFAR-32 pixel-space); the *size* of the trajectory's lead over baselines is new information, but so is the fact that a bare VAE encode/decode error already gets 0.87 | one aMUSEd sampling config; some baselines (DiT) share machinery with the probe |
| E85 thumbnail/colour control (exploratory, `scripts/thumbnail_baseline_multigen.py`) | is aMUSEd's separability just a gross colour/contrast difference | same 12-statistic 32px thumbnail logistic as E78 | both generators | SD1.5 0.668 [0.582, 0.756]; **aMUSEd 0.675 [0.597, 0.768]** -- nearly identical, both moderate | exploratory | aMUSEd's strong separability under VAE/CIFAR/DiT/trajectory is **not** explained by gross colour statistics; whatever those probes detect is a finer-grained image property (plausibly VQGAN-decoder texture/quantization artifacts) | 12 hand-picked statistics only |
| **E86 incremental-information test** (Phase 5, `scripts/incremental_information_test.py`) | does the frozen SD1.5 representation add discriminative information beyond a VAE+CIFAR-32 baseline model | baseline = VAE(4 scalars)+CIFAR32(30 feat.); extended = baseline + {full ALL_v1(652), guidance(92), eps-norm curve(18), eps_mean(1)}; identical content-grouped folds shared by both models per repeat; paired content-bootstrap of the AUROC difference | same 60-pair sets, both generators | **SD1.5**: baseline 0.916; **+full trajectory: 0.860 (diff -0.044, CI [-0.093, -0.003] -- adding SynthImage *hurts*)**; +guidance -0.036 (CI crosses 0, no effect); +eps curve/+eps_mean: no effect (CI crosses 0 both). **aMUSEd**: baseline 0.910; **+full trajectory: 0.993 (diff +0.081, CI [0.037, 0.135] -- adds information)**; +guidance: +0.013 (CI [-0.039, 0.067], **no effect**); +eps curve: +0.046 (CI [-0.002, 0.102], **not significant**); +eps_mean: no effect | **exploratory but decision-relevant** (not pre-registered; frozen after design, run once, not re-tuned on the result) | Beyond a generic VAE+CIFAR baseline, the **only** configuration that adds information for aMUSEd is the full 652-feature representation on 60 pairs (p/n ~ 11) -- every interpretable sub-family (guidance, eps curve, eps scalar) adds nothing beyond VAE+CIFAR, and for SD1.5 the full representation actively *hurts*. That pattern (gain confined to the highest-dimensional, least interpretable configuration; absent from every simpler, hypothesis-driven family) is the signature of high-dimensional overfitting inside the CV fold, not of a located mechanism. It is not proof of overfitting, but it is not evidence *against* that reading either, and no result here isolates *which* columns carry the +0.081 | n=60, single classifier family, one C value; a truly independent held-out generator/content set would be needed to distinguish overfitting from a real diffuse signal |
| E87 supporting cross-probe check (exploratory, `scripts/matched_crossprobe_multigen.py`) | do other probes' Class-B quantities also separate aMUSEd strongly | 30 cross-probe quantities, content-grouped CV, both generators | same 60-pair sets | **DiT-XL/2: sd15 0.741, amused 0.940** [0.905, 0.979]; **CIFAR-32: sd15 0.816, amused 0.748** | exploratory | DiT (shares SD1.5's VAE family) separates aMUSEd almost as strongly as the full 652-feature trajectory (0.940 vs 0.994) using only 30 quantities; CIFAR-32 (no VAE at all) is weaker on aMUSEd than on SD1.5. Consistent with E84/E85: the dominant aMUSEd cue is plausibly decoder/VAE-vs-VQGAN incompatibility, visible to any VAE-based probe, not a SD1.5-diffusion-specific mechanism | as E84 |

### Reading E81–E87 together

Every individual-generator AUROC for aMUSEd is high (ALL_v1 0.994; even the weakest baseline, thumbnail colour, is 0.675 > chance). Taken alone this
looks like the strongest positive result in the project. But the three tests designed to distinguish a *real, generalizing, SynthImage-specific* signal
from *aMUSEd just being easy to separate by many unrelated means* all point the same direction:

1. **No shared direction** (H_gen-transfer): a classifier trained on real-vs-SD1.5 is inverted on real-vs-aMUSEd (0.393); the reverse transfer is inconclusive (0.651, CI-low 0.5497).
2. **No incremental information in any interpretable sub-representation** (E86): guidance, eps-norm curve and eps-mean each add ~0 beyond VAE+CIFAR for aMUSEd; only the full 652-column representation adds anything, and only on 60 pairs.
3. **The founding mechanistic lead does not replicate** (H_eps[aMUSEd]): 60% direction-consistent, p=0.155, versus 70%/p=0.003 for SD1.5.
4. **Generic, SD1.5-unrelated probes match or exceed the full trajectory** on aMUSEd specifically (VAE 0.869, DiT-Class-B 0.940), and a pure colour-statistic baseline is flat across both generators (0.668 vs 0.675) -- so the aMUSEd effect is not "the whole image looks different," it's something texture/decoder-level, and multiple unrelated machinery (VAE alone; a pixel-space DDPM; a class-conditional transformer) all pick some of it up.

Read against the addendum's fixed interpretation table (SD1.5 supported / aMUSEd supported / transfer not supported): **"two generator-specific signals; no shared direction."**
Combined with the incremental-information result, the more precise statement is: aMUSEd images are separable by many unrelated low-level probes, SD1.5's
full trajectory captures a bit more separating information than a VAE+CIFAR baseline for aMUSEd specifically, but nothing in this experiment shows that
extra information is (a) attributable to an interpretable trajectory family, (b) the same mechanism that separates SD1.5-on-SD1.5, or (c) not an
artifact of fitting 652 correlated columns to 60 pairs.

## Feature-level audit of the matched-content results (E88)

*Scope: which of the 652 v1 columns drive the real-vs-SD1.5 and real-vs-aMUSEd matched-pair results (E75, E83).
No new features, no new model family — the same StandardScaler+LogisticRegression(C=0.1) pipeline, plus closed-form
univariate statistics and a 500-resample bootstrap. Full deliverable: `FEATURE_AUDIT.md`; code:
`scripts/feature_audit_matched.py`, `scripts/incremental_information_test_spatialfft.py`; tables:
`results/feature_audit/`.*

| exp | question | result | interpretation |
|---|---|---|---|
| E88a per-task coefficients | which features carry each task | SD1.5 top-30 dominated by `rich_guidance_moments` (11) + `rich_latent_geometry` (7); mean univariate AUROC 0.564, mean \|paired d\| 0.19, mean bootstrap sign-stability 0.96. aMUSEd top-30 dominated by `rich_score_fft` (13) + `rich_score_spatial` (7); mean univariate AUROC **0.824**, mean \|paired d\| **1.07**, sign-stability **1.00** | SD1.5's signal is weak and diffuse; aMUSEd's is strong, individually decisive, and essentially perfectly bootstrap-stable |
| E88b cross-task comparison | do the two coefficient vectors agree | Pearson r = 0.117 (shuffle-null mean 0.00, sd 0.038 — ~3 SD above null, p=0.003); sign agreement 55.5% (null 48.5%); **top-10 overlap 0/10, top-25 overlap 0/25**; of 19 features independently "strong" (top-100) in both, 8/19 (42%) reverse sign | a small, statistically real, but practically tiny shared component; the features that actually decide each task are almost entirely disjoint |
| E88c clean-residual comparison | does agreement improve once source-identity-confounded families (E60) are excluded | restricted to guidance+round-trip (94 cols, the only families with near-chance E60 real-source AUROC): r rises to **0.49**, top-25 overlap **44%** | the most cross-generator-consistent part of the representation — but E86 already showed this exact subset adds zero information beyond VAE+CIFAR-32 on aMUSEd |
| E88d top-k truncation (held-out, content-grouped CV) | is the aMUSEd 0.994 distributed or concentrated | SD1.5: k=1 -> 0.513, peaks 0.92 at k=75-150, **degrades to 0.800 at k=652**. aMUSEd: k=1 -> **0.857**, k=2 -> **0.978**, flat at ~1.00 from k=50 to k=652 | aMUSEd is carried by 1-2 features and is stable under dimensionality; SD1.5 is diffuse and is diluted by noise columns as more are added |
| E88e targeted incremental-information follow-up to E86 | does the specific family located in E88a/d (`rich_score_spatial`+`rich_score_fft`, 66 cols, pre-specified by this audit) explain E86's "+full trajectory only" result | vs VAE+CIFAR-32 baseline: **aMUSEd +0.064 [0.016, 0.120] (adds info)**; SD1.5 -0.037 [-0.076,-0.004] (hurts) — nearly matches E86's full-652-column aMUSEd increment (+0.081) using <11% of the columns | resolves E86's open overfitting question: the incremental gain is not diffuse high-dimensional noise-fitting, it is concentrated in a specific, identifiable family |

**Reading.** The aMUSEd result is real, reproducible and concentrated — not an overfitting artifact — but the
concentrating families (score-map spatial autocorrelation and 2-D frequency spectrum) are exactly the kind of
generic texture/decoder cue a VQGAN-based generator would be expected to leave, consistent with E84/E85/E87's
independent finding that a bare VAE encode/decode error (0.87) and a DiT probe sharing SD1.5's VAE family (0.94)
also separate aMUSEd strongly while a colour-only baseline does not (0.675). SD1.5's own separation runs on a
different, weaker, diffuse mechanism (guidance-family response) that this audit shows is genuinely disjoint from
aMUSEd's — 0% top-10/25 feature overlap, 42% sign reversal among jointly-strong features. **This audit does not
change the hypothesis status set in `SCIENTIFIC_AUDIT.md` §7 ("unsupported"); it replaces an open overfitting
question with a located, mechanistic, non-provenance explanation, which if anything sharpens that conclusion.**
See `FEATURE_AUDIT.md` for full detail.

## Raw inverse-trajectory subspace analysis (E89–E96)

*Scope: a methodological reset away from the 652-dimensional handcrafted v1 representation, testing the broader
hypothesis directly on the raw tensors the frozen SD1.5 probe already computes (latent states, conditional/
unconditional predicted score, derived guidance and latent-displacement tensors), using only linear/subspace
methods (SVD, principal angles, projection energy) with explicit permutation nulls. Full detail, all numbers,
and limitations: `RAW_TRAJECTORY_ANALYSIS.md`. Code: `scripts/extract_raw_trajectory.py`,
`scripts/raw_trajectory_{normcompare,core,paired_cosine,baselines,feature_link,plots}.py`,
`src/analysis/raw_trajectory.py`. 10 new regression tests: `tests/test_raw_trajectory.py`.*

| exp | question | design | data | result | conf./expl. | interpretation |
|---|---|---|---|---|---|---|
| E89 raw tensor extraction | persist tensors the frozen probe already computes | `capture_predictions=True` (existing flag), identical 256px/6-step/human-caption protocol; per-image `.npz` (z, cond, uncond only), resumable, one MPS process | 180 matched images | 180/180 cached, ~32MB, all pass shape/completeness assertions; one unexplained 5.7-min stall on a single batch (self-recovered, no data effect) | confirmatory (process check) | raw-trajectory analysis is possible without any protocol change |
| E90 normalization comparison | which paired-difference normalization to use | 3 normalizations (raw / per-sample-L2-normalized / relative-to-real-RMS) compared on the `guidance` tensor before committing | 60 triplets | raw shows *more* apparent cross-generator alignment (mean top-5 angle 61.8°) than sample-normalized/relative (80.5°/80.6°) | exploratory (pre-registered choice, not tuned) | raw chosen for its literal interpretation, not because it "worked better"; this makes the subsequent null result (below) a stronger, not weaker, negative |
| E91 within/cross-generator subspace structure | is there low-dimensional shared structure between SD1.5 and aMUSEd raw trajectory differences | SVD per generator per tensor; principal angles + projection energy at k=1,2,3,5,10 (pre-specified, not searched) | 60 triplets, 5 tensor types | neither generator's deltas are low-rank (effective rank 43-58/60; top-5 PCs capture 10-28% variance); cross-generator k=5: angles 50.6-77.1°, projection energy 2.8-15.8% | exploratory | face-value numbers suggest weak alignment; nulls (below) show this is not distinguishable from chance |
| **E92 pair-breaking null** | does subspace alignment require correct real/fake content pairing | 300 permutations/tensor: independently shuffle which real is subtracted per generator, recompute cross-generator stats | 60 triplets, 5 tensor types | observed k=5 mean principal angle indistinguishable from (often *less* aligned than) mismatched-pairing null for every tensor: z p=0.890, cond p=0.370, uncond p=0.277, guidance p=0.503, dz p=0.743 | **confirmatory null test** | the centered-PCA cross-generator alignment does NOT depend on correct content pairing at all -- it is not a content-specific phenomenon |
| E93 random-subspace null | is observed alignment above pure chance given n=60<<p | up to 300 random k=5 orthonormal subspaces per tensor | same | observed projection energy (2.8-15.8%) vastly exceeds random (~0.02%), p=0.0 for every tensor | confirmatory but uninformative | expected n<<p artifact; every Phase-7 baseline shows the identical pattern (below), so this null does not support a trajectory-specific claim |
| E94 strict cross-generator projection test + content-grouped bootstrap | fit basis on one generator, freeze, project the other; is the point estimate stable under content resampling | k=5, 300 bootstrap resamples of the 60 content ids, same resample applied to both matrices | same | several tensors' point estimates fall outside their own 95% bootstrap CI (e.g. dz: point 50.6°, CI [55.3,65.0]) -- real n=60 instability, reported not hidden; combined with E92, no tensor/direction shows above-chance shared variance | confirmatory | no basis learned on one generator explains more of the other than wrong-pairing chance already would |
| **E95 paired per-content cosine (a genuinely positive, distinct result)** | for the SAME content, do SD1.5's and aMUSEd's raw perturbations point the same way | un-centered cosine(Delta_SD_i, Delta_AM_i) per content; content-shuffle null (2000 permutations) | 60 triplets, 5 tensor types | mean paired cosine 0.47-0.57, **100% of 60 pairs positive** for every tensor, p<0.0005 (Wilcoxon p~1.6e-11); mean-vector cosine (uncentered population average) tracks it closely (0.24-0.59) | **confirmatory** | this is real and large, but is overwhelmingly a shared, roughly content-independent MEAN direction ("any synthetic image differs from real in a broadly similar way") that centered PCA (E91-E94) removes before testing -- not a content-specific covariance-structure mechanism |
| **E96 image-space/VAE/CIFAR-32 baselines (Phase 7)** | is any of the above trajectory-specific | identical statistics + both nulls on 32x32 RGB, VAE latent z0 (already cached, zero extra compute), 32x32 grayscale FFT magnitude, reused CIFAR-32 Class-B 30-feature vectors | same 60 triplets | pair-breaking null: ALL FOUR baselines also indistinguishable from chance (p=0.65-1.00); random-subspace null: ALL FOUR also trivially "significant" (p=0.0, same n<<p artifact); paired cosine: ALL FOUR also positive and significant (0.34-0.49, p=0.0) -- `vae_latent_z0` (0.477 paired, 0.243 mean-vector) reproduces the raw trajectory's exact qualitative pattern from a single VAE encoding step with NO diffusion trajectory at all; `rgb32`/`fft32` show a *more* specific (content-driven, negative mean-vector-cosine) version of the same effect | **confirmatory** | the raw inverse trajectory shows no cross-generator structure exceeding generic low-level image representations by either test; the positive paired-cosine effect is reproduced by the bare VAE latent alone |
| E97 feature-link (Phase 8) | do the 652 v1 top-30 features correspond to leading raw PCA components | raw trajectory -> PCA (already fit) -> correlate top-5 per-generator PCs' per-content scores against E88 top-30 feature differences, 1500 Spearman tests | 60 triplets | max \|r\|=0.485 (aMUSEd, z-tensor PC2 vs rich_latent_t3_skew); only 11/1500 exceed 0.4 | exploratory | the handcrafted top-30 features are not simple linear projections of the leading raw-trajectory directions; whatever they capture is narrower/higher-order than a coarse 5-component linear summary |

### Reading E89-E97 together

The broad hypothesis ("do architecturally different generators induce shared higher-level structure in the raw
inverse trajectory") gets a clean, multi-test answer: **there is a real, large, statistically solid shared
effect (E95) between SD1.5 and aMUSEd's raw trajectory perturbations — but it is a near-constant, content-
independent mean-direction effect, not evidence of shared content-specific dynamics (E91-E94 all null against
the pair-breaking test), and it is not specific to the inverse trajectory (E96: the bare VAE latent alone, with
no diffusion steps at all, reproduces it; raw RGB pixels and image-frequency content show an even more specific
version).** This is a materially stronger and more precise negative result than `FEATURE_AUDIT.md`'s coefficient
comparison: it is not merely that one particular 652-column summary fails to transfer, but that the raw tensors
themselves show no cross-generator structure beyond (a) a shared population-level offset already present in the
VAE encoding alone and (b) an n≪p artifact present in every representation tested, trajectory or not.

## SynthImage v2: literature-anchored stage decomposition (E98–E106)

*Scope: a full methodological reset away from the 652-feature handcrafted representation, per
`SYNTHIMAGE_PROJECT_STATE.md`'s pivot. New research question: where does forensic information enter the SD1.5
pipeline (VAE / score / trajectory / round-trip), how much does each later stage add beyond the earlier ones, and
which signals survive realistic transformations or partial AI editing? Full literature verification (paper
abstracts/HTML fetched and transcribed, not reconstructed from memory) in `LITERATURE_FEATURE_PANEL.md`; full
results in `STAGE_DECOMPOSITION_RESULTS.md`. 10 core scalar features (not 652), each directly reproduced from a
cited paper or a small, pre-declared adaptation thereof; the classifier is the same simple
StandardScaler+LogisticRegression used throughout the project.*

| exp | question | type | design | data | result | interpretation |
|---|---|---|---|---|---|---|
| E98 literature verification | exact definitions of AEROBLADE, DIRE, LaRE², FakeInversion, DiffPath, diffusion-OOD reconstruction (Graham et al.) | **literature reproduction (verification)** | WebSearch+WebFetch against arXiv abstract/HTML pages; every formula transcribed and cited, not approximated from memory | 6 papers | AEROBLADE: LPIPS layer-2, VGG16, training-free. DIRE: pixel L1 residual of ADM-based 20-step DDIM round trip, fed to a trained ResNet-50. LaRE²: single-step known-noise MSE at fixed t=200, e=4 ensemble, feeding a trained EGRE+CNN. DiffPath-1D: `Σ_t‖∂_t ε_θ‖²` finite-difference curvature over the DDIM score trajectory. FakeInversion: trained ResNet-50 on [image, decoded noise, decoded reconstruction] from text-conditioned SD1.5 inversion | frozen panel below reproduces each method's core *scalar quantity*; none of the papers' trained classifiers/refinement modules are reproduced (Phase-8 restriction: no learned representation this pass) |
| E99 frozen v2 panel | 10 core features across 4 stages | design (pre-declared) | S1.1-1.3 VAE (LPIPS-AE, pixel-MSE-AE, latent-MSE-AE); S2.1-2.2 score (LaRE@t200, score-norm@step0); S3.1-3.2 trajectory (DiffPath curvature, path length); S4.1-4.3 round-trip (pixel-L1, LPIPS, latent-MSE) | n/a | `LITERATURE_FEATURE_PANEL.md` | every feature is exact-reproduction or explicitly labeled adaptation; timesteps/steps pre-declared before any result was seen |
| E100 two-pass extraction + MPS bug fix | can the panel be computed on the existing 180-image matched set without a protocol change | **implementation** | Pass 1: everything needing the SD1.5 UNet. Pass 2: LPIPS-only. Reuses frozen 256px/6-step/human-caption protocol | 180 images | 180/180 extracted, 0 NaN. **Bug found and fixed**: interleaving LPIPS (MPS float32) and the SD1.5 UNet (MPS float16) calls in one process corrupted every subsequent UNet output to NaN, persisting across images — isolated during smoke-testing (2/2 images), fixed by strict two-process separation (not a retry/reset workaround) | resumable, deterministic; `tests/` were not extended for this bug specifically since it is an MPS-runtime issue, not a project logic error — documented instead in the script docstring |
| **E101 Phase 3: per-feature/stage results** | how much signal exists at each stage | confirmatory | content-grouped 5x10 CV, paired Cohen's d, bootstrap CI, univariate AUROC | 60 triplets | SD1.5 stage AUROC: vae 0.639, score 0.621, **trajectory 0.753 (highest)**, roundtrip 0.696. aMUSEd stage AUROC: **vae 0.981 (near-ceiling)**, score 0.845, trajectory 0.892, roundtrip 0.955 | trajectory is SD1.5's strongest stage; VAE alone is already close to ceiling for aMUSEd and nothing clearly beats it |
| **E102 Phase 4: incremental information (central experiment)** | does each later stage add information beyond the earlier ones | confirmatory | cumulative stage sets, paired OOF predictions (same folds), paired content-bootstrap CI on the AUROC difference; 3 targeted comparisons | 60 triplets | **SD1.5: VAE->+score adds nothing (CI crosses 0); +score->+trajectory adds +0.125 [0.056,0.204] (robust); +roundtrip adds +0.040 [0.003,0.080] (small).** **aMUSEd: nothing adds anything beyond VAE alone (every CI crosses 0, including the trajectory and round-trip comparisons)** | trajectory contributes genuinely new information beyond the VAE specifically for SD1.5; for aMUSEd the entire forensic effect is already present at the VAE stage |
| **E103 Phase 5: cross-generator direction** | does each feature's real->fake direction match across generators | confirmatory | paired Cohen's d sign per feature per generator; content-grouped transfer AUROC | 60 triplets | 7/10 features sign-agree; `latent_mse_ae`/`latent_mse_roundtrip` reverse weakly (both non-significant for SD1.5); **`path_length` reverses with large effect in BOTH directions** (SD1.5 d=-0.33, aMUSEd d=+1.39). Transfer AUROC only 0.58 (SD1.5->aMUSEd) / 0.60 (aMUSEd->SD1.5) despite 70% sign agreement | `path_length` treated as generator-specific per the task's own rule, not reconciled; direction agreement does not imply quantitative transfer |
| E104 Phase 6: small robustness pilot | do different mechanism classes degrade differently under JPEG/blur/resize/crop | confirmatory (small n) | 10 content ids x 3 generators x 5 conditions (clean, jpeg-50, resize-0.5, blur-1.0, center-crop-0.8), pre-declared before any result seen | 150 images | aMUSEd's VAE-only signal is transform-invariant (0.97-1.00 across all 4 conditions); SD1.5's trajectory-dependent full-panel signal survives qualitatively (0.70-0.78 across conditions vs 0.77 clean, still far above VAE-alone's 0.43-0.66). Every feature's before/after ranking correlates 0.66-1.00 (Spearman) across transforms | no feature/stage collapsed at this n; the SD1.5-trajectory / aMUSEd-VAE mechanism split (E101-102) persists under transformation |
| E105 Phase 7: AI-edit strength pilot | does any feature move monotonically with degree of generative intervention | **exploratory** | img2img partial-noise/denoise strength continuum (0/0.3/0.6/0.9), 8 content ids, global (not localized) edit -- localized/masked inpainting judged infeasible without a new, unvalidated checkpoint download (documented, not silently skipped) | 32 images | 6/10 features show significant monotonic Spearman trends (\|r\|=0.34-0.72), every significant trend in the direction consistent with that feature's established binary real/fake sign (E103) | convergent-validity evidence for the panel; explicitly not an "AI percentage" claim (n=8, no classifier, no held-out test) |
| E106 reading E98-E105 together | overall mechanism classification | synthesis | Outcome A-E decision tree (pre-specified) | all of the above | **Outcome E: different stages dominate for different generators.** SD1.5 = Outcome C (trajectory adds robust information beyond VAE). aMUSEd = Outcome A (VAE explains essentially everything; nothing else adds). Neither generator shows Outcome B or D | this is the cleanest, most literature-grounded, and most decisive positive+negative result pair in the project to date; see `STAGE_DECOMPOSITION_RESULTS.md` for full detail and `SYNTHIMAGE_PROJECT_STATE.md` for the resulting update to project direction |

## Third-generator decision experiment: does trajectory curvature generalize beyond SD1.5? (E107–E110)

*Scope: a narrow, preregistered decision experiment testing whether SD1.5's E102 trajectory-adds-information
result reflects a broader diffusion-model property or is SD1.5/probe-specific. Full blocker chain and exact
frozen configuration: `PREREGISTRATION_pixart_v1.md`. Full results: `PIXART_STAGE_DECOMPOSITION.md`. No feature
change, no classifier change — pure protocol reuse on a third generator's images.*

| exp | question | type | design | data | result | interpretation |
|---|---|---|---|---|---|---|
| E107 generator feasibility chain | can PixArt-Sigma (or an equally-distinct architecture) run here | **infrastructure (documented, not silently bypassed)** | checked PixArt-Sigma/alpha (disk: T5-XXL ~19GB vs 19GB free), SD3-medium (gated, no token), Stable Cascade lite (downloaded ~10GB, failed to load: diffusers/repo version incompatibility) | n/a | all three blocked for independent, concrete, verified reasons; SDXL substituted (not gated, no custom components, verified loadable) | preregistration amended twice (disk/gating chain, then Stable Cascade load failure) before any generation, per the task's own "document, do not silently substitute" rule |
| E108 generation, amended twice for machine-load/image-validity reasons | generate 60 SDXL counterparts | **implementation** | 1024px (task default) -> stopped at 4/60 (~3min/img, unacceptable sustained load, user intervened mid-run) -> 512px (fixed load, but produced a severely tiled/duplicated-subject image on 1/2 validation samples, a known SDXL sub-native-resolution failure mode, not fixed by explicit micro-conditioning) -> **768px frozen** (clean on validation, ~71-95s/img, ~2.4x faster than 1024px). Generated in 5 small paced batches (10-12 images each) with load checks between batches, not one continuous block | 60 images | 60/60 generated, 768x768, 0 degenerate, spot-checked visually | both amendments made on image-validity/machine-stewardship grounds, before any feature was extracted from any SDXL image and before any classification result was seen |
| E109 frozen panel extraction | measure the unchanged 10-feature v2 panel on SDXL images | confirmatory (protocol reuse) | same two-pass extractor (`extract_stage_panel.py`/`compute_lpips_panel.py`), same 256px canonicalization, same human COCO captions (no BLIP) | 60 images | 60/60 extracted, 0 NaN | zero code change required to add a third generator, confirming the frozen extractor is generator-agnostic (already established by aMUSEd) |
| **E110 primary preregistered test + secondary analyses** | does trajectory add information beyond VAE+score for SDXL; feature-level comparison; transfer | **confirmatory (preregistered)** | identical content-grouped 5x10 CV, identical bootstrap, identical code path as E102-E103 | 60 real-vs-SDXL pairs | **PRIMARY: ΔAUROC(VAE+score+trajectory vs VAE+score) = +0.053, 95% CI [0.016, 0.092] -- excludes zero.** VAE alone 0.855 (does not saturate, unlike aMUSEd's 0.981). `diffpath_curvature` d=-0.665 (larger than SD1.5's -0.542; aMUSEd's -0.175 is n.s.). `path_length` sign matches SD1.5 (-0.36 vs -0.33), disagrees with aMUSEd's reversal (+1.39) -- the same feature already flagged as SD1.5/aMUSEd's one large reversal. 9/10 features sign-agree SD1.5<->SDXL vs 7/10 aMUSEd<->SDXL (same 3 features disagree in both comparisons). Transfer AUROC SD1.5<->SDXL 0.77-0.90, far stronger than SD1.5<->aMUSEd's 0.58-0.60 from the prior phase | **Scenario A** (SDXL behaves like SD1.5): the trajectory result is not an SD1.5-checkpoint idiosyncrasy -- it replicates on a substantially larger, separately-trained, differently-VAE'd UNet diffusion model. Confound: SD1.5 and SDXL share Stable-Diffusion-family lineage, so this cannot yet separate "diffusion property" from "SD-lineage property" -- exactly why a true Diffusion Transformer (blocked this pass, E107) remains the more decisive, not-yet-run test |

### Reading E107-E110 together

The active hypothesis is **strengthened**, not settled. Every angle of this experiment (primary incremental-AUROC
test, raw curvature effect size, feature-level sign agreement, cross-generator transfer) converges on the same
answer: SDXL's trajectory behaves like SD1.5's, not like aMUSEd's. This rules out the narrowest form of the
"SD1.5-checkpoint-specific" null hypothesis. It does not yet rule out the "Stable-Diffusion-lineage-specific"
null hypothesis, since SDXL is still a UNet-based, SD-family model. The single most informative next step
remains an architecturally independent diffusion model (ideally a DiT), not scaling the robustness study around
curvature yet -- recorded as the recommended next experiment in `PIXART_STAGE_DECOMPOSITION.md`.

## Fourth-generator architecture-disambiguation experiment: genuine PixArt-Sigma DiT (E111–E113)

*Scope: the decisive follow-up to E107-E110, finally testing a generator outside the Stable Diffusion UNet
lineage. Full configuration and every engineering blocker/fix: `PREREGISTRATION_DIT_GENERATOR_V1.md`. Full
results: `DIT_STAGE_DECOMPOSITION.md`. No feature or classifier change — pure protocol reuse on a fourth
generator's images, computed entirely with the frozen SD1.5 probe (the DiT generator supplies only RGB pixels).*

| exp | question | type | design | data | result | interpretation |
|---|---|---|---|---|---|---|
| E111 genuine DiT generation, resolved disk blocker via GGUF quantization | generate 60 real PixArt-Sigma (Diffusion Transformer) counterparts | **implementation** | Official PixArt-Sigma transformer+VAE, unmodified; disclosed adaptation: community GGUF-quantized T5-XXL text encoder (3.39GB vs official 19GB, resolving E107's disk blocker). 512x512, 20 steps, guidance 4.5, `DPMSolverMultistepScheduler`. Generation split across two environments: 15/60 locally (validated), 45/60 on an HPC A40 GPU cluster after the user offered compute access explicitly to reduce local machine load (repeated MPS swap-thrashing incidents during earlier local attempts) | 60 images | 60/60 generated, 512x512, 0 corrupted (PIL verify), 9-image random QC contact sheet shows coherent, correctly-captioned, non-degenerate photographs | **8 execution-environment bugs found and fixed on the HPC path, none touching the frozen protocol**: missing job working directory, missing `sentencepiece`/`protobuf` packages, `HF_HUB_OFFLINE=1` breaking tokenizer cache resolution (fixed via explicit `local_files_only=True`), a missing cached `config.json` for the GGUF repo in offline mode (fixed by one online cache-warming pass), a diffusers-version regression where `PixArtSigmaPipeline.__init__` crashes on `vae=None` (fixed by loading the real, tiny VAE in the caption-encoding phase instead of `None` — no effect on embeddings), a device-placement bug in `encode_prompt()`'s automatic execution-device detection, and a `negative_prompt`/`negative_prompt_embeds` conflict in a stricter `check_inputs()`. Every fix is execution-only (device selection, package installation, explicit kwargs) — steps/guidance/resolution/scheduler/seed/captions never changed |
| E112 frozen panel extraction | measure the unchanged 10-feature v2 panel on genuine DiT images | confirmatory (protocol reuse) | same two-pass extractor (`extract_stage_panel.py`/`compute_lpips_panel.py`), same 256px canonicalization, same human COCO captions | 60 images | 60/60 extracted, 0 NaN (300/300 total rows across all 5 generators) | zero code change required for the fourth generator, confirming the frozen extractor remains fully generator-agnostic |
| **E113 primary preregistered test + secondary analyses** | does trajectory add information beyond VAE+score for a genuine DiT; feature-level comparison across 4 generators; full transfer matrix | **confirmatory (preregistered)** | identical content-grouped 5x10 CV, identical bootstrap, identical code path as E102/E110 | 60 real-vs-pixart_dit pairs | **PRIMARY: ΔAUROC(VAE+score+trajectory vs VAE+score) = −0.005, 95% CI [−0.029, 0.018] — includes zero, point estimate slightly negative.** But `diffpath_curvature` alone remains strong and sign-consistent with SD1.5 (d=−0.54) and SDXL (d=−0.67) at d=−0.53; and trajectory *does* add significantly on top of score alone (+0.132, CI [0.032,0.232]), just not on top of VAE+score, and actively subtracts on top of VAE alone (−0.025, CI [−0.048,−0.007]). VAE alone reaches 0.876 (comparable to SDXL's 0.855, well above SD1.5's 0.639). 4/10 features sign-agree with SD1.5, 6/10 with SDXL (down from SD1.5<->SDXL's 9/10) — `lare_t200`, `latent_mse_ae`, `path_length`, `pixel_mse_ae` reverse against both. Full transfer matrix (12 directions): DiT<->{SD1.5,SDXL,aMUSEd} ranges 0.67-0.80, weaker than SD1.5<->SDXL (0.77-0.90) but stronger than SD1.5<->aMUSEd (0.58-0.60); one striking outlier, pixart_dit->amused = 0.949, the single strongest transfer direction in the entire matrix | **Scenario C (intermediate)**, per the preregistration's own example ("trajectory adds only in one targeted comparison"). Not a clean replication (Scenario A) nor a clean failure (Scenario B, since curvature itself is neither weak nor inconsistent) — the raw `diffpath_curvature` effect survives the architecture change essentially unchanged, but how much it *adds* on top of other stages does not, most plausibly due to VAE+score already occupying more of the available headroom for this generator than for SD1.5 |

### Reading E111-E113 together

The trajectory-curvature hypothesis is **modulated by architecture, not falsified**. Moving from two UNet-based
Stable Diffusion models (SD1.5, SDXL) to a genuine Diffusion Transformer changes the *incremental* result (the
primary preregistered test now fails) while leaving the *raw* `diffpath_curvature` effect size essentially
unchanged (d=-0.53, in the same range as d=-0.54/-0.67 for SD1.5/SDXL, and unlike aMUSEd's non-significant
d=-0.18). This is a more nuanced outcome than either of the two clean scenarios the preregistration anticipated,
and per its own decision rule the next step is one targeted ceiling-controlled mechanism test (matching VAE-alone
prediction confidence between generators before re-running the primary comparison), not a fifth generator —
recorded as the recommended next experiment in `DIT_STAGE_DECOMPOSITION.md`.

## VAE-curvature information-redundancy mechanistic follow-up (E114)

*Scope: **exploratory/mechanistic, not an independent preregistered confirmation** — designed after seeing
E113's discrepancy (PixArt's primary incremental test null despite a large, stable raw curvature effect size)
specifically to explain it. Plan frozen in `ANALYSIS_PLAN_VAE_CURVATURE_REDUNDANCY.md` before any result below
was inspected. Full results: `VAE_CURVATURE_REDUNDANCY.md`. No new generation, no new feature extraction — pure
CPU statistics on the already-cached 300-row v2 panel; no HPC needed for this phase.*

| exp | question | type | design | data | result | interpretation |
|---|---|---|---|---|---|---|
| **E114 cross-fitted VAE-curvature residualization + proper scoring rules + transfer diagnostic** | is `diffpath_curvature`'s real-vs-fake signal redundant with static VAE-reconstruction information, and does this redundancy explain PixArt's E113 null result | **exploratory (mechanistic follow-up, not a confirmatory test of a preregistered hypothesis)** | label-blind content-grouped cross-fitted ridge residualization (`C ~ Ridge(VAE feats)`, alpha=1.0, train-fold-only fit) per generator; paired Cohen's d on raw vs residual curvature; proper-scoring-rule (AUROC/logloss/Brier) incremental comparison using curvature alone (`path_length` excluded, since it reverses sign across architectures); effect-vector diagnostic for the PixArt→aMUSEd 0.949 transfer | same 60 content ids × 4 generators (sd15/sdxl/pixart_dit/amused vs real), already-cached v2 panel | **Working hypothesis not supported as stated.** SD1.5 retains a significant residual curvature effect after VAE residualization (d=−0.542→−0.337, CI still excludes zero, 62% retention) but **SDXL collapses just as much as PixArt** (10.5% vs 11.4% retention, both CIs cross zero) — contradicting the hypothesized SD1.5/SDXL-retain-vs-PixArt-collapses grouping. Proper scoring rules: curvature alone significantly *degrades* AUROC on top of VAE alone for PixArt (Δ=−0.021, CI excludes zero) and adds nothing on any metric on top of VAE+score (the E113 null replicates under logloss/Brier too); SD1.5 gets a real, CI-excluding-zero boost from curvature on top of VAE+score (AUROC and Brier); aMUSEd gets a small but real 3-metric improvement despite a weak raw effect. PixArt→aMUSEd transfer diagnostic: effect-vector cosine similarity 0.797, driven by VAE/round-trip features (`lpips_roundtrip`, `lpips_ae` are the top-2 |weight| features in the pixart_dit-trained classifier), not trajectory features — explains the high 0.949 transfer as a VAE/round-trip-level artifact, not shared diffusion dynamics | **Outcome R4 (mixed/nonlinear)**, with an identifiable failure mode: the linear residualization model itself behaves consistently (R² 0.30–0.41, stable across VAE-feature and learned-discriminant residualization) but the *hypothesized architecture grouping* does not hold — the data instead splits SD1.5 (low VAE-redundancy) from {SDXL, PixArt} (both high VAE-redundancy), an axis that tracks something other than UNet-vs-DiT (plausibly VAE capacity/generation era, not directly tested). PixArt's E113 null is best explained by curvature's VAE-redundancy (shared with SDXL, not PixArt-specific) combined with `path_length`'s already-documented PixArt-specific sign reversal removing the complementary contribution that appears to rescue SDXL |
