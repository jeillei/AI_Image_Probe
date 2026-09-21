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
