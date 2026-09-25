# PUBLIC_REPO_AUDIT — repository inventory for public release

Complete classification of every meaningful top-level directory/file, performed before any further cleanup
decision in this pass. Verified with `du -sh`, `find . -type f -size +5M`, `git ls-files`, `git status --short`,
`git status --ignored`. Git-tracked repository size: **17MB, 475 files**. On-disk working directory (including
everything gitignored): several GB, dominated by `data/` (raw datasets, generated images, caches — see below).

## KEEP — public core

| path | notes |
|---|---|
| `README.md`, `pyproject.toml`, `uv.lock`, `.gitignore` | root entrypoints/config |
| `FINAL_VALIDATION_PLAN.md`, `PUBLIC_RELEASE_CHECKLIST.md`, `SYNTHIMAGE_PROJECT_STATE.md`, `PUBLIC_REPO_AUDIT.md` | root status/audit docs |
| `docs/FINAL_RESULTS.md`, `docs/METHODS.md`, `docs/REPRODUCIBILITY.md` | primary public documentation |
| `src/` (292K) | frozen probe (`src/probes/sd15.py`), frozen v2 feature panel (`src/features/panel_v2.py`), transform suite (`src/corruption/`), data utilities (`src/data/captioning.py`) — everything imported by the final pipeline |
| `scripts/reproduce/` (new) | `final_analysis.py` (headline reproduction, no GPU), `analyze_image.py` (single-image measurement CLI) — the two canonical public entrypoints |
| ~21 scripts at `scripts/` root | the final-reproduction subset (generation/extraction/analysis for the current 4-generator, mechanism, and validation pipeline) — see `scripts/README.md` for the exact list |
| `scripts/README.md`, `data/README.md` | indexes/provenance docs |
| `tests/` (96K) | scientific-correctness test suite, 20 tests |
| `hpc/*.pbs` (12K, 3 files) | generic, sanitized (env-var placeholder) job templates — optional acceleration infrastructure, documented as such |

## KEEP — compact scientific result

| path | size (tracked) | notes |
|---|---:|---|
| `results/stage_decomposition/` | ~1.5MB tracked | four-generator comparison, feature/stage tables, `panel_features_dit.json` (the compact final feature table — 60 content ids × 5 generators) |
| `results/vae_curvature_redundancy/` | 784K | mechanism-analysis tables + figures |
| `results/path_length_mechanism/` | ~similar | the decisive mechanism result's tables + figures |
| `results/final_validation/` | 3.6MB | robustness (Q1-Q3) + AI-edit tables, `panel_features_final.json` (4,380-row final validation feature table), figures |
| `results/summary/` | 420K | the 7 headline figures + pipeline diagram used in README |
| `results/pixart/`, `results/crossprobe/cifar32.json`, a handful of QC contact-sheet PNGs | small | referenced from `docs/research_history/` reports |

All of the above are small, final, directly support a specific number or figure in `docs/FINAL_RESULTS.md`, and
were kept deliberately, not because they happened to exist.

## ARCHIVE — research history

`docs/research_history/` (already populated this project phase): 17 historical protocol/report markdown files
(preregistrations, `RESEARCH_REPORT.md`, phase-result documents), plus `docs/research_history/obsolete_scripts/`
(3 genuinely obsolete debug scripts, moved with corrected import depth).

**Disclosed compromise**: ~68 additional Python scripts at `scripts/` root are classified as research-history
by *role* (superseded v1-era experiments, one-off pilots since scaled up) but were deliberately **not physically
moved**, because every script in this codebase shares a `sys.path.insert(..., parents[1])` import pattern that
assumes `scripts/` is exactly one level below the repo root — moving ~68 files into a nested archive directory
would have silently broken that pattern for each one, which the task's own "do not move files if it would break
references" instruction weighs against a purely cosmetic reorganization. Instead, `scripts/README.md` indexes
every script by role (final-reproduction / research-history / obsolete) so the distinction is legible without
the move. This is flagged here as a deliberate, reasoned exception, not an oversight.

## IGNORE — reproducible/generated local artifacts (all gitignored, never tracked)

| path (local, not in git) | approx. size | why |
|---|---:|---|
| `data/RRDataset_original_train_val.tar.gz` | 3.3G | third-party raw dataset archive |
| `data/rr_partial/`, `data/rr_stage_b_real/`, `data/rr_pilot/`, `data/aigc_*`, `data/controlled_coco/` | ~1.7G combined | earlier-phase raw/downloaded image sets, superseded by `data/content_matched/` |
| `data/stage_panel_cache/` | 455M | per-image reconstruction cache (regenerable, `extract_stage_panel.py`) |
| `data/raw_trajectory_cache/` | 32M | v1-era raw tensor cache |
| `data/content_matched/` | 144M | the current matched-content images (real + all 5 generators) — regenerable via `scripts/build_content_matched.py` + per-generator scripts |
| `results/features/`, `results/core200/`, `results/self_conditioning/`, `results/prompt_ensemble/`, `results/content_matched/*.json`, `results/robustness_screen/` | ~140M combined | v1-era raw per-run feature dumps, already covered by existing `.gitignore` patterns (confirmed never tracked) |
| `.venv/`, `__pycache__/`, `.pytest_cache/` | varies | standard Python/tooling artifacts |

None of these were ever committed (verified via `git ls-files`) — no untracking action was needed for this
category, only confirming `.gitignore` coverage (done, see §5 below).

## REMOVE — accidental/debugging/dead artifacts

| item | action taken |
|---|---|
| `configs/` (empty directory, 0 files, never tracked) | removed this pass — served no purpose |
| 3 obsolete debug scripts (`benchmark_batch2.py`, `optimize_sd15.py`, `validate_sd15.py`) | already moved to `docs/research_history/obsolete_scripts/` in the prior cleanup pass |
| 10 large (13MB-1.6MB) v1-era raw feature dumps, 5 log files leaking a local absolute path, 2 `.pbs` files hardcoding a real HPC username/path | already untracked/sanitized in the prior cleanup pass (see `git log`) |

No new REMOVE-category items were found in this pass.

## MANUAL REVIEW

| item | status |
|---|---|
| **License** | no `LICENSE` file exists in the working tree or git history (checked directly, `git log --all --diff-filter=A --name-only \| grep -i license` returns nothing). **Not chosen on the owner's behalf.** The single remaining blocker before publication — see `PUBLIC_RELEASE_CHECKLIST.md`. |
| **COCO image redistribution** | `data/README.md` already documents that this project does not redistribute COCO images, only ids + derived features, per COCO's CC terms — no further action needed, flagged here as already resolved. |

## Notebooks

None exist (`find . -iname "*.ipynb"` returns nothing outside `.venv/`) — the notebook-audit requirement is
trivially satisfied; nothing to clean.

## Duplicate/stale file search

Searched for `*old*`, `*copy*`, `*backup*`, `README2*`, `*_v2.md` at the repository root: none found. Exactly
one current README, METHODS, FINAL_RESULTS, REPRODUCIBILITY, and PROJECT_STATE document exists, as required.
