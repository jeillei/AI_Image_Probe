# PUBLIC_RELEASE_CHECKLIST

Each item below was actually verified during the final cleanup pass, not assumed. Verification method noted
inline. Status as of this document's last update.

- [x] **Final validation completed** — ⚠️ **partial**: `FINAL_VALIDATION_PLAN.md` frozen and the extraction job
  submitted to HPC; Track A (transformation robustness) and Track B (AI-edit continuum) results were **pending**
  (job queued, not yet run) when this checklist was last updated. `docs/FINAL_RESULTS.md` §8–9 explicitly say
  "pending," not a fabricated result. **Action before publication: re-run this checklist once that job
  completes and fill in §8–9.**
- [x] **Headline numbers reproduced from saved results** — verified: `uv run python
  scripts/dit_stage_decomposition_analysis.py` and `scripts/path_length_mechanism_analysis.py` both re-ran
  successfully against the already-committed `results/` feature tables during this cleanup pass, reproducing the
  numbers quoted in `docs/FINAL_RESULTS.md` §5–7.
- [x] **README reflects final evidence** — rewritten from scratch this pass; every headline claim traces to a
  specific result in `docs/FINAL_RESULTS.md`; §8–9 (robustness/AI-edit) explicitly marked pending, not asserted.
- [x] **No unsupported detector claims** — checked: README/`docs/FINAL_RESULTS.md` explicitly disclaim a
  universal detector, generator-independent deployment performance, "curvature alone proves provenance," and any
  "percent AI" claim (§11 / README "Limitations").
- [x] **No secrets/tokens** — verified via `git grep` for API-key/secret/password/private-key/HF-token patterns
  across all tracked files (post-cleanup): zero matches beyond the word "secrets" appearing in a `.gitignore`
  comment.
- [x] **No private filesystem paths** — verified via `git grep` for `/Users/<name>`, `/home/<name>`,
  the HPC hostname, and the HPC username across all tracked files: zero matches (previously found in 2 tracked
  `.pbs` files and 5 tracked `.log` files; both fixed — `.pbs` files now use `${HPC_SCRATCH_ROOT:?...}` env-var
  placeholders, the `.log` files were untracked).
- [x] **No model checkpoints tracked** — verified: `.gitignore` excludes `*.safetensors`/`*.ckpt`/`*.gguf`,
  `.cache/`, `hf-cache/`, `torch-cache/`; `git ls-files` shows none present.
- [x] **No huge accidental files** — verified: total tracked repository size reduced from the original
  ~70MB (including several large v1-era raw feature dumps, 13MB/12MB/6.7MB/etc.) to **13MB total**, largest
  single tracked file 744KB (a QC contact-sheet image). The large v1-era dumps were untracked (`git rm --cached`)
  but remain in git history, not deleted from disk.
- [x] **Dataset licensing/redistribution considered** — `data/README.md` documents COCO's CC licensing terms and
  states explicitly that this project does not redistribute the underlying images, only ids and derived scalar
  features; generated images are not committed (regenerable via deterministic seeding, documented in the same
  file).
- [x] **Dependency installation tested** — `uv sync` ran clean (`Resolved 81 packages`, `Checked 59 packages`,
  no errors) in a fresh check during this pass.
- [x] **Minimal reproduction tested** — `scripts/dit_stage_decomposition_analysis.py` and
  `scripts/path_length_mechanism_analysis.py` both ran end-to-end against committed feature tables with no GPU
  and no model download, per `docs/REPRODUCIBILITY.md`'s minimal path.
- [x] **Tests pass** — `uv run pytest`: **20/20 passed**, including 5 new tests added this pass specifically
  covering scientifically load-bearing behavior (frozen feature definitions, content-grouped CV never splits a
  matched pair, cross-fitted residualization never lets a held-out fold leak into its own fit, manifest schema).
- [x] **Links/references resolve** — checked cross-references from the new top-level docs (`README.md`,
  `FINAL_VALIDATION_PLAN.md`, `docs/METHODS.md`, `docs/REPRODUCIBILITY.md`) into `docs/research_history/`
  after the file moves; all rewritten to the correct `docs/research_history/...` path.
- [x] **Figures render** — every figure referenced from `README.md` was visually inspected after generation
  (`results/summary/01_pipeline_diagram.png` through `05_vae_residualization_result.png`).
- [x] **Git working tree understood/clean** — `git status` reviewed before every commit in this pass; no
  unexplained modifications.
- [ ] **License present OR license choice explicitly flagged** — **no LICENSE file exists in this repository or
  its git history** (checked directly). Not chosen on the owner's behalf, per instruction. **LICENSE CHOICE
  REQUIRED BEFORE PUBLIC RELEASE.**
- [ ] **Repository is ready for manual GitHub publication** — see "GitHub readiness" verdict at the end of this
  cleanup task's handoff. Blocked on the two open items above (Track A/B completion, license choice).

## Known remaining manual actions

1. **License decision** (blocking) — pick a license (or explicitly decide "no license / all rights reserved")
   and add a `LICENSE` file.
2. **Complete Track A/B final validation** (blocking for full scientific completeness, not for code/hygiene
   readiness) — the HPC extraction job was queued at the time of this cleanup pass; once it completes, fill in
   `docs/FINAL_RESULTS.md` §8–9 with the actual retention/response numbers and re-verify this checklist's first
   item.
3. Consider whether `docs/research_history/obsolete_scripts/` should be dropped entirely rather than kept (they
   are currently kept per "do not delete anything scientifically important without preserving it," but they are
   explicitly labeled low-value).
4. `git push`/publish is a separate, explicit action this task does not take — nothing in this repository has
   been pushed to a remote as part of this cleanup.
