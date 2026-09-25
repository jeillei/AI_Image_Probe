# PUBLIC_RELEASE_CHECKLIST

Each item below was actually verified during this pass (a dedicated final repository-cleanup task, run after
the science was already complete), not assumed. See `PUBLIC_REPO_AUDIT.md` for the full inventory this
checklist is based on.

- [x] **Final README reviewed** — every headline claim traces to a specific result in `docs/FINAL_RESULTS.md`;
  read start-to-finish as a stranger this pass (§ installation → quick reproduction → own-image analysis →
  layout → hardware → limitations → research history → status → license), nothing found confusing or requiring
  undocumented local knowledge.
- [x] **Final scientific claims match FINAL_RESULTS** — checked: README's 6 headline findings and Limitations
  section match `docs/FINAL_RESULTS.md` §5–11 exactly; no claim in either exceeds what §10–11 disclaim (no
  universal detector, no generator-independent deployment claim, no "curvature alone proves provenance," no
  "percent AI" claim).
- [x] **Git-tracked files audited** — full inventory in `PUBLIC_REPO_AUDIT.md`: 475 tracked files, 17MB total.
  Every top-level path classified KEEP (public core / compact result) / ARCHIVE / IGNORE / REMOVE / MANUAL
  REVIEW; no item left unclassified.
- [x] **`.gitignore` tested** — `git check-ignore -v` run against representative files in both directions this
  pass: confirmed `scripts/reproduce/*.py`, `results/final_validation/panel_features_final.json`,
  `results/summary/*.png`, `data/README.md`, and `PUBLIC_REPO_AUDIT.md` are **not** ignored; confirmed `.venv`,
  `data/content_matched`, `data/stage_panel_cache`, `results/core200/*.json`, and `.DS_Store` **are** ignored.
- [x] **Secrets scan passed** — `git grep` for API-key/secret/password/private-key/HF-token patterns across all
  tracked files, re-run this pass including the two new `scripts/reproduce/` files and `PUBLIC_REPO_AUDIT.md`:
  zero matches beyond the word "secrets" in prose/`.gitignore` comments.
- [x] **Private paths removed** — `git grep` for `/Users/<name>`, `/home/<name>`, the remote-machine hostname,
  and the remote-machine username across all tracked files: zero matches. One remaining literal SSH alias
  (`ssh vandal-codex`) was found and removed from `FINAL_VALIDATION_PLAN.md` this pass (replaced with neutral
  "a remote GPU machine" framing); `docs/REPRODUCIBILITY.md`'s hardware framing was also rewritten this pass to
  present GPU/cluster acceleration as optional, not a prerequisite (see §6–7 of this task).
- [x] **Raw datasets excluded** — verified: `data/` is fully gitignored except `data/README.md`; `git ls-files`
  confirms no COCO images, no third-party dataset archives are tracked. `data/README.md` documents provenance,
  licensing (COCO's CC terms), and regeneration instructions.
- [x] **Model weights excluded** — verified: `.gitignore` excludes `*.safetensors`/`*.ckpt`/`*.gguf`/`*.pt`/
  `*.pth`, `.cache/`, `hf-cache/`, `torch-cache/`; `git ls-files` shows none tracked.
- [x] **Heavyweight caches excluded** — verified: `data/stage_panel_cache/` (455M on disk), `data/
  raw_trajectory_cache/` (32M), and all v1-era raw per-run feature dumps under `results/` are gitignored, never
  tracked (confirmed via `git status --ignored`).
- [x] **Large tracked files justified** — `git ls-files` + `du`: only one tracked file exceeds 1MB
  (`results/final_validation/panel_features_final.json`, 3.3MB — the compact final feature table behind
  §8–9's numbers); none exceed 5MB. Every file >100KB has a stated purpose in `PUBLIC_REPO_AUDIT.md`.
- [x] **Minimal reproduction works** — `uv run python scripts/reproduce/final_analysis.py` (the new single
  canonical entrypoint, created this pass) ran end-to-end this pass: all 8 steps succeeded, ~4 minutes, no GPU,
  no model download, zero unexpected diffs to committed outputs (fully deterministic).
- [x] **Clean install tested** — `uv sync`: `Resolved 81 packages`, `Checked 59 packages`, no errors, re-run
  this pass.
- [x] **Tests pass** — `uv run pytest`: **20/20 passed** (re-run this pass after all changes).
- [x] **Public image-analysis command smoke-tested** — `scripts/reproduce/analyze_image.py` (new this pass):
  `--help` reviewed, missing-file error path tested (clean `argparse` error, exit code 2), and a full real run
  against a sample image tested both with and without `--skip-lpips` — all 10 frozen features computed
  correctly, no NaN, plausible values.
- [x] **Final figures regenerate/render** — all 8 figures under `results/summary/` (pipeline diagram + 7
  headline results) visually inspected; the two newest (`06_path_length_robustness.png`,
  `07_ai_edit_response_comparison.png`) confirmed to regenerate byte-identically via
  `scripts/reproduce/final_analysis.py`.
- [x] **Links/references resolve** — cross-references from every top-level doc into `docs/research_history/`
  checked and correct; `README.md`'s repository-layout tree matches the actual tree (including the new
  `scripts/reproduce/` and `PUBLIC_REPO_AUDIT.md` entries added this pass).
- [x] **No obvious stale duplicate docs** — searched for `*old*`/`*copy*`/`*backup*`/`README2*`/`*_v2.md`:
  none found. Exactly one current README, METHODS, FINAL_RESULTS, REPRODUCIBILITY, and PROJECT_STATE document.
- [x] **Research history preserved** — `docs/research_history/` holds 17 historical protocol/report documents
  plus 3 archived obsolete scripts; nothing was deleted, only reorganized (see `PUBLIC_REPO_AUDIT.md`'s
  disclosed compromise on the ~68 research-history-role scripts kept in place to avoid breaking their import
  paths, indexed instead via `scripts/README.md`).
- [x] **Repository status understood** — `git status` clean at every commit boundary this pass; `git log`
  reviewed; no unexplained modifications.
- [x] **No notebooks requiring audit** — `find . -iname "*.ipynb"` (excluding `.venv/`): zero results.
- [ ] **License resolved OR explicitly listed as final manual blocker** — **no `LICENSE` file exists in this
  repository or its git history** (checked directly this pass again). Not chosen on the owner's behalf.
  **LICENSE CHOICE REQUIRED BEFORE PUBLIC RELEASE — this is the only unchecked item.**

## Remaining manual actions

1. **License decision** (blocking) — pick a license (or explicitly decide "no license / all rights reserved")
   and add a `LICENSE` file. The only remaining blocker to publication.
2. Create/attach a GitHub remote, then `git add . && git commit && git push` — a separate, explicit action this
   task does not take. See the exact command sequence in the task handoff.
