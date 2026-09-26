# scripts/ index

**Start with `scripts/reproduce/`** — `final_analysis.py` reproduces every headline result from committed data
(no GPU), and `analyze_image.py` runs the frozen measurement panel on your own image. Both are documented in the
root `README.md`.

Everything below is a normal package import (`from synthimage.probes.sd15 import SD15Probe`, etc. — no
`sys.path` manipulation). The frozen probe/feature/corruption code itself lives in `src/synthimage/`, installed
as an editable package (`uv sync`).

Earlier project phases used ~95 scripts and a much larger `results/` tree; they were removed to keep this a
portfolio-sized repo, but nothing is truly gone — see the "Research history" note in the root `README.md` for how
to retrieve any of it from the `research-history-archive` git tag.

## Dataset / generation

`build_caption_matched.py`, `build_sdxl_manifest.py`, `build_pixart_dit_manifest.py`,
`build_final_validation_manifest.py`, `build_final_validation_combined_manifest.py`, `generate_sdxl.py`,
`generate_pixart_dit.py`, `generate_ai_edit_scaled.py`

## Feature extraction (frozen v2 panel)

`extract_stage_panel.py` (pass 1: everything needing the SD1.5 UNet), `compute_lpips_panel.py` (pass 2:
LPIPS-only, run as a separate process — see that script's docstring for why).

## Analysis / plots

`stage_decomposition_analysis.py` (imports its CV/bootstrap/statistics functions from
`synthimage.analysis.cv` and re-exports them — every other analysis script below imports from
`stage_decomposition_analysis.py`/`vae_curvature_redundancy_analysis.py` rather than duplicating this code),
`stage_decomposition_plots.py`, `pixart_stage_decomposition_analysis.py`, `pixart_stage_decomposition_plots.py`,
`dit_stage_decomposition_analysis.py`, `dit_stage_decomposition_plots.py`,
`vae_curvature_redundancy_analysis.py`, `vae_curvature_redundancy_plots.py`,
`path_length_mechanism_analysis.py`, `path_length_mechanism_plots.py`, `final_validation_analysis.py`,
`final_validation_plots.py`, `make_pipeline_diagram.py` (regenerates the README's pipeline figure)

**Naming note (read this before trusting a filename):** `pixart_stage_decomposition_*.py` is named after the
generator this project *originally intended* to run — the actual images analyzed by that script are **SDXL**,
substituted after PixArt-Sigma was blocked at the time (disk space; see
`docs/research_history/PREREGISTRATION_pixart_v1.md`). The genuine PixArt-Sigma Diffusion Transformer run is
`generate_pixart_dit.py` / `dit_stage_decomposition_*.py` (generator label `pixart_dit` throughout). Every
result table and figure uses the `sdxl` / `pixart_dit` generator labels unambiguously — only the *filenames* of
the SDXL-substitute scripts still carry the earlier working name.

## Reviewer validation (post-v1.0 extension)

`reviewer_validation/conditioning_ablation.py` and `reviewer_validation/probe_swap.py` — two stress tests of the
v1.0 `path_length` result (caption-conditioning ablation; SD1.5-vs-SDXL probe swap), run after the original study
was tagged `v1.0`. See `docs/research_history/reviewer_validation/REVIEWER_VALIDATION_PLAN.md` for the
preregistered protocol and `REVIEWER_VALIDATION_RESULTS.md` for the outcome.
