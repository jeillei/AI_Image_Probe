# scripts/ index

This directory accumulated ~95 scripts across the project's full history. Rather than move files and risk
breaking the `sys.path.insert(0, Path(__file__).resolve().parents[1])` import pattern every script in this
directory shares (which assumes `scripts/` is exactly one level below the repo root), this index tells you which
scripts matter for which purpose. Nothing below has been relocated.

## Final reproduction (the current, valid scientific pipeline)

These are the only scripts needed to reproduce every result in `docs/FINAL_RESULTS.md`. See
`docs/REPRODUCIBILITY.md` for the exact command sequence.

**Dataset / generation:**
`build_content_matched.py`, `build_sdxl_manifest.py`, `build_pixart_dit_manifest.py`,
`build_final_validation_manifest.py`, `build_final_validation_combined_manifest.py`, `generate_sdxl.py`,
`generate_pixart_dit.py`, `generate_ai_edit_scaled.py`

**Feature extraction (frozen v2 panel):**
`extract_stage_panel.py`, `compute_lpips_panel.py`, `extract_detector_features.py` (its `load()` function is
reused by the transform-aware manifests above), `thumbnail_baseline_multigen.py` (Stage-0 image-space control,
reused not recomputed — see `docs/research_history/LITERATURE_FEATURE_PANEL.md`)

**Analysis / plots:**
`stage_decomposition_analysis.py` (the core CV/bootstrap library every later analysis script imports),
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

## Research history (earlier phases, not needed to reproduce the final result, kept for the record)

The project's largest and most valuable finding was methodological: the original 652-feature representation and
its benchmark were confounded (`docs/research_history/SCIENTIFIC_AUDIT.md`), which is why they were abandoned
for the frozen 10-feature v2 panel above. Scripts from that phase and other exploratory dead ends:

`add_rr_real_to_stage_b.py`, `ai_edit_pilot_analysis.py`, `analyze_controlled_depth.py`, `analyze_image.py`,
`analyze_rich_checkpoint.py`, `analyze_sensitivity.py`, `audit_real_source.py`, `benchmark_stage_a_rich.py`,
`build_coco_controlled.py`, `build_coco_karpathy_controlled.py`, `build_controlled_detector_manifest.py`,
`build_core_manifests.py`, `build_master_manifest.py`, `build_prompt_ensemble.py`,
`build_robustness_pilot_manifest.py`, `build_robustness_screen.py`, `build_stage_b_manifest.py`,
`caption_baseline.py`, `caption_cohort.py`, `controlled_depth_sweep.py`, `crossed_eval.py`,
`crossprobe_analysis.py`, `dataset_audit_tables.py`, `detect.py`, `download_aigc_subset.py`,
`download_pilot_real.py`, `evaluate_content_matched.py`, `evaluate_content_matched_multigen.py`,
`evaluate_controlled_roundtrip.py`, `evaluate_core200.py`, `evaluate_cross_source.py`,
`evaluate_matched_crossprobe.py`, `evaluate_prompt_ensemble.py`, `evaluate_self_conditioning.py`,
`extract_crossprobe.py`, `extract_raw_trajectory.py`, `feature_audit_matched.py`, `generate_ai_edit_pilot.py`
(superseded by `generate_ai_edit_scaled.py`), `generate_cascade.py`, `incremental_information_test.py`,
`incremental_information_test_spatialfft.py`, `logo_shortcut_baseline.py`, `matched_baseline_comparison.py`,
`matched_crossprobe_multigen.py`, `matched_subset_eval.py`, `materialize_signature_parquet.py`,
`plot_prompt_ensemble.py`, `prepare_controlled_detector_features.py`, `raw_trajectory_baselines.py`,
`raw_trajectory_core.py`, `raw_trajectory_feature_link.py`, `raw_trajectory_normcompare.py`,
`raw_trajectory_paired_cosine.py`, `raw_trajectory_plots.py`, `residualized_by_family.py`, `residualized_eval.py`,
`robustness_pilot_analysis.py` (superseded by the scaled Track A analysis), `run_pilot.py`,
`sample_extraction_manifest.py`, `sd15_optimized_subset.py`, `sd15_sensitivity.py`,
`seed_caption_cache_from_features.py`, `seed_prompt_features.py`, `seed_rich_full_cache.py`,
`seed_robustness_clean_cache.py`, `stage_a_followup_diagnostics.py`, `thumbnail_baseline.py` (single-generator
predecessor of `thumbnail_baseline_multigen.py`), `train_detector.py`, `train_stage_a_experimental_detector.py`,
`vae_recon_baseline.py`, plus the sequential job-runner shells `run_core_jobs.sh`, `run_followup_jobs.sh`,
`run_followup2_jobs.sh`, `run_followup3_jobs.sh` (v1-era dataset construction pipeline).

## Obsolete / superseded debug scripts

Moved to `docs/research_history/obsolete_scripts/`: `benchmark_batch2.py`, `optimize_sd15.py`,
`validate_sd15.py` — one-off diagnostics whose findings are already fully captured in prose elsewhere.
