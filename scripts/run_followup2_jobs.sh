#!/bin/bash
# Exploratory controls on the content-matched set (run after run_followup_jobs.sh): VAE baseline, JPEG75 and band128 pipelines on BOTH classes.
set -u; cd "$(dirname "$0")/.."; mkdir -p results/content_matched/analysis
until grep -q "all follow-up done" results/followup_driver.log; do sleep 30; done
echo "[$(date +%T)] followup2 start"
uv run python scripts/vae_recon_baseline.py > results/content_matched/logs/vae.log 2>&1; echo "[$(date +%T)] vae done"
for c in canon_crop256_jpeg75 canon_band128; do caffeinate -i uv run python scripts/extract_detector_features.py --manifest data/content_matched/$c.csv --output results/content_matched/$c.json --rich --batch-size 2 > results/content_matched/logs/$c.log 2>&1; echo "[$(date +%T)] $c done"; done
echo "[$(date +%T)] all followup2 done"
