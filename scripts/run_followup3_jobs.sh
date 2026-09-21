#!/bin/bash
# Cross-generator matched pair (Addendum 1): aMUSEd fakes on the same 60 contents/captions -> manifest -> v1 features (only new rows are extracted).
set -u; cd "$(dirname "$0")/.."
until grep -q "all followup2 done" results/followup2_driver.log 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] followup3 start"
caffeinate -i uv run python scripts/build_content_matched.py --stage fake --generator amused --count 60 > results/content_matched/logs/amused.log 2>&1; echo "[$(date +%T)] amused fakes done"
uv run python scripts/build_content_matched.py --stage manifest --count 60 > results/content_matched/logs/manifest2.log 2>&1
caffeinate -i uv run python scripts/extract_detector_features.py --manifest data/content_matched/manifest.csv --output results/content_matched/v1_humancaption.json --rich --batch-size 2 > results/content_matched/logs/v1_amused.log 2>&1; echo "[$(date +%T)] all followup3 done"
