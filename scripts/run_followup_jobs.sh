#!/bin/bash
# (Run after run_core_jobs.sh has finished.)  content-matched fakes (SD1.5) -> manifest -> v1 features (human caption) -> DiT probe on core-200 + content-matched.
set -u; cd "$(dirname "$0")/.."; mkdir -p results/content_matched/logs
echo "[$(date +%T)] follow-up start"
caffeinate -i uv run python scripts/build_content_matched.py --stage fake --count 60 > results/content_matched/logs/fake.log 2>&1
uv run python scripts/build_content_matched.py --stage manifest --count 60 > results/content_matched/logs/manifest.log 2>&1
echo "[$(date +%T)] fakes done"
caffeinate -i uv run python scripts/extract_detector_features.py --manifest data/content_matched/manifest.csv --output results/content_matched/v1_humancaption.json --rich --batch-size 2 > results/content_matched/logs/v1.log 2>&1
echo "[$(date +%T)] v1 content-matched done"
caffeinate -i uv run python scripts/extract_crossprobe.py --probe dit --device mps --batch 2 --manifest data/core200/core200_frozen.csv --output results/crossprobe/dit_core200.json > results/content_matched/logs/dit_core200.log 2>&1
echo "[$(date +%T)] dit core200 done"
caffeinate -i uv run python scripts/extract_crossprobe.py --probe dit --device mps --batch 2 --manifest data/content_matched/manifest.csv --output results/crossprobe/dit_content_matched.json > results/content_matched/logs/dit_cm.log 2>&1
echo "[$(date +%T)] all follow-up done"
