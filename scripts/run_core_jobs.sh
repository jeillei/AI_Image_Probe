#!/bin/bash
# Sequential, resumable extraction of all core-200 conditions (one model process at a time on MPS).
# Re-running skips finished rows.  Approx 35 min/condition at batch 2 on Apple MPS (~11 s/image).
set -u; cd "$(dirname "$0")/.."; mkdir -p results/core200/logs
run(){ name=$1; shift; echo "[$(date +%T)] start $name"; caffeinate -i uv run python scripts/extract_detector_features.py --manifest data/core200/$name.csv --output results/core200/$name.json --rich --batch-size 2 "$@" > results/core200/logs/$name.log 2>&1; echo "[$(date +%T)] done $name rc=$?"; }
run canon_crop256
run cond_null --null-conditioning
run canon_crop256_jpeg75
run cond_shuffled
run canon_band128
run cond_generic
