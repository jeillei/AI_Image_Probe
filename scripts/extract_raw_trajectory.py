"""Persist the raw per-timestep tensors the frozen SD1.5 v1 probe already computes internally, for the
content-matched 180-image set.  No protocol change: identical canonicalization (`load()` from
extract_detector_features.py, i.e. squash-resize to 256px), identical 6-step DDIM inversion, identical
human-caption conditioning, `capture_predictions=True` (already a supported, existing probe flag -- it is
what the `--rich` feature extraction already turns on internally; those tensors were simply never written to
disk before).  Resumable (skips images whose .npz already exists); one probe process; not committed to git
(results/raw_trajectory/cache/ is in .gitignore).

Per image we keep only what Phases 1-8 of RAW_TRAJECTORY_ANALYSIS.md need:
  z          forward latent states,     shape (7,4,32,32)  -- z[0] is the clean encoded latent (=VAE-only baseline)
  cond       conditional predicted score, shape (6,4,32,32) -- cond[k] evaluated at z[k]
  uncond     unconditional predicted score, shape (6,4,32,32) -- uncond[k] evaluated at z[k]
guidance (cond-uncond) and latent displacement (diff(z)) are trivial derived quantities and are NOT stored
separately.  The pixel-space reconstruction and the reverse/reconstruction-path latents are not needed by any
planned analysis and are dropped to keep the cache small (~300KB/image, ~55MB total for 180 images)."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scripts.extract_detector_features import load
from src.probes.sd15 import SD15Probe

def out_path(cache_dir: Path, row: dict) -> Path:
    # real/sd15/amused share the same content-id-derived filename in different subdirectories
    # (data/content_matched/{real,sd15,amused}/<content_id>.png); the cache key must include the
    # generator or collisions silently overwrite one class's cache with another's.
    return cache_dir / f"{row['generator']}__{row['content_id']}.npz"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/content_matched/manifest.csv")
    p.add_argument("--cache-dir", default="data/raw_trajectory_cache")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))[: a.limit]
    pending = [r for r in rows if not out_path(cache, r).exists()]
    print(f"{len(rows) - len(pending)}/{len(rows)} already cached; {len(pending)} to extract")
    if not pending:
        return
    probe = SD15Probe(a.steps)
    for start in range(0, len(pending), a.batch_size):
        group = pending[start:start + a.batch_size]
        t = time.time()
        xs = [load(r["path"], a.size, 0, seed=17) for r in group]
        caps = [r["caption"] for r in group]
        results = (probe.invert_reconstruct_batch(xs, caps, capture_predictions=True) if len(group) > 1
                   else [probe.invert_reconstruct(xs[0], caps[0], mode="caption", guidance=1.0, capture_predictions=True)])
        for r, res in zip(group, results):
            z = np.asarray(res["forward"], dtype=np.float32)          # (7,4,32,32)
            cond = np.asarray(res["cond_scores"], dtype=np.float32)   # (6,4,32,32)
            uncond = np.asarray(res["uncond_scores"], dtype=np.float32)  # (6,4,32,32)
            assert z.shape == (a.steps + 1, 4, a.size // 8, a.size // 8), z.shape
            assert cond.shape == uncond.shape == (a.steps, 4, a.size // 8, a.size // 8), (cond.shape, uncond.shape)
            np.savez_compressed(out_path(cache, r), z=z, cond=cond, uncond=uncond,
                                path=r["path"], content_id=r["content_id"], generator=r["generator"], label=int(r["label"]))
        elapsed = (time.time() - t) / len(group)
        print(len(group), "imgs,", round(elapsed, 2), "s/img,", start + len(group), "/", len(pending), flush=True)
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

if __name__ == "__main__":
    main()
