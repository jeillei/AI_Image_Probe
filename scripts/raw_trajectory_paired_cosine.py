"""Phase 4, paired statistic: for the SAME content id, does the direction of SD1.5's raw-trajectory
perturbation align with aMUSEd's perturbation more than chance?  This is the one statistic in this analysis
that is sensitive to content correspondence (unlike the subspace-level stats in raw_trajectory_core.py, which
are invariant to row order and are instead tested with the pair-breaking null).  Null: permute which AM
content is compared against which SD content (content-shuffle) before computing per-row cosine."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from src.analysis.raw_trajectory import load_cache, assert_triplets_complete, stack_tensor, TENSOR_KEYS, norm_raw

OUT = Path("results/raw_trajectory/nulls"); OUT.mkdir(parents=True, exist_ok=True)
N_SHUFFLE = 2000
rng = np.random.default_rng(1)
cache = load_cache("data/raw_trajectory_cache")
cids = sorted({cid for (g, cid) in cache if all((gg, cid) in cache for gg in ("real", "sd15", "amused"))})
assert_triplets_complete(cache, cids)

def cosine_rows(A, B):
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8); b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(1)

rows = []; per_content_rows = []
for key in TENSOR_KEYS:
    real = stack_tensor(cache, "real", cids, key)
    Dsd = norm_raw(stack_tensor(cache, "sd15", cids, key), real)
    Dam = norm_raw(stack_tensor(cache, "amused", cids, key), real)
    obs = cosine_rows(Dsd, Dam)
    for cid, c in zip(cids, obs): per_content_rows.append({"tensor": key, "content_id": cid, "paired_cosine": float(c)})
    null_means = np.array([cosine_rows(Dsd, Dam[rng.permutation(len(cids))]).mean() for _ in range(N_SHUFFLE)])
    w = wilcoxon(obs) if not np.allclose(obs, 0) else (np.nan, np.nan)
    rows.append({"tensor": key, "mean_paired_cosine": float(obs.mean()), "median_paired_cosine": float(np.median(obs)),
                "frac_positive": float((obs > 0).mean()), "null_mean_of_means": float(null_means.mean()),
                "null_sd_of_means": float(null_means.std()), "empirical_p_two_sided": float(np.mean(np.abs(null_means - null_means.mean()) >= abs(obs.mean() - null_means.mean()))),
                "wilcoxon_stat": float(w[0]) if w[0] == w[0] else None, "wilcoxon_p": float(w[1]) if w[1] == w[1] else None})
    print(key, rows[-1])
pd.DataFrame(rows).to_csv(OUT / "paired_cosine_summary.csv", index=False)
pd.DataFrame(per_content_rows).to_csv(OUT / "paired_cosine_per_content.csv", index=False)
print("\nDONE"); print(pd.DataFrame(rows).to_string(index=False))
