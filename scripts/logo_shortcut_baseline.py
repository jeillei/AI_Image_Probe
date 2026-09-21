"""E79: like-for-like re-run of the Stage-B headline protocol (LOGO with disjoint real folds, 450 images: 200 AIGC-real + 10x25 fakes)
with shortcut baselines.  C fixed at 0.1 (E52 tuned regularisation; this is intentionally less flexible).  Bootstrap over held-out generators' images."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import *
from src.analysis.families import COARSE
from src.analysis.metadata import META_COLS
d = load_features("results/features/stage_b_cross_source_rich.json"); d = d[(d.label == 1) | (d.source == "aigc_benchmark_real")].reset_index(drop=True)
gens = sorted(d[d.label == 1].generator.unique()); real_idx = d.index[d.label == 0].values; allf = feature_cols(d)
sets = {"ALL_v1(652)": allf, "legacy51": cols_for(d, COARSE["legacy51"]), "metadata_geometry(4)": ["min_side", "aspect", "is_square", "is_pow2_side"], "metadata_all(7)": META_COLS, "bpp_only(1)": ["bpp"], "is_square_only(1)": ["is_square"]}
res = []
for seed in range(5):
    rng = np.random.default_rng(seed); perm = rng.permutation(real_idx); folds = np.array_split(perm, len(gens))
    for name, cols in sets.items():
        for g, rf in zip(gens, folds):
            te = d.loc[list(d.index[d.generator == g]) + list(rf)]; tr = d.drop(te.index); m = make(.1).fit(np.nan_to_num(tr[cols].values), tr.label.values)
            res.append({"seed": seed, "feature_set": name, "held_generator": g, "auroc": roc_auc_score(te.label, m.predict_proba(np.nan_to_num(te[cols].values))[:, 1])})
r = pd.DataFrame(res); r.to_csv("results/audit/logo_shortcut_baseline_cells.csv", index=False)
per_seed = r.groupby(["feature_set", "seed"]).auroc.median().reset_index(); s = per_seed.groupby("feature_set").auroc.agg(["mean", "std"]).rename(columns={"mean": "median_LOGO_AUROC(mean over 5 real-fold seeds)", "std": "sd_over_seeds"})
s["worst_generator(mean over seeds)"] = r.groupby(["feature_set", "held_generator"]).auroc.mean().groupby("feature_set").min(); s = s.round(3); s.to_csv("results/audit/logo_shortcut_baseline_summary.csv"); print(s.to_string())
print(r.groupby(["feature_set", "held_generator"]).auroc.mean().unstack(0).round(2).to_string())
