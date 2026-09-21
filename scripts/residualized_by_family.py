"""E80: crossed macro AUROC per coarse feature group, raw vs geometry-residualized (exploratory)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.analysis.evalkit import *
from src.analysis.families import COARSE
d = load_features("results/features/stage_b_cross_source_rich.json"); d["log_min_side"] = np.log(d.min_side)
NUI = ["log_min_side", "aspect", "is_square", "is_pow2_side", "is_png"]; gens = sorted(d[d.label == 1].generator.unique()); srcs = sorted(d[d.label == 0].source.unique()); rows = []
groups = {k: cols_for(d, v) for k, v in COARSE.items()}; groups["ALL_v1"] = feature_cols(d)
for name, cols in groups.items():
    for mode in ("raw", "geometry_residualized"):
        a = []
        for rs in srcs:
            for g in gens:
                te = d[(d.generator == g) | ((d.label == 0) & (d.source == rs))]; tr = d.drop(te.index); Xtr = np.nan_to_num(tr[cols].values); Xte = np.nan_to_num(te[cols].values)
                if mode != "raw": sc = StandardScaler().fit(tr[NUI]); r = Ridge(alpha=1.0).fit(sc.transform(tr[NUI]), Xtr); Xtr = Xtr - r.predict(sc.transform(tr[NUI])); Xte = Xte - r.predict(sc.transform(te[NUI]))
                a.append(roc_auc_score(te.label, make(.1).fit(Xtr, tr.label.values).predict_proba(Xte)[:, 1]))
        rows.append({"feature_group": name, "mode": mode, "n_features": len(cols), "macro_mean": np.mean(a), "median": np.median(a), "worst": np.min(a), "frac_above_0.5": np.mean(np.array(a) > .5)})
r = pd.DataFrame(rows); r.to_csv("results/audit/crossed/residualized_by_family.csv", index=False); print(r.pivot(index="feature_group", columns="mode", values=["macro_mean", "frac_above_0.5"]).round(3).to_string())
