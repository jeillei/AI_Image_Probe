"""E78: how much crossed real-vs-fake signal survives linear removal of acquisition metadata?
Ridge regression of every trajectory feature on nuisance columns is fit on the TRAINING split only (both classes), then removed from train and test.
Two nuisance sets: geometry(min_side,aspect,is_square,is_pow2,is_png) and geometry+file-complexity(bpp,qtable).  Exploratory."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.analysis.evalkit import *
d = load_features("results/features/stage_b_cross_source_rich.json"); d["log_min_side"] = np.log(d.min_side); d["qtable_mean"] = d.qtable_mean.fillna(0); d["log_bpp"] = np.log(d.bpp)
NUI = {"geometry": ["log_min_side", "aspect", "is_square", "is_pow2_side", "is_png"], "geometry+complexity": ["log_min_side", "aspect", "is_square", "is_pow2_side", "is_png", "log_bpp", "qtable_mean"]}
gens = sorted(d[d.label == 1].generator.unique()); srcs = sorted(d[d.label == 0].source.unique()); allf = feature_cols(d); out = []
def run(name, nui):
    cells = []
    for rs in srcs:
        for g in gens:
            te = d[(d.generator == g) | ((d.label == 0) & (d.source == rs))]; tr = d.drop(te.index); y = te.label.values
            Xtr = np.nan_to_num(tr[allf].values); Xte = np.nan_to_num(te[allf].values)
            if nui:
                sc = StandardScaler().fit(tr[nui]); Ntr = sc.transform(tr[nui]); Nte = sc.transform(te[nui]); r = Ridge(alpha=1.0).fit(Ntr, Xtr); Xtr = Xtr - r.predict(Ntr); Xte = Xte - r.predict(Nte)
            m = make(.1).fit(Xtr, tr.label.values); p = m.predict_proba(Xte)[:, 1]; cells.append({"held_real": rs, "held_generator": g, "auroc": roc_auc_score(y, p)})
    c = pd.DataFrame(cells); c.to_csv(f"results/audit/crossed/residualized_{name}.csv", index=False); a = c.auroc.values
    return {"nuisance": name, "macro_mean": a.mean(), "median": np.median(a), "worst": a.min(), "frac_above_0.5": (a > .5).mean()}
for name, nui in [("none", None), *NUI.items()]: out.append(run(name, nui)); print(out[-1], flush=True)
pd.DataFrame(out).to_csv("results/audit/crossed/residualized_summary.csv", index=False)
