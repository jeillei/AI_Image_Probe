"""E65: resize/JPEG-matched control.  Keep only images whose native side is 256 (so the 256px
canonicalizer is an identity resample), square, JPEG at the benchmark's common quantization
(excludes ADM PNG).  Real = AIGC-benchmark 256px reals; fake = BigGAN, DALLE2, GLIDE, VQDM.
Leave-one-generator-out; real images are split into folds (single real source -> cannot be held out)."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from src.analysis.evalkit import *
from src.analysis.families import COARSE
d = load_features("results/features/stage_b_cross_source_rich.json")
m = d[(d.min_side == 256) & (d.is_square == 1) & (d.is_png == 0) & ((d.label == 1) | (d.source == "aigc_benchmark_real"))].reset_index(drop=True)
print(m.groupby("source").size().to_dict(), "qtable range", m.qtable_mean.min(), m.qtable_mean.max())
gens = sorted(m[m.label == 1].generator.unique()); allf = feature_cols(d); res = []
sets = {"ALL_v1": allf, "legacy51": cols_for(d, COARSE["legacy51"]), "rich608": [c for c in allf if c.startswith("rich_")], "bpp_only(file-size shortcut)": ["bpp"]}
for name, cols in sets.items():
    for g in gens:
        te_f = m[m.generator == g]; real = m[m.label == 0]; aucs = []; ps = []; ys = []
        for r, (tr_i, te_i) in enumerate(KFold(5, shuffle=True, random_state=0).split(real)):
            tr = pd.concat([m[(m.label == 1) & (m.generator != g)], real.iloc[tr_i]]); te = pd.concat([te_f, real.iloc[te_i]])
            mod = make(.1).fit(np.nan_to_num(tr[cols].values), tr.label.values); p = mod.predict_proba(np.nan_to_num(te[cols].values))[:, 1]
            aucs.append(roc_auc_score(te.label, p))
        res.append({"feature_set": name, "held_generator": g, "auroc_mean_over_real_folds": float(np.mean(aucs)), "auroc_min_fold": float(np.min(aucs)), "n_fake": len(te_f), "n_real_total": len(real)})
r = pd.DataFrame(res); r.to_csv("results/audit/matched256_logo.csv", index=False)
print(r.pivot(index="held_generator", columns="feature_set", values="auroc_mean_over_real_folds").round(3)); print(r.groupby("feature_set").auroc_mean_over_real_folds.agg(["mean", "median", "min"]).round(3))
