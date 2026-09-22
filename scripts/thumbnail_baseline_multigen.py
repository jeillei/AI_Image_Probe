"""Extends the exploratory thumbnail/colour baseline (E78 for SD1.5) to per-generator numbers including aMUSEd,
using the identical 12-statistic feature definition and content-grouped CV; not re-tuned."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import make
m = pd.read_csv("data/content_matched/manifest.csv")
def feats(p):
    im = Image.open(p).convert("RGB").resize((32, 32), Image.Resampling.LANCZOS); a = np.asarray(im, float) / 255; hsv = np.asarray(im.convert("HSV"), float) / 255; l = a.mean(2)
    return [*a.mean((0, 1)), *a.std((0, 1)), hsv[..., 1].mean(), hsv[..., 1].std(), hsv[..., 2].mean(), l.std(), np.abs(np.diff(l, axis=0)).mean(), np.abs(np.diff(l, axis=1)).mean()]
m["feat"] = [feats(p) for p in m.path]; rng = np.random.default_rng(0); out = {}
for gen in ["sd15", "amused"]:
    x = m[(m.label == 0) | (m.generator == gen)].reset_index(drop=True); X = np.array(list(x.feat)); y = x.label.values; ids = np.array(sorted(x.content_id.unique())); oof = np.zeros(len(x)); a = []
    for r in range(10):
        perm = np.random.default_rng(r).permutation(ids); fo = x.content_id.map({c: i % 5 for i, c in enumerate(perm)}).values; p = np.zeros(len(x))
        for k in range(5): te = fo == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y, p)); oof += p / 10
    by = {c: np.where(x.content_id.values == c)[0] for c in ids}; bs = [roc_auc_score(y[i], oof[i]) for i in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(1000))]
    out[gen] = {"thumb32_12stats_auroc": float(np.mean(a)), "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}
json.dump(out, open("results/content_matched/analysis/thumbnail_baseline_multigen.json", "w"), indent=1); print(json.dumps(out, indent=1))
