"""E80 (exploratory control): 12 global colour/contrast statistics of a 32x32 thumbnail -> logistic, on the content-matched set (content-grouped CV).
Tests whether the matched-set separation is just a coarse 'look' (saturation/contrast/brightness) of CFG-guided generations."""
from __future__ import annotations
import sys, json
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
X = np.array([feats(p) for p in m.path]); y = m.label.values; ids = np.array(sorted(m.content_id.unique())); rng = np.random.default_rng(0); oof = np.zeros(len(m)); a = []
for r in range(10):
    perm = np.random.default_rng(r).permutation(ids); fo = m.content_id.map({c: i % 5 for i, c in enumerate(perm)}).values; p = np.zeros(len(m))
    for k in range(5): te = fo == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    a.append(roc_auc_score(y, p)); oof += p / 10
by = {c: np.where(m.content_id.values == c)[0] for c in ids}; bs = [roc_auc_score(y[i], oof[i]) for i in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(1000))]
out = {"thumb32_12stats_auroc": float(np.mean(a)), "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}; json.dump(out, open("results/content_matched/analysis/thumbnail_baseline.json", "w")); print(out)
