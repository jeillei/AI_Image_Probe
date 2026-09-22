"""Extends E77's independent-probe check to aMUSEd: DiT-XL/2 (transformer, ImageNet, shared SD-family VAE) and
CIFAR-10 DDPM-32 (VAE-free, pixel-space) Class-B (30 quantities) AUROC per generator, content-grouped CV."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import make
from src.probes.base import CROSS_PROBE_COLUMNS as Q
m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id))
def load(f): r = json.load(open(f)); return pd.DataFrame([{"path": x["path"], "label": x["label"], "generator": x["generator"], "content_id": cid.get(x["path"], ""), **{k: x["features"][k] for k in Q}} for x in r])
rng = np.random.default_rng(0); out = {}
for pname, f in [("DiT-XL/2 classB30", "results/crossprobe/dit_content_matched.json"), ("CIFAR32 classB30", "results/crossprobe/cifar32_content_matched.json")]:
    d = load(f); out[pname] = {}
    for gen in ["sd15", "amused"]:
        x = d[(d.label == 0) | (d.generator == gen)].reset_index(drop=True); ids = np.array(sorted(x.content_id.unique())); X = np.nan_to_num(x[Q].values); y = x.label.values; oof = np.zeros(len(x)); a = []
        for r in range(10):
            perm = np.random.default_rng(r).permutation(ids); fo = x.content_id.map({c: i % 5 for i, c in enumerate(perm)}).values; p = np.zeros(len(x))
            for k in range(5): te = fo == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
            a.append(roc_auc_score(y, p)); oof += p / 10
        by = {c: np.where(x.content_id.values == c)[0] for c in ids}; bs = [roc_auc_score(y[i], oof[i]) for i in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(1000))]
        out[pname][gen] = {"auroc": float(np.mean(a)), "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}
json.dump(out, open("results/content_matched/analysis/crossprobe_baseline_multigen.json", "w"), indent=1); print(json.dumps(out, indent=1))
