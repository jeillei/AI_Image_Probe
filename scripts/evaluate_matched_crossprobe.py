"""E78 (exploratory): does an independent probe (DiT-XL/2 null-class) separate content-matched real vs SD1.5 fakes using the 30 Class-B quantities?
Content-grouped 5-fold x10, content bootstrap.  SD1.5 Class-B (its legacy noise/roundtrip/endpoint/geometry columns) is the like-for-like reference."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import make
from src.probes.base import CROSS_PROBE_COLUMNS as Q
m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id)); gen = sys.argv[1] if len(sys.argv) > 1 else "sd15"
def load(f): 
    r = json.load(open(f)); return pd.DataFrame([{"path": x["path"], "label": x["label"], "generator": x["generator"], "content_id": cid.get(x["path"], ""), **{k: x["features"][k] for k in Q}} for x in r])
rng = np.random.default_rng(0); out = {}; PROBES = {"SD1.5 classB30": "results/content_matched/v1_humancaption.json", "DiT classB30": "results/crossprobe/dit_content_matched.json", "CIFAR32 classB30 (VAE-free, 32px)": "results/crossprobe/cifar32_content_matched.json", "church256 classB30 (VAE-free, 256px)": "results/crossprobe/church256_content_matched.json"}
for name, f in PROBES.items():
    if not Path(f).exists(): print("skip", name); continue
    d = load(f)
    if len(d) < 120: print("skip incomplete", name, len(d)); continue
    d = d[(d.label == 0) | (d.generator == gen)]; ok = d.groupby("content_id").label.nunique(); d = d[d.content_id.isin(ok[ok == 2].index)].reset_index(drop=True); ids = np.array(sorted(d.content_id.unique())); X = np.nan_to_num(d[Q].values); y = d.label.values; oof = np.zeros(len(d)); a = []
    for r in range(10):
        perm = np.random.default_rng(r).permutation(ids); fo = d.content_id.map({c: i % 5 for i, c in enumerate(perm)}).values; p = np.zeros(len(d))
        for k in range(5): te = fo == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y, p)); oof += p / 10
    by = {c: np.where(d.content_id.values == c)[0] for c in ids}; bs = [roc_auc_score(y[ix], oof[ix]) for ix in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(1000))]
    pw = d.pivot_table(index="content_id", columns="label", values="eps_mean"); df = (pw[1] - pw[0]).dropna()
    out[name] = {"auroc": float(np.mean(a)), "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])], "n_pairs": len(ids), "eps_mean_frac_fake_lower": float((df < 0).mean())}
json.dump(out, open(f"results/content_matched/analysis/matched_crossprobe_{gen}.json", "w"), indent=1); print(json.dumps(out, indent=1))
