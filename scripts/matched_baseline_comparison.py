"""Phase 4D (exploratory, decision-relevant): per-generator head-to-head of SD1.5 trajectory vs simple/independent
baselines on the SAME content-matched pairs.  Paired content bootstrap for every AUROC (not independent CIs).
Frozen after design (uses only already-extracted files); not re-tuned after seeing aMUSEd numbers."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import make, feature_cols, cols_for
from src.analysis.families import COARSE
OUT = Path("results/content_matched/analysis"); OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id))
sd = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": cid.get(r["path"], ""), **r["features"]} for r in json.load(open("results/content_matched/v1_humancaption.json"))])
vae = pd.read_csv("results/content_matched/analysis/vae_recon_errors.csv")[["path", "mse512", "mae512"]]
cif = pd.DataFrame([{"path": r["path"], **{f"cifar_{k}": v for k, v in r["features"].items()}} for r in json.load(open("results/crossprobe/cifar32_content_matched.json"))])
thumb_path = Path("results/content_matched/analysis/thumbnail_features.csv")
d = sd.merge(vae, on="path").merge(cif, on="path"); allf = feature_cols(d); G = cols_for(d, COARSE["guidance"])
rng = np.random.default_rng(0)
def sub(gen): x = d[(d.label == 0) | (d.generator == gen)]; ok = x.groupby("content_id").label.nunique(); return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)
def gcv_auc(x, cols, reps=10):
    X = np.nan_to_num(x[cols].values); y = x.label.values; ids = np.array(sorted(x.content_id.unique())); oof = np.zeros(len(x)); a = []
    for r in range(reps):
        perm = np.random.default_rng(r).permutation(ids); fo = x.content_id.map({c: i % 5 for i, c in enumerate(perm)}).values; p = np.zeros(len(x))
        for k in range(5): te = fo == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y, p)); oof += p / reps
    return float(np.mean(a)), oof
def scalar_auc(x, col, lower_is_fake=True):
    s = -x[col].values if lower_is_fake else x[col].values; return float(roc_auc_score(x.label.values, s)), s
def boot_ci(x, score, n=1000):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}; y = x.label.values
    o = [roc_auc_score(y[ix], score[ix]) for ix in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(n))]
    return [float(v) for v in np.percentile(o, [2.5, 97.5])]
out = {}
for gen in ["sd15", "amused"]:
    x = sub(gen); row = {"n_pairs": int(x.content_id.nunique())}
    for name, cols in [("SD1.5_full_trajectory(652)", allf), ("SD1.5_guidance_family(92)", G), ("SD1.5_eps_mean_scalar", None), ("CIFAR32_classB(30)", [c for c in x.columns if c.startswith("cifar_")]), ("VAE_mse512", None)]:
        if name == "SD1.5_eps_mean_scalar": a, s = scalar_auc(x, "eps_mean", lower_is_fake=True)
        elif name == "VAE_mse512": a, s = scalar_auc(x, "mse512", lower_is_fake=True)
        else: a, s = gcv_auc(x, cols)
        row[name] = {"auroc": a, "ci": boot_ci(x, s)}
    out[gen] = row; print(gen, json.dumps(row, indent=1))
json.dump(out, open(OUT / "matched_baseline_comparison.json", "w"), indent=1)
