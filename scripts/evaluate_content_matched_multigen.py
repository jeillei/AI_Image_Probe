"""Addendum-1 evaluation: per-generator H_primary / H_eps and cross-generator transfer (train real+SD1.5 -> test real+aMUSEd, and reverse).
Estimator and decision rules identical to PREREGISTRATION_content_matched_v1.md.  Exploratory extras are labelled."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import *
from src.analysis.families import COARSE
OUT = Path("results/content_matched/analysis"); OUT.mkdir(parents=True, exist_ok=True)
src = sys.argv[1] if len(sys.argv) > 1 else "results/content_matched/v1_humancaption.json"; tag = sys.argv[2] if len(sys.argv) > 2 else "multigen"
rows = json.load(open(src)); m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id))
d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": cid.get(r["path"], ""), **r["features"]} for r in rows]); d = d[d.content_id != ""]
gens = sorted(d[d.label == 1].generator.unique()); allf = feature_cols(d); G = cols_for(d, COARSE["guidance"]); rng = np.random.default_rng(0)
def sub(g): 
    x = d[(d.label == 0) | (d.generator == g)]; ok = x.groupby("content_id").label.nunique(); return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)
def folds(ids, r, k=5): perm = np.random.default_rng(r).permutation(ids); return {c: i % k for i, c in enumerate(perm)}
def boot(x, score, n=1000):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}; y = x.label.values; o = []
    for _ in range(n): ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids))]); o.append(roc_auc_score(y[ix], score[ix]))
    return [float(v) for v in np.percentile(o, [2.5, 97.5])]
def rule(lo): return "SUPPORTS(ci_lo>0.55)" if lo > .55 else ("DISCONFIRMS(ci includes 0.5)" if lo <= .5 else "INCONCLUSIVE(0.5<ci_lo<=0.55)")
def gcv(x, cols, reps=10):
    X = np.nan_to_num(x[cols].values); y = x.label.values; ids = np.array(sorted(x.content_id.unique())); oof = np.zeros(len(x)); a = []
    for r in range(reps):
        f = x.content_id.map(folds(ids, r)).values; p = np.zeros(len(x))
        for k in range(5): te = f == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y, p)); oof += p / reps
    return float(np.mean(a)), oof
res = {}
for g in gens:
    x = sub(g); r = {"n_pairs": int(x.content_id.nunique())}
    for name, cols in [("ALL_v1", allf), ("guidance", G), ("legacy51", cols_for(d, COARSE["legacy51"]))]:
        a, oof = gcv(x, cols); lo, hi = boot(x, oof); r[name] = {"auroc": a, "ci": [lo, hi], **({"decision": rule(lo)} if name in ("ALL_v1", "guidance") else {})}
    pw = x.pivot_table(index="content_id", columns="label", values="eps_mean"); df = (pw[1] - pw[0]).dropna(); neg = int((df < 0).sum()); r["H_eps"] = {"n_fake_lower": neg, "n": len(df), "p": float(binomtest(neg, len(df), .5).pvalue), "mean_diff": float(df.mean())}
    r["exploratory_family_auroc"] = {k: gcv(x, cols_for(d, v), reps=3)[0] for k, v in COARSE.items()}; res[g] = r
# cross-generator transfer
def transfer(gtr, gte, reps=10):
    both = d[(d.label == 0) | d.generator.isin([gtr, gte])]; ids = np.array(sorted(both.content_id.unique())); X = np.nan_to_num(both[allf].values); oofs = []; aucs = []
    te_mask = ((both.label == 0) | (both.generator == gte)).values; tr_mask = ((both.label == 0) | (both.generator == gtr)).values; y = both.label.values; oof = np.zeros(len(both))
    for r in range(reps):
        f = both.content_id.map(folds(ids, r)).values; p = np.full(len(both), np.nan)
        for k in range(5):
            tr = tr_mask & (f != k); te = te_mask & (f == k); p[te] = make(.1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te_mask], p[te_mask])); oof += np.nan_to_num(p) / reps
    x = both[te_mask].reset_index(drop=True); s = oof[te_mask]; lo, hi = boot(x, s); return {"auroc": float(np.mean(aucs)), "ci": [lo, hi], "decision": rule(lo)}
if len(gens) >= 2 and {"sd15", "amused"} <= set(gens): res["H_gen-transfer sd15->amused"] = transfer("sd15", "amused"); res["H_gen-transfer amused->sd15"] = transfer("amused", "sd15")
json.dump(res, open(OUT / f"preregistered_results_{tag}.json", "w"), indent=1); print(json.dumps({k: (v if not isinstance(v, dict) or "exploratory_family_auroc" not in v else {kk: vv for kk, vv in v.items() if kk != "exploratory_family_auroc"}) for k, v in res.items()}, indent=1))
for g in gens: print(g, {k: round(v, 3) for k, v in res[g]["exploratory_family_auroc"].items()})
