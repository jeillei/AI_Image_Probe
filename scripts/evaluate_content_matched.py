"""Evaluate the content-matched pilot EXACTLY per PREREGISTRATION_content_matched_v1.md (grouped CV, content bootstrap, sign test, transfer)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import *
from src.analysis.families import COARSE, family
OUT = Path("results/content_matched/analysis"); OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open("results/content_matched/v1_humancaption.json")); m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id))
d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": cid[r["path"]], "caption": r["caption"], **r["features"]} for r in rows])
cnt = d.groupby("content_id").label.nunique(); keep = cnt[cnt == 2].index; d = d[d.content_id.isin(keep)].reset_index(drop=True); ids = np.array(sorted(d.content_id.unique())); print("complete pairs:", len(ids), "images:", len(d))
allf = feature_cols(d); G = cols_for(d, COARSE["guidance"]); rng = np.random.default_rng(0)
def grouped_cv(cols, reps=10, folds=5):
    X = np.nan_to_num(d[cols].values); y = d.label.values; oof = np.zeros(len(d)); aucs = []
    for r in range(reps):
        perm = np.random.default_rng(r).permutation(ids); fold_of = {c: i % folds for i, c in enumerate(perm)}; f = d.content_id.map(fold_of).values; p = np.zeros(len(d))
        for k in range(folds):
            te = f == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p)); oof += p / reps
    return np.mean(aucs), oof
def content_boot(oof, n=1000):
    y = d.label.values; by = {c: np.where(d.content_id.values == c)[0] for c in ids}; out = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(ids, len(ids))]); out.append(roc_auc_score(y[idx], oof[idx]))
    return tuple(np.percentile(out, [2.5, 97.5]))
res = {}
for name, cols in [("ALL_v1", allf), ("guidance", G), ("legacy51", cols_for(d, COARSE["legacy51"]))] + [(k, cols_for(d, v)) for k, v in COARSE.items() if k not in ("legacy51", "guidance")]:
    a, oof = grouped_cv(cols); lo, hi = content_boot(oof); res[name] = {"auroc": float(a), "ci_lo": lo, "ci_hi": hi, "n_features": len(cols)}
    if name in ("ALL_v1", "guidance"): res[name]["decision"] = "SUPPORTS(ci_lo>0.55)" if lo > .55 else ("DISCONFIRMS(ci includes 0.5)" if lo <= .5 else "INCONCLUSIVE(0.5<ci_lo<=0.55)")
# H_eps: paired sign test on eps_mean (fake - real)
pw = d.pivot_table(index="content_id", columns="label", values="eps_mean"); diff = (pw[1] - pw[0]).dropna(); neg = int((diff < 0).sum()); bt = binomtest(neg, len(diff), .5)
res["H_eps"] = {"n_pairs": len(diff), "n_fake_lower": neg, "frac_fake_lower": neg / len(diff), "sign_test_p_two_sided": bt.pvalue, "mean_diff": float(diff.mean())}
# H_transfer: classifier fit on the confounded 550-image corpus
base = load_features("results/features/stage_b_cross_source_rich.json"); mm = make(.1).fit(np.nan_to_num(base[allf].values), base.label.values); p = mm.predict_proba(np.nan_to_num(d[allf].values))[:, 1]
res["H_transfer(confounded-corpus classifier -> matched set)"] = {"auroc": float(roc_auc_score(d.label, p)), "ci": [float(x) for x in content_boot(p)], "mean_score_real": float(p[d.label == 0].mean()), "mean_score_fake": float(p[d.label == 1].mean())}
# exploratory: paired effect sizes per family (fake - real)
pe = []
for c in allf:
    q = d.pivot_table(index="content_id", columns="label", values=c); df_ = (q[1] - q[0]).dropna(); pe.append({"feature": c, "family": family(c), "paired_d": float(df_.mean() / (df_.std() + 1e-12)), "frac_fake_higher": float((df_ > 0).mean())})
pe = pd.DataFrame(pe); pe["abs_d"] = pe.paired_d.abs(); pe.sort_values("abs_d", ascending=False).to_csv(OUT / "paired_effects.csv", index=False); pe.groupby("family").abs_d.median().sort_values(ascending=False).to_csv(OUT / "paired_effect_by_family.csv")
json.dump(res, open(OUT / "preregistered_results.json", "w"), indent=1); print(json.dumps(res, indent=1)[:3500]); print(pe.groupby("family").abs_d.median().sort_values(ascending=False).round(2).to_string())
