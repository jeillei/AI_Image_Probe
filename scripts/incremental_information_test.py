"""Phase 5 (exploratory, decision-relevant): does the frozen SynthImage representation add
information beyond generic low-level probes (VAE reconstruction error + CIFAR-32 unconditional probe)?

Baseline model  = VAE(4 scalars: mse256,mae256,mse512,mae512) + CIFAR-32 Class-B (30 features).
Extended models = baseline + {full ALL_v1 (652), guidance family (92), eps-norm curve (legacy_noise, 18),
                               eps_mean alone (1 scalar)}  -- each tested separately, not searched over.
Same content-grouped 5-fold CV (10 repeats, same fold assignment shared by both models in each repeat) so the
AUROC difference per bootstrap draw is a genuine paired comparison, not two independent estimates.
Run per generator (sd15, amused).  Frozen after design; not re-tuned on results."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import make, feature_cols, cols_for
from src.analysis.families import COARSE
from src.probes.base import CROSS_PROBE_COLUMNS as Q

def load_matched(gen: str) -> pd.DataFrame:
    m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id))
    sd = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": cid.get(r["path"], ""),
                        **r["features"]} for r in json.load(open("results/content_matched/v1_humancaption.json"))])
    vae = pd.read_csv("results/content_matched/analysis/vae_recon_errors.csv")[["path", "mse256", "mae256", "mse512", "mae512"]]
    cif = pd.DataFrame([{"path": r["path"], **{f"cifar_{k}": v for k, v in r["features"].items()}} for r in json.load(open("results/crossprobe/cifar32_content_matched.json"))])
    d = sd.merge(vae, on="path").merge(cif, on="path")
    d = d[(d.label == 0) | (d.generator == gen)]; ok = d.groupby("content_id").label.nunique()
    return d[d.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)

def paired_grouped_cv(d, cols_a, cols_b, reps=10, folds=5, seed=0):
    """Same fold assignment for both column sets each repeat -> paired OOF scores."""
    y = d.label.values; ids = np.array(sorted(d.content_id.unique())); Xa = np.nan_to_num(d[cols_a].values); Xb = np.nan_to_num(d[cols_b].values)
    oof_a = np.zeros(len(d)); oof_b = np.zeros(len(d)); aucs_a = []; aucs_b = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(ids); fo = d.content_id.map({c: i % folds for i, c in enumerate(perm)}).values
        pa = np.zeros(len(d)); pb = np.zeros(len(d))
        for k in range(folds):
            te = fo == k
            pa[te] = make(.1).fit(Xa[~te], y[~te]).predict_proba(Xa[te])[:, 1]
            pb[te] = make(.1).fit(Xb[~te], y[~te]).predict_proba(Xb[te])[:, 1]
        aucs_a.append(roc_auc_score(y, pa)); aucs_b.append(roc_auc_score(y, pb)); oof_a += pa / reps; oof_b += pb / reps
    return float(np.mean(aucs_a)), float(np.mean(aucs_b)), oof_a, oof_b

def paired_boot(d, oof_a, oof_b, n=2000, seed=0):
    y = d.label.values; ids = np.array(sorted(d.content_id.unique())); by = {c: np.where(d.content_id.values == c)[0] for c in ids}; rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        diffs.append(roc_auc_score(y[ix], oof_b[ix]) - roc_auc_score(y[ix], oof_a[ix]))
    return float(np.mean(diffs)), [float(v) for v in np.percentile(diffs, [2.5, 97.5])]

def main():
    out = {}
    for gen in ["sd15", "amused"]:
        d = load_matched(gen); allf = feature_cols(d)
        VAE = ["mse256", "mae256", "mse512", "mae512"]; CIF = [c for c in d.columns if c.startswith("cifar_")]
        BASE = VAE + CIF
        EXT = {"+full_trajectory(652)": BASE + allf, "+guidance_family(92)": BASE + cols_for(d, COARSE["guidance"]),
               "+eps_norm_curve(18)": BASE + cols_for(d, ["legacy_noise"]),
               "+eps_mean_scalar(1)": BASE + ["eps_mean"]}
        res = {"n_pairs": int(d.content_id.nunique())}
        for name, cols in EXT.items():
            a, b, oa, ob = paired_grouped_cv(d, BASE, cols); md, ci = paired_boot(d, oa, ob)
            res[name] = {"baseline(VAE+CIFAR)_auroc": a, "extended_auroc": b, "paired_diff_mean": md, "paired_diff_ci": ci,
                         "adds_information": bool(ci[0] > 0)}
        out[gen] = res; print(gen, json.dumps(res, indent=1))
    json.dump(out, open("results/content_matched/analysis/incremental_information_test.json", "w"), indent=1)
main()
