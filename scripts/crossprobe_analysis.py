"""E73-E77: cross-probe analysis of Class-B quantities (SD1.5 cached legacy columns vs a second probe).
A directional replication (+ joint-label permutation null), B agreement residual, C generator-probe affinity, D weight-vector transfer."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import load_features, make, boot_auc, _prep
from src.probes.base import CROSS_PROBE_COLUMNS as Q
ap = argparse.ArgumentParser(); ap.add_argument("--other", default="results/crossprobe/cifar32.json"); ap.add_argument("--name", default="cifar32"); ap.add_argument("--out", default="results/crossprobe/analysis"); a = ap.parse_args()
out = Path(a.out) / a.name; out.mkdir(parents=True, exist_ok=True)
sd = load_features("results/features/stage_b_cross_source_rich.json"); ot = pd.DataFrame([{"path": r["path"], **{f"{k}": v for k, v in r["features"].items()}} for r in json.load(open(a.other))])
d = sd[["path", "label", "generator", "source", "min_side", "aspect", "is_square", "is_pow2_side", "bpp", *Q]].merge(ot[["path", *Q]], on="path", suffixes=("_sd", "_ot")); print("n merged", len(d))
S = [q + "_sd" for q in Q]; O = [q + "_ot" for q in Q]; gens = sorted(d[d.label == 1].generator.unique()); srcs = sorted(d[d.label == 0].source.unique())
def z(df, cols, ref): m = ref[cols].mean(); s = ref[cols].std() + 1e-9; return ((df[cols] - m) / s).values
def cohen(df, cols):
    f = df[df.label == 1][cols]; r = df[df.label == 0][cols]; sp = np.sqrt((f.var() + r.var()) / 2) + 1e-12; return ((f.mean() - r.mean()) / sp).values
res = {}
# ---- A. directional replication ------------------------------------------------------------------
rng = np.random.default_rng(0); A = []
for rs in srcs + ["ALL_REAL"]:
    sub = d[(d.label == 1) | ((d.label == 0) & ((d.source == rs) if rs != "ALL_REAL" else True))].reset_index(drop=True)
    e1, e2 = cohen(sub, S), cohen(sub, O); obs = spearmanr(e1, e2).statistic; sign = float(np.mean(np.sign(e1) == np.sign(e2)))
    null = []; y = sub.label.values.copy()
    for _ in range(500):
        sub2 = sub.copy(); sub2["label"] = rng.permutation(y); null.append(spearmanr(cohen(sub2, S), cohen(sub2, O)).statistic)
    A.append({"real_reference": rs, "spearman_effect_vectors": obs, "sign_agreement": sign, "perm_p_two_sided": float(np.mean(np.abs(null) >= abs(obs))), "null_sd": float(np.std(null))})
# source-shift replication: is the AIGC-real vs RR-real shift the same in both probes?
rr = d[d.label == 0].copy(); rr["label"] = (rr.source == "rrdataset_real").astype(int); e1, e2 = cohen(rr, S), cohen(rr, O)
A.append({"real_reference": "SOURCE_SHIFT(aigc_real->rr_real)", "spearman_effect_vectors": spearmanr(e1, e2).statistic, "sign_agreement": float(np.mean(np.sign(e1) == np.sign(e2))), "perm_p_two_sided": np.nan, "null_sd": np.nan})
pd.DataFrame(A).to_csv(out / "A_directional_replication.csv", index=False); print(pd.DataFrame(A).round(3))
# ---- B & D crossed folds -----------------------------------------------------------------------------
rows = []
for rs in srcs:
    for g in gens:
        te = d[(d.generator == g) | ((d.label == 0) & (d.source == rs))]; tr = d.drop(te.index); y = te.label.values
        row = {"held_real": rs, "held_generator": g}
        ms = make(.1).fit(_prep(tr[S]), tr.label); mo = make(.1).fit(_prep(tr[O]), tr.label)
        row["auroc_own_sd30"] = roc_auc_score(y, ms.predict_proba(_prep(te[S]))[:, 1]); row["auroc_own_other30"] = roc_auc_score(y, mo.predict_proba(_prep(te[O]))[:, 1])
        # D. transfer: weights learned on probe P applied to probe Q standardized with Q's *training* stats (features analogous by construction)
        def transfer(src_cols, dst_cols):
            m = make(.1).fit(_prep(tr[src_cols]), tr.label); w = m[-1].coef_[0]; b = m[-1].intercept_[0]; zt = z(te, dst_cols, tr); return roc_auc_score(y, np.nan_to_num(zt) @ w + b)
        row["D_transfer_sd->other"] = transfer(S, O); row["D_transfer_other->sd"] = transfer(O, S)
        # B. agreement residual: ridge other->sd on TRAIN REALS only
        trr = tr[tr.label == 0]; rg = Ridge(alpha=10).fit(z(trr, O, trr), z(trr, S, trr)); res_tr = z(trr, S, trr) - rg.predict(z(trr, O, trr)); sdv = res_tr.std(0) + 1e-9
        rt = (z(te, S, trr) - rg.predict(z(te, O, trr))) / sdv; score = np.abs(rt).mean(1); row["B_agreement_residual_auroc(fake higher)"] = roc_auc_score(y, score); rows.append(row)
R = pd.DataFrame(rows); R.to_csv(out / "BD_crossed_cells.csv", index=False)
sm = R.drop(columns=["held_real", "held_generator"]).agg(["mean", "median", "min"]).round(3); sm.to_csv(out / "BD_summary.csv"); print(sm.to_string())
for c in R.columns[2:]:
    piv = R.pivot(index="held_real", columns="held_generator", values=c); fig, ax = plt.subplots(figsize=(9, 2.4)); im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=40, ha="right"); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{a.name}: {c}"); fig.colorbar(im); fig.tight_layout(); fig.savefig(out / f"{c.replace('>','_').replace('(','_').replace(')','')}.png", dpi=120); plt.close(fig)
# ---- C. affinity matrix ------------------------------------------------------------------------------
real = d[d.label == 0]; scal = ["pixel_roundtrip_mse", "eps_mean", "endpoint_norm_per_dim", "path_length", "endpoint_kurtosis"]; aff = {}
for probe, sfx in [("SD1.5", "_sd"), (a.name, "_ot")]:
    for s in scal:
        col = s + sfx; aff[(probe, s)] = {src: float((d[d.source == src][col].mean() - real[col].mean()) / (real[col].std() + 1e-9)) for src in srcs + gens}
M = pd.DataFrame(aff); M.to_csv(out / "C_affinity_matrix_zscore_vs_real.csv"); print(M.round(2).to_string())
fig, ax = plt.subplots(figsize=(1 + .55 * M.shape[1], 4.5)); im = ax.imshow(M.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{p}\n{s}" for p, s in M.columns], rotation=60, ha="right", fontsize=7); ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
fig.colorbar(im, label="mean z-score vs pooled real (probe-normalized)"); ax.set_title("Source x probe compatibility"); fig.tight_layout(); fig.savefig(out / "C_affinity_matrix.png", dpi=130)
# ---- real-source separability by probe (does the low-res control probe also encode source identity?) ---------------
from src.analysis.evalkit import cv_binary
rr_ = d[d.label == 0].reset_index(drop=True); ys = (rr_.source == "rrdataset_real").astype(int).values; rs_out = {}
for nm, cols in [("SD1.5_classB30", S), (f"{a.name}_classB30", O)]:
    aa, oof = cv_binary(_prep(rr_[cols]), ys, .1, reps=5); rs_out[nm] = {"auroc": float(aa.mean()), "ci": [float(x) for x in boot_auc(ys, oof, 500)]}
json.dump(rs_out, open(out / "real_source_by_probe.json", "w"), indent=1); print(rs_out)
