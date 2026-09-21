"""E60-E63: real-only source audit + acquisition-metadata shortcut baselines (no GPU; uses cached v1 features)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.evalkit import *
from src.analysis.families import COARSE, family
from src.analysis.metadata import META_COLS
OUT = Path("results/audit"); OUT.mkdir(parents=True, exist_ok=True)
d = load_features("results/features/stage_b_cross_source_rich.json")
real = d[d.label == 0].reset_index(drop=True); y = (real.source == "rrdataset_real").astype(int).values
allf = feature_cols(d); fams = sorted({family(c) for c in allf})
rows = []
def run(name, cols, kind):
    for c in (0.01, 0.1):
        a, _ = cv_binary(real[cols].values, y, c); rows.append({"feature_set": name, "kind": kind, "n_features": len(cols), "C": c, "auroc_mean": a.mean(), "auroc_sd_over_reps": a.std()})
run("ALL_v1", allf, "all"); run("legacy51", cols_for(d, COARSE["legacy51"]), "coarse")
run("rich608", [c for c in allf if c.startswith("rich_")], "coarse")
for k, v in COARSE.items(): run(k, cols_for(d, v), "coarse")
for f in fams: run(f, cols_for(d, [f]), "family")
run("metadata_only", META_COLS, "metadata")
run("metadata_no_qtable", [c for c in META_COLS if c != "qtable_mean"], "metadata")
run("metadata_geometry_only(min_side,aspect,is_square,is_pow2)", ["min_side", "aspect", "is_square", "is_pow2_side"], "metadata")
res = pd.DataFrame(rows); res.to_csv(OUT / "real_source_auroc_by_family.csv", index=False)
# CI for headline sets (pooled OOF bootstrap at C=0.1)
ci = []
for name, cols in [("ALL_v1", allf), ("legacy51", cols_for(d, COARSE["legacy51"])), ("rich608", [c for c in allf if c.startswith("rich_")]), ("metadata_only", META_COLS)]:
    a, oof = cv_binary(real[cols].values, y, .1); lo, hi = boot_auc(y, oof); ci.append({"feature_set": name, "auroc_oof": roc_auc_score(y, oof), "ci_lo": lo, "ci_hi": hi})
pd.DataFrame(ci).to_csv(OUT / "real_source_headline_ci.csv", index=False)
# permutation null (label shuffle) for ALL_v1
rng = np.random.default_rng(0); null = [cv_binary(real[allf].values, rng.permutation(y), .1, reps=1)[0][0] for _ in range(30)]
json.dump({"perm_null_mean": float(np.mean(null)), "perm_null_max": float(np.max(null)), "n_perm": 30}, open(OUT / "real_source_perm_null.json", "w"))
# effect sizes
def cohen(a, b): s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12; return (a.mean() - b.mean()) / s
eff = pd.DataFrame({"feature": allf, "family": [family(c) for c in allf], "d_rr_minus_aigc": [cohen(real.loc[y == 1, c].values, real.loc[y == 0, c].values) for c in allf]})
eff["abs_d"] = eff.d_rr_minus_aigc.abs(); eff.sort_values("abs_d", ascending=False).to_csv(OUT / "real_source_effect_sizes.csv", index=False)
fe = eff.groupby("family").abs_d.agg(median_abs_d="median", max_abs_d="max", frac_gt_0p5=lambda x: float((x > .5).mean()), n="size").sort_values("median_abs_d", ascending=False)
fe.to_csv(OUT / "real_source_effect_by_family.csv")
# pseudo-source multiclass by native size bucket
def bucket(r):
    if r.source == "rrdataset_real": return "rr_real"
    return "aigc_256" if r.min_side == 256 else ("aigc_257-599" if r.min_side < 600 else "aigc_600+")
real["bucket"] = real.apply(bucket, axis=1); print(real.bucket.value_counts().to_dict())
mc = []
for name, cols in [("ALL_v1", allf), ("legacy51", cols_for(d, COARSE["legacy51"])), ("metadata_only", META_COLS), ("roundtrip", cols_for(d, ["roundtrip"]))]:
    b, conf, classes = cv_multiclass(real[cols].values, real.bucket.values); mc.append({"feature_set": name, "balanced_acc": b.mean(), "chance": 1 / len(classes)})
    if name == "ALL_v1":
        pd.DataFrame(conf, index=classes, columns=classes).to_csv(OUT / "real_bucket_confusion_ALL_v1.csv")
        fig, ax = plt.subplots(figsize=(4.6, 4)); im = ax.imshow(conf, vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=35, ha="right"); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)): ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center", color="w" if conf[i, j] < .6 else "k")
        ax.set_title("Real-only source/size-bucket confusion (SD1.5 v1, logistic)"); ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "real_bucket_confusion.png", dpi=140); plt.close(fig)
pd.DataFrame(mc).to_csv(OUT / "real_bucket_multiclass.csv", index=False)
# does trajectory predict native resolution *within the AIGC-real source*? (resize-history leakage)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold; from scipy.stats import spearmanr
a = real[real.source != "rrdataset_real"].reset_index(drop=True); t = np.log(a.min_side.values); X = np.nan_to_num(a[allf].values)
def ridge_oof(X):
    p = np.zeros(len(t))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X): 
        m = make_pipeline(StandardScaler(), Ridge(alpha=3000)).fit(X[tr], t[tr]); p[te] = m.predict(X[te])
    return spearmanr(p, t).statistic
json.dump({"spearman_oof_logMinSide_within_aigc_real": ridge_oof(X), "n": len(a)}, open(OUT / "real_resolution_regression.json", "w"))
# figure: AUROC by family
f = res[(res.C == .1) & res.kind.isin(["family", "coarse", "metadata", "all"])].sort_values("auroc_mean")
fig, ax = plt.subplots(figsize=(7, 7)); colr = f.kind.map({"family": "#4477aa", "coarse": "#66ccee", "metadata": "#ee6677", "all": "#222222"}); ax.barh(f.feature_set, f.auroc_mean, color=colr); ax.axvline(.5, color="k", ls="--", lw=.8)
ax.set_xlabel("AUROC: AIGC-benchmark real vs RR real (5x5-fold CV, C=0.1)"); ax.set_title("Where does real-source identity live?"); fig.tight_layout(); fig.savefig(OUT / "real_source_auroc_by_family.png", dpi=140)
print(res[res.C == .1].sort_values("auroc_mean", ascending=False).round(3).to_string(index=False)); print(pd.DataFrame(ci).round(3)); print(pd.DataFrame(mc).round(3)); print(fe.round(2)); print(open(OUT / "real_source_perm_null.json").read(), open(OUT / "real_resolution_regression.json").read())
