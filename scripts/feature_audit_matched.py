"""Feature-level audit of the frozen 652-column v1 representation on the content-matched sets
(real vs SD1.5, real vs aMUSEd).  No new features, no new model family: the same
StandardScaler + LogisticRegression(C=0.1, class_weight=balanced) already used throughout the
project.  Produces coefficient tables, bootstrap stability, univariate effect sizes, cross-task
comparison, a "clean-residual" comparison restricted to families with near-chance real-source-identity
AUROC, and a top-k truncation curve (distributed-vs-concentrated diagnostic for the aMUSEd result)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, binomtest
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.analysis.evalkit import make, feature_cols, cols_for
from src.analysis.families import family, COARSE

OUT = Path("results/feature_audit"); OUT.mkdir(parents=True, exist_ok=True)
RNG_SEED = 0
N_BOOT = 500

# ---------- data ----------
m = pd.read_csv("data/content_matched/manifest.csv"); cid = dict(zip(m.path, m.content_id))
rows = json.load(open("results/content_matched/v1_humancaption.json"))
d_full = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"],
                        "content_id": cid.get(r["path"], ""), **r["features"]} for r in rows])
d_full = d_full[d_full.content_id != ""]
ALLF = feature_cols(d_full)
assert len(ALLF) == 652, len(ALLF)

def sub(gen: str) -> pd.DataFrame:
    x = d_full[(d_full.label == 0) | (d_full.generator == gen)]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].sort_values(["content_id", "label"]).reset_index(drop=True)

TASKS = {"sd15": sub("sd15"), "amused": sub("amused")}
for g, x in TASKS.items(): assert x.content_id.nunique() == 60 and len(x) == 120, (g, len(x))

# ---------- 1. feature list / families ----------
fam_table = pd.DataFrame({"feature": ALLF, "family": [family(c) for c in ALLF]})
coarse_of = {}
for k, v in COARSE.items():
    for fm in v: coarse_of.setdefault(fm, []).append(k)
fam_table["coarse_groups"] = fam_table.family.map(lambda f: ",".join(coarse_of.get(f, [f])))
fam_table.to_csv(OUT / "feature_list_by_family.csv", index=False)
fam_counts = fam_table.family.value_counts()
fam_counts.to_csv(OUT / "family_counts.csv")

# ---------- helpers ----------
def fit_coefs(x: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.nan_to_num(x[cols].values); y = x.label.values
    pipe = make(0.1).fit(X, y)
    return pipe.named_steps["logisticregression"].coef_[0]

def univariate(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = x.label.values.astype(bool); out = []
    piv = x.pivot_table(index="content_id", columns="label", values=cols)
    for c in cols:
        v = np.nan_to_num(x[c].values.astype(float))
        auc = roc_auc_score(y, v)
        auc_dir = max(auc, 1 - auc)  # direction-free separability
        pair = piv[c] if c in piv.columns.get_level_values(0) else None
        diff = (piv[(c, 1)] - piv[(c, 0)]).dropna() if (c, 1) in piv.columns and (c, 0) in piv.columns else pd.Series(dtype=float)
        paired_d = float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if len(diff) > 1 else np.nan
        frac_fake_higher = float((diff > 0).mean()) if len(diff) else np.nan
        out.append({"feature": c, "univariate_auroc": float(auc), "univariate_auroc_direction_free": float(auc_dir),
                    "paired_cohens_d": paired_d, "frac_pairs_fake_higher": frac_fake_higher})
    return pd.DataFrame(out).set_index("feature")

def bootstrap_stability(x: pd.DataFrame, cols: list[str], full_coef: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> pd.DataFrame:
    ids = np.array(sorted(x.content_id.unique())); rng = np.random.default_rng(seed)
    sign0 = np.sign(full_coef)
    same_sign = np.zeros(len(cols)); mags = np.zeros((n_boot, len(cols)))
    for b in range(n_boot):
        samp = rng.choice(ids, len(ids), replace=True)
        xb = pd.concat([x[x.content_id == c] for c in samp], ignore_index=True)
        cb = fit_coefs(xb, cols)
        same_sign += (np.sign(cb) == sign0)
        mags[b] = cb
    sign_stability = same_sign / n_boot
    mag_std = mags.std(0); mag_iqr = np.percentile(mags, 75, axis=0) - np.percentile(mags, 25, axis=0)
    ci_lo, ci_hi = np.percentile(mags, [2.5, 97.5], axis=0)
    return pd.DataFrame({"feature": cols, "sign_stability": sign_stability, "boot_coef_std": mag_std,
                         "boot_coef_iqr": mag_iqr, "boot_ci_lo": ci_lo, "boot_ci_hi": ci_hi}).set_index("feature")

# ---------- 2. per-task coefficient / univariate / bootstrap tables ----------
task_tables = {}
for gen, x in TASKS.items():
    coef = fit_coefs(x, ALLF)
    uni = univariate(x, ALLF)
    boot = bootstrap_stability(x, ALLF, coef)
    t = pd.DataFrame({"feature": ALLF, "family": [family(c) for c in ALLF], "coef": coef}).set_index("feature")
    t = t.join(uni).join(boot)
    t["abs_coef"] = t.coef.abs()
    t["abs_coef_rank"] = t.abs_coef.rank(ascending=False, method="min").astype(int)
    t["sign"] = np.sign(t.coef).astype(int)
    t = t.sort_values("abs_coef_rank")
    t.to_csv(OUT / f"coef_table_{gen}.csv")
    task_tables[gen] = t
    print(gen, "top 5 by |coef|:")
    print(t.head(5)[["family", "coef", "univariate_auroc_direction_free", "sign_stability"]].to_string())

# ---------- 3. cross-task comparison ----------
# task_tables are each sorted by their own abs_coef_rank; reindex to a common column order (ALLF)
# before any elementwise comparison so pandas doesn't complain about (and silently we don't get) misaligned rows.
c1 = task_tables["sd15"].coef.reindex(ALLF); c2 = task_tables["amused"].coef.reindex(ALLF)
pear = pearsonr(c1.values, c2.values); spear = spearmanr(c1.values, c2.values)
sign_agree = float((np.sign(c1.values) == np.sign(c2.values)).mean())
# permutation null for sign agreement / correlation (label-shuffle refit is expensive x2x652; use coefficient-vector shuffle null instead, documented as such)
rng = np.random.default_rng(1)
null_sign = []; null_pear = []
for _ in range(1000):
    perm = rng.permutation(len(c2))
    null_sign.append(float((np.sign(c1) == np.sign(c2.values[perm])).mean()))
    null_pear.append(pearsonr(c1, c2.values[perm])[0])
overlaps = {}
for k in (10, 25, 50):
    top1 = set(task_tables["sd15"].nsmallest(k, "abs_coef_rank").index)
    top2 = set(task_tables["amused"].nsmallest(k, "abs_coef_rank").index)
    overlaps[f"top{k}_overlap_n"] = len(top1 & top2)
    overlaps[f"top{k}_overlap_frac"] = len(top1 & top2) / k
    overlaps[f"top{k}_shared_same_sign"] = int(sum(1 for f in (top1 & top2) if task_tables["sd15"].loc[f, "sign"] == task_tables["amused"].loc[f, "sign"]))

cross = pd.DataFrame({"sd15_coef": c1, "amused_coef": c2, "sd15_abs_rank": task_tables["sd15"].abs_coef_rank.reindex(ALLF),
                      "amused_abs_rank": task_tables["amused"].abs_coef_rank.reindex(ALLF), "family": task_tables["sd15"].family.reindex(ALLF),
                      "sd15_sign": task_tables["sd15"].sign.reindex(ALLF), "amused_sign": task_tables["amused"].sign.reindex(ALLF)})
cross["sign_agree"] = cross.sd15_sign.values == cross.amused_sign.values
cross["min_abs_rank"] = cross[["sd15_abs_rank", "amused_abs_rank"]].min(1)
cross.sort_values("min_abs_rank").to_csv(OUT / "cross_task_coefficients.csv")

# strong features that reverse: both in top-100 by |coef| in their own task, opposite sign
strong_both = cross[(cross.sd15_abs_rank <= 100) & (cross.amused_abs_rank <= 100)]
reversed_strong = strong_both[~strong_both.sign_agree].sort_values("min_abs_rank")
shared_strong = strong_both[strong_both.sign_agree].sort_values("min_abs_rank")
reversed_strong.to_csv(OUT / "reversed_strong_features.csv")
shared_strong.to_csv(OUT / "shared_strong_features.csv")

summary = {"pearson_r": float(pear[0]), "pearson_p": float(pear[1]), "spearman_r": float(spear[0]), "spearman_p": float(spear[1]),
           "sign_agreement": sign_agree, "sign_agreement_null_mean": float(np.mean(null_sign)), "sign_agreement_null_sd": float(np.std(null_sign)),
           "pearson_null_mean": float(np.mean(null_pear)), "pearson_null_sd": float(np.std(null_pear)),
           **overlaps, "n_strong_both_top100": len(strong_both), "n_reversed_among_strong_both": len(reversed_strong),
           "n_shared_same_sign_among_strong_both": len(shared_strong)}
json.dump(summary, open(OUT / "cross_task_summary.json", "w"), indent=1)
print("CROSS-TASK SUMMARY:", json.dumps(summary, indent=1))

# ---------- 4. family attribution of top features ----------
fam_attr = {}
for gen in TASKS:
    t = task_tables[gen]
    for k in (10, 30, 50):
        fam_attr[(gen, k)] = t.nsmallest(k, "abs_coef_rank").family.value_counts().to_dict()
json.dump({f"{g}_top{k}": v for (g, k), v in fam_attr.items()}, open(OUT / "family_attribution_top_features.json", "w"), indent=1)

# ---------- 5. clean-residual comparison (exclude families with real-source AUROC > 0.60) ----------
rs = pd.read_csv("results/audit/real_source_auroc_by_family.csv")
rs_fam = rs[(rs.kind == "family") & (rs.C == 0.1)].set_index("feature_set").auroc_mean
CONFOUND_THRESH = 0.60
clean_families = sorted(rs_fam[rs_fam <= CONFOUND_THRESH].index)  # near-chance for real-source identity
CLEANF = [c for c in ALLF if family(c) in clean_families]
print("clean-residual families (real-source AUROC <=", CONFOUND_THRESH, "):", clean_families, "->", len(CLEANF), "columns")
clean_tables = {}
for gen, x in TASKS.items():
    coef = fit_coefs(x, CLEANF); uni = univariate(x, CLEANF); boot = bootstrap_stability(x, CLEANF, coef, n_boot=300)
    t = pd.DataFrame({"feature": CLEANF, "family": [family(c) for c in CLEANF], "coef": coef}).set_index("feature").join(uni).join(boot)
    t["abs_coef"] = t.coef.abs(); t["abs_coef_rank"] = t.abs_coef.rank(ascending=False, method="min").astype(int); t["sign"] = np.sign(t.coef).astype(int)
    t = t.sort_values("abs_coef_rank"); t.to_csv(OUT / f"coef_table_clean_{gen}.csv"); clean_tables[gen] = t
cc1 = clean_tables["sd15"].coef.reindex(CLEANF); cc2 = clean_tables["amused"].coef.reindex(CLEANF)
clean_summary = {"n_features": len(CLEANF), "families": clean_families,
                 "pearson_r": float(pearsonr(cc1.values, cc2.values)[0]), "spearman_r": float(spearmanr(cc1.values, cc2.values)[0]),
                 "sign_agreement": float((np.sign(cc1.values) == np.sign(cc2.values)).mean())}
for k in (10, 25):
    t1 = set(clean_tables["sd15"].nsmallest(min(k, len(CLEANF)), "abs_coef_rank").index)
    t2 = set(clean_tables["amused"].nsmallest(min(k, len(CLEANF)), "abs_coef_rank").index)
    clean_summary[f"top{k}_overlap_n"] = len(t1 & t2)
json.dump(clean_summary, open(OUT / "clean_residual_cross_task_summary.json", "w"), indent=1)
print("CLEAN-RESIDUAL SUMMARY:", json.dumps(clean_summary, indent=1))

# ---------- 6. top-k truncation curve (distributed vs concentrated), held-out content-grouped CV ----------
def grouped_cv_auc(x, cols, reps=10, folds=5, seed=0):
    X = np.nan_to_num(x[cols].values); y = x.label.values; ids = np.array(sorted(x.content_id.unique())); a = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(ids); fo = x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values; p = np.zeros(len(x))
        for k in range(folds): te = fo == k; p[te] = make(.1).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y, p))
    return float(np.mean(a))
K = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 250, 400, 652]
trunc_rows = []
for gen, x in TASKS.items():
    order = task_tables[gen].sort_values("abs_coef_rank").index.tolist()  # ranked on ALL data (in-fold ranking would be more rigorous but this matches "which features the full fit relies on")
    for k in K:
        auc = grouped_cv_auc(x, order[:k])
        trunc_rows.append({"generator": gen, "k": k, "held_out_auroc": auc})
        print(gen, "top-k=", k, "held-out grouped-CV AUROC:", round(auc, 3))
pd.DataFrame(trunc_rows).to_csv(OUT / "topk_truncation_curve.csv", index=False)

# ---------- 7. top-30 tables with cross-reference ----------
top30 = {}
for gen in TASKS:
    other = "amused" if gen == "sd15" else "sd15"
    t = task_tables[gen].nsmallest(30, "abs_coef_rank").copy()
    t["rank_in_other_task"] = task_tables[other].loc[t.index, "abs_coef_rank"]
    t["other_task_coef"] = task_tables[other].loc[t.index, "coef"]
    t["other_task_sign_agree"] = np.sign(t.coef) == np.sign(t.other_task_coef)
    cols = ["family", "coef", "sign", "univariate_auroc_direction_free", "paired_cohens_d", "sign_stability", "rank_in_other_task", "other_task_coef", "other_task_sign_agree"]
    t[cols].to_csv(OUT / f"top30_{gen}.csv")
    top30[gen] = t[cols]
    print(f"\n=== TOP 30 for {gen} ===")
    print(t[cols].round(3).to_string())

print("\nDONE. Outputs in", OUT)
