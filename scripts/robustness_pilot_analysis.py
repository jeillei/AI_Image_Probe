"""Phase 6 analysis: for the small robustness pilot (10 content ids x 3 generators x 5 conditions), report per
feature/stage how much effect size and AUROC are retained relative to the SAME 10-content-id clean baseline
(not the full-60 Phase-3 numbers, to keep the before/after comparison apples-to-apples), plus per-image
before/after correlation and incremental-information re-evaluation under each transform. n=10/generator/condition
-- explicitly a small pilot; CIs are wide and are reported, not hidden."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.features.panel_v2 import CORE_FEATURE_NAMES, STAGE_OF, STAGE_ORDER
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_decomposition_analysis import make, grouped_cv_auc, content_boot_ci, cohen_paired

OUT = Path("results/stage_decomposition"); OUT.mkdir(parents=True, exist_ok=True)


def sub(d, gen, cond):
    x = d[(d.condition == cond) & ((d.label == 0) | (d.generator == gen))]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)


def main():
    rows = json.load(open("results/stage_decomposition/panel_features_robustness.json"))
    d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": r["content_id"],
                       "condition": r.get("transform") or "clean", **r["features"]} for r in rows])
    for c in CORE_FEATURE_NAMES: assert d[c].notna().all(), f"NaN in {c}"
    conds = sorted(d.condition.unique()); print("conditions:", conds, "n_content:", d.content_id.nunique())

    retain_rows = []
    for gen in ("sd15", "amused"):
        for cond in conds:
            x = sub(d, gen, cond)
            if len(x) < 6: continue
            for feat in CORE_FEATURE_NAMES:
                dval, _ = cohen_paired(x, feat)
                auc = roc_auc_score(x.label, x[feat]); auc = max(auc, 1 - auc)
                retain_rows.append({"generator": gen, "condition": cond, "feature": feat, "stage": STAGE_OF[feat],
                                    "paired_cohens_d": dval, "univariate_auroc": auc, "n_pairs": x.content_id.nunique()})
    retain = pd.DataFrame(retain_rows)
    clean = retain[retain.condition == "clean"].set_index(["generator", "feature"])[["paired_cohens_d", "univariate_auroc"]]
    clean.columns = ["clean_d", "clean_auroc"]
    retain = retain.join(clean, on=["generator", "feature"])
    retain["d_retained_frac"] = retain.paired_cohens_d / (retain.clean_d.replace(0, np.nan))
    retain["auroc_change"] = retain.univariate_auroc - retain.clean_auroc
    retain.to_csv(OUT / "robustness_pilot_retained_effect.csv", index=False)

    # before/after per-image correlation (clean vs transformed, same image, across BOTH generators+real pooled by content x generator)
    corr_rows = []
    piv = d.pivot_table(index=["generator", "content_id"], columns="condition", values=CORE_FEATURE_NAMES)
    for feat in CORE_FEATURE_NAMES:
        for cond in conds:
            if cond == "clean": continue
            a = piv[(feat, "clean")]; b = piv[(feat, cond)]
            ok = a.notna() & b.notna()
            if ok.sum() < 4: continue
            r, p = spearmanr(a[ok], b[ok])
            corr_rows.append({"feature": feat, "stage": STAGE_OF[feat], "condition": cond, "spearman_r_clean_vs_transformed": float(r), "n": int(ok.sum())})
    pd.DataFrame(corr_rows).to_csv(OUT / "robustness_pilot_before_after_correlation.csv", index=False)

    # incremental info re-evaluated per condition (small n; fewer reps)
    cum = {"vae": [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "vae"]}
    cum["vae+score"] = cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "score"]
    cum["vae+score+trajectory"] = cum["vae+score"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "trajectory"]
    cum["vae+score+trajectory+roundtrip"] = cum["vae+score+trajectory"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "roundtrip"]
    incr_rows = []
    for gen in ("sd15", "amused"):
        for cond in conds:
            x = sub(d, gen, cond)
            if x.content_id.nunique() < 8: continue
            for name, cols in cum.items():
                auc, _ = grouped_cv_auc(x, cols, reps=5, folds=5)
                incr_rows.append({"generator": gen, "condition": cond, "stage_set": name, "n_content": x.content_id.nunique(), "grouped_cv_auroc": auc})
    incr = pd.DataFrame(incr_rows); incr.to_csv(OUT / "robustness_pilot_incremental_info.csv", index=False)

    print(retain.groupby(["generator", "condition", "stage"]).paired_cohens_d.mean().round(3).to_string())
    print(incr.pivot_table(index=["generator", "stage_set"], columns="condition", values="grouped_cv_auroc").round(3).to_string())


if __name__ == "__main__":
    main()
