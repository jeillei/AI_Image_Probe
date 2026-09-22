"""Third-generator (SDXL, substituted for PixArt-Sigma) decision experiment: Phases 3-5 + cross-generator
transfer, exactly mirroring scripts/stage_decomposition_analysis.py's design (content-grouped 5x10 CV, same
classifier, same bootstrap), applied to real-vs-SDXL and joined with the existing SD1.5/aMUSEd results for the
three-generator comparison table.  No feature/classifier change; this is purely an evaluation-protocol reuse."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.features.panel_v2 import CORE_FEATURE_NAMES, STAGE_OF, STAGE_ORDER
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_decomposition_analysis import (make, sub, cohen_paired, boot_ci, grouped_cv_auc, content_boot_ci,
    paired_grouped_cv, paired_diff_ci)

OUT = Path("results/stage_decomposition/pixart"); OUT.mkdir(parents=True, exist_ok=True)
GEN = "sdxl"


def main():
    rows = json.load(open("results/stage_decomposition/panel_features_pixart.json"))
    d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": r["content_id"], **r["features"]} for r in rows])
    assert set(CORE_FEATURE_NAMES) <= set(d.columns)
    x = sub(d, GEN)
    assert x.content_id.nunique() == 60 and len(x) == 120, (x.content_id.nunique(), len(x))
    for c in CORE_FEATURE_NAMES: assert d[c].notna().all(), f"NaN in {c}"
    print("real vs", GEN, ":", len(x), "rows,", x.content_id.nunique(), "content ids")

    # ---- Phase 3: per-feature univariate ----
    uni_rows = []
    for feat in CORE_FEATURE_NAMES:
        dval, diffs = cohen_paired(x, feat)
        lo_d, hi_d = boot_ci(diffs.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc = roc_auc_score(x.label, x[feat]); auc_df = max(auc, 1 - auc)
        _, oof = grouped_cv_auc(x, [feat], reps=5)
        lo_a, hi_a = content_boot_ci(x, oof)
        uni_rows.append({"generator": GEN, "feature": feat, "stage": STAGE_OF[feat], "paired_cohens_d": dval,
                         "d_ci_lo": lo_d, "d_ci_hi": hi_d, "univariate_auroc_direction_free": auc_df,
                         "grouped_cv_auroc": roc_auc_score(x.label, oof), "auroc_ci_lo": lo_a, "auroc_ci_hi": hi_a,
                         "frac_pairs_fake_higher": float((diffs > 0).mean())})
    uni = pd.DataFrame(uni_rows); uni.to_csv(OUT / "univariate_feature_results_sdxl.csv", index=False)
    print(uni.round(3).to_string(index=False))

    # ---- Phase 3: stage-level ----
    stage_rows = []
    for stage in STAGE_ORDER:
        cols = [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == stage]
        auc, oof = grouped_cv_auc(x, cols); lo, hi = content_boot_ci(x, oof)
        stage_rows.append({"generator": GEN, "stage": stage, "n_features": len(cols), "grouped_cv_auroc": auc, "ci_lo": lo, "ci_hi": hi})
    stage_df = pd.DataFrame(stage_rows); stage_df.to_csv(OUT / "stage_level_results_sdxl.csv", index=False)
    print(stage_df.round(3).to_string(index=False))

    # ---- Phase 4: incremental information (THE PRIMARY TEST) ----
    cum = {"vae": [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "vae"]}
    cum["vae+score"] = cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "score"]
    cum["vae+score+trajectory"] = cum["vae+score"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "trajectory"]
    cum["vae+score+trajectory+roundtrip"] = cum["vae+score+trajectory"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "roundtrip"]
    incr_rows = []; names = list(cum); oofs = {}
    for name in names:
        auc, oof = grouped_cv_auc(x, cum[name]); oofs[name] = oof
        incr_rows.append({"generator": GEN, "stage_set": name, "n_features": len(cum[name]), "grouped_cv_auroc": auc})
    for a, b in zip(names[:-1], names[1:]):
        md, ci = paired_diff_ci(x, oofs[a], oofs[b])
        incr_rows.append({"generator": GEN, "stage_set": f"{b} vs {a} (paired diff)", "paired_diff_mean": md, "diff_ci_lo": ci[0], "diff_ci_hi": ci[1], "adds_information": bool(ci[0] > 0)})
    # THE primary preregistered comparison, called out explicitly
    md_primary, ci_primary = paired_diff_ci(x, oofs["vae+score"], oofs["vae+score+trajectory"])
    primary = {"generator": GEN, "comparison": "PRIMARY: vae+score+trajectory vs vae+score", "auroc_vae_score": float(roc_auc_score(x.label, oofs["vae+score"])),
              "auroc_vae_score_trajectory": float(roc_auc_score(x.label, oofs["vae+score+trajectory"])),
              "paired_diff_mean": md_primary, "diff_ci_lo": ci_primary[0], "diff_ci_hi": ci_primary[1], "adds_information": bool(ci_primary[0] > 0)}
    targets = [("vae", "vae+trajectory", cum["vae"], cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "trajectory"]),
               ("vae", "vae+roundtrip", cum["vae"], cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "roundtrip"]),
               ("score", "score+trajectory", [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "score"], [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] in ("score", "trajectory")])]
    for na, nb, ca, cb in targets:
        auc_a, auc_b, oa, ob = paired_grouped_cv(x, ca, cb); md, ci = paired_diff_ci(x, oa, ob)
        incr_rows.append({"generator": GEN, "stage_set": f"TARGETED: {nb} vs {na}", "grouped_cv_auroc": auc_b, "baseline_auroc": auc_a,
                          "paired_diff_mean": md, "diff_ci_lo": ci[0], "diff_ci_hi": ci[1], "adds_information": bool(ci[0] > 0)})
    incr_df = pd.DataFrame(incr_rows); incr_df.to_csv(OUT / "incremental_information_sdxl.csv", index=False)
    json.dump(primary, open(OUT / "PRIMARY_RESULT.json", "w"), indent=1)
    print("\nPRIMARY RESULT:", json.dumps(primary, indent=1))
    print(incr_df.round(3).to_string(index=False))

    # ---- Phase 5 + three-generator table + transfer ----
    prior = pd.read_csv("results/stage_decomposition/univariate_feature_results.csv")  # sd15 + amused, from the earlier phase
    three = prior.pivot(index="feature", columns="generator", values="paired_cohens_d")
    three["sdxl"] = uni.set_index("feature").paired_cohens_d
    three["stage"] = [STAGE_OF[f] for f in three.index]
    three = three[["stage", "sd15", "amused", "sdxl"]]
    three["sign_sd15_sdxl_agree"] = np.sign(three.sd15) == np.sign(three.sdxl)
    three["sign_amused_sdxl_agree"] = np.sign(three.amused) == np.sign(three.sdxl)
    three.to_csv(OUT / "three_generator_feature_table.csv")
    print("\nTHREE-GENERATOR TABLE:\n", three.round(3).to_string())

    # transfer: train on one real-vs-generator pair, test on another, content-grouped, same code pattern as Phase 5
    def transfer(train_gen, test_gen, reps=10, folds=5, seed=0):
        real = d[d.label == 0]; tr_fake = d[(d.label == 1) & (d.generator == train_gen)]; te_fake = d[(d.label == 1) & (d.generator == test_gen)]
        ids = np.array(sorted(set(real.content_id) & set(tr_fake.content_id) & set(te_fake.content_id)))
        aucs = []
        for r in range(reps):
            perm = np.random.default_rng(seed + r).permutation(ids); fo = {c: i % folds for i, c in enumerate(perm)}
            preds = []; ys = []
            for k in range(folds):
                tr_ids = [c for c in ids if fo[c] != k]; te_ids = [c for c in ids if fo[c] == k]
                tr = pd.concat([real[real.content_id.isin(tr_ids)], tr_fake[tr_fake.content_id.isin(tr_ids)]])
                te = pd.concat([real[real.content_id.isin(te_ids)], te_fake[te_fake.content_id.isin(te_ids)]])
                mdl = make().fit(np.nan_to_num(tr[CORE_FEATURE_NAMES].values), tr.label.values)
                p = mdl.predict_proba(np.nan_to_num(te[CORE_FEATURE_NAMES].values))[:, 1]
                preds.append(p); ys.append(te.label.values)
            y_all = np.concatenate(ys); p_all = np.concatenate(preds); aucs.append(roc_auc_score(y_all, p_all))
        return float(np.mean(aucs))
    transfer_rows = {"sd15_to_sdxl": transfer("sd15", "sdxl"), "sdxl_to_sd15": transfer("sdxl", "sd15"),
                     "amused_to_sdxl": transfer("amused", "sdxl"), "sdxl_to_amused": transfer("sdxl", "amused")}
    json.dump(transfer_rows, open(OUT / "transfer_results.json", "w"), indent=1)
    print("\nTRANSFER:", json.dumps(transfer_rows, indent=1))


if __name__ == "__main__":
    main()
