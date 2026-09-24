"""Fourth-generator (genuine PixArt-Sigma, Diffusion Transformer) architecture-disambiguation experiment: Phases
3-5 + cross-generator transfer, exactly mirroring scripts/pixart_stage_decomposition_analysis.py's design
(content-grouped 5x10 CV, same classifier, same bootstrap), applied to real-vs-pixart_dit and joined with the
existing SD1.5/aMUSEd/SDXL results for a four-generator comparison table and a full pairwise transfer matrix.
No feature/classifier change; this is purely an evaluation-protocol reuse. See PREREGISTRATION_DIT_GENERATOR_V1.md."""
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

OUT = Path("results/stage_decomposition/dit_generator"); OUT.mkdir(parents=True, exist_ok=True)
GEN = "pixart_dit"
ALL_GENERATORS = ["sd15", "amused", "sdxl", "pixart_dit"]


def main():
    rows = json.load(open("results/stage_decomposition/panel_features_dit.json"))
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
    uni = pd.DataFrame(uni_rows); uni.to_csv(OUT / "univariate_feature_results_dit.csv", index=False)
    print(uni.round(3).to_string(index=False))

    # ---- Phase 3: stage-level ----
    stage_rows = []
    for stage in STAGE_ORDER:
        cols = [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == stage]
        auc, oof = grouped_cv_auc(x, cols); lo, hi = content_boot_ci(x, oof)
        stage_rows.append({"generator": GEN, "stage": stage, "n_features": len(cols), "grouped_cv_auroc": auc, "ci_lo": lo, "ci_hi": hi})
    stage_df = pd.DataFrame(stage_rows); stage_df.to_csv(OUT / "stage_level_results_dit.csv", index=False)
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
    incr_df = pd.DataFrame(incr_rows); incr_df.to_csv(OUT / "incremental_information_dit.csv", index=False)
    json.dump(primary, open(OUT / "PRIMARY_RESULT.json", "w"), indent=1)
    print("\nPRIMARY RESULT:", json.dumps(primary, indent=1))
    print(incr_df.round(3).to_string(index=False))

    # ---- Phase 5 + four-generator table + full pairwise transfer matrix ----
    prior = pd.read_csv("results/stage_decomposition/univariate_feature_results.csv")  # sd15 + amused, from the earlier phase
    sdxl_uni = pd.read_csv("results/stage_decomposition/pixart/univariate_feature_results_sdxl.csv")  # sdxl, from the third-generator phase
    four = prior.pivot(index="feature", columns="generator", values="paired_cohens_d")
    four["sdxl"] = sdxl_uni.set_index("feature").paired_cohens_d
    four["pixart_dit"] = uni.set_index("feature").paired_cohens_d
    four["stage"] = [STAGE_OF[f] for f in four.index]
    four = four[["stage", "sd15", "amused", "sdxl", "pixart_dit"]]
    four["sign_sd15_dit_agree"] = np.sign(four.sd15) == np.sign(four.pixart_dit)
    four["sign_sdxl_dit_agree"] = np.sign(four.sdxl) == np.sign(four.pixart_dit)
    four["sign_amused_dit_agree"] = np.sign(four.amused) == np.sign(four.pixart_dit)
    four.to_csv(OUT / "four_generator_feature_table.csv")
    print("\nFOUR-GENERATOR TABLE:\n", four.round(3).to_string())

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

    # full pairwise transfer matrix across all 4 generators (12 off-diagonal directions), plus the 6 new
    # DiT-involving directions the task specifically requires (SD1.5<->DiT, SDXL<->DiT, aMUSEd<->DiT)
    transfer_rows = {}
    for tr_gen in ALL_GENERATORS:
        for te_gen in ALL_GENERATORS:
            if tr_gen == te_gen:
                continue
            transfer_rows[f"{tr_gen}_to_{te_gen}"] = transfer(tr_gen, te_gen)
    json.dump(transfer_rows, open(OUT / "transfer_results.json", "w"), indent=1)
    print("\nFULL TRANSFER MATRIX:", json.dumps(transfer_rows, indent=1))


if __name__ == "__main__":
    main()
