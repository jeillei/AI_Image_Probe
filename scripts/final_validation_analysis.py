"""Final validation analysis: Track A (realistic-transformation robustness) + Track B (AI-edit continuum).
See FINAL_VALIDATION_PLAN.md, frozen before this script was run. Reuses the exact CV/bootstrap/classifier
machinery from stage_decomposition_analysis.py and vae_curvature_redundancy_analysis.py -- no new model family,
no new feature, no per-condition hyperparameter tuning."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_decomposition_analysis import make, cohen_paired, boot_ci, grouped_cv_auc, paired_diff_ci
from vae_curvature_redundancy_analysis import cv_oof_probs, metric_val, paired_metric_diff_ci
from src.corruption.robustness_suite import condition_id as _condition_id

OUT = Path("results/final_validation"); OUT.mkdir(parents=True, exist_ok=True)
GENS_PRIMARY = ["sd15", "sdxl"]
GENS_SECONDARY = ["amused", "pixart_dit"]
VAE_FEATS = ["lpips_ae", "pixel_mse_ae", "latent_mse_ae"]
SCORE_FEATS = ["lare_t200", "score_norm_step0"]
CURV, PLEN = "diffpath_curvature", "path_length"
FEATS_OF_INTEREST = ["lpips_ae", CURV, PLEN]


def load_all():
    clean_rows = json.load(open("results/stage_decomposition/panel_features_dit.json"))
    clean_df = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"],
                              "content_id": r["content_id"], "transform": "clean", "transform_value": None,
                              "condition": "clean", **r["features"]} for r in clean_rows])
    tf_rows = json.load(open("results/final_validation/panel_features_final.json"))
    tf_df = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"],
                           "content_id": r["content_id"], "transform": r.get("transform") or "clean",
                           "transform_value": r.get("transform_value"),
                           "condition": _condition_id(r.get("transform"), r.get("transform_value")) if r.get("transform") else "",
                           **r["features"]} for r in tf_rows])
    all_df = pd.concat([clean_df, tf_df[tf_df["condition"] != ""]], ignore_index=True)
    track_b = tf_df[tf_df.generator.str.startswith("edit_s")].copy()
    track_b["strength"] = track_b.generator.str.replace("edit_s", "").astype(float)
    return all_df, track_b, clean_df


def sub_condition(d, gen, condition):
    x = d[((d.label == 0) & (d["condition"] == condition)) | ((d.generator == gen) & (d["condition"] == condition))]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)


# ---------------- Q1: does the feature itself survive? ----------------
def q1_feature_survival(d):
    conditions = sorted(d["condition"].unique())
    rows = []
    for gen in GENS_PRIMARY + GENS_SECONDARY:
        x_clean = sub_condition(d, gen, "clean")
        for feat in FEATS_OF_INTEREST:
            d_clean, _ = cohen_paired(x_clean, feat)
            for cond in conditions:
                x = sub_condition(d, gen, cond)
                if len(x) == 0: continue
                d_val, diffs = cohen_paired(x, feat)
                lo, hi = boot_ci(diffs.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
                auc = roc_auc_score(x.label, x[feat]); auc_df = max(auc, 1 - auc)
                frac_match = float((np.sign(diffs.values) == np.sign(d_clean)).mean())
                # clean -> transformed Spearman (per-image, same content+label, before/after)
                merged = x_clean[["content_id", "label", feat]].merge(
                    x[["content_id", "label", feat]], on=["content_id", "label"], suffixes=("_clean", "_tf"))
                sp = stats.spearmanr(merged[f"{feat}_clean"], merged[f"{feat}_tf"]).statistic if len(merged) > 2 else float("nan")
                retention = float(d_val / d_clean) if d_clean != 0 else float("nan")
                rows.append({"generator": gen, "feature": feat, "condition": cond, "d_clean": d_clean,
                            "d_transformed": d_val, "d_ci_lo": lo, "d_ci_hi": hi, "auroc_direction_free": auc_df,
                            "frac_matched_clean_direction": frac_match, "clean_to_transformed_spearman": sp,
                            "effect_retention": retention,
                            "status": ("collapsed" if (lo < 0 < hi or np.sign(d_val) != np.sign(d_clean)) else
                                       ("degraded" if abs(d_val) < 0.7 * abs(d_clean) else "survived"))})
    df = pd.DataFrame(rows); df.to_csv(OUT / "q1_feature_survival.csv", index=False)
    return df


# ---------------- Q2: does incremental information survive? ----------------
def q2_incremental_survival(d):
    conditions = sorted(d["condition"].unique())
    rows = []
    for gen in GENS_PRIMARY:
        for cond in conditions:
            x = sub_condition(d, gen, cond)
            if len(x) == 0: continue
            sets = {"VAE": VAE_FEATS, "VAE+path_length": VAE_FEATS + [PLEN],
                   "VAE+curvature": VAE_FEATS + [CURV],
                   "VAE+score": VAE_FEATS + SCORE_FEATS, "VAE+score+path_length": VAE_FEATS + SCORE_FEATS + [PLEN],
                   "VAE+score+curvature": VAE_FEATS + SCORE_FEATS + [CURV]}
            oofs = {name: cv_oof_probs(x, cols, reps=5) for name, cols in sets.items()}
            for base, ext in [("VAE", "VAE+path_length"), ("VAE", "VAE+curvature"),
                              ("VAE+score", "VAE+score+path_length"), ("VAE+score", "VAE+score+curvature")]:
                for metric in ["auroc", "logloss", "brier"]:
                    md, ci = paired_metric_diff_ci(x, oofs[base], oofs[ext], metric, n=300)
                    rows.append({"generator": gen, "condition": cond, "comparison": f"{ext} vs {base}",
                                "metric": metric, "delta_mean": md, "delta_ci_lo": ci[0], "delta_ci_hi": ci[1]})
    df = pd.DataFrame(rows); df.to_csv(OUT / "q2_incremental_survival.csv", index=False)
    return df


# ---------------- Q3: clean-trained transfer (deployment-style, no recalibration) ----------------
def q3_clean_trained_transfer(d):
    conditions = sorted(d["condition"].unique())
    rows = []
    for gen in GENS_PRIMARY:
        x_clean = sub_condition(d, gen, "clean")
        y_clean = x_clean.label.values
        for name, cols in [("VAE", VAE_FEATS), ("VAE+path_length", VAE_FEATS + [PLEN]),
                           ("VAE+score+path_length", VAE_FEATS + SCORE_FEATS + [PLEN])]:
            model = make().fit(np.nan_to_num(x_clean[cols].values), y_clean)
            for cond in conditions:
                x = sub_condition(d, gen, cond)
                if len(x) == 0: continue
                p = model.predict_proba(np.nan_to_num(x[cols].values))[:, 1]
                rows.append({"generator": gen, "feature_set": name, "condition": cond,
                            "auroc": metric_val("auroc", x.label.values, p),
                            "logloss": metric_val("logloss", x.label.values, p),
                            "brier": metric_val("brier", x.label.values, p)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "q3_clean_trained_transfer.csv", index=False)
    return df


# ---------------- Track B: AI-edit continuum ----------------
def track_b_analysis(track_b, clean_df):
    strengths = [0.0, 0.3, 0.6, 0.9]
    baseline = clean_df[clean_df.generator == "real"][["content_id"] + FEATS_OF_INTEREST].copy()
    baseline["strength"] = 0.0
    tb = pd.concat([baseline, track_b[["content_id", "strength"] + FEATS_OF_INTEREST]], ignore_index=True)
    tb = tb[tb.content_id.isin(baseline.content_id)]  # keep only contents with a strength=0 baseline

    curve_rows = []
    for feat in FEATS_OF_INTEREST:
        for s in strengths:
            vals = tb[tb.strength == s][feat].dropna().values
            if len(vals) == 0: continue
            lo, hi = boot_ci(vals, np.mean)
            curve_rows.append({"feature": feat, "strength": s, "mean": float(np.mean(vals)), "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    curve_df = pd.DataFrame(curve_rows); curve_df.to_csv(OUT / "track_b_population_curves.csv", index=False)

    trend_rows = []
    for feat in FEATS_OF_INTEREST:
        piv = tb.pivot_table(index="content_id", columns="strength", values=feat)
        piv = piv.dropna()
        rhos = []
        for cid, row in piv.iterrows():
            rho = stats.spearmanr(strengths, [row[s] for s in strengths]).statistic
            rhos.append(rho)
        rhos = np.array(rhos)
        pooled_x = np.concatenate([[s] * len(piv) for s in strengths])
        pooled_y = np.concatenate([piv[s].values for s in strengths])
        pooled = stats.spearmanr(pooled_x, pooled_y)
        trend_rows.append({"feature": feat, "n_contents": len(piv), "pooled_spearman_r": float(pooled.statistic),
                           "pooled_p": float(pooled.pvalue), "frac_monotonic": float((np.abs(rhos) > 0.99).mean()),
                           "mean_within_content_spearman": float(np.nanmean(rhos))})
    trend_df = pd.DataFrame(trend_rows); trend_df.to_csv(OUT / "track_b_trend_test.csv", index=False)
    return curve_df, trend_df


def main():
    d, track_b, clean_df = load_all()
    print("loaded", len(d), "Track A rows,", len(track_b), "Track B rows")
    q1 = q1_feature_survival(d); print("\nQ1 done:", len(q1), "rows")
    print(q1[q1.generator.isin(GENS_PRIMARY)].groupby(["generator", "feature"]).status.value_counts().to_string())
    q2 = q2_incremental_survival(d); print("\nQ2 done:", len(q2), "rows")
    q3 = q3_clean_trained_transfer(d); print("\nQ3 done:", len(q3), "rows")
    curve_df, trend_df = track_b_analysis(track_b, clean_df)
    print("\nTrack B trend test:\n", trend_df.round(3).to_string(index=False))
    print("\nAll parts complete. Outputs in", OUT)


if __name__ == "__main__":
    main()
