"""VAE-curvature information-redundancy mechanistic follow-up. See ANALYSIS_PLAN_VAE_CURVATURE_REDUNDANCY.md,
frozen before any output here was inspected. Exploratory/mechanistic, NOT an independent preregistered
confirmation -- motivated directly by DIT_STAGE_DECOMPOSITION.md's primary-test discrepancy. Reuses the existing
content-grouped CV machinery and classifier from scripts/stage_decomposition_analysis.py verbatim; no new model
family, no feature recomputation, no remote compute needed (pure CPU statistics on already-cached features)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_decomposition_analysis import make, sub, cohen_paired, boot_ci

OUT = Path("results/vae_curvature_redundancy"); OUT.mkdir(parents=True, exist_ok=True)
REPS, FOLDS, SEED, N_BOOT = 10, 5, 0, 1000
GENS = ["sd15", "sdxl", "pixart_dit", "amused"]
VAE_FEATS = ["lpips_ae", "pixel_mse_ae", "latent_mse_ae"]
SCORE_FEATS = ["lare_t200", "score_norm_step0"]
CURV = "diffpath_curvature"


def fold_assignment(x, r, folds=FOLDS, seed=SEED):
    ids = np.array(sorted(x.content_id.unique()))
    perm = np.random.default_rng(seed + r).permutation(ids)
    return x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values


# ---------------- Part 1: simple geometry of VAE vs curvature ----------------
def part1_correlations(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen)
        for vf in VAE_FEATS:
            for subset_name, mask in [("pooled", np.ones(len(x), bool)), ("real_only", x.label.values == 0), ("generated_only", x.label.values == 1)]:
                xs = x[mask]
                pear = stats.pearsonr(xs[vf], xs[CURV])
                spear = stats.spearmanr(xs[vf], xs[CURV])
                rows.append({"generator": gen, "vae_feature": vf, "subset": subset_name, "n": int(mask.sum()),
                            "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
                            "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "correlation_tables.csv", index=False)
    return df


# ---------------- Part 3: cross-fitted VAE-only discriminant (S_vae) ----------------
def part3_s_vae(x):
    X = np.nan_to_num(x[VAE_FEATS].values); y = x.label.values; n = len(x)
    oof_matrix = np.zeros((REPS, n))
    for r in range(REPS):
        fo = fold_assignment(x, r)
        for k in range(FOLDS):
            te = fo == k
            oof_matrix[r, te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return oof_matrix, oof_matrix.mean(axis=0)


# ---------------- Part 4: cross-fitted curvature residualization (label-blind, vs raw VAE feats) ----------------
def cross_fitted_residualize(x, C_obs, predictor_matrix, alpha=1.0):
    n = len(x)
    pred_matrix = np.full((REPS, n), np.nan)
    for r in range(REPS):
        fo = fold_assignment(x, r)
        for k in range(FOLDS):
            te = fo == k; tr = ~te
            scaler = StandardScaler().fit(predictor_matrix[tr])
            Xtr = scaler.transform(predictor_matrix[tr]); Xte = scaler.transform(predictor_matrix[te])
            ridge = Ridge(alpha=alpha).fit(Xtr, C_obs[tr])
            pred_matrix[r, te] = ridge.predict(Xte)
    resid_matrix = C_obs[None, :] - pred_matrix
    return pred_matrix, resid_matrix, np.nanmean(pred_matrix, axis=0), C_obs - np.nanmean(pred_matrix, axis=0)


# ---------------- Part 7: residualize against S_vae itself (rep-varying predictor) ----------------
def cross_fitted_residualize_on_svae(x, C_obs, s_vae_oof_matrix, alpha=1.0):
    n = len(x)
    pred_matrix = np.full((REPS, n), np.nan)
    for r in range(REPS):
        fo = fold_assignment(x, r)
        s = s_vae_oof_matrix[r].reshape(-1, 1)
        for k in range(FOLDS):
            te = fo == k; tr = ~te
            scaler = StandardScaler().fit(s[tr])
            Xtr = scaler.transform(s[tr]); Xte = scaler.transform(s[te])
            ridge = Ridge(alpha=alpha).fit(Xtr, C_obs[tr])
            pred_matrix[r, te] = ridge.predict(Xte)
    resid_matrix = C_obs[None, :] - pred_matrix
    return pred_matrix, resid_matrix, np.nanmean(pred_matrix, axis=0), C_obs - np.nanmean(pred_matrix, axis=0)


def cohen_paired_array(x, values):
    xx = x.copy(); xx["_val"] = values
    pw = xx.pivot_table(index="content_id", columns="label", values="_val")
    diff = (pw[1] - pw[0]).dropna()
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12)), diff


# ---------------- Part 5+6+7: raw vs residual effect table + regression performance ----------------
def parts_4_5_6_7(d):
    effect_rows = []; regress_rows = []
    predicted_records = []; svae_records = []; s_vae_store = {}
    for gen in GENS:
        x = sub(d, gen)
        C_obs = x[CURV].values.astype(float)
        V = np.nan_to_num(x[VAE_FEATS].values.astype(float))

        # raw curvature effect (established direction reference)
        d_raw, diffs_raw = cohen_paired(x, CURV)
        lo_raw, hi_raw = boot_ci(diffs_raw.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc_raw = roc_auc_score(x.label, C_obs); auc_raw_df = max(auc_raw, 1 - auc_raw)
        established_sign = np.sign(d_raw) if d_raw != 0 else 1.0
        frac_raw = float((np.sign(diffs_raw.values) == established_sign).mean())
        effect_rows.append({"generator": gen, "curvature_variant": "raw", "paired_cohens_d": d_raw,
                            "d_ci_lo": lo_raw, "d_ci_hi": hi_raw, "univariate_auroc_direction_free": auc_raw_df,
                            "frac_matched_established_direction": frac_raw, "effect_retention_vs_raw": 1.0})

        # Part 4: residualize vs raw VAE features (label-blind)
        pred_m, resid_m, pred_mean, resid_mean = cross_fitted_residualize(x, C_obs, V)
        d_res, diffs_res = cohen_paired_array(x, resid_mean)
        lo_res, hi_res = boot_ci(diffs_res.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc_res = roc_auc_score(x.label, resid_mean); auc_res_df = max(auc_res, 1 - auc_res)
        frac_res = float((np.sign(diffs_res.values) == established_sign).mean())
        retention = float(abs(d_res) / (abs(d_raw) + 1e-12))
        effect_rows.append({"generator": gen, "curvature_variant": "vae_residualized", "paired_cohens_d": d_res,
                            "d_ci_lo": lo_res, "d_ci_hi": hi_res, "univariate_auroc_direction_free": auc_res_df,
                            "frac_matched_established_direction": frac_res, "effect_retention_vs_raw": retention})

        # Part 6: cross-fitted regression performance, pooled + within-class
        r2_pool = float(1 - np.sum((C_obs - pred_mean) ** 2) / np.sum((C_obs - C_obs.mean()) ** 2))
        mae_pool = float(np.mean(np.abs(C_obs - pred_mean)))
        pear_pool = float(stats.pearsonr(C_obs, pred_mean).statistic)
        regress_rows.append({"generator": gen, "predictor": "raw_VAE_features", "subset": "pooled",
                             "cross_fitted_r2": r2_pool, "mae": mae_pool, "pearson_pred_obs": pear_pool})
        for subset_name, mask in [("real_only", x.label.values == 0), ("generated_only", x.label.values == 1)]:
            co, pm = C_obs[mask], pred_mean[mask]
            r2 = float(1 - np.sum((co - pm) ** 2) / np.sum((co - co.mean()) ** 2)) if co.std() > 0 else float("nan")
            mae = float(np.mean(np.abs(co - pm)))
            pear = float(stats.pearsonr(co, pm).statistic) if co.std() > 0 and pm.std() > 0 else float("nan")
            regress_rows.append({"generator": gen, "predictor": "raw_VAE_features", "subset": subset_name,
                                 "cross_fitted_r2": r2, "mae": mae, "pearson_pred_obs": pear})

        predicted_records.append(pd.DataFrame({"generator": gen, "content_id": x.content_id.values, "label": x.label.values,
                                                "curvature_observed": C_obs, "curvature_predicted_from_vae": pred_mean,
                                                "curvature_residual_vs_vae": resid_mean}))
        np.savez_compressed(OUT / f"curvature_residual_repeats_{gen}.npz", pred_matrix=pred_m, resid_matrix=resid_m)

        # Part 3: S_vae (needed here for Part 7)
        s_oof_matrix, s_oof_mean = part3_s_vae(x)
        s_vae_store[gen] = (s_oof_matrix, s_oof_mean)
        np.savez_compressed(OUT / f"s_vae_oof_repeats_{gen}.npz", oof_matrix=s_oof_matrix)

        # Part 7: residualize vs S_vae (label-informed through training-fold VAE classifier)
        pred_s, resid_s, pred_s_mean, resid_s_mean = cross_fitted_residualize_on_svae(x, C_obs, s_oof_matrix)
        d_svae, diffs_svae = cohen_paired_array(x, resid_s_mean)
        lo_svae, hi_svae = boot_ci(diffs_svae.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc_svae = roc_auc_score(x.label, resid_s_mean); auc_svae_df = max(auc_svae, 1 - auc_svae)
        frac_svae = float((np.sign(diffs_svae.values) == established_sign).mean())
        retention_svae = float(abs(d_svae) / (abs(d_raw) + 1e-12))
        effect_rows.append({"generator": gen, "curvature_variant": "svae_residualized_label_informed",
                            "paired_cohens_d": d_svae, "d_ci_lo": lo_svae, "d_ci_hi": hi_svae,
                            "univariate_auroc_direction_free": auc_svae_df,
                            "frac_matched_established_direction": frac_svae, "effect_retention_vs_raw": retention_svae})
        svae_records.append(pd.DataFrame({"generator": gen, "content_id": x.content_id.values, "label": x.label.values,
                                          "s_vae_oof_mean": s_oof_mean, "curvature_predicted_from_svae": pred_s_mean,
                                          "curvature_residual_vs_svae": resid_s_mean}))
        r2_s = float(1 - np.sum((C_obs - pred_s_mean) ** 2) / np.sum((C_obs - C_obs.mean()) ** 2))
        regress_rows.append({"generator": gen, "predictor": "S_vae_learned_discriminant", "subset": "pooled",
                             "cross_fitted_r2": r2_s, "mae": float(np.mean(np.abs(C_obs - pred_s_mean))),
                             "pearson_pred_obs": float(stats.pearsonr(C_obs, pred_s_mean).statistic)})
        print(gen, "parts 3/4/6/7 done", flush=True)

    pd.DataFrame(effect_rows).to_csv(OUT / "raw_vs_residual_effect_table.csv", index=False)
    pd.DataFrame(regress_rows).to_csv(OUT / "vae_to_curvature_regression_performance.csv", index=False)
    pd.concat(predicted_records, ignore_index=True).to_csv(OUT / "curvature_predicted_vs_observed.csv", index=False)
    pd.concat(svae_records, ignore_index=True).to_csv(OUT / "svae_residual_curvature.csv", index=False)
    s_vae_all = pd.concat([pd.DataFrame({"generator": g, "content_id": sub(d, g).content_id.values,
                                         "label": sub(d, g).label.values, "s_vae_oof_mean": m}) for g, (_, m) in s_vae_store.items()], ignore_index=True)
    s_vae_all.to_csv(OUT / "s_vae_oof.csv", index=False)
    return pd.DataFrame(effect_rows), s_vae_store


# ---------------- Part 8: proper scoring rules, nested feature-set comparison ----------------
def cv_oof_probs(x, cols, reps=REPS, folds=FOLDS):
    X = np.nan_to_num(x[cols].values); y = x.label.values; n = len(x)
    oof = np.zeros(n)
    for r in range(reps):
        fo = fold_assignment(x, r, folds)
        p = np.zeros(n)
        for k in range(folds):
            te = fo == k
            p[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        oof += p / reps
    return oof


def metric_val(metric, y, p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    if metric == "auroc": return roc_auc_score(y, p)
    if metric == "logloss": return log_loss(y, pc, labels=[0, 1])
    if metric == "brier": return brier_score_loss(y, p)


def paired_metric_diff_ci(x, oof_a, oof_b, metric, n=N_BOOT, seed=0):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}
    y = x.label.values; rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        diffs.append(metric_val(metric, y[ix], oof_b[ix]) - metric_val(metric, y[ix], oof_a[ix]))
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def part8_proper_scoring(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen); y = x.label.values
        feats = {"VAE": VAE_FEATS, "VAE+curvature": VAE_FEATS + [CURV],
                "VAE+score": VAE_FEATS + SCORE_FEATS, "VAE+score+curvature": VAE_FEATS + SCORE_FEATS + [CURV]}
        oofs = {name: cv_oof_probs(x, cols) for name, cols in feats.items()}
        for name, oof in oofs.items():
            rows.append({"generator": gen, "feature_set": name, "auroc": metric_val("auroc", y, oof),
                        "logloss": metric_val("logloss", y, oof), "brier": metric_val("brier", y, oof)})
        for base, ext in [("VAE", "VAE+curvature"), ("VAE+score", "VAE+score+curvature")]:
            for metric in ["auroc", "logloss", "brier"]:
                md, ci = paired_metric_diff_ci(x, oofs[base], oofs[ext], metric)
                rows.append({"generator": gen, "feature_set": f"Δ({ext} - {base})", "metric": metric,
                            "delta_mean": md, "delta_ci_lo": ci[0], "delta_ci_hi": ci[1],
                            "improves": bool((ci[1] < 0) if metric != "auroc" else (ci[0] > 0))})
        print(gen, "part 8 done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "incremental_proper_scoring.csv", index=False)
    return df


# ---------------- Part 9: conditional curvature coefficient ----------------
def part9_conditional_coefficient(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen)
        cols = VAE_FEATS + [CURV]
        X = np.nan_to_num(x[cols].values); y = x.label.values
        curv_idx = cols.index(CURV)
        coefs = []
        for r in range(REPS):
            fo = fold_assignment(x, r)
            for k in range(FOLDS):
                te = fo == k
                pipe = make().fit(X[~te], y[~te])
                lr = pipe.named_steps["logisticregression"]
                coefs.append(float(lr.coef_[0][curv_idx]))
        coefs = np.array(coefs)
        rows.append({"generator": gen, "mean_std_coef": float(coefs.mean()), "sd_std_coef": float(coefs.std(ddof=1)),
                    "frac_same_sign": float((np.sign(coefs) == np.sign(coefs.mean())).mean()), "n_fold_fits": len(coefs)})
        print(gen, "part 9 done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "conditional_curvature_coefficients.csv", index=False)
    return df


# ---------------- Part 11: PixArt -> aMUSEd effect-vector diagnostic ----------------
def part11_pixart_amused_diagnostic(d):
    four = pd.read_csv("results/stage_decomposition/dit_generator/four_generator_feature_table.csv", index_col=0)
    a = four["pixart_dit"].values; b = four["amused"].values
    pear = stats.pearsonr(a, b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    contrib = a * b / (np.linalg.norm(a) * np.linalg.norm(b))
    contrib_df = pd.DataFrame({"feature": four.index, "stage": four["stage"], "d_pixart_dit": a, "d_amused": b,
                               "contribution_to_cosine": contrib}).sort_values("contribution_to_cosine", ascending=False)
    contrib_df.to_csv(OUT / "pixart_amused_effect_vector_diagnostic.csv", index=False)

    # refit real-vs-pixart_dit 10-feature classifier, record fold-level standardized coefficients
    from src.features.panel_v2 import CORE_FEATURE_NAMES
    x = sub(d, "pixart_dit")
    X = np.nan_to_num(x[CORE_FEATURE_NAMES].values); y = x.label.values
    coef_rows = []
    for r in range(REPS):
        fo = fold_assignment(x, r)
        for k in range(FOLDS):
            te = fo == k
            pipe = make().fit(X[~te], y[~te])
            lr = pipe.named_steps["logisticregression"]
            coef_rows.append(lr.coef_[0])
    coef_mean = np.mean(coef_rows, axis=0); coef_sd = np.std(coef_rows, axis=0, ddof=1)
    weight_df = pd.DataFrame({"feature": CORE_FEATURE_NAMES, "mean_std_coef": coef_mean, "sd_std_coef": coef_sd,
                              "abs_mean_std_coef": np.abs(coef_mean)}).sort_values("abs_mean_std_coef", ascending=False)
    weight_df.to_csv(OUT / "pixart_dit_classifier_weights.csv", index=False)

    summary = {"pearson_correlation_effect_vectors": float(pear.statistic), "pearson_p": float(pear.pvalue),
              "cosine_similarity_effect_vectors": cos, "top3_contributing_features": contrib_df.head(3)["feature"].tolist(),
              "top3_classifier_weight_features": weight_df.head(3)["feature"].tolist()}
    json.dump(summary, open(OUT / "pixart_amused_effect_vector_diagnostic.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return contrib_df, weight_df, summary


def main():
    rows = json.load(open("results/stage_decomposition/panel_features_dit.json"))
    d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": r["content_id"], **r["features"]} for r in rows])
    assert len(d) == 300 and d.groupby("content_id").size().eq(5).all(), "expected 300 rows, 5 per content_id"
    for c in VAE_FEATS + [CURV] + SCORE_FEATS: assert d[c].notna().all(), f"NaN present in {c}"
    print("data sufficiency check passed:", len(d), "rows,", d.content_id.nunique(), "content ids, 0 NaN in required columns")

    corr = part1_correlations(d); print("\nPart 1 done:\n", corr.round(3).to_string(index=False))
    effect_df, s_vae_store = parts_4_5_6_7(d); print("\nParts 4/5/6/7 done:\n", effect_df.round(3).to_string(index=False))
    scoring_df = part8_proper_scoring(d); print("\nPart 8 done:\n", scoring_df.round(4).to_string(index=False))
    coef_df = part9_conditional_coefficient(d); print("\nPart 9 done:\n", coef_df.round(3).to_string(index=False))
    contrib_df, weight_df, summary = part11_pixart_amused_diagnostic(d)
    print("\nAll parts complete. Outputs in", OUT)


if __name__ == "__main__":
    main()
