"""path_length mechanism decomposition: does path_length (alone, or jointly with curvature) explain why the
trajectory stage added information for SDXL but not PixArt? See ANALYSIS_PLAN_PATH_LENGTH_MECHANISM.md, frozen
before any output here was inspected. Exploratory/mechanistic stopping-point experiment, not an independent
preregistered confirmation. C_resid is reused verbatim from results/vae_curvature_redundancy/ (NOT recomputed);
P_resid is computed fresh here with the identical residualization procedure. No new feature, no remote compute needed (pure CPU
statistics on already-cached features)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from stage_decomposition_analysis import make, sub, cohen_paired, boot_ci
from vae_curvature_redundancy_analysis import fold_assignment, cross_fitted_residualize, cohen_paired_array, cv_oof_probs, metric_val, paired_metric_diff_ci

OUT = Path("results/path_length_mechanism"); OUT.mkdir(parents=True, exist_ok=True)
REPS, FOLDS, SEED, N_BOOT = 10, 5, 0, 1000
GENS = ["sd15", "sdxl", "pixart_dit", "amused"]
VAE_FEATS = ["lpips_ae", "pixel_mse_ae", "latent_mse_ae"]
SCORE_FEATS = ["lare_t200", "score_norm_step0"]
CURV, PLEN = "diffpath_curvature", "path_length"

# ---------------- Part 1: raw path_length effect ----------------
def part1_raw_effect(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen)
        dval, diffs = cohen_paired(x, PLEN)
        lo, hi = boot_ci(diffs.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc = roc_auc_score(x.label, x[PLEN]); auc_df = max(auc, 1 - auc)
        rows.append({"generator": gen, "paired_cohens_d": dval, "d_ci_lo": lo, "d_ci_hi": hi,
                    "univariate_auroc_direction_free": auc_df, "frac_pairs_matched_sign": float((np.sign(diffs.values) == np.sign(dval)).mean())})
    df = pd.DataFrame(rows); df.to_csv(OUT / "raw_path_length_effect.csv", index=False)
    return df

# ---------------- Part 2: VAE-path_length dependence ----------------
def part2_correlations(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen)
        for vf in VAE_FEATS:
            for subset_name, mask in [("pooled", np.ones(len(x), bool)), ("real_only", x.label.values == 0), ("generated_only", x.label.values == 1)]:
                xs = x[mask]
                pear = stats.pearsonr(xs[vf], xs[PLEN]); spear = stats.spearmanr(xs[vf], xs[PLEN])
                rows.append({"generator": gen, "vae_feature": vf, "subset": subset_name, "n": int(mask.sum()),
                            "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
                            "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "path_length_correlation_tables.csv", index=False)
    return df

def part2_regression_performance(d, pred_records):
    rows = []
    for gen in GENS:
        pr = pred_records[gen]; P_obs = pr["P_obs"]; pred_mean = pr["pred_mean"]
        r2 = float(1 - np.sum((P_obs - pred_mean) ** 2) / np.sum((P_obs - P_obs.mean()) ** 2))
        mae = float(np.mean(np.abs(P_obs - pred_mean))); pear = float(stats.pearsonr(P_obs, pred_mean).statistic)
        rows.append({"generator": gen, "subset": "pooled", "cross_fitted_r2": r2, "mae": mae, "pearson_pred_obs": pear})
        for subset_name, mask in [("real_only", pr["label"] == 0), ("generated_only", pr["label"] == 1)]:
            po, pm = P_obs[mask], pred_mean[mask]
            r2s = float(1 - np.sum((po - pm) ** 2) / np.sum((po - po.mean()) ** 2)) if po.std() > 0 else float("nan")
            rows.append({"generator": gen, "subset": subset_name, "cross_fitted_r2": r2s,
                        "mae": float(np.mean(np.abs(po - pm))),
                        "pearson_pred_obs": float(stats.pearsonr(po, pm).statistic) if po.std() > 0 and pm.std() > 0 else float("nan")})
    df = pd.DataFrame(rows); df.to_csv(OUT / "vae_to_path_length_regression_performance.csv", index=False)
    return df

# ---------------- Part 3: residual path_length effect ----------------
def part3_residualize(d):
    effect_rows = []; pred_records = {}
    for gen in GENS:
        x = sub(d, gen)
        P_obs = x[PLEN].values.astype(float)
        V = np.nan_to_num(x[VAE_FEATS].values.astype(float))
        d_raw, diffs_raw = cohen_paired(x, PLEN)
        lo_raw, hi_raw = boot_ci(diffs_raw.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc_raw = roc_auc_score(x.label, P_obs); auc_raw_df = max(auc_raw, 1 - auc_raw)
        established_sign = np.sign(d_raw) if d_raw != 0 else 1.0
        frac_raw = float((np.sign(diffs_raw.values) == established_sign).mean())
        effect_rows.append({"generator": gen, "variant": "raw", "paired_cohens_d": d_raw, "d_ci_lo": lo_raw, "d_ci_hi": hi_raw,
                            "univariate_auroc_direction_free": auc_raw_df, "frac_matched_established_direction": frac_raw, "effect_retention_vs_raw": 1.0})

        pred_m, resid_m, pred_mean, resid_mean = cross_fitted_residualize(x, P_obs, V)
        d_res, diffs_res = cohen_paired_array(x, resid_mean)
        lo_res, hi_res = boot_ci(diffs_res.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
        auc_res = roc_auc_score(x.label, resid_mean); auc_res_df = max(auc_res, 1 - auc_res)
        frac_res = float((np.sign(diffs_res.values) == established_sign).mean())
        retention = float(abs(d_res) / (abs(d_raw) + 1e-12))
        effect_rows.append({"generator": gen, "variant": "vae_residualized", "paired_cohens_d": d_res, "d_ci_lo": lo_res, "d_ci_hi": hi_res,
                            "univariate_auroc_direction_free": auc_res_df, "frac_matched_established_direction": frac_res, "effect_retention_vs_raw": retention})
        np.savez_compressed(OUT / f"path_length_residual_repeats_{gen}.npz", pred_matrix=pred_m, resid_matrix=resid_m)
        pred_records[gen] = {"content_id": x.content_id.values, "label": x.label.values, "P_obs": P_obs,
                             "pred_mean": pred_mean, "resid_mean": resid_mean}
        print(gen, "part 3 done", flush=True)
    effect_df = pd.DataFrame(effect_rows); effect_df.to_csv(OUT / "raw_vs_residual_path_length_effect.csv", index=False)
    pred_df = pd.concat([pd.DataFrame({"generator": g, "content_id": r["content_id"], "label": r["label"],
                                       "path_length_observed": r["P_obs"], "path_length_predicted_from_vae": r["pred_mean"],
                                       "path_length_residual_vs_vae": r["resid_mean"]}) for g, r in pred_records.items()], ignore_index=True)
    pred_df.to_csv(OUT / "path_length_predicted_vs_observed.csv", index=False)
    return effect_df, pred_records

def load_c_resid():
    """Reuse C_resid verbatim from the curvature-redundancy phase -- not recomputed."""
    df = pd.read_csv("results/vae_curvature_redundancy/curvature_predicted_vs_observed.csv")
    return df

# ---------------- Part 4: residual trajectory model (C_resid alone / P_resid alone / both) ----------------
def part4_residual_trajectory_model(d, pred_records):
    c_resid_df = load_c_resid()
    rows = []
    for gen in GENS:
        x = sub(d, gen).reset_index(drop=True)
        # content_id is not unique within a generator's subset (real + fake share the same content_id), so
        # join on (content_id, label), not content_id alone.
        cr = c_resid_df[c_resid_df.generator == gen][["content_id", "label", "curvature_residual_vs_vae"]]
        pr = pred_records[gen]
        pr_df = pd.DataFrame({"content_id": pr["content_id"], "label": pr["label"], "path_length_resid": pr["resid_mean"]})
        merged = x[["content_id", "label"]].merge(cr, on=["content_id", "label"], how="left").merge(pr_df, on=["content_id", "label"], how="left")
        assert merged["curvature_residual_vs_vae"].notna().all() and merged["path_length_resid"].notna().all()
        c_vals = merged["curvature_residual_vs_vae"].values.astype(float)
        p_vals = merged["path_length_resid"].values.astype(float)
        y = x.label.values
        feats = {"C_resid_alone": c_vals.reshape(-1, 1), "P_resid_alone": p_vals.reshape(-1, 1),
                 "C_resid_and_P_resid": np.column_stack([c_vals, p_vals])}
        oofs = {}
        for name, X in feats.items():
            n = len(x); oof = np.zeros(n)
            for r in range(REPS):
                fo = fold_assignment(x, r)
                p = np.zeros(n)
                for k in range(FOLDS):
                    te = fo == k
                    p[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
                oof += p / REPS
            oofs[name] = oof
            rows.append({"generator": gen, "feature_set": name, "auroc": metric_val("auroc", y, oof),
                        "logloss": metric_val("logloss", y, oof), "brier": metric_val("brier", y, oof)})
        for base, ext in [("C_resid_alone", "C_resid_and_P_resid"), ("P_resid_alone", "C_resid_and_P_resid")]:
            for metric in ["auroc", "logloss", "brier"]:
                md, ci = paired_metric_diff_ci(x, oofs[base], oofs[ext], metric)
                rows.append({"generator": gen, "feature_set": f"Δ({ext} - {base})", "metric": metric,
                            "delta_mean": md, "delta_ci_lo": ci[0], "delta_ci_hi": ci[1]})
        print(gen, "part 4 done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "residual_trajectory_model_results.csv", index=False)
    return df

# ---------------- Part 5: SDXL vs PixArt feature-addition comparison (central diagnostic) ----------------
def part5_feature_addition(d):
    rows = []
    for gen in ["sdxl", "pixart_dit"]:
        x = sub(d, gen); y = x.label.values
        sets_novae = {"VAE": VAE_FEATS, "VAE+curvature": VAE_FEATS + [CURV], "VAE+path_length": VAE_FEATS + [PLEN],
                     "VAE+curvature+path_length": VAE_FEATS + [CURV, PLEN]}
        sets_score = {"VAE+score": VAE_FEATS + SCORE_FEATS, "VAE+score+curvature": VAE_FEATS + SCORE_FEATS + [CURV],
                     "VAE+score+path_length": VAE_FEATS + SCORE_FEATS + [PLEN],
                     "VAE+score+curvature+path_length": VAE_FEATS + SCORE_FEATS + [CURV, PLEN]}
        all_sets = {**sets_novae, **sets_score}
        oofs = {name: cv_oof_probs(x, cols) for name, cols in all_sets.items()}
        for name, oof in oofs.items():
            rows.append({"generator": gen, "feature_set": name, "auroc": metric_val("auroc", y, oof),
                        "logloss": metric_val("logloss", y, oof), "brier": metric_val("brier", y, oof)})
        pairs = [("VAE", "VAE+curvature"), ("VAE", "VAE+path_length"), ("VAE", "VAE+curvature+path_length"),
                ("VAE+score", "VAE+score+curvature"), ("VAE+score", "VAE+score+path_length"), ("VAE+score", "VAE+score+curvature+path_length")]
        for base, ext in pairs:
            for metric in ["auroc", "logloss", "brier"]:
                md, ci = paired_metric_diff_ci(x, oofs[base], oofs[ext], metric)
                rows.append({"generator": gen, "feature_set": f"Δ({ext} - {base})", "metric": metric,
                            "delta_mean": md, "delta_ci_lo": ci[0], "delta_ci_hi": ci[1]})
        print(gen, "part 5 done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "sdxl_pixart_feature_addition_comparison.csv", index=False)
    return df

# ---------------- Part 6: coefficient stability ----------------
def part6_coefficient_stability(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen); y = x.label.values
        for label, cols in [("VAE+curvature+path_length", VAE_FEATS + [CURV, PLEN]),
                            ("VAE+score+curvature+path_length", VAE_FEATS + SCORE_FEATS + [CURV, PLEN])]:
            X = np.nan_to_num(x[cols].values)
            c_idx, p_idx = cols.index(CURV), cols.index(PLEN)
            c_coefs, p_coefs = [], []
            for r in range(REPS):
                fo = fold_assignment(x, r)
                for k in range(FOLDS):
                    te = fo == k
                    pipe = make().fit(X[~te], y[~te])
                    lr = pipe.named_steps["logisticregression"]
                    c_coefs.append(float(lr.coef_[0][c_idx])); p_coefs.append(float(lr.coef_[0][p_idx]))
            c_coefs = np.array(c_coefs); p_coefs = np.array(p_coefs)
            rows.append({"generator": gen, "model": label, "feature": "curvature", "mean_std_coef": float(c_coefs.mean()),
                        "sd_std_coef": float(c_coefs.std(ddof=1)), "frac_same_sign": float((np.sign(c_coefs) == np.sign(c_coefs.mean())).mean())})
            rows.append({"generator": gen, "model": label, "feature": "path_length", "mean_std_coef": float(p_coefs.mean()),
                        "sd_std_coef": float(p_coefs.std(ddof=1)), "frac_same_sign": float((np.sign(p_coefs) == np.sign(p_coefs.mean())).mean())})
        print(gen, "part 6 done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "coefficient_stability.csv", index=False)
    return df

# ---------------- Part 7: curvature x path_length interaction ----------------
def part7_interaction(d):
    rows = []
    for gen in GENS:
        x = sub(d, gen); y = x.label.values
        c = x[CURV].values.astype(float); p = x[PLEN].values.astype(float)
        interaction = c * p
        base_sets = {"VAE+curvature+path_length": np.column_stack([np.nan_to_num(x[VAE_FEATS].values), c, p]),
                    "VAE+score+curvature+path_length": np.column_stack([np.nan_to_num(x[VAE_FEATS + SCORE_FEATS].values), c, p])}
        ext_sets = {"VAE+curvature+path_length": np.column_stack([np.nan_to_num(x[VAE_FEATS].values), c, p, interaction]),
                   "VAE+score+curvature+path_length": np.column_stack([np.nan_to_num(x[VAE_FEATS + SCORE_FEATS].values), c, p, interaction])}
        for name in base_sets:
            def oof_for(X):
                n = len(x); oof = np.zeros(n)
                for r in range(REPS):
                    fo = fold_assignment(x, r)
                    pr_ = np.zeros(n)
                    for k in range(FOLDS):
                        te = fo == k
                        pr_[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
                    oof += pr_ / REPS
                return oof
            oof_base = oof_for(base_sets[name]); oof_ext = oof_for(ext_sets[name])
            for metric in ["auroc", "logloss", "brier"]:
                md, ci = paired_metric_diff_ci(x, oof_base, oof_ext, metric)
                rows.append({"generator": gen, "base_model": name, "metric": metric, "delta_mean": md,
                            "delta_ci_lo": ci[0], "delta_ci_hi": ci[1]})
        print(gen, "part 7 done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "interaction_analysis.csv", index=False)
    return df

# ---------------- Parts 9/10: aMUSEd control + PixArt->aMUSEd path_length diagnostic ----------------
def part10_transfer_ablation(d):
    from synthimage.features.panel_v2 import CORE_FEATURE_NAMES
    def transfer(train_gen, test_gen, cols, reps=REPS, folds=FOLDS, seed=0):
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
                mdl = make().fit(np.nan_to_num(tr[cols].values), tr.label.values)
                p = mdl.predict_proba(np.nan_to_num(te[cols].values))[:, 1]
                preds.append(p); ys.append(te.label.values)
            y_all = np.concatenate(ys); p_all = np.concatenate(preds); aucs.append(roc_auc_score(y_all, p_all))
        return float(np.mean(aucs))

    full = CORE_FEATURE_NAMES
    no_plen = [f for f in CORE_FEATURE_NAMES if f != PLEN]
    no_traj = [f for f in CORE_FEATURE_NAMES if f not in (CURV, PLEN)]

    x = sub(d, "pixart_dit"); y = x.label.values
    X = np.nan_to_num(x[full].values); p_idx = full.index(PLEN)
    coefs = []
    for r in range(REPS):
        fo = fold_assignment(x, r)
        for k in range(FOLDS):
            te = fo == k
            pipe = make().fit(X[~te], y[~te])
            coefs.append(float(pipe.named_steps["logisticregression"].coef_[0][p_idx]))
    coefs = np.array(coefs)

    result = {"pixart_path_length_standardized_coef_mean": float(coefs.mean()), "pixart_path_length_standardized_coef_sd": float(coefs.std(ddof=1)),
             "transfer_pixart_to_amused_full_panel": transfer("pixart_dit", "amused", full),
             "transfer_pixart_to_amused_no_path_length": transfer("pixart_dit", "amused", no_plen),
             "transfer_pixart_to_amused_no_trajectory": transfer("pixart_dit", "amused", no_traj)}
    json.dump(result, open(OUT / "pixart_amused_path_length_diagnostic.json", "w"), indent=1)
    pd.DataFrame([result]).to_csv(OUT / "pixart_amused_path_length_diagnostic.csv", index=False)
    print(json.dumps(result, indent=1))
    return result

def main():
    rows = json.load(open("results/stage_decomposition/panel_features_dit.json"))
    d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": r["content_id"], **r["features"]} for r in rows])
    assert len(d) == 300 and d.groupby("content_id").size().eq(5).all(), "expected 300 rows, 5 per content_id"
    for c in VAE_FEATS + [CURV, PLEN] + SCORE_FEATS: assert d[c].notna().all(), f"NaN present in {c}"
    print("data sufficiency check passed:", len(d), "rows,", d.content_id.nunique(), "content ids, 0 NaN in required columns")

    r1 = part1_raw_effect(d); print("\nPart 1 (raw path_length effect):\n", r1.round(3).to_string(index=False))
    r2a = part2_correlations(d); print("\nPart 2 correlations done")
    effect_df, pred_records = part3_residualize(d); print("\nPart 3 (raw vs residual path_length):\n", effect_df.round(3).to_string(index=False))
    r2b = part2_regression_performance(d, pred_records); print("\nPart 2 regression performance:\n", r2b.round(3).to_string(index=False))
    r4 = part4_residual_trajectory_model(d, pred_records); print("\nPart 4 done")
    r5 = part5_feature_addition(d); print("\nPart 5 (SDXL vs PixArt central diagnostic) done")
    r6 = part6_coefficient_stability(d); print("\nPart 6:\n", r6.round(3).to_string(index=False))
    r7 = part7_interaction(d); print("\nPart 7 (interaction):\n", r7.round(4).to_string(index=False))
    r10 = part10_transfer_ablation(d)
    print("\nAll parts complete. Outputs in", OUT)

if __name__ == "__main__":
    main()
