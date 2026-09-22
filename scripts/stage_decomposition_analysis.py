"""Phases 3-5: per-feature univariate stats, stage-level AUROC, incremental-information comparisons, and
cross-generator direction consistency for the v2 literature-anchored panel.  Content-grouped throughout (60
triplets, never 180 independent rows). Simple StandardScaler+LogisticRegression(C=0.1) only -- no new model
family. Stage-0 thumbnail baseline is reused, not recomputed, from the prior phase's cached output."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features.panel_v2 import CORE_FEATURE_NAMES, STAGE_OF, STAGE_ORDER

OUT = Path("results/stage_decomposition"); OUT.mkdir(parents=True, exist_ok=True)
C = 0.1
REPS = 10
FOLDS = 5
N_BOOT = 1000
rng_global = np.random.default_rng(0)


def make():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=17))


def sub(d, gen):
    x = d[(d.label == 0) | (d.generator == gen)]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)


def cohen_paired(x, feat):
    pw = x.pivot_table(index="content_id", columns="label", values=feat)
    diff = (pw[1] - pw[0]).dropna()
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12)), diff


def boot_ci(vals, stat_fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    arr = [stat_fn(np.random.default_rng(seed + 1 + i).choice(vals, len(vals), replace=True)) for i in range(n)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def grouped_cv_auc(x, cols, reps=REPS, folds=FOLDS, seed=0):
    X = np.nan_to_num(x[cols].values); y = x.label.values; ids = np.array(sorted(x.content_id.unique()))
    oof = np.zeros(len(x)); aucs = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(ids)
        fo = x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values
        p = np.zeros(len(x))
        for k in range(folds):
            te = fo == k
            p[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p)); oof += p / reps
    return float(np.mean(aucs)), oof


def content_boot_ci(x, score, n=N_BOOT, seed=0):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}
    y = x.label.values; rng = np.random.default_rng(seed)
    vals = [roc_auc_score(y[ix], score[ix]) for ix in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_grouped_cv(x, cols_a, cols_b, reps=REPS, folds=FOLDS, seed=0):
    X_a = np.nan_to_num(x[cols_a].values); X_b = np.nan_to_num(x[cols_b].values); y = x.label.values
    ids = np.array(sorted(x.content_id.unique())); oof_a = np.zeros(len(x)); oof_b = np.zeros(len(x)); aucs_a = []; aucs_b = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(ids)
        fo = x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values
        pa = np.zeros(len(x)); pb = np.zeros(len(x))
        for k in range(folds):
            te = fo == k
            pa[te] = make().fit(X_a[~te], y[~te]).predict_proba(X_a[te])[:, 1]
            pb[te] = make().fit(X_b[~te], y[~te]).predict_proba(X_b[te])[:, 1]
        aucs_a.append(roc_auc_score(y, pa)); aucs_b.append(roc_auc_score(y, pb)); oof_a += pa / reps; oof_b += pb / reps
    return float(np.mean(aucs_a)), float(np.mean(aucs_b)), oof_a, oof_b


def paired_diff_ci(x, oof_a, oof_b, n=N_BOOT, seed=0):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}
    y = x.label.values; rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        diffs.append(roc_auc_score(y[ix], oof_b[ix]) - roc_auc_score(y[ix], oof_a[ix]))
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main():
    rows = json.load(open("results/stage_decomposition/panel_features.json"))
    d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": r["content_id"], **r["features"]} for r in rows])
    assert set(CORE_FEATURE_NAMES) <= set(d.columns), set(CORE_FEATURE_NAMES) - set(d.columns)
    assert len(d) == 180 and d.groupby("content_id").size().eq(3).all(), "expected 180 rows, 3 per content_id"
    for c in CORE_FEATURE_NAMES: assert d[c].notna().all(), f"NaN present in {c}"
    print("loaded", len(d), "rows,", d.content_id.nunique(), "content ids")

    # ---------------- Phase 3: per-feature univariate stats ----------------
    uni_rows = []
    for gen in ("sd15", "amused"):
        x = sub(d, gen)
        for feat in CORE_FEATURE_NAMES:
            d_val, diffs = cohen_paired(x, feat)
            lo_d, hi_d = boot_ci(diffs.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
            auc = roc_auc_score(x.label, x[feat]); auc_df = max(auc, 1 - auc)
            _, oof = grouped_cv_auc(x, [feat], reps=5)
            lo_a, hi_a = content_boot_ci(x, oof)
            uni_rows.append({"generator": gen, "feature": feat, "stage": STAGE_OF[feat], "paired_cohens_d": d_val,
                             "d_ci_lo": lo_d, "d_ci_hi": hi_d, "univariate_auroc_direction_free": auc_df,
                             "grouped_cv_auroc": roc_auc_score(x.label, oof), "auroc_ci_lo": lo_a, "auroc_ci_hi": hi_a,
                             "frac_pairs_fake_higher": float((diffs > 0).mean())})
        print(gen, "univariate done", flush=True)
    uni = pd.DataFrame(uni_rows); uni.to_csv(OUT / "univariate_feature_results.csv", index=False)

    # ---------------- Phase 3: stage-level AUROC ----------------
    stage_rows = []
    for gen in ("sd15", "amused"):
        x = sub(d, gen)
        for stage in STAGE_ORDER:
            cols = [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == stage]
            auc, oof = grouped_cv_auc(x, cols)
            lo, hi = content_boot_ci(x, oof)
            stage_rows.append({"generator": gen, "stage": stage, "n_features": len(cols), "grouped_cv_auroc": auc, "ci_lo": lo, "ci_hi": hi})
        print(gen, "stage-level done", flush=True)
    stage_df = pd.DataFrame(stage_rows); stage_df.to_csv(OUT / "stage_level_results.csv", index=False)

    # ---------------- Phase 4: incremental information ----------------
    cum = {"vae": [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "vae"]}
    cum["vae+score"] = cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "score"]
    cum["vae+score+trajectory"] = cum["vae+score"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "trajectory"]
    cum["vae+score+trajectory+roundtrip"] = cum["vae+score+trajectory"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "roundtrip"]
    incr_rows = []
    for gen in ("sd15", "amused"):
        x = sub(d, gen)
        names = list(cum); ooofs = {}
        for name in names:
            auc, oof = grouped_cv_auc(x, cum[name]); ooofs[name] = oof
            incr_rows.append({"generator": gen, "stage_set": name, "n_features": len(cum[name]), "grouped_cv_auroc": auc})
        for a, b in zip(names[:-1], names[1:]):
            md, ci = paired_diff_ci(x, ooofs[a], ooofs[b])
            incr_rows.append({"generator": gen, "stage_set": f"{b} vs {a} (paired diff)", "paired_diff_mean": md, "diff_ci_lo": ci[0], "diff_ci_hi": ci[1], "adds_information": bool(ci[0] > 0)})
        # targeted comparisons
        targets = [("vae", "vae+trajectory", cum["vae"], cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "trajectory"]),
                   ("vae", "vae+roundtrip", cum["vae"], cum["vae"] + [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "roundtrip"]),
                   ("score", "score+trajectory", [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] == "score"], [f for f in CORE_FEATURE_NAMES if STAGE_OF[f] in ("score", "trajectory")])]
        for na, nb, ca, cb in targets:
            auc_a, auc_b, oa, ob = paired_grouped_cv(x, ca, cb)
            md, ci = paired_diff_ci(x, oa, ob)
            incr_rows.append({"generator": gen, "stage_set": f"TARGETED: {nb} vs {na}", "grouped_cv_auroc": auc_b,
                              "baseline_auroc": auc_a, "paired_diff_mean": md, "diff_ci_lo": ci[0], "diff_ci_hi": ci[1], "adds_information": bool(ci[0] > 0)})
        print(gen, "incremental-info done", flush=True)
    incr_df = pd.DataFrame(incr_rows); incr_df.to_csv(OUT / "incremental_information.csv", index=False)

    # ---------------- Phase 5: cross-generator direction ----------------
    x_sd = sub(d, "sd15"); x_am = sub(d, "amused")
    dir_rows = []
    for feat in CORE_FEATURE_NAMES:
        d_sd, _ = cohen_paired(x_sd, feat); d_am, _ = cohen_paired(x_am, feat)
        dir_rows.append({"feature": feat, "stage": STAGE_OF[feat], "d_sd15": d_sd, "d_amused": d_am,
                         "sign_agree": bool(np.sign(d_sd) == np.sign(d_am))})
    dir_df = pd.DataFrame(dir_rows); dir_df.to_csv(OUT / "cross_generator_direction.csv", index=False)
    sign_agree_rate = dir_df.sign_agree.mean()
    # transfer: train on one generator's full panel, test on the other, content-grouped
    both = pd.concat([x_sd.assign(_g="sd15"), x_am.assign(_g="amused")])

    def transfer(train_gen, test_gen, reps=REPS, folds=FOLDS, seed=0):
        real = d[d.label == 0]; tr_fake = d[(d.label == 1) & (d.generator == train_gen)]; te_fake = d[(d.label == 1) & (d.generator == test_gen)]
        ids = np.array(sorted(set(real.content_id) & set(tr_fake.content_id) & set(te_fake.content_id)))
        aucs = []; oof = np.zeros(0)
        for r in range(reps):
            perm = np.random.default_rng(seed + r).permutation(ids); fo = {c: i % folds for i, c in enumerate(perm)}
            preds = []; ys = []
            for k in range(folds):
                tr_ids = [c for c in ids if fo[c] != k]; te_ids = [c for c in ids if fo[c] == k]
                tr = pd.concat([real[real.content_id.isin(tr_ids)], tr_fake[tr_fake.content_id.isin(tr_ids)]])
                te = pd.concat([real[real.content_id.isin(te_ids)], te_fake[te_fake.content_id.isin(te_ids)]])
                m = make().fit(np.nan_to_num(tr[CORE_FEATURE_NAMES].values), tr.label.values)
                p = m.predict_proba(np.nan_to_num(te[CORE_FEATURE_NAMES].values))[:, 1]
                preds.append(p); ys.append(te.label.values)
            y_all = np.concatenate(ys); p_all = np.concatenate(preds); aucs.append(roc_auc_score(y_all, p_all))
        return float(np.mean(aucs))
    transfer_sd_am = transfer("sd15", "amused"); transfer_am_sd = transfer("amused", "sd15")
    summary = {"sign_agreement_rate_10_features": float(sign_agree_rate), "transfer_sd15_to_amused_auroc": transfer_sd_am, "transfer_amused_to_sd15_auroc": transfer_am_sd}
    json.dump(summary, open(OUT / "cross_generator_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(uni.round(3).to_string(index=False))
    print(stage_df.round(3).to_string(index=False))
    print(incr_df.round(3).to_string(index=False))
    print(dir_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
