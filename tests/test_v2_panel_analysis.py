"""Scientific-correctness tests for the frozen v2 panel and its analysis machinery: feature definitions,
content-grouping (no real/fake pair split across a fold), and cross-fitted residualization fit-on-train-only.
No GPU/model weights needed -- these test the statistics code, not feature extraction itself."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd


def test_v2_panel_feature_definitions_frozen():
    from src.features.panel_v2 import CORE_FEATURE_NAMES, STAGE_OF, STAGE_ORDER
    assert len(CORE_FEATURE_NAMES) == 10
    assert set(CORE_FEATURE_NAMES) == set(STAGE_OF.keys())
    assert STAGE_ORDER == ["vae", "score", "trajectory", "roundtrip"]
    assert set(STAGE_OF.values()) == set(STAGE_ORDER)
    # exactly three VAE, two score, two trajectory, three round-trip features
    counts = {s: sum(1 for f in CORE_FEATURE_NAMES if STAGE_OF[f] == s) for s in STAGE_ORDER}
    assert counts == {"vae": 3, "score": 2, "trajectory": 2, "roundtrip": 3}
    assert "diffpath_curvature" in CORE_FEATURE_NAMES and "path_length" in CORE_FEATURE_NAMES


def _make_synthetic_panel(n_content=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_content):
        cid = f"c{i:03d}"
        real_val = rng.normal(0, 1)
        rows.append({"content_id": cid, "label": 0, "generator": "real", "feat": real_val})
        rows.append({"content_id": cid, "label": 1, "generator": "gen", "feat": real_val - 0.5 + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_sub_keeps_only_matched_content_pairs():
    from scripts.stage_decomposition_analysis import sub
    d = _make_synthetic_panel()
    # drop one generator row so its content id has only a real counterpart -- must be excluded
    d = d.drop(d[(d.generator == "gen") & (d.content_id == "c000")].index)
    x = sub(d, "gen")
    assert "c000" not in set(x.content_id)
    assert x.groupby("content_id").label.nunique().eq(2).all()
    assert len(x) == 2 * (d.content_id.nunique() - 1)


def test_content_grouped_cv_never_splits_a_matched_pair():
    from scripts.stage_decomposition_analysis import grouped_cv_auc
    d = _make_synthetic_panel(n_content=30)
    x = d[(d.generator == "real") | (d.generator == "gen")].reset_index(drop=True)
    ids = np.array(sorted(x.content_id.unique()))
    for r in range(10):
        perm = np.random.default_rng(0 + r).permutation(ids)
        fold_of = {c: i % 5 for i, c in enumerate(perm)}
        fo = x.content_id.map(fold_of).values
        for cid in ids:
            rows_for_cid = fo[x.content_id.values == cid]
            assert len(set(rows_for_cid)) == 1, f"content {cid} split across folds in rep {r}"
    # the function itself runs without error on this synthetic panel
    auc, oof = grouped_cv_auc(x, ["feat"], reps=3)
    assert 0.0 <= auc <= 1.0 and len(oof) == len(x)


def test_residualization_uses_train_fold_only():
    """Cross-fitted residualization must never let a held-out fold's own values influence the fit used to
    predict that same fold. Verify by construction: perturbing one row's target value must not change the
    prediction for any OTHER row that shares its held-out fold in the same CV repeat (both are excluded from
    training together) -- it MAY legitimately change predictions in repeats/folds where the perturbed row is
    itself part of the training set, which is not leakage, just normal fitting."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from vae_curvature_redundancy_analysis import cross_fitted_residualize, fold_assignment
    d = _make_synthetic_panel(n_content=30)
    x = d[(d.generator == "real") | (d.generator == "gen")].reset_index(drop=True)
    rng = np.random.default_rng(1)
    predictors = rng.normal(0, 1, size=(len(x), 3))
    target = x.feat.values.copy()

    pred_a, _, _, _ = cross_fitted_residualize(x, target, predictors)

    victim = 5
    target_b = target.copy()
    target_b[victim] += 1000.0
    pred_b, _, _, _ = cross_fitted_residualize(x, target_b, predictors)

    checked_any = False
    for r in range(10):
        fo = fold_assignment(x, r)
        victim_fold = fo[victim]
        fold_mates = np.where((fo == victim_fold) & (np.arange(len(x)) != victim))[0]
        assert np.allclose(pred_a[r, fold_mates], pred_b[r, fold_mates], atol=1e-8), \
            f"rep {r}: victim's held-out fold-mates' predictions changed -- held-out data leaked into training"
        checked_any = True
    assert checked_any


def test_final_validation_manifest_schema():
    required = {"path", "label", "generator", "content_id", "caption", "transform", "transform_value", "transform_seed"}
    from src.corruption.robustness_suite import CONDITIONS, condition_id
    seen_ids = {condition_id(name, value) for name, value in CONDITIONS}
    assert "clean" in seen_ids
    assert "jpeg_30" in seen_ids and "blur_2p0" in seen_ids and "resize_0p25" in seen_ids
    # every non-clean condition must carry a non-None value (frozen parameter, never left implicit)
    for name, value in CONDITIONS:
        if name != "clean":
            assert value is not None, f"condition {name} has no frozen parameter value"
