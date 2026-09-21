"""E66-E72: evaluate acquisition-normalization pipelines and conditioning ablations on the frozen core-200 set.
Baseline = v1 native squash-resize + BLIP caption (already cached).  Partial conditions are reported as partial."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.analysis.evalkit import *
from src.analysis.families import COARSE, family
from src.analysis.metadata import META_COLS
OUT = Path("results/core200/analysis"); OUT.mkdir(parents=True, exist_ok=True)
core = pd.read_csv("data/core200/core200_frozen.csv", keep_default_na=False); paths = core.path.tolist()
base_all = load_features("results/features/stage_b_cross_source_rich.json"); base = base_all[base_all.path.isin(paths)].set_index("path").loc[paths].reset_index()
CONDS = {"BASELINE(v1 squash+BLIP)": None, "canon_crop256": "canon_crop256", "canon_crop256_jpeg75": "canon_crop256_jpeg75", "canon_band128": "canon_band128",
         "cond_null(empty)": "cond_null", "cond_generic('a photo')": "cond_generic", "cond_shuffled(other image's BLIP)": "cond_shuffled"}
def load_cond(name):
    if name is None: return base
    f = Path(f"results/core200/{name}.json")
    if not f.exists(): return None
    d = load_features(str(f)); d = d[d.path.isin(paths)].drop_duplicates("path").set_index("path"); n = len(d)
    if n < len(paths): print(f"  PARTIAL {name}: {n}/{len(paths)}"); 
    return d.reset_index() if n == len(paths) else None
def real_source_auc(d, cols, reps=5):
    r = d[d.label == 0].reset_index(drop=True); y = (r.source == "rrdataset_real").astype(int).values; a, oof = cv_binary(r[cols].values, y, .1, reps=reps); lo, hi = boot_auc(y, oof, 500); return a.mean(), lo, hi
rows = []; mats = {}
allf = feature_cols(base); groups = {"ALL_v1": allf, "legacy51": cols_for(base, COARSE["legacy51"]), **{k: cols_for(base, v) for k, v in COARSE.items() if k not in ("legacy51",)}}
gens = sorted(base[base.label == 1].generator.unique()); srcs = sorted(base[base.label == 0].source.unique())
meta = crossed_matrix(base, ["min_side", "aspect", "is_square", "is_pow2_side"], srcs, gens, n_boot=300); meta.to_csv(OUT / "crossed_metadata_geometry_core200.csv", index=False); ms = summarize_matrix(meta); ms.update({"condition": "metadata_geometry(4 features)", "feature_set": "metadata_geometry"}); rows.append(ms)
for cname, key in CONDS.items():
    d = load_cond(key)
    if d is None: print("skip (not yet complete):", cname); continue
    print("evaluating", cname)
    for gname in ["ALL_v1", "legacy51", "rich_alignment" if False else "trajectory_noise", "guidance", "latent_endpoint_stats", "spatial", "fft", "crosstime", "roundtrip" if False else "reconstruction_roundtrip"]:
        cols = groups[gname]; a, lo, hi = real_source_auc(d, cols); rec = {"condition": cname, "feature_set": gname, "real_source_auroc": a, "real_source_ci_lo": lo, "real_source_ci_hi": hi}
        if gname in ("ALL_v1", "legacy51", "guidance", "latent_endpoint_stats"):
            m = crossed_matrix(d, cols, srcs, gens, n_boot=300); m.to_csv(OUT / f"crossed_{key or 'baseline'}_{gname}.csv", index=False); rec.update(summarize_matrix(m)); mats[(cname, gname)] = m
        rows.append(rec)
res = pd.DataFrame(rows); res.to_csv(OUT / "summary.csv", index=False); print(res.round(3).to_string(index=False))
# caption-dependence per feature family: rank corr across images between BLIP baseline and each conditioning variant
dep = []
for cname, key in CONDS.items():
    if not (key or "").startswith("cond_"): continue
    d = load_cond(key)
    if d is None: continue
    for fam in sorted({family(c) for c in allf}):
        cs = [c for c in cols_for(base, [fam]) if base[c].std() > 0 and d[c].std() > 0]
        rc = [spearmanr(base[c], d[c]).statistic for c in cs]; dep.append({"variant": cname, "family": fam, "median_rank_corr_with_BLIP": float(np.nanmedian(rc)), "n": len(cs)})
if dep: dd = pd.DataFrame(dep); dd.to_csv(OUT / "caption_dependence_by_family.csv", index=False); print(dd.pivot(index="family", columns="variant", values="median_rank_corr_with_BLIP").round(2))
