"""Freeze the 200-image core set and write one extraction manifest per experimental condition.
Baseline (v1 native + BLIP) rows are already cached, so they are NOT re-extracted.
Selection is deterministic and label/feature-blind: robustness-screen originals + evenly strided extra reals."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.corruption.canonical import CANON
D = Path("data/core200"); D.mkdir(parents=True, exist_ok=True)
m = pd.read_csv("data/stage_b_cross_source/manifest.csv", keep_default_na=False)
rob = set(json.load(open("data/robustness_screen/manifest.csv.json"))["originals"]); core = m[m.image_id.isin(rob)].copy()
for src in ["aigc_benchmark_real", "rrdataset_real"]:
    pool = m[(m.real_source == src) & ~m.image_id.isin(rob)].sort_values("path"); core = pd.concat([core, pool.iloc[::max(1, len(pool) // 40)].head(40)])
core = core.sort_values("path").reset_index(drop=True); assert len(core) == 200, len(core)
core["source"] = np.where(core.label == 0, core.real_source, core.generator)
feats = {r["path"]: r for r in json.load(open("results/features/stage_b_cross_source_rich.json"))}
core["blip_caption"] = [feats[p]["caption"] for p in core.path]
core.to_csv(D / "core200_frozen.csv", index=False)
base = core[["path", "label", "generator", "content_id", "split"]].copy(); base["transform_seed"] = 17
def write(name, df): df.to_csv(D / f"{name}.csv", index=False); print(name, len(df))
for c in CANON: write(c, base.assign(transform=c, transform_value="", extractor_protocol_version=f"sd15_blip_256_ddim6_rich_v1__{c}"))
# conditioning ablation on the frozen v1-native image path (transform=clean now resizes identically to v1)
cb = base.assign(transform="clean", transform_value="")
write("cond_generic", cb.assign(caption="a photo", extractor_protocol_version="v1_native__cond_generic"))
rng = np.random.default_rng(17); n = len(core)
while True:
    perm = rng.permutation(n)
    if not (perm == np.arange(n)).any(): break
write("cond_shuffled", cb.assign(caption=core.blip_caption.values[perm], extractor_protocol_version="v1_native__cond_shuffled"))
write("cond_null", cb.assign(extractor_protocol_version="v1_native__cond_null"))   # run with --null-conditioning
print(core.source.value_counts().to_dict())
