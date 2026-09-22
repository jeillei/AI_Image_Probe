"""Phase 8: interpret the raw-trajectory PCA components using the existing 652 v1 features -- AFTER, not
before, computing the components (raw trajectory -> discovered component -> correlate with named features,
never the reverse).  For each generator, project the (centered) delta matrix onto its own top-5 PCs (per
tensor type), then Spearman-correlate each PC's per-content score against the per-content DIFFERENCE
(fake-real) of every one of that generator's E88 top-30 handcrafted features (results/feature_audit/)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.analysis.raw_trajectory import load_cache, assert_triplets_complete, stack_tensor, TENSOR_KEYS, norm_raw, economy_svd

OUT = Path("results/raw_trajectory/projection"); OUT.mkdir(parents=True, exist_ok=True)
cache = load_cache("data/raw_trajectory_cache")
cids = sorted({cid for (g, cid) in cache if all((gg, cid) in cache for gg in ("real", "sd15", "amused"))})
assert_triplets_complete(cache, cids)

feat_rows = json.load(open("results/content_matched/v1_humancaption.json"))
manifest = pd.read_csv("data/content_matched/manifest.csv"); cid_of_path = dict(zip(manifest.path, manifest.content_id))
feat_df = pd.DataFrame([{"path": r["path"], "generator": r["generator"], "content_id": cid_of_path.get(r["path"], ""), **r["features"]} for r in feat_rows])

def feat_diff(gen: str, feature: str) -> np.ndarray:
    real = feat_df[(feat_df.generator == "real") & feat_df.content_id.isin(cids)].set_index("content_id").loc[cids, feature]
    fake = feat_df[(feat_df.generator == gen) & feat_df.content_id.isin(cids)].set_index("content_id").loc[cids, feature]
    return (fake.values - real.values).astype(float)

results = []
for gen, top30_file in [("sd15", "results/feature_audit/top30_sd15.csv"), ("amused", "results/feature_audit/top30_amused.csv")]:
    top_features = pd.read_csv(top30_file, index_col=0).index.tolist()
    real = stack_tensor(cache, "real", cids, "guidance")  # focus on 'guidance' tensor: the SD1.5-dominant family in E88 is
    fake = stack_tensor(cache, gen, cids, "guidance")     # rich_guidance_moments; aMUSEd's is rich_score_spatial/fft (scored below too)
    for tensor_key in TENSOR_KEYS:
        real_t = stack_tensor(cache, "real", cids, tensor_key); fake_t = stack_tensor(cache, gen, cids, tensor_key)
        D = norm_raw(fake_t, real_t); Dc = D - D.mean(0)
        U, S, Vt = economy_svd(Dc)
        for pc in range(min(5, Vt.shape[0])):
            score = Dc @ Vt[pc]  # per-content projection onto this PC
            for feature in top_features:
                r, p = spearmanr(score, feat_diff(gen, feature))
                results.append({"generator": gen, "tensor": tensor_key, "pc": pc + 1, "feature": feature, "spearman_r": float(r), "spearman_p": float(p)})
    print(gen, "done", flush=True)

df = pd.DataFrame(results); df.to_csv(OUT / "pc_feature_correlations.csv", index=False)
strong = df[df.spearman_r.abs() > 0.5].sort_values("spearman_r", key=abs, ascending=False)
strong.to_csv(OUT / "pc_feature_correlations_strong.csv", index=False)
print("\nSTRONG correlations (|r|>0.5):", len(strong))
print(strong.head(40).to_string(index=False))
# summary: best tensor/PC match per top feature
best = df.loc[df.groupby(["generator", "feature"]).spearman_r.apply(lambda s: s.abs().idxmax())]
best.to_csv(OUT / "pc_feature_best_match_per_feature.csv", index=False)
print("\nBest raw-PC match per top-30 feature:"); print(best.sort_values(["generator"]).to_string(index=False))
