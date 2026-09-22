"""Phase 7 analysis: does any frozen v2 feature move monotonically with img2img edit strength (0=real,
0.3/0.6/0.9)?  n=8 content ids -- explicitly exploratory, no AUROC/classifier claim, Spearman
correlation(feature, strength) within each content id's own 4-point sequence, aggregated across content ids."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from src.features.panel_v2 import CORE_FEATURE_NAMES, STAGE_OF

OUT = Path("results/stage_decomposition"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = json.load(open("results/stage_decomposition/panel_features_ai_edit.json"))
    d = pd.DataFrame([{"content_id": r["content_id"], "strength": float(r["strength"]), **r["features"]} for r in rows])
    d = d.sort_values(["content_id", "strength"])
    print(d[["content_id", "strength"]].to_string())
    res = []
    for feat in CORE_FEATURE_NAMES:
        # per-content-id monotonic trend (Spearman over the 4-point strength sequence)
        per_content = []
        for cid, g in d.groupby("content_id"):
            if g.strength.nunique() < 4: continue
            r, p = spearmanr(g.strength, g[feat])
            per_content.append(r)
        # pooled: rank strength vs feature across ALL 32 points (mixed-content), a coarser aggregate check
        r_pooled, p_pooled = spearmanr(d.strength, d[feat])
        res.append({"feature": feat, "stage": STAGE_OF[feat], "mean_per_content_spearman": float(np.nanmean(per_content)),
                    "frac_content_monotonic_positive": float(np.mean(np.array(per_content) > 0)),
                    "pooled_spearman": float(r_pooled), "pooled_p": float(p_pooled), "n_content": len(per_content)})
    out = pd.DataFrame(res).sort_values("pooled_spearman", key=abs, ascending=False)
    out.to_csv(OUT / "ai_edit_pilot_monotonicity.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
