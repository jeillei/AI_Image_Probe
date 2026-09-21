"""E64-E66: crossed (held real source x held generator) matrices, metadata shortcut baselines,
and a resize-matched (256-native, same-JPEG-state) control.  Cached features only."""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.analysis.evalkit import *
from src.analysis.families import COARSE
from src.analysis.metadata import META_COLS
def heat(m, title, path):
    piv = m.pivot(index="held_real", columns="held_generator", values="auroc"); fig, ax = plt.subplots(figsize=(9, 2.6 + .4 * len(piv)))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="RdBu", aspect="auto"); ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=40, ha="right"); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title); fig.colorbar(im, label="AUROC (0.5 = chance)"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", default="results/features/stage_b_cross_source_rich.json"); ap.add_argument("--out", default="results/audit/crossed"); ap.add_argument("--tag", default="v1_native"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); d = load_features(a.features)
    gens = sorted(d[d.label == 1].generator.unique()); srcs = sorted(d[d.label == 0].source.unique()); allf = feature_cols(d); summ = {}
    sets = {"ALL_v1": allf, "legacy51": cols_for(d, COARSE["legacy51"]), "metadata_only": META_COLS, "metadata_geometry": ["min_side", "aspect", "is_square", "is_pow2_side"]}
    if a.tag != "v1_native": sets = {"ALL_v1": allf, "legacy51": cols_for(d, COARSE["legacy51"])}
    for name, cols in sets.items():
        m = crossed_matrix(d, cols, srcs, gens); m.to_csv(out / f"crossed_{a.tag}_{name}.csv", index=False); summ[name] = summarize_matrix(m); heat(m, f"{a.tag} {name}: held real source x held generator", out / f"crossed_{a.tag}_{name}.png")
    # leave-one-generator-out with *random real folds* (classic protocol) for comparison
    json.dump(summ, open(out / f"crossed_{a.tag}_summary.json", "w"), indent=1); print(pd.DataFrame(summ).T.round(3).to_string())
main()
