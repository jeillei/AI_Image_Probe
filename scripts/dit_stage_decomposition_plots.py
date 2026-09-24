"""Required plots for DIT_STAGE_DECOMPOSITION.md (genuine PixArt-Sigma DiT, fourth generator)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN = Path("results/stage_decomposition"); PX = IN / "pixart"; DT = IN / "dit_generator"; OUT = DT / "plots"; OUT.mkdir(parents=True, exist_ok=True)
STAGE_ORDER = ["vae", "score", "trajectory", "roundtrip"]
GENS = ["sd15", "amused", "sdxl", "pixart_dit"]
GEN_COLORS = {"sd15": "#4477aa", "amused": "#ee6677", "sdxl": "#228833", "pixart_dit": "#aa3377"}

stage_sd_am = pd.read_csv(IN / "stage_level_results.csv")
stage_sdxl = pd.read_csv(PX / "stage_level_results_sdxl.csv")
stage_dit = pd.read_csv(DT / "stage_level_results_dit.csv")
stage = pd.concat([stage_sd_am, stage_sdxl, stage_dit], ignore_index=True)
incr_sd_am = pd.read_csv(IN / "incremental_information.csv")
incr_sdxl = pd.read_csv(PX / "incremental_information_sdxl.csv")
incr_dit = pd.read_csv(DT / "incremental_information_dit.csv")
four = pd.read_csv(DT / "four_generator_feature_table.csv", index_col=0)
transfer = __import__("json").load(open(DT / "transfer_results.json"))

# 1. stage-wise AUROC across all four generators
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.2
for i, gen in enumerate(GENS):
    s = stage[stage.generator == gen].set_index("stage").reindex(STAGE_ORDER)
    x = np.arange(len(STAGE_ORDER)) + (i - 1.5) * w
    ax.bar(x, s.grouped_cv_auroc, width=w, label=gen, color=GEN_COLORS[gen],
          yerr=[s.grouped_cv_auroc - s.ci_lo, s.ci_hi - s.grouped_cv_auroc], capsize=3)
ax.set_xticks(range(len(STAGE_ORDER))); ax.set_xticklabels(STAGE_ORDER); ax.axhline(.5, color="k", ls="--", lw=.7)
ax.set_ylabel("content-grouped CV AUROC"); ax.legend(); ax.set_title("Stage-level AUROC, four generators")
fig.tight_layout(); fig.savefig(OUT / "01_stage_auroc_four_generators.png", dpi=140); plt.close(fig)

# 2. incremental AUROC gain with bootstrap CI (cumulative, all four generators)
cum_order = ["vae", "vae+score", "vae+score+trajectory", "vae+score+trajectory+roundtrip"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for gen, incr in (("sd15", incr_sd_am), ("amused", incr_sd_am), ("sdxl", incr_sdxl), ("pixart_dit", incr_dit)):
    i = incr[(incr.generator == gen) & incr.stage_set.isin(cum_order)].set_index("stage_set").reindex(cum_order)
    ax.plot(range(4), i.grouped_cv_auroc, "o-", color=GEN_COLORS[gen], label=gen)
ax.set_xticks(range(4)); ax.set_xticklabels(["VAE", "+score", "+traj.", "+r.trip"], rotation=20)
ax.axhline(.5, color="k", ls="--", lw=.7); ax.set_ylabel("grouped CV AUROC"); ax.legend()
ax.set_title("Incremental gain across stages, four generators")
fig.tight_layout(); fig.savefig(OUT / "02_incremental_gain_four_generators.png", dpi=140); plt.close(fig)

# 3. diffpath_curvature effect across all four generators
fig, ax = plt.subplots(figsize=(5.5, 4))
vals = [four.loc["diffpath_curvature", g] for g in GENS]
ax.bar(GENS, vals, color=[GEN_COLORS[g] for g in GENS])
ax.axhline(0, color="k", lw=.7); ax.set_ylabel("paired Cohen's d"); ax.set_title("diffpath_curvature effect by generator")
fig.tight_layout(); fig.savefig(OUT / "03_diffpath_curvature_four_generators.png", dpi=140); plt.close(fig)

# 4. full pairwise transfer matrix heatmap
mat = pd.DataFrame(index=GENS, columns=GENS, dtype=float)
for g in GENS: mat.loc[g, g] = np.nan
for k, v in transfer.items():
    tr, te = k.rsplit("_to_", 1)
    mat.loc[tr, te] = v
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(mat.values.astype(float), cmap="viridis", vmin=0.5, vmax=1.0)
ax.set_xticks(range(4)); ax.set_xticklabels(GENS, rotation=30, ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(GENS)
ax.set_xlabel("test generator"); ax.set_ylabel("train generator")
for i in range(4):
    for j in range(4):
        v = mat.values[i, j]
        if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.85 else "black", fontsize=9)
fig.colorbar(im, ax=ax, label="AUROC"); ax.set_title("Cross-generator transfer (train→test)")
fig.tight_layout(); fig.savefig(OUT / "04_transfer_matrix_four_generators.png", dpi=140); plt.close(fig)

print("plots written to", OUT)
