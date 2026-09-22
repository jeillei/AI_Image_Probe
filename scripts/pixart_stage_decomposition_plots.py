"""Required plots for PIXART_STAGE_DECOMPOSITION.md (SDXL substitute)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN = Path("results/stage_decomposition"); PX = IN / "pixart"; OUT = PX / "plots"; OUT.mkdir(parents=True, exist_ok=True)
STAGE_ORDER = ["vae", "score", "trajectory", "roundtrip"]
GEN_COLORS = {"sd15": "#4477aa", "amused": "#ee6677", "sdxl": "#228833"}

stage_sd_am = pd.read_csv(IN / "stage_level_results.csv")
stage_sdxl = pd.read_csv(PX / "stage_level_results_sdxl.csv")
stage = pd.concat([stage_sd_am, stage_sdxl], ignore_index=True)
incr_sd_am = pd.read_csv(IN / "incremental_information.csv")
incr_sdxl = pd.read_csv(PX / "incremental_information_sdxl.csv")
three = pd.read_csv(PX / "three_generator_feature_table.csv", index_col=0)

# 1. stage-wise AUROC across all three generators
fig, ax = plt.subplots(figsize=(7, 4.2))
w = 0.25
for i, gen in enumerate(("sd15", "amused", "sdxl")):
    s = stage[stage.generator == gen].set_index("stage").reindex(STAGE_ORDER)
    x = np.arange(len(STAGE_ORDER)) + (i - 1) * w
    ax.bar(x, s.grouped_cv_auroc, width=w, label=gen, color=GEN_COLORS[gen],
          yerr=[s.grouped_cv_auroc - s.ci_lo, s.ci_hi - s.grouped_cv_auroc], capsize=3)
ax.set_xticks(range(len(STAGE_ORDER))); ax.set_xticklabels(STAGE_ORDER); ax.axhline(.5, color="k", ls="--", lw=.7)
ax.set_ylabel("content-grouped CV AUROC"); ax.legend(); ax.set_title("Stage-level AUROC, three generators")
fig.tight_layout(); fig.savefig(OUT / "01_stage_auroc_three_generators.png", dpi=140); plt.close(fig)

# 2. incremental AUROC gain with bootstrap CI (cumulative, sdxl vs sd15 vs amused)
cum_order = ["vae", "vae+score", "vae+score+trajectory", "vae+score+trajectory+roundtrip"]
fig, ax = plt.subplots(figsize=(7, 4.5))
for gen, incr in (("sd15", incr_sd_am), ("amused", incr_sd_am), ("sdxl", incr_sdxl)):
    i = incr[(incr.generator == gen) & incr.stage_set.isin(cum_order)].set_index("stage_set").reindex(cum_order)
    ax.plot(range(4), i.grouped_cv_auroc, "o-", color=GEN_COLORS[gen], label=gen)
ax.set_xticks(range(4)); ax.set_xticklabels(["VAE", "+score", "+traj.", "+r.trip"], rotation=20)
ax.axhline(.5, color="k", ls="--", lw=.7); ax.set_ylabel("grouped CV AUROC"); ax.legend()
ax.set_title("Incremental gain across stages, three generators")
fig.tight_layout(); fig.savefig(OUT / "02_incremental_gain_three_generators.png", dpi=140); plt.close(fig)

# 3. diffpath_curvature effect across SD1.5, aMUSEd, SDXL
fig, ax = plt.subplots(figsize=(5, 4))
vals = [three.loc["diffpath_curvature", g] for g in ("sd15", "amused", "sdxl")]
ax.bar(["sd15", "amused", "sdxl"], vals, color=[GEN_COLORS[g] for g in ("sd15", "amused", "sdxl")])
ax.axhline(0, color="k", lw=.7); ax.set_ylabel("paired Cohen's d"); ax.set_title("diffpath_curvature effect by generator")
fig.tight_layout(); fig.savefig(OUT / "03_diffpath_curvature_three_generators.png", dpi=140); plt.close(fig)

print("plots written to", OUT)
