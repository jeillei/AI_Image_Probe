"""Required plots for the final validation phase. See docs/research_history/FINAL_VALIDATION_PLAN.md."""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path("results/final_validation"); PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
GEN_COLORS = {"sd15": "#4477aa", "sdxl": "#228833"}

# 1. path_length effect retention across all 14 transform conditions, SD1.5 + SDXL
q1 = pd.read_csv(OUT / "q1_feature_survival.csv")
p = q1[(q1.feature == "path_length") & (q1.generator.isin(["sd15", "sdxl"])) & (q1.condition != "clean")]
conds = sorted(p.condition.unique())
fig, ax = plt.subplots(figsize=(11, 5))
w = 0.35; xi = np.arange(len(conds))
for i, gen in enumerate(["sd15", "sdxl"]):
    g = p[p.generator == gen].set_index("condition").reindex(conds)
    off = (i - 0.5) * w
    ax.bar(xi + off, g.d_transformed, width=w, label=gen, color=GEN_COLORS[gen],
          yerr=[g.d_transformed - g.d_ci_lo, g.d_ci_hi - g.d_transformed], capsize=2)
d_clean_sd15 = q1[(q1.feature == "path_length") & (q1.generator == "sd15") & (q1.condition == "clean")].d_clean.iloc[0]
d_clean_sdxl = q1[(q1.feature == "path_length") & (q1.generator == "sdxl") & (q1.condition == "clean")].d_clean.iloc[0]
ax.axhline(d_clean_sd15, color=GEN_COLORS["sd15"], ls="--", lw=1, alpha=.6, label="sd15 clean")
ax.axhline(d_clean_sdxl, color=GEN_COLORS["sdxl"], ls="--", lw=1, alpha=.6, label="sdxl clean")
ax.axhline(0, color="k", lw=.7)
ax.set_xticks(xi); ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("paired Cohen's d"); ax.legend(fontsize=8, ncol=2)
ax.set_title("path_length effect under realistic transformations (dashed = clean baseline)")
fig.tight_layout(); fig.savefig(PLOTS / "01_path_length_robustness.png", dpi=140); plt.close(fig)

# 2. AI-edit response curves: static (lpips_ae) vs dynamical (curvature, path_length), normalized
curves = pd.read_csv(OUT / "track_b_population_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
for feat, color in [("lpips_ae", "#4477aa"), ("diffpath_curvature", "#aa3377"), ("path_length", "#cc6633")]:
    g = curves[curves.feature == feat].sort_values("strength")
    ax.plot(g.strength, g["mean"], "o-", color=color, label=feat)
    ax.fill_between(g.strength, g.ci_lo, g.ci_hi, color=color, alpha=0.15)
ax.set_xlabel("img2img edit strength"); ax.set_ylabel("raw feature value"); ax.legend(fontsize=8)
ax.set_title("Raw response vs. edit strength")

ax = axes[1]
for feat, color in [("lpips_ae", "#4477aa"), ("diffpath_curvature", "#aa3377"), ("path_length", "#cc6633")]:
    g = curves[curves.feature == feat].sort_values("strength")
    lo, hi = g["mean"].min(), g["mean"].max()
    norm = (g["mean"] - lo) / (hi - lo + 1e-12)
    ax.plot(g.strength, norm, "o-", color=color, label=feat)
ax.set_xlabel("img2img edit strength"); ax.set_ylabel("normalized response (clean→heaviest-edit scale)")
ax.set_title("Normalized comparison (visualization only)"); ax.legend(fontsize=8)
fig.suptitle("Static (VAE) vs. dynamical (trajectory) response to increasing AI-edit strength")
fig.tight_layout(); fig.savefig(PLOTS / "02_ai_edit_response_comparison.png", dpi=140); plt.close(fig)

print("plots written to", PLOTS)
