"""Required plots for PATH_LENGTH_MECHANISM.md. See ANALYSIS_PLAN_PATH_LENGTH_MECHANISM.md."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path("results/path_length_mechanism"); PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
GENS = ["sd15", "sdxl", "pixart_dit", "amused"]
GEN_COLORS = {"sd15": "#4477aa", "sdxl": "#228833", "pixart_dit": "#aa3377", "amused": "#ee6677"}

# 1. raw vs VAE-residualized path_length effect, four generators
effect = pd.read_csv(OUT / "raw_vs_residual_path_length_effect.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.35
raw = effect[effect.variant == "raw"].set_index("generator").reindex(GENS)
res = effect[effect.variant == "vae_residualized"].set_index("generator").reindex(GENS)
xi = np.arange(len(GENS))
ax.bar(xi - w / 2, raw.paired_cohens_d, width=w, label="raw path_length", color="#888888",
      yerr=[raw.paired_cohens_d - raw.d_ci_lo, raw.d_ci_hi - raw.paired_cohens_d], capsize=3)
ax.bar(xi + w / 2, res.paired_cohens_d, width=w, label="VAE-residualized path_length", color="#4477aa",
      yerr=[res.paired_cohens_d - res.d_ci_lo, res.d_ci_hi - res.paired_cohens_d], capsize=3)
ax.axhline(0, color="k", lw=.7); ax.set_xticks(xi); ax.set_xticklabels(GENS); ax.set_ylabel("paired Cohen's d")
ax.legend(); ax.set_title("Raw vs VAE-residualized path_length effect size")
fig.tight_layout(); fig.savefig(PLOTS / "01_raw_vs_residual_path_length.png", dpi=140); plt.close(fig)

# 2. residual curvature vs residual path_length geometry, four generators
c_resid_df = pd.read_csv("results/vae_curvature_redundancy/curvature_predicted_vs_observed.csv")
p_resid_df = pd.read_csv(OUT / "path_length_predicted_vs_observed.csv")
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
for ax, gen in zip(axes, GENS):
    cr = c_resid_df[c_resid_df.generator == gen][["content_id", "label", "curvature_residual_vs_vae"]]
    pr = p_resid_df[p_resid_df.generator == gen][["content_id", "label", "path_length_residual_vs_vae"]]
    m = cr.merge(pr, on=["content_id", "label"])
    real = m[m.label == 0]; fake = m[m.label == 1]
    ax.scatter(real.curvature_residual_vs_vae, real.path_length_residual_vs_vae, s=18, color="gray", alpha=0.6, label="real")
    ax.scatter(fake.curvature_residual_vs_vae, fake.path_length_residual_vs_vae, s=18, color=GEN_COLORS[gen], alpha=0.7, label=gen)
    ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
    ax.set_title(gen); ax.set_xlabel("C_resid"); ax.legend(fontsize=8)
axes[0].set_ylabel("P_resid")
fig.suptitle("Residual trajectory geometry: curvature vs path_length after removing VAE-predictable structure")
fig.tight_layout(); fig.savefig(PLOTS / "02_residual_trajectory_geometry.png", dpi=140); plt.close(fig)

# 3. SDXL vs PixArt incremental contributions: curvature, path_length, both (on top of VAE)
r5 = pd.read_csv(OUT / "sdxl_pixart_feature_addition_comparison.csv")
deltas = r5[r5.feature_set.isin(["Δ(VAE+curvature - VAE)", "Δ(VAE+path_length - VAE)", "Δ(VAE+curvature+path_length - VAE)"]) & (r5.metric == "auroc")]
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["+curvature", "+path_length", "+both"]
xi = np.arange(len(labels)); w = 0.35
for i, gen in enumerate(["sdxl", "pixart_dit"]):
    g = deltas[deltas.generator == gen].set_index("feature_set").reindex(
        ["Δ(VAE+curvature - VAE)", "Δ(VAE+path_length - VAE)", "Δ(VAE+curvature+path_length - VAE)"])
    off = (i - 0.5) * w
    ax.bar(xi + off, g.delta_mean, width=w, label=gen, color=GEN_COLORS[gen],
          yerr=[g.delta_mean - g.delta_ci_lo, g.delta_ci_hi - g.delta_mean], capsize=3)
ax.axhline(0, color="k", lw=.7); ax.set_xticks(xi); ax.set_xticklabels(labels)
ax.set_ylabel("ΔAUROC vs VAE alone"); ax.legend(); ax.set_title("SDXL vs PixArt: incremental AUROC on top of VAE")
fig.tight_layout(); fig.savefig(PLOTS / "03_sdxl_pixart_incremental_contributions.png", dpi=140); plt.close(fig)

# 4. coefficient plot: curvature vs path_length standardized coefficients, VAE+curvature+path_length model
coef = pd.read_csv(OUT / "coefficient_stability.csv")
sub_coef = coef[coef.model == "VAE+curvature+path_length"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
w = 0.35; xi = np.arange(len(GENS))
curv = sub_coef[sub_coef.feature == "curvature"].set_index("generator").reindex(GENS)
plen = sub_coef[sub_coef.feature == "path_length"].set_index("generator").reindex(GENS)
ax.bar(xi - w / 2, curv.mean_std_coef, width=w, yerr=curv.sd_std_coef, label="curvature", color="#4477aa", capsize=3)
ax.bar(xi + w / 2, plen.mean_std_coef, width=w, yerr=plen.sd_std_coef, label="path_length", color="#cc6677", capsize=3)
ax.axhline(0, color="k", lw=.7); ax.set_xticks(xi); ax.set_xticklabels(GENS)
ax.set_ylabel("mean standardized coefficient (± fold SD)"); ax.legend()
ax.set_title("Conditional coefficients, VAE+curvature+path_length model")
fig.tight_layout(); fig.savefig(PLOTS / "04_coefficients.png", dpi=140); plt.close(fig)

print("plots written to", PLOTS)
