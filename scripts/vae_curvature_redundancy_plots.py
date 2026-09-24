"""Required plots for VAE_CURVATURE_REDUNDANCY.md. See ANALYSIS_PLAN_VAE_CURVATURE_REDUNDANCY.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_decomposition_analysis import sub

OUT = Path("results/vae_curvature_redundancy"); PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
GENS = ["sd15", "sdxl", "pixart_dit", "amused"]
GEN_COLORS = {"sd15": "#4477aa", "sdxl": "#228833", "pixart_dit": "#aa3377", "amused": "#ee6677"}
CURV = "diffpath_curvature"

rows = json.load(open("results/stage_decomposition/panel_features_dit.json"))
d = pd.DataFrame([{"path": r["path"], "label": r["label"], "generator": r["generator"], "content_id": r["content_id"], **r["features"]} for r in rows])

# 1. VAE (lpips_ae) vs curvature scatter, all four generators, real/generated distinguished
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
for ax, gen in zip(axes, GENS):
    x = sub(d, gen)
    real = x[x.label == 0]; fake = x[x.label == 1]
    ax.scatter(real.lpips_ae, real[CURV], s=18, color="gray", alpha=0.6, label="real")
    ax.scatter(fake.lpips_ae, fake[CURV], s=18, color=GEN_COLORS[gen], alpha=0.7, label=gen)
    ax.set_title(gen); ax.set_xlabel("lpips_ae"); ax.legend(fontsize=8)
axes[0].set_ylabel("diffpath_curvature")
fig.suptitle("VAE (lpips_ae) vs curvature, real vs generated")
fig.tight_layout(); fig.savefig(PLOTS / "01_vae_vs_curvature_scatter.png", dpi=140); plt.close(fig)

# 2. raw vs VAE-residualized effect size, all four generators
effect = pd.read_csv(OUT / "raw_vs_residual_effect_table.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.35
raw = effect[effect.curvature_variant == "raw"].set_index("generator").reindex(GENS)
res = effect[effect.curvature_variant == "vae_residualized"].set_index("generator").reindex(GENS)
xi = np.arange(len(GENS))
ax.bar(xi - w / 2, raw.paired_cohens_d, width=w, label="raw curvature", color="#888888",
      yerr=[raw.paired_cohens_d - raw.d_ci_lo, raw.d_ci_hi - raw.paired_cohens_d], capsize=3)
ax.bar(xi + w / 2, res.paired_cohens_d, width=w, label="VAE-residualized curvature", color="#cc6677",
      yerr=[res.paired_cohens_d - res.d_ci_lo, res.d_ci_hi - res.paired_cohens_d], capsize=3)
ax.axhline(0, color="k", lw=.7); ax.set_xticks(xi); ax.set_xticklabels(GENS); ax.set_ylabel("paired Cohen's d")
ax.legend(); ax.set_title("Raw vs VAE-residualized curvature effect size")
fig.tight_layout(); fig.savefig(PLOTS / "02_raw_vs_residual_effect.png", dpi=140); plt.close(fig)

# 3. cross-fitted predicted vs observed curvature, all four generators
pred_obs = pd.read_csv(OUT / "curvature_predicted_vs_observed.csv")
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=False, sharey=False)
for ax, gen in zip(axes, GENS):
    g = pred_obs[pred_obs.generator == gen]
    real = g[g.label == 0]; fake = g[g.label == 1]
    ax.scatter(real.curvature_observed, real.curvature_predicted_from_vae, s=18, color="gray", alpha=0.6, label="real")
    ax.scatter(fake.curvature_observed, fake.curvature_predicted_from_vae, s=18, color=GEN_COLORS[gen], alpha=0.7, label=gen)
    lo, hi = g.curvature_observed.min(), g.curvature_observed.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=.7)
    ax.set_title(gen); ax.set_xlabel("observed curvature"); ax.legend(fontsize=8)
axes[0].set_ylabel("predicted curvature (from VAE features, cross-fitted)")
fig.suptitle("Cross-fitted VAE-predicted vs observed curvature")
fig.tight_layout(); fig.savefig(PLOTS / "03_predicted_vs_observed_curvature.png", dpi=140); plt.close(fig)

# 4. incremental proper-scoring comparison: VAE vs VAE+curvature
scoring = pd.read_csv(OUT / "incremental_proper_scoring.csv")
base = scoring[scoring.feature_set == "VAE"].set_index("generator").reindex(GENS)
ext = scoring[scoring.feature_set == "VAE+curvature"].set_index("generator").reindex(GENS)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric, better in zip(axes, ["auroc", "logloss", "brier"], ["higher", "lower", "lower"]):
    xi = np.arange(len(GENS)); w = 0.35
    ax.bar(xi - w / 2, base[metric], width=w, label="VAE", color="#888888")
    ax.bar(xi + w / 2, ext[metric], width=w, label="VAE+curvature", color="#4477aa")
    ax.set_xticks(xi); ax.set_xticklabels(GENS, rotation=20); ax.set_title(f"{metric} ({better} = better)"); ax.legend(fontsize=8)
fig.suptitle("VAE vs VAE+curvature: proper scoring rules")
fig.tight_layout(); fig.savefig(PLOTS / "04_incremental_proper_scoring.png", dpi=140); plt.close(fig)

# 5. PixArt vs aMUSEd effect-vector plot (produced: it does help explain the 0.949 transfer)
diag = pd.read_csv(OUT / "pixart_amused_effect_vector_diagnostic.csv")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(diag.d_pixart_dit, diag.d_amused, s=40, color="#aa3377")
for _, row in diag.iterrows():
    ax.annotate(row.feature, (row.d_pixart_dit, row.d_amused), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
lims = [min(diag.d_pixart_dit.min(), diag.d_amused.min()) - 0.2, max(diag.d_pixart_dit.max(), diag.d_amused.max()) + 0.2]
ax.plot(lims, lims, "k--", lw=.5, alpha=.5)
ax.set_xlabel("paired Cohen's d, pixart_dit"); ax.set_ylabel("paired Cohen's d, amused")
ax.set_title("Effect-vector alignment: pixart_dit vs amused (10 features)")
fig.tight_layout(); fig.savefig(PLOTS / "05_pixart_amused_effect_vectors.png", dpi=140); plt.close(fig)

print("plots written to", PLOTS)
