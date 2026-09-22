"""Required plots for RAW_TRAJECTORY_ANALYSIS.md: singular-value spectra, cumulative variance, principal-angle
curves, cross-generator projection-energy curves vs k, null distributions with observed statistic marked,
mean-difference-vector alignment, baseline-representation equivalents, and one illustrative 2D PCA scatter
(NOT primary evidence)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.analysis.raw_trajectory import load_cache, assert_triplets_complete, stack_tensor, norm_raw, economy_svd

IN = Path("results/raw_trajectory"); OUT = IN / "plots"; OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"z": "#4477aa", "cond": "#66ccee", "uncond": "#228833", "guidance": "#ccbb44", "dz": "#ee6677"}

spectra = pd.read_csv(IN / "svd" / "singular_value_spectra_raw.csv")
cross = pd.read_csv(IN / "svd" / "cross_generator_subspace_raw.csv")
pairbreak = pd.read_csv(IN / "nulls" / "pairbreaking_null_raw.csv")
randsub = pd.read_csv(IN / "nulls" / "randomsubspace_null_raw.csv")
paired = pd.read_csv(IN / "nulls" / "paired_cosine_summary.csv")
meancos = pd.read_csv(IN / "svd" / "mean_vector_cosine_raw.csv")
baselines = pd.read_csv(IN / "baselines" / "baseline_cross_generator_subspace.csv")
baseline_null = pd.read_csv(IN / "baselines" / "baseline_pairbreaking_null.csv")

# 1. singular value spectra
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, gen in zip(axes, ("sd15", "amused")):
    for key, c in COLORS.items():
        s = spectra[(spectra.tensor == key) & (spectra.generator == gen)]
        ax.plot(s.component, s.singular_value, "o-", color=c, label=key, ms=3)
    ax.set_title(f"real vs {gen}"); ax.set_xlabel("component"); ax.set_yscale("log")
axes[0].set_ylabel("singular value (log)"); axes[0].legend(fontsize=8)
fig.suptitle("Singular-value spectra of paired difference matrices (raw normalization)")
fig.tight_layout(); fig.savefig(OUT / "01_singular_value_spectra.png", dpi=140); plt.close(fig)

# 2. cumulative explained variance
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, gen in zip(axes, ("sd15", "amused")):
    for key, c in COLORS.items():
        s = spectra[(spectra.tensor == key) & (spectra.generator == gen)]
        ax.plot(s.component, s.cum_var_frac, "o-", color=c, label=key, ms=3)
    ax.axhline(.95, color="k", ls="--", lw=.7); ax.set_title(f"real vs {gen}"); ax.set_xlabel("component")
axes[0].set_ylabel("cumulative variance fraction"); axes[0].legend(fontsize=8)
fig.suptitle("Cumulative explained variance (within-generator PCA)")
fig.tight_layout(); fig.savefig(OUT / "02_cumulative_variance.png", dpi=140); plt.close(fig)

# 3. principal-angle curves vs k
fig, ax = plt.subplots(figsize=(6, 4))
for key, c in COLORS.items():
    s = cross[cross.tensor == key].sort_values("k")
    ax.plot(s.k, s.mean_principal_angle_deg, "o-", color=c, label=key)
ax.axhline(90, color="k", ls="--", lw=.7, label="90 deg = orthogonal")
ax.set_xlabel("k (subspace dimension)"); ax.set_ylabel("mean principal angle (deg)"); ax.legend(fontsize=8)
ax.set_title("Principal angles: SD1.5 vs aMUSEd difference subspaces")
fig.tight_layout(); fig.savefig(OUT / "03_principal_angles_vs_k.png", dpi=140); plt.close(fig)

# 4. cross-generator projection-energy curves vs k
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for key, c in COLORS.items():
    s = cross[cross.tensor == key].sort_values("k")
    axes[0].plot(s.k, s.projection_energy_AM_on_SD_basis, "o-", color=c, label=key)
    axes[1].plot(s.k, s.projection_energy_SD_on_AM_basis, "o-", color=c, label=key)
axes[0].set_title("aMUSEd projected onto SD1.5-derived basis"); axes[1].set_title("SD1.5 projected onto aMUSEd-derived basis")
for ax in axes: ax.set_xlabel("k"); ax.set_ylim(0, 1)
axes[0].set_ylabel("fraction of energy captured"); axes[0].legend(fontsize=8)
fig.suptitle("Phase 5: strict cross-generator projection test")
fig.tight_layout(); fig.savefig(OUT / "04_cross_projection_energy.png", dpi=140); plt.close(fig)

# 5. null distributions with observed marked (k=5, pair-breaking null, principal angle)
fig, axes = plt.subplots(1, len(COLORS), figsize=(16, 3.2), sharey=True)
for ax, (key, c) in zip(axes, COLORS.items()):
    row = pairbreak[(pairbreak.tensor == key) & (pairbreak.k == 5)].iloc[0]
    null = np.random.default_rng(0).normal(row.null_mean_deg, row.null_sd_deg, 3000)  # display approx (matches summary stats saved)
    ax.hist(null, bins=30, color="#bbbbbb"); ax.axvline(row.observed_mean_angle_deg, color="crimson", lw=2)
    ax.set_title(f"{key}\np={row.empirical_p_more_aligned_than_null:.3f}", fontsize=9)
axes[0].set_ylabel("count (pair-break null, k=5)")
fig.suptitle("Null 1 (pair-breaking): observed mean principal angle (red) vs null")
fig.tight_layout(); fig.savefig(OUT / "05_pairbreaking_null.png", dpi=140); plt.close(fig)

# random-subspace null
fig, axes = plt.subplots(1, len(COLORS), figsize=(16, 3.2), sharey=True)
for ax, (key, c) in zip(axes, COLORS.items()):
    row = randsub[(randsub.tensor == key) & (randsub.k == 5)]
    if len(row) == 0: continue
    row = row.iloc[0]
    null = np.random.default_rng(1).normal(row.random_subspace_mean, row.random_subspace_sd, 3000)
    ax.hist(null, bins=30, color="#bbbbbb"); ax.axvline(row.observed_projection_energy_AM_on_SD, color="crimson", lw=2)
    ax.set_title(f"{key}\np={row.empirical_p_exceeds_random:.3f}", fontsize=9)
axes[0].set_ylabel("count (random-subspace null, k=5)")
fig.suptitle("Null 3 (random subspace): observed AM-on-SD projection energy (red) vs random k=5 subspaces")
fig.tight_layout(); fig.savefig(OUT / "06_randomsubspace_null.png", dpi=140); plt.close(fig)

# 6. mean-difference-vector alignment (cosine) bar
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(meancos.tensor, meancos.mean_vector_cosine, color=[COLORS[k] for k in meancos.tensor])
ax.axhline(0, color="k", lw=.7); ax.set_ylabel("cosine(mean Delta_SD, mean Delta_AM)")
ax.set_title("Mean-difference-vector alignment"); fig.tight_layout(); fig.savefig(OUT / "07_mean_vector_cosine.png", dpi=140); plt.close(fig)

# 7. baseline representations equivalent (principal angle vs k)
fig, ax = plt.subplots(figsize=(6, 4))
reps = baselines.representation.unique()
for rep in reps:
    s = baselines[baselines.representation == rep].sort_values("k")
    ax.plot(s.k, s.mean_principal_angle_deg, "o-", label=rep)
guid = cross[cross.tensor == "guidance"].sort_values("k")
ax.plot(guid.k, guid.mean_principal_angle_deg, "ks--", label="raw trajectory: guidance (reference)")
ax.axhline(90, color="k", ls=":", lw=.7)
ax.set_xlabel("k"); ax.set_ylabel("mean principal angle (deg)"); ax.legend(fontsize=8)
ax.set_title("Phase 7: image-space / VAE / CIFAR-32 baselines vs raw trajectory")
fig.tight_layout(); fig.savefig(OUT / "08_baseline_comparison.png", dpi=140); plt.close(fig)

# 8. illustrative 2D PCA scatter (guidance tensor, SD1.5 vs aMUSEd deltas jointly) -- illustration only
from src.analysis.raw_trajectory import TENSOR_KEYS
cache = load_cache("data/raw_trajectory_cache")
cids = sorted({cid for (g, cid) in cache if all((gg, cid) in cache for gg in ("real", "sd15", "amused"))})
real = stack_tensor(cache, "real", cids, "guidance")
Dsd = norm_raw(stack_tensor(cache, "sd15", cids, "guidance"), real); Dam = norm_raw(stack_tensor(cache, "amused", cids, "guidance"), real)
both = np.vstack([Dsd, Dam]); both_c = both - both.mean(0)
U, S, Vt = economy_svd(both_c)
proj = both_c @ Vt[:2].T
fig, ax = plt.subplots(figsize=(5, 5))
n = len(cids)
ax.scatter(*proj[:n].T, color="#4477aa", label="SD1.5 - real", alpha=.7)
ax.scatter(*proj[n:].T, color="#ee6677", label="aMUSEd - real", alpha=.7)
ax.set_xlabel("joint PC1"); ax.set_ylabel("joint PC2"); ax.legend()
ax.set_title("Illustration only (not evidence): joint 2D PCA of guidance deltas")
fig.tight_layout(); fig.savefig(OUT / "09_illustrative_2dpca.png", dpi=140); plt.close(fig)
print("plots written to", OUT)
