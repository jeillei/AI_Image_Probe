"""Required plots for STAGE_DECOMPOSITION_RESULTS.md."""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN = Path("results/stage_decomposition"); OUT = IN / "plots"; OUT.mkdir(parents=True, exist_ok=True)
STAGE_COLORS = {"vae": "#4477aa", "score": "#66ccee", "trajectory": "#ccbb44", "roundtrip": "#ee6677"}
STAGE_ORDER = ["vae", "score", "trajectory", "roundtrip"]

uni = pd.read_csv(IN / "univariate_feature_results.csv")
stage = pd.read_csv(IN / "stage_level_results.csv")
incr = pd.read_csv(IN / "incremental_information.csv")
direction = pd.read_csv(IN / "cross_generator_direction.csv")

# 1. stage-wise effect-size comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, gen in zip(axes, ("sd15", "amused")):
    u = uni[uni.generator == gen].sort_values(["stage", "feature"])
    colors = [STAGE_COLORS[s] for s in u.stage]
    ax.barh(u.feature, u.paired_cohens_d, color=colors, xerr=[u.paired_cohens_d - u.d_ci_lo, u.d_ci_hi - u.paired_cohens_d])
    ax.axvline(0, color="k", lw=.7); ax.set_title(f"real vs {gen}"); ax.set_xlabel("paired Cohen's d")
fig.suptitle("Stage-wise effect size (10 core v2 features)"); fig.tight_layout()
fig.savefig(OUT / "01_effect_size_by_stage.png", dpi=140); plt.close(fig)

# 2. stage-wise AUROC with CI
fig, ax = plt.subplots(figsize=(6, 4))
w = 0.35
for i, gen in enumerate(("sd15", "amused")):
    s = stage[stage.generator == gen].set_index("stage").loc[STAGE_ORDER]
    x = np.arange(len(STAGE_ORDER)) + (i - 0.5) * w
    ax.bar(x, s.grouped_cv_auroc, width=w, label=gen, yerr=[s.grouped_cv_auroc - s.ci_lo, s.ci_hi - s.grouped_cv_auroc], capsize=3)
ax.set_xticks(range(len(STAGE_ORDER))); ax.set_xticklabels(STAGE_ORDER); ax.axhline(.5, color="k", ls="--", lw=.7)
ax.set_ylabel("content-grouped CV AUROC"); ax.legend(); ax.set_title("Stage-level AUROC (3-column panels)")
fig.tight_layout(); fig.savefig(OUT / "02_stage_auroc.png", dpi=140); plt.close(fig)

# 3. incremental-performance gain from each stage
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
cum_order = ["vae", "vae+score", "vae+score+trajectory", "vae+score+trajectory+roundtrip"]
for ax, gen in zip(axes, ("sd15", "amused")):
    i = incr[(incr.generator == gen) & incr.stage_set.isin(cum_order)].set_index("stage_set").loc[cum_order]
    ax.plot(range(4), i.grouped_cv_auroc, "o-", color="#333333")
    ax.set_xticks(range(4)); ax.set_xticklabels(["VAE", "+score", "+traj.", "+r.trip"], rotation=20)
    ax.axhline(.5, color="k", ls="--", lw=.7); ax.set_ylim(0.45, 1.02); ax.set_title(gen); ax.set_ylabel("grouped CV AUROC")
fig.suptitle("Incremental information across cumulative stage sets"); fig.tight_layout()
fig.savefig(OUT / "03_incremental_gain.png", dpi=140); plt.close(fig)

# 4. SD1.5 vs aMUSEd effect-direction comparison
fig, ax = plt.subplots(figsize=(6, 6))
colors = [STAGE_COLORS[s] for s in direction.stage]
ax.scatter(direction.d_sd15, direction.d_amused, c=colors, s=70)
for _, r in direction.iterrows(): ax.annotate(r.feature, (r.d_sd15, r.d_amused), fontsize=7, xytext=(3, 3), textcoords="offset points")
lim = max(direction.d_sd15.abs().max(), direction.d_amused.abs().max()) * 1.15
ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6); ax.plot([-lim, lim], [-lim, lim], "k:", lw=.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("paired Cohen's d (SD1.5)"); ax.set_ylabel("paired Cohen's d (aMUSEd)")
ax.set_title("Cross-generator effect direction (sign-agreement quadrants)")
fig.tight_layout(); fig.savefig(OUT / "04_cross_generator_direction.png", dpi=140); plt.close(fig)

print("core plots written to", OUT)

# 5. feature robustness under the small transformation pilot
retain = pd.read_csv(IN / "robustness_pilot_retained_effect.csv")
conds = [c for c in retain.condition.unique() if c != "clean"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, gen in zip(axes, ("sd15", "amused")):
    r = retain[(retain.generator == gen) & retain.condition.isin(conds)]
    piv = r.pivot_table(index="feature", columns="condition", values="paired_cohens_d")
    clean_d = retain[(retain.generator == gen) & (retain.condition == "clean")].set_index("feature").paired_cohens_d
    order = clean_d.abs().sort_values(ascending=False).index
    piv = piv.loc[order]
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30)
    ax.set_title(f"{gen} (rows ordered by |clean effect|)")
fig.colorbar(im, ax=axes, label="paired Cohen's d under transform", shrink=.8)
fig.suptitle("Phase 6: feature effect size retained under transformation (n=10 pilot)")
fig.savefig(OUT / "05_robustness_pilot.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# 6. AI-edit strength plot
edit = pd.read_csv(IN / "ai_edit_pilot_monotonicity.csv")
import json as _json
rows = _json.load(open(IN / "panel_features_ai_edit.json"))
ed = pd.DataFrame([{"content_id": r["content_id"], "strength": float(r["strength"]), **r["features"]} for r in rows])
top = edit.head(4).feature.tolist()
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, feat in zip(axes, top):
    for cid, g in ed.sort_values("strength").groupby("content_id"):
        ax.plot(g.strength, g[feat], "o-", alpha=.5, color="#4477aa")
    ax.set_title(feat, fontsize=9); ax.set_xlabel("img2img strength")
fig.suptitle("Phase 7 (exploratory, n=8): feature vs. degree of generative intervention")
fig.tight_layout(); fig.savefig(OUT / "06_ai_edit_strength.png", dpi=140); plt.close(fig)
print("all plots written")
