"""Summarize rich SD1.5 trajectory features without fitting a detector.

The 80-image checkpoint is a discovery/control cohort, so this script produces
generator-stratified descriptive tables and a small set of pre-specified curve
plots.  It deliberately does not rank individual columns by AUROC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURVES = {
    "score_norm": "rich_score_norm_t",
    "score_change": "rich_score_change_t",
    "score_alignment": "rich_cond_dz_align_t",
    "guidance_alignment": "rich_guidance_dz_align_t",
    "latent_step": "rich_latent_step_t",
    "score_fft_high_low": "rich_score_t{}_fft_high_low",
}


def values_for_curve(frame: pd.DataFrame, pattern: str) -> list[str]:
    if "{}" in pattern:
        cols = [pattern.format(i) for i in range(6)]
    else:
        cols = [f"{pattern}{i}" for i in range(6)]
    return [c for c in cols if c in frame]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/analysis/rich_stage_a")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text())
    flat = [{k: v for k, v in row.items() if k not in {"features", "caption"}} | row["features"] for row in rows]
    df = pd.DataFrame(flat)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = ["real"] + sorted(g for g in df.loc[df.label == 1, "generator"].unique())
    summaries: list[dict[str, object]] = []
    for curve, pattern in CURVES.items():
        cols = values_for_curve(df, pattern)
        for group in groups:
            subset = df[df.label.eq(0)] if group == "real" else df[df.generator.eq(group)]
            for step, col in enumerate(cols):
                x = subset[col].to_numpy(float)
                summaries.append({"curve": curve, "group": group, "step": step, "n": len(x), "mean": x.mean(), "median": np.median(x), "std": x.std(ddof=0)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "curve_summary.csv", index=False)
    for curve in CURVES:
        part = summary[summary.curve.eq(curve)]
        fig, ax = plt.subplots(figsize=(7, 4))
        for group in groups:
            q = part[part.group.eq(group)]
            if q.empty:
                continue
            sem = q["std"].to_numpy() / np.sqrt(q["n"].to_numpy())
            ax.plot(q.step, q["mean"], marker="o", label=group)
            ax.fill_between(q.step, q["mean"] - sem, q["mean"] + sem, alpha=.14)
        ax.set(title=curve.replace("_", " "), xlabel="inverse step", ylabel="mean ± SEM")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"{curve}.png", dpi=160)
        plt.close(fig)
    family_counts = pd.Series({"legacy": len([c for c in df if not c.startswith("rich_")]), "rich": len([c for c in df if c.startswith("rich_")])})
    (out / "metadata.json").write_text(json.dumps({"images": len(df), "groups": groups, "columns": int(len(df.columns)), "family_counts": family_counts.to_dict()}, indent=2))
    print(json.dumps(json.loads((out / "metadata.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
