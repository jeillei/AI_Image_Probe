"""Generate measured source-level acquisition table for DATASET_AUDIT.md (no hand-typed numbers)."""
import pandas as pd, numpy as np
m = pd.read_csv("results/audit/file_metadata.csv"); m["src"] = m.real_source.fillna(m.generator)
g = m.groupby("src").agg(n=("path", "size"), fmt=("format", lambda x: "/".join(sorted(set(x)))), min_side_med=("min_side", "median"), min_side_range=("min_side", lambda x: f"{x.min()}-{x.max()}"),
    square=("is_square", "mean"), pow2=("is_pow2_side", "mean"), qtab=("qtable_mean", "mean"), bpp_med=("bpp", "median"), label=("label", "first")).round(2)
g = g.sort_values(["label", "min_side_med"]); g.to_csv("results/audit/source_acquisition_table.csv")
with open("results/audit/source_acquisition_table.md", "w") as f:
    f.write("| source | label | n | format | median min-side | min-side range | frac square | frac pow2-sides | JPEG q-table mean | median bpp |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for s, r in g.iterrows(): f.write(f"| {s} | {'fake' if r.label else 'real'} | {r.n} | {r.fmt} | {r.min_side_med:.0f} | {r.min_side_range} | {r.square:.2f} | {r.pow2:.2f} | {'-' if np.isnan(r.qtab) else round(r.qtab,1)} | {r.bpp_med:.2f} |\n")
rr = pd.read_csv("results/audit/rr_partial_inventory.csv"); t = rr.groupby("cls").agg(n=("w", "size"), png=("fmt", lambda x: (x == "PNG").mean()), square=("w", lambda x: 0)).round(2)
t["square"] = rr.groupby("cls").apply(lambda x: (x.w == x.h).mean()).round(2); t["max_side"] = rr.groupby("cls").apply(lambda x: max(x.w.max(), x.h.max())); t["median_bytes"] = rr.groupby("cls").bytes.median()
t.to_csv("results/audit/rr_class_confounds.csv"); print(open("results/audit/source_acquisition_table.md").read()); print(t)
