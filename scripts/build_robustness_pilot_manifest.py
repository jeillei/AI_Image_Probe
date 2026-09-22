"""Phase 6: build a small, pre-declared robustness-pilot manifest.  10 content ids (deterministic, first-N by
content_id sort -- label/feature-blind selection, not picked to favor a result) x 3 generators x 5 conditions
(clean, jpeg_50, resize_0.5 downsample-upsample, blur_1.0, center_crop_0.8 -- matching this project's existing
src/corruption/robustness_suite.py CONDITIONS tuple exactly, chosen before any pilot result was seen)."""
from __future__ import annotations
import csv, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.corruption.robustness_suite import condition_id

N_CONTENT = 10
CONDITIONS = [("clean", None), ("jpeg", 50), ("resize", 0.5), ("blur", 1.0), ("center_crop", 0.8)]


def seed(text): return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def main():
    m = pd.read_csv("data/content_matched/manifest.csv")
    content_ids = sorted(m.content_id.unique())[:N_CONTENT]
    rows = []
    for _, r in m[m.content_id.isin(content_ids)].iterrows():
        for name, value in CONDITIONS:
            rows.append({"path": r.path, "label": r.label, "generator": r.generator, "content_id": r.content_id,
                        "caption": r.caption, "transform": name if name != "clean" else "", "transform_value": "" if value is None else value,
                        "transform_seed": seed(f"{r.content_id}:{name}:{value}"), "condition_id": condition_id(name, value) if name != "clean" else "clean"})
    out = pd.DataFrame(rows)
    out.to_csv("data/robustness_pilot_v2/manifest.csv", index=False)
    print(len(out), "rows;", out.content_id.nunique(), "content ids;", out.generator.value_counts().to_dict(), out.condition_id.value_counts().to_dict())


if __name__ == "__main__":
    Path("data/robustness_pilot_v2").mkdir(parents=True, exist_ok=True)
    main()
