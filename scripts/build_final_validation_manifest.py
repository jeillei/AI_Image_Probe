"""Track A of docs/research_history/FINAL_VALIDATION_PLAN.md: build the full-scale transformation-robustness manifest. All 60 matched
content ids x {real, sd15, sdxl} (primary) x every non-clean condition in
src/corruption/robustness_suite.py::CONDITIONS (frozen, unchanged). No new image files -- transforms are applied
on-the-fly from the original clean paths by scripts/extract_detector_features.py::load(), exactly as the existing
n=10 robustness pilot manifest already does at smaller scale."""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.corruption.robustness_suite import CONDITIONS, condition_id

PRIMARY_GENERATORS = ["real", "sd15", "sdxl"]
SECONDARY_GENERATORS = ["amused", "pixart_dit"]


def seed(text): return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def main():
    m = pd.read_csv("data/content_matched/manifest.csv")
    non_clean = [(name, value) for name, value in CONDITIONS if name != "clean"]
    rows = []
    for gen in PRIMARY_GENERATORS + SECONDARY_GENERATORS:
        g = m[m.generator == gen].sort_values("content_id")
        for _, r in g.iterrows():
            for name, value in non_clean:
                rows.append({"path": r.path, "label": r.label, "generator": r.generator, "content_id": r.content_id,
                            "caption": r.caption, "transform": name, "transform_value": value,
                            "transform_seed": seed(f"{r.content_id}:{name}:{value}"), "condition_id": condition_id(name, value)})
    out = pd.DataFrame(rows)
    Path("data/final_validation").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/final_validation/track_a_manifest.csv", index=False)
    print(len(out), "rows;", out.content_id.nunique(), "content ids;", out.generator.value_counts().to_dict())
    print(out.condition_id.nunique(), "conditions:", sorted(out.condition_id.unique()))


if __name__ == "__main__":
    main()
