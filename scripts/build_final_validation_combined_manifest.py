"""Combine Track A (transformation robustness) and Track B (scaled AI-edit) manifests into one manifest for the
frozen SD1.5-probe extractor. Track B's strength=0.0 rows are dropped (they are the unedited real images,
already fully extracted and cached under generator="real"/transform="clean") rather than re-extracted --
resumable, avoids redundant compute. Track B's label is corrected here (the strength>0 rows are always
label=1/generated; the pilot script's own manifest.csv sets label=1 unconditionally, which is only correct for
strength>0)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

track_a = pd.read_csv("data/final_validation/track_a_manifest.csv")

track_b_raw = pd.read_csv("data/ai_edit_scaled/manifest.csv")
track_b = track_b_raw[track_b_raw.strength > 0].copy()
track_b["label"] = 1
track_b["transform"] = ""
track_b["transform_value"] = ""
track_b["transform_seed"] = 17
track_b["condition_id"] = track_b["strength"].apply(lambda s: f"edit_s{s}")
track_b = track_b[["path", "label", "generator", "content_id", "caption", "transform", "transform_value", "transform_seed", "condition_id"]]

combined = pd.concat([track_a, track_b], ignore_index=True)
Path("data/final_validation").mkdir(parents=True, exist_ok=True)
combined.to_csv("data/final_validation/combined_manifest.csv", index=False)
print(len(track_a), "Track A rows +", len(track_b), "Track B rows =", len(combined), "total")
print(combined.generator.value_counts().to_dict())
