"""Extend data/content_matched/manifest.csv with the 60 SDXL rows (once generated), reusing exactly the same
content_id/caption as the existing real/sd15/amused rows -- no new captioning, no prompt changes."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

m = pd.read_csv("data/content_matched/manifest.csv")
real = m[m.generator == "real"].sort_values("content_id")
rows = []
for r in real.itertuples():
    p = Path("data/content_matched/sdxl") / f"{r.content_id}.png"
    if not p.exists():
        continue
    rows.append({"path": str(p), "label": 1, "generator": "sdxl", "content_id": r.content_id, "split": "",
                "caption": r.caption, "real_source": "", "transform_seed": 17})
out = pd.DataFrame(rows)
existing = m[m.generator != "sdxl"]
full = pd.concat([existing, out], ignore_index=True)
full.to_csv("data/content_matched/manifest.csv", index=False)
print(len(out), "sdxl rows added;", len(full), "total rows;", full.generator.value_counts().to_dict())
