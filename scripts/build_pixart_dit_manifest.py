"""Extend data/caption_matched/manifest.csv with the 60 pixart_dit rows (genuine PixArt-Sigma DiT), reusing
exactly the same content_id/caption as the existing real/sd15/amused/sdxl rows -- no new captioning, no prompt
changes."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

m = pd.read_csv("data/caption_matched/manifest.csv")
real = m[m.generator == "real"].sort_values("content_id")
rows = []
for r in real.itertuples():
    p = Path("data/caption_matched/pixart_dit") / f"{r.content_id}.png"
    if not p.exists():
        continue
    rows.append({"path": str(p), "label": 1, "generator": "pixart_dit", "content_id": r.content_id, "split": "",
                "caption": r.caption, "real_source": "", "transform_seed": 17})
out = pd.DataFrame(rows)
existing = m[m.generator != "pixart_dit"]
full = pd.concat([existing, out], ignore_index=True)
full.to_csv("data/caption_matched/manifest.csv", index=False)
print(len(out), "pixart_dit rows added;", len(full), "total rows;", full.generator.value_counts().to_dict())
