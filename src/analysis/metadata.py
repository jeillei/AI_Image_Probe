"""Acquisition metadata (file-level) used ONLY as a shortcut baseline / diagnostic,
never as a feature of the SynthImage representation."""
from __future__ import annotations
import os
import numpy as np, pandas as pd
from PIL import Image

def file_meta(path: str) -> dict:
    im = Image.open(path); w, h = im.size; fmt = (im.format or "").lower()
    q = getattr(im, "quantization", None) or {}
    qmean = float(np.mean([np.mean(list(t)) for t in q.values()])) if q else np.nan  # higher = harsher JPEG
    b = os.path.getsize(path)
    return {"width": w, "height": h, "format": fmt, "bytes": b, "bpp": 8*b/(w*h), "qtable_mean": qmean,
            "min_side": min(w, h), "aspect": w/h, "is_png": int(fmt == "png"),
            "is_square": int(w == h), "is_pow2_side": int(w in (256, 512, 1024) and h in (256, 512, 1024))}

META_COLS = ["min_side", "aspect", "bpp", "qtable_mean", "is_png", "is_square", "is_pow2_side"]

def build(manifest_csv: str, out_csv: str) -> pd.DataFrame:
    m = pd.read_csv(manifest_csv)
    rows = [file_meta(p) for p in m.path]
    d = pd.concat([m[["image_id", "path", "label", "generator", "real_source"]], pd.DataFrame(rows)], axis=1)
    d.to_csv(out_csv, index=False); return d
