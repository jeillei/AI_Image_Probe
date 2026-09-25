"""Canonical image loading for the frozen v2 feature panel. Every image -- real or generated, any
generator -- passes through this exact function before any feature is computed, applying an optional
realistic-transformation condition first (see synthimage.corruption.robustness_suite). This is the single
canonicalization path referenced throughout docs/METHODS.md."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageOps
from synthimage.corruption.robustness_suite import apply_condition


def load(path: str, size: int, seed: int, transform: str | None = None, transform_value=None) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if transform:
        im = apply_condition(im, transform, float(transform_value) if transform_value not in (None, "") else None, seed)
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
    else:
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 127.5 - 1
