"""Acquisition-normalization pipelines (SynthImage v1 audit, E66+).

Each pipeline maps an EXIF-transposed RGB image of any size to a 256x256 RGB image using a
processing path that is identical for every source.  None uses labels, source or metadata.
The frozen v1 pipeline (squash-resize, no crop, no re-encode) is NOT modified; these are additive.
  canon_crop256       centre-crop to square (no aspect distortion) -> LANCZOS 256
  canon_crop256_jpeg75 as above, then one controlled JPEG re-encode (q=75, 4:2:0)
  canon_band128       centre-crop -> LANCZOS 128 -> BICUBIC 256  (equalises spectral support)
"""
from __future__ import annotations
import io
from PIL import Image
S = 256
def _crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size; k = min(w, h); l = (w - k) // 2; t = (h - k) // 2; return im.crop((l, t, l + k, t + k))
def _jpeg(im: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO(); im.save(b, format="JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB").copy()
def apply_canon(img: Image.Image, name: str) -> Image.Image:
    img = _crop_square(img.convert("RGB"))
    if name == "canon_crop256": return img.resize((S, S), Image.Resampling.LANCZOS)
    if name == "canon_crop256_jpeg75": return _jpeg(img.resize((S, S), Image.Resampling.LANCZOS), 75)
    if name == "canon_band128": return img.resize((S // 2, S // 2), Image.Resampling.LANCZOS).resize((S, S), Image.Resampling.BICUBIC)
    raise ValueError(f"unknown canonical pipeline {name}")
CANON = ("canon_crop256", "canon_crop256_jpeg75", "canon_band128")
