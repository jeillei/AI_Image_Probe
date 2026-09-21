"""Metadata-free, matched image-circulation corruption chains."""
from __future__ import annotations
import io
import random
from PIL import Image, ImageEnhance, ImageFilter

def _codec(img: Image.Image, fmt: str, quality: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, format=fmt, quality=quality, method=6 if fmt == "WEBP" else 0)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB").copy()

def _resize(img: Image.Image, scale: float, base: int) -> Image.Image:
    w, h = img.size
    side = max(24, int(min(w, h) * scale))
    a = img.resize((side, side), Image.Resampling.LANCZOS)
    return a.resize((base, base), Image.Resampling.LANCZOS)

def apply_chain(image: Image.Image, severity: int, seed: int, base_size: int = 256) -> Image.Image:
    """Apply randomized operations.  Caller supplies identically distributed seeds."""
    rng = random.Random(seed)
    img = image.convert("RGB").resize((base_size, base_size), Image.Resampling.LANCZOS)
    if severity == 0:
        return _codec(img, "JPEG", 95)  # normalize encoding for both classes
    n = {1: 1, 2: rng.randint(2, 3), 3: rng.randint(4, 6), 4: 7}[severity]
    ops = ["jpeg", "webp", "resample", "crop", "blur", "sharp", "colour", "screen"]
    for op in rng.sample(ops * 2, n):
        if op == "jpeg": img = _codec(img, "JPEG", rng.randint(45 if severity >= 3 else 70, 92))
        elif op == "webp": img = _codec(img, "WEBP", rng.randint(45 if severity >= 3 else 70, 90))
        elif op == "resample": img = _resize(img, rng.uniform(.35, .82), base_size)
        elif op == "crop":
            c = rng.uniform(.78, .96); k = int(base_size*c); x=rng.randint(0,base_size-k); y=rng.randint(0,base_size-k)
            img = img.crop((x,y,x+k,y+k)).resize((base_size,base_size),Image.Resampling.LANCZOS)
        elif op == "blur": img = img.filter(ImageFilter.GaussianBlur(rng.uniform(.25, 1.1)))
        elif op == "sharp": img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=rng.randint(50,130), threshold=3))
        elif op == "colour":
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(.9,1.1)); img=ImageEnhance.Color(img).enhance(rng.uniform(.9,1.1))
        else: img = _resize(img, rng.uniform(.72,.95), base_size)
    return _codec(img, "JPEG", rng.randint(50, 82) if severity >= 3 else 90)
