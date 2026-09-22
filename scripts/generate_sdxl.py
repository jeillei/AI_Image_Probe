"""Generate the third-generator (SDXL, final substitute for PixArt-Sigma after two independent infeasibility
blockers -- see PREREGISTRATION_pixart_v1.md) counterparts for the 60 existing content-matched COCO content ids.
Exact frozen configuration -- do not change after seeing outputs.  Resumable (skips existing files); one MPS
process.  Generator label used everywhere downstream: "sdxl"."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd, torch
from diffusers import StableDiffusionXLPipeline

STEPS = 25
GUIDANCE = 7.5
SIZE = 512  # amended from 1024 for machine-load reasons before further generation; see PREREGISTRATION_pixart_v1.md "Amendment"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/content_matched/sdxl"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev == "mps" else torch.float32
    m = pd.read_csv("data/content_matched/manifest.csv"); real = m[m.generator == "real"].sort_values("content_id")
    if a.limit: real = real.head(a.limit)
    pending = [r for r in real.itertuples() if not (out / f"{r.content_id}.png").exists()]
    print(f"{len(real) - len(pending)}/{len(real)} already done; {len(pending)} to generate")
    if not pending:
        return
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, use_safetensors=True).to(dev)
    for r in pending:
        seed = int(hashlib.sha256(f"sdxl:{r.content_id}".encode()).hexdigest()[:8], 16)
        gen = torch.Generator("cpu").manual_seed(seed)
        img = pipe(prompt=r.caption, negative_prompt="", height=SIZE, width=SIZE, num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE, generator=gen).images[0]
        img.convert("RGB").save(out / f"{r.content_id}.png")
        print("generated", r.content_id, flush=True)


if __name__ == "__main__":
    main()
