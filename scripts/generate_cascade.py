"""Generate the third-generator (Stable Cascade, substituted for infeasible PixArt-Sigma; see
PREREGISTRATION_pixart_v1.md) counterparts for the 60 existing content-matched COCO content ids.  Exact frozen
configuration -- do not change after seeing outputs.  Resumable (skips existing files); one MPS process."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd, torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline, StableCascadeUNet

PRIOR_STEPS = 20
PRIOR_GUIDANCE = 4.0
DECODER_STEPS = 10
DECODER_GUIDANCE = 0.0
SIZE = 512


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/content_matched/cascade"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = pd.read_csv("data/content_matched/manifest.csv"); real = m[m.generator == "real"].sort_values("content_id")
    if a.limit: real = real.head(a.limit)
    pending = [r for r in real.itertuples() if not (out / f"{r.content_id}.png").exists()]
    print(f"{len(real) - len(pending)}/{len(real)} already done; {len(pending)} to generate")
    if not pending:
        return
    prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior_lite", torch_dtype=torch.bfloat16)
    decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder="decoder_lite", torch_dtype=torch.bfloat16)
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet, torch_dtype=torch.bfloat16).to(dev)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet, torch_dtype=torch.bfloat16).to(dev)
    for r in pending:
        seed = int(hashlib.sha256(f"cascade:{r.content_id}".encode()).hexdigest()[:8], 16)
        gen = torch.Generator("cpu").manual_seed(seed)
        prior_out = prior(prompt=r.caption, negative_prompt="", height=SIZE, width=SIZE, guidance_scale=PRIOR_GUIDANCE,
                          num_images_per_prompt=1, num_inference_steps=PRIOR_STEPS, generator=gen)
        gen2 = torch.Generator("cpu").manual_seed(seed)  # deterministic decoder noise, independent draw from the same seed policy
        img = decoder(image_embeddings=prior_out.image_embeddings.to(torch.bfloat16), prompt=r.caption, negative_prompt="",
                      guidance_scale=DECODER_GUIDANCE, num_inference_steps=DECODER_STEPS, output_type="pil", generator=gen2).images[0]
        img.convert("RGB").save(out / f"{r.content_id}.png")
        print("generated", r.content_id, flush=True)


if __name__ == "__main__":
    main()
