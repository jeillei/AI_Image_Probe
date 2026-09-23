"""Generate the fourth-generator (genuine PixArt-Sigma, a Diffusion Transformer) counterparts for the 60
existing content-matched COCO content ids.  Exact frozen configuration per PREREGISTRATION_DIT_GENERATOR_V1.md
-- do not change after seeing outputs.  Resumable (skips existing files); one MPS process.  Generator label used
everywhere downstream: "pixart_dit".

Uses a community GGUF-quantized T5-XXL text encoder (city96/t5-v1_1-xxl-encoder-gguf, Q5_K_M) in place of the
official fp32 T5 weights -- PixArt-Sigma's own transformer and VAE are official, unmodified. See the
preregistration for why this is a disclosed, validated adaptation and not a substitution of the denoiser."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd, torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import PixArtSigmaPipeline

STEPS = 20
GUIDANCE = 4.5
SIZE = 512
TRANSFORMER_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
T5_GGUF_REPO = "city96/t5-v1_1-xxl-encoder-gguf"
T5_GGUF_FILE = "t5-v1_1-xxl-encoder-Q5_K_M.gguf"
T5_TOKENIZER_ID = "google/t5-v1_1-xxl"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/content_matched/pixart_dit"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = pd.read_csv("data/content_matched/manifest.csv"); real = m[m.generator == "real"].sort_values("content_id")
    if a.limit: real = real.head(a.limit)
    pending = [r for r in real.itertuples() if not (out / f"{r.content_id}.png").exists()]
    print(f"{len(real) - len(pending)}/{len(real)} already done; {len(pending)} to generate")
    if not pending:
        return
    tok = T5Tokenizer.from_pretrained(T5_TOKENIZER_ID)
    enc = T5EncoderModel.from_pretrained(T5_GGUF_REPO, gguf_file=T5_GGUF_FILE, torch_dtype=torch.float16)
    pipe = PixArtSigmaPipeline.from_pretrained(TRANSFORMER_ID, text_encoder=enc, tokenizer=tok, torch_dtype=torch.float16).to(dev)
    # NOTE: enable_model_cpu_offload() was tried and reverted -- it assumes a discrete GPU with separate VRAM
    # to offload *from*; on Apple Silicon's unified memory, "CPU" and "MPS" share the same physical pool, so
    # offloading adds shuffling overhead without freeing anything, and measurably made swap usage worse (19GB
    # used, up from 5GB baseline) rather than better. See PREREGISTRATION_DIT_GENERATOR_V1.md's amendment log.
    for r in pending:
        seed = int(hashlib.sha256(f"pixart_dit:{r.content_id}".encode()).hexdigest()[:8], 16)
        gen = torch.Generator("cpu").manual_seed(seed)
        img = pipe(prompt=r.caption, negative_prompt="", height=SIZE, width=SIZE, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen).images[0]
        img.convert("RGB").save(out / f"{r.content_id}.png")
        print("generated", r.content_id, flush=True)


if __name__ == "__main__":
    main()
