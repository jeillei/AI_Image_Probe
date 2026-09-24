"""Generate the fourth-generator (genuine PixArt-Sigma, a Diffusion Transformer) counterparts for the 60
existing content-matched COCO content ids.  Exact frozen configuration per PREREGISTRATION_DIT_GENERATOR_V1.md
-- do not change after seeing outputs.  Resumable (skips existing files); one MPS process.  Generator label used
everywhere downstream: "pixart_dit".

Uses a community GGUF-quantized T5-XXL text encoder (city96/t5-v1_1-xxl-encoder-gguf, Q5_K_M) in place of the
official fp32 T5 weights -- PixArt-Sigma's own transformer and VAE are official, unmodified. See the
preregistration for why this is a disclosed, validated adaptation and not a substitution of the denoiser.

TWO-PHASE structure (added after observing severe swap thrashing when the ~9.5GB de-quantized T5 encoder stayed
resident in memory for the entire 60-image denoising loop, competing with other processes on this machine's
unified memory). transformers' GGUF loader fully de-quantizes T5 to fp16 at load time (not lazily per forward
pass), so its memory cost is paid once regardless of how many images use it. Phase 1 encodes every caption's
(fixed, resolution-independent) text embedding ONCE, caches it to disk, then frees the encoder. Phase 2 loads
only the much smaller transformer+VAE and runs generation using the cached embeddings. This changes only *when*
text encoding happens, not any generation-quality parameter: identical prompts, identical embeddings (same
model, same weights, same call), identical seeds/steps/guidance/scheduler/resolution."""
from __future__ import annotations
import argparse, gc, hashlib, os
from pathlib import Path
import pandas as pd, torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import PixArtSigmaPipeline, Transformer2DModel, AutoencoderKL

# HF_HUB_OFFLINE=1 alone does not reliably make T5Tokenizer.from_pretrained() resolve
# its vocab file from the local cache in this transformers version (fails with
# "Either model_file or model_proto must be specified" -- vocab_file silently None);
# passing local_files_only=True explicitly on every from_pretrained call does work
# and is used whenever HF_HUB_OFFLINE=1 is set (the offline HPC execution path).
LOCAL_FILES_ONLY = bool(os.environ.get("HF_HUB_OFFLINE"))

STEPS = 20
GUIDANCE = 4.5
SIZE = 512
TRANSFORMER_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
T5_GGUF_REPO = "city96/t5-v1_1-xxl-encoder-gguf"
T5_GGUF_FILE = "t5-v1_1-xxl-encoder-Q5_K_M.gguf"
T5_TOKENIZER_ID = "google/t5-v1_1-xxl"
EMBED_CACHE = Path("data/content_matched/pixart_dit_embeds")


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def phase1_encode_captions(rows, dev):
    """Encode every unique caption + the single shared empty negative prompt once; cache to disk; free T5."""
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not (EMBED_CACHE / f"{r.content_id}.pt").exists()]
    neg_cached = (EMBED_CACHE / "_negative.pt").exists()
    if not missing and neg_cached:
        print("Phase 1: all caption embeddings already cached, skipping T5 load entirely.")
        return
    print(f"Phase 1: encoding {len(missing)} captions (+ shared negative prompt)")
    tok = T5Tokenizer.from_pretrained(T5_TOKENIZER_ID, local_files_only=LOCAL_FILES_ONLY)
    enc = T5EncoderModel.from_pretrained(T5_GGUF_REPO, gguf_file=T5_GGUF_FILE, torch_dtype=torch.float16,
                                          local_files_only=LOCAL_FILES_ONLY).to(dev)
    # Some diffusers versions unconditionally read self.vae.config.block_out_channels in
    # PixArtSigmaPipeline.__init__, so vae=None crashes the constructor even though encode_prompt()
    # never touches the VAE. Loading the real (tiny, 0.34GB) VAE here instead of passing None is a
    # constructor-compatibility workaround only -- it changes no embedding/generation-quality output,
    # and its memory cost is negligible next to the ~9.5GB T5 encoder this phase split exists to isolate.
    vae_stub = AutoencoderKL.from_pretrained(TRANSFORMER_ID, subfolder="vae", torch_dtype=torch.float16,
                                              local_files_only=LOCAL_FILES_ONLY)
    # borrow PixArtSigmaPipeline's own encode_prompt for exact parity with a normal call (same truncation/padding)
    tmp_pipe = PixArtSigmaPipeline.from_pretrained(TRANSFORMER_ID, text_encoder=enc, tokenizer=tok,
                                                   transformer=None, vae=vae_stub, torch_dtype=torch.float16,
                                                   local_files_only=LOCAL_FILES_ONLY)
    if not neg_cached:
        # any prompt works here -- only the negative_prompt="" outputs (index 2,3) are used/cached
        # device= passed explicitly: with transformer=None, the pipeline's automatic execution-device
        # detection falls back to CPU while the text_encoder itself was explicitly placed on dev,
        # causing a device-mismatch crash inside T5's embedding lookup.
        _, _, ne, na = tmp_pipe.encode_prompt(rows[0].caption, negative_prompt="", device=dev)
        torch.save({"embeds": ne.cpu(), "mask": na.cpu()}, EMBED_CACHE / "_negative.pt")
    for r in missing:
        pe, pa, _, _ = tmp_pipe.encode_prompt(r.caption, negative_prompt="", device=dev)
        torch.save({"embeds": pe.cpu(), "mask": pa.cpu()}, EMBED_CACHE / f"{r.content_id}.pt")
        print("encoded", r.content_id, flush=True)
    del tmp_pipe, enc, tok, vae_stub
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    print("Phase 1 done; T5 encoder freed.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/content_matched/pixart_dit"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = pick_device()
    m = pd.read_csv("data/content_matched/manifest.csv"); real = m[m.generator == "real"].sort_values("content_id")
    if a.limit: real = real.head(a.limit)
    rows = list(real.itertuples())
    pending = [r for r in rows if not (out / f"{r.content_id}.png").exists()]
    print(f"{len(rows) - len(pending)}/{len(rows)} images already done; {len(pending)} to generate")
    if not pending:
        return
    phase1_encode_captions(pending, dev)
    print("Phase 2: loading transformer+VAE only (no T5 in memory)")
    transformer = Transformer2DModel.from_pretrained(TRANSFORMER_ID, subfolder="transformer", torch_dtype=torch.float16,
                                                      local_files_only=LOCAL_FILES_ONLY)
    vae = AutoencoderKL.from_pretrained(TRANSFORMER_ID, subfolder="vae", torch_dtype=torch.float16,
                                         local_files_only=LOCAL_FILES_ONLY)
    tok = T5Tokenizer.from_pretrained(T5_TOKENIZER_ID, local_files_only=LOCAL_FILES_ONLY)  # tokenizer object is required by the pipeline constructor but not used (we pass embeds directly)
    pipe = PixArtSigmaPipeline.from_pretrained(TRANSFORMER_ID, transformer=transformer, vae=vae,
                                               text_encoder=None, tokenizer=tok, torch_dtype=torch.float16,
                                               local_files_only=LOCAL_FILES_ONLY).to(dev)
    neg = torch.load(EMBED_CACHE / "_negative.pt")
    for r in pending:
        cache = torch.load(EMBED_CACHE / f"{r.content_id}.pt")
        seed = int(hashlib.sha256(f"pixart_dit:{r.content_id}".encode()).hexdigest()[:8], 16)
        gen = torch.Generator("cpu").manual_seed(seed)
        # negative_prompt=None overrides __call__'s default negative_prompt="", which this diffusers
        # version's check_inputs() otherwise treats as conflicting with negative_prompt_embeds.
        img = pipe(prompt_embeds=cache["embeds"].to(dev), prompt_attention_mask=cache["mask"].to(dev),
                   negative_prompt=None,
                   negative_prompt_embeds=neg["embeds"].to(dev), negative_prompt_attention_mask=neg["mask"].to(dev),
                   height=SIZE, width=SIZE, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen).images[0]
        img.convert("RGB").save(out / f"{r.content_id}.png")
        print("generated", r.content_id, flush=True)


if __name__ == "__main__":
    main()
