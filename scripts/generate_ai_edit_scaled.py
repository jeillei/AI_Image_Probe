"""Track B of docs/research_history/FINAL_VALIDATION_PLAN.md: scale the n=8 img2img-strength pilot (scripts/generate_ai_edit_pilot.py)
to the full 60 matched content ids. Reuses the EXACT same img2img mechanism, strengths, steps, guidance, and
seed policy as the original pilot -- no new editing model, no parameter change. This is a controlled-intervention
scaling, not a new pilot; see STAGE_DECOMPOSITION_RESULTS.md Phase 7 for why img2img-strength (not localized
inpainting) was chosen and disclosed as a scope limitation, unchanged here."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from diffusers import DDIMScheduler

STRENGTHS = (0.3, 0.6, 0.9)
STEPS = 25
GUIDANCE = 7.5

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def img2img(probe, image_hwc: np.ndarray, prompt: str, strength: float, seed: int) -> np.ndarray:
    sched = DDIMScheduler.from_config(probe.pipe.scheduler.config)
    sched.set_timesteps(STEPS, device=probe.device)
    z0 = probe.encode_image(image_hwc)
    init_step = int(STEPS * strength)
    t_start = max(STEPS - init_step, 0)
    timesteps = sched.timesteps[t_start:]
    gen = torch.Generator("cpu").manual_seed(seed)
    noise = torch.randn(z0.shape, generator=gen, dtype=torch.float32).to(probe.device, probe.dtype)
    z = sched.add_noise(z0, noise, timesteps[:1]) if len(timesteps) else z0.clone()
    p, n = probe.embeds(prompt, "caption")
    for t in timesteps:
        model_input = torch.cat([z, z]); emb = torch.cat([n, p])
        with torch.inference_mode():
            out = probe.pipe.unet(model_input, t[None].expand(2), encoder_hidden_states=emb).sample
        eu, ec = out.chunk(2)
        eps = eu + GUIDANCE * (ec - eu)
        z = sched.step(eps, t, z).prev_sample
    return probe.decode_latent(z)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/ai_edit_scaled"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv("data/content_matched/manifest.csv")
    real = m[m.generator == "real"].sort_values("content_id")
    if a.limit: real = real.head(a.limit)
    from synthimage.probes.sd15 import SD15Probe
    probe = SD15Probe(6, device=pick_device())
    rows = []
    for _, r in real.iterrows():
        img = np.asarray(Image.open(r.path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 127.5 - 1
        rows.append({"content_id": r.content_id, "strength": 0.0, "path": r.path, "caption": r.caption})
        for s in STRENGTHS:
            dst = out / f"{r.content_id}__s{s}.png"
            if not dst.exists():
                seed = int(hashlib.sha256(f"{r.content_id}:{s}".encode()).hexdigest()[:8], 16)
                edited = img2img(probe, img, r.caption, s, seed)
                Image.fromarray(((edited.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(dst)
                print("generated", dst.name, flush=True)
            rows.append({"content_id": r.content_id, "strength": s, "path": str(dst), "caption": r.caption})
    df = pd.DataFrame(rows)
    df["label"] = 1
    df["generator"] = df["strength"].apply(lambda s: f"edit_s{s}" if s > 0 else "real")
    df.to_csv(out / "manifest.csv", index=False)
    print(len(real), "content ids;", len(df), "total rows;", "done")

if __name__ == "__main__":
    main()
