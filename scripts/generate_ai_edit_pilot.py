"""Phase 7 AI-edit feasibility pilot: an img2img "strength" continuum as the feasible, clean substitute for a
localized-inpainting edit continuum.

Why img2img-strength, not masked inpainting: the task's ideal structure (localized object edit / background
replacement / controlled inpainting) requires an inpainting-specific checkpoint (e.g. runwayml/stable-diffusion-
inpainting) that is NOT among this project's already-downloaded, already-validated weights and would require a
new ~4-5GB uncontrolled download plus fresh validation before use -- out of proportion to a feasibility pilot.
img2img strength is a standard, well-understood SD1.5 mechanism (partially noise the image to an intermediate
timestep, then denoise from there) that gives a precise, continuous "degree of generative intervention" knob
(strength in (0,1]: 0 = unchanged, 1 = full regeneration) using ONLY the checkpoint already in use everywhere
else in this project. It is a genuine substitute for the requested continuum, not a downgrade of the question
being asked ("does a frozen feature move monotonically with degree of generative intervention"), but it is
GLOBAL, not localized -- there is no edited/unedited region within one image, which the task's masked-edit
examples would have provided. This is disclosed as a limitation in STAGE_DECOMPOSITION_RESULTS.md.

Pre-declared: 8 content ids (first-8 by sorted content_id -- deterministic, label/feature-blind), strengths
0.3/0.6/0.9 (light/medium/heavy), 25 inference steps, guidance 7.5 (standard SD1.5 defaults, not tuned),
generator seed = hash(content_id) (same seeding convention as scripts/build_content_matched.py)."""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from PIL import Image
from diffusers import DDIMScheduler

N_CONTENT = 8
STRENGTHS = (0.3, 0.6, 0.9)
STEPS = 25
GUIDANCE = 7.5


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
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/ai_edit_pilot"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv("data/content_matched/manifest.csv")
    real = m[m.generator == "real"].sort_values("content_id").head(N_CONTENT)
    from src.probes.sd15 import SD15Probe
    probe = SD15Probe(6)  # UNet/VAE reused; step count for img2img set separately (STEPS) from the frozen inversion protocol
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
    pd.DataFrame(rows).assign(label=1, generator=[f"edit_s{r['strength']}" if r["strength"] > 0 else "real" for r in rows]).to_csv(out / "manifest.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
