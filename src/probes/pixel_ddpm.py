"""Pixel-space, unconditional, VAE-free DDPM probe (architecturally independent of SD1.5).

Default checkpoint google/ddpm-cifar10-32 (locally cached).  Images are canonicalized exactly as SD1.5 v1
(squash-resize to 256) and then LANCZOS-downsampled to the probe's native size, so the probe sees a
deterministic function of the same canonical pixels.  Inversion uses DDIMInverseScheduler and the reverse uses DDIMScheduler,
mirroring the SD1.5 protocol (no text, so cond_gap is 0).
"""
from __future__ import annotations
import numpy as np, torch
from PIL import Image, ImageOps
from diffusers import DDPMPipeline, DDIMScheduler, DDIMInverseScheduler

class PixelDDPMProbe:
    def __init__(self, model_id="google/ddpm-cifar10-32", steps=6, size=32, device="cpu", tag="cifar32"):
        self.name = tag; self.protocol_version = f"{tag}_ddim{steps}_from_v1canon256_v0"; self.canonical_size = size; self.device = device; self.steps = steps
        self.pipe = DDPMPipeline.from_pretrained(model_id, local_files_only=False).to(device); self.unet = self.pipe.unet.eval()
        cfg = self.pipe.scheduler.config
        self.sched = DDIMScheduler.from_config(cfg, clip_sample=False); self.inv = DDIMInverseScheduler.from_config(cfg, clip_sample=False)
        self.sched.set_timesteps(steps, device=device); self.inv.set_timesteps(steps, device=device)
    def preprocess(self, path: str) -> np.ndarray:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)   # == SD1.5 v1 canonical
        im = im.resize((self.canonical_size, self.canonical_size), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 127.5 - 1
    @torch.inference_mode()
    def measure(self, images, conditioning=None):
        x0 = torch.from_numpy(np.stack([i.transpose(2, 0, 1) for i in images])).to(self.device); z = x0.clone(); b = len(images)
        fwd = [z.numpy() if self.device == "cpu" else z.cpu().numpy()]; eps_n = []
        for t in self.inv.timesteps:
            e = self.unet(z, t.expand(b)).sample; z = self.inv.step(e, t, z).prev_sample; fwd.append(z.cpu().numpy()); eps_n.append(e.float().square().mean(dim=(1, 2, 3)).sqrt().cpu().numpy())
        rev = [z.cpu().numpy()]
        for t in self.sched.timesteps:
            e = self.unet(z, t.expand(b)).sample; z = self.sched.step(e, t, z, eta=0.0).prev_sample; rev.append(z.cpu().numpy())
        rec = z.cpu().numpy()
        return [{"z0": x0.cpu().numpy()[i], "forward": np.stack([s[i] for s in fwd]), "reverse": np.stack([s[i] for s in rev]), "reconstruction": rec[i],
                 "eps_norm": [float(s[i]) for s in eps_n], "cond_gap": [0.0] * len(eps_n), "partial_roundtrips": {}} for i in range(b)]
