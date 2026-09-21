"""DiT-XL/2-256 probe: class-conditional latent *transformer* (ImageNet), eps-prediction with learned sigma.

Architecturally independent of SD1.5 in denoiser (transformer vs UNet), conditioning (null class token vs BLIP text),
and training set (ImageNet-1k vs LAION).  CAVEAT: it uses an SD-family VAE (sd-vae-ft-ema), so VAE-compatibility
is shared with SD1.5 -- disclosed in MULTIPROBE_PLAN.md.  Conditioning here is the *null class* (id 1000) so no caption is needed.
Same canonical pixels as SD1.5 v1 (squash-resize 256), same DDIM-inverse/DDIM 6-step protocol.
"""
from __future__ import annotations
import numpy as np, torch
from PIL import Image, ImageOps
from diffusers import DiTPipeline, DDIMScheduler, DDIMInverseScheduler

class DiTProbe:
    def __init__(self, steps=6, device="cpu", model_id="facebook/DiT-XL-2-256", dtype=None):
        self.name = "dit_xl2_256_nullcls"; self.protocol_version = f"{self.name}_ddim{steps}_from_v1canon256_v0"; self.canonical_size = 256; self.device = device; self.steps = steps
        self.dtype = dtype or (torch.float16 if device == "mps" else torch.float32)
        self.pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(device); self.tf = self.pipe.transformer.eval(); self.vae = self.pipe.vae.eval()
        cfg = self.pipe.scheduler.config; self.sched = DDIMScheduler.from_config(cfg, clip_sample=False); self.inv = DDIMInverseScheduler.from_config(cfg, clip_sample=False)
        self.sched.set_timesteps(steps, device=device); self.inv.set_timesteps(steps, device=device); self.null = 1000
    def preprocess(self, path: str) -> np.ndarray:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS); return np.asarray(im, dtype=np.float32) / 127.5 - 1
    def _eps(self, z, t, b):
        out = self.tf(z, timestep=t.expand(b), class_labels=torch.full((b,), self.null, device=self.device, dtype=torch.long)).sample; return out[:, :z.shape[1]]
    @torch.inference_mode()
    def measure(self, images, conditioning=None):
        b = len(images); x = torch.from_numpy(np.stack([i.transpose(2, 0, 1) for i in images])).to(self.device, self.dtype)
        z0 = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor; z = z0.clone(); fwd = [z.float().cpu().numpy()]; eps_n = []
        for t in self.inv.timesteps:
            e = self._eps(z, t, b); z = self.inv.step(e, t, z).prev_sample; fwd.append(z.float().cpu().numpy()); eps_n.append(e.float().square().mean(dim=(1, 2, 3)).sqrt().cpu().numpy())
        rev = [z.float().cpu().numpy()]
        for t in self.sched.timesteps:
            e = self._eps(z, t, b); z = self.sched.step(e, t, z, eta=0.0).prev_sample; rev.append(z.float().cpu().numpy())
        rec = self.vae.decode(z / self.vae.config.scaling_factor).sample.float().cpu().numpy()
        return [{"z0": z0.float().cpu().numpy()[i], "forward": np.stack([s[i] for s in fwd]), "reverse": np.stack([s[i] for s in rev]), "reconstruction": rec[i],
                 "eps_norm": [float(s[i]) for s in eps_n], "cond_gap": [0.0] * len(eps_n), "partial_roundtrips": {}} for i in range(b)]
