"""Real pretrained diffusion probe with deterministic DDIM-style inversion."""
from __future__ import annotations
import numpy as np
import torch
from diffusers import DDPMPipeline, DDIMScheduler

class CifarDDPMProbe:
    def __init__(self, steps: int = 20, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # Cache first: subsequent experiments must not silently require network.
        self.pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", local_files_only=True).to(self.device)
        self.pipe.unet.eval()
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, clip_sample=False)
        self.scheduler.set_timesteps(steps, device=self.device)
        self.ts = self.scheduler.timesteps.detach().cpu().numpy().astype(int)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.inference_mode()
    def invert_and_reconstruct(self, x0: np.ndarray) -> dict:
        """x0 is CHW in [-1,1]. Forward inversion and reverse DDIM reconstruction."""
        x = torch.from_numpy(x0[None]).float().to(self.device)
        forward = [x.detach().cpu().numpy()[0]]
        # timesteps are descending; forward uses ascending adjacent alpha values
        asc = list(self.ts[::-1])
        for t, nxt in zip(asc[:-1], asc[1:]):
            eps = self.pipe.unet(x, torch.tensor([t], device=self.device)).sample
            at, an = self.alphas[t], self.alphas[nxt]
            pred0 = (x - (1-at).sqrt()*eps) / at.sqrt()
            x = an.sqrt()*pred0 + (1-an).sqrt()*eps
            forward.append(x.detach().cpu().numpy()[0])
        endpoint = x
        # conventional reverse scheduler from the endpoint
        rev = [endpoint.detach().cpu().numpy()[0]]
        for t in self.scheduler.timesteps:
            eps = self.pipe.unet(x, t[None]).sample
            x = self.scheduler.step(eps, t, x, eta=0.0).prev_sample
            rev.append(x.detach().cpu().numpy()[0])
        return {"forward": np.stack(forward), "reverse": np.stack(rev), "reconstruction": x.detach().cpu().numpy()[0]}
