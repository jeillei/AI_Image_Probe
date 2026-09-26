"""Conditioned latent-DDIM inversion probe for SDXL -- a second, independent measuring instrument used only for
the reviewer-validation cross-probe stress test. See
docs/research_history/reviewer_validation/REVIEWER_VALIDATION_PLAN.md for the full configuration table and why
each choice was made (matched scheduler/steps/guidance/canonicalization to SD15Probe; SDXL's own dual-encoder
conditioning and micro-conditioning kept genuine, not simplified).

Structurally parallel to SD15Probe -- same public interface (encode_image, decode_latent, embeds,
invert_reconstruct, .device, .dtype) so every function in synthimage.features.panel_v2 applies to this probe
completely unchanged."""
from __future__ import annotations
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDIMScheduler, DDIMInverseScheduler

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


class SDXLProbe:
    def __init__(self, steps: int = 6, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "mps" else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
        # SDXL's own VAE is documented to be numerically unstable in float16 (produces NaNs); load it separately
        # in float32 while the UNet/text encoders stay at self.dtype -- verified NaN-free in engineering
        # validation (see REVIEWER_VALIDATION_PLAN.md).
        self.pipe.vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, clip_sample=False)
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing("max")
        self.pipe.vae.enable_slicing()
        self.pipe.unet.eval(); self.pipe.vae.eval()
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config, clip_sample=False)
        self.set_steps(steps)

    def set_steps(self, steps: int) -> None:
        self.steps = steps
        self.pipe.scheduler.set_timesteps(steps, device=self.device)
        self.inverse_scheduler.set_timesteps(steps, device=self.device)
        self.ts = self.pipe.scheduler.timesteps.detach().cpu().numpy().astype(int)

    @torch.inference_mode()
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """HWC RGB [-1,1] -> correctly scaled SDXL latent (VAE runs in float32; result cast to probe dtype)."""
        x = torch.from_numpy(image.transpose(2, 0, 1)[None]).to(self.device, torch.float32)
        z = self.pipe.vae.encode(x).latent_dist.mean * self.pipe.vae.config.scaling_factor
        return z.to(self.dtype)

    @torch.inference_mode()
    def decode_latent(self, z: torch.Tensor) -> np.ndarray:
        x = self.pipe.vae.decode(z.to(torch.float32) / self.pipe.vae.config.scaling_factor).sample
        return x.float().cpu().numpy()[0]

    def _add_time_ids(self, batch: int) -> torch.Tensor:
        """(original_size, crop_top_left, target_size) micro-conditioning: tells the model this is a full,
        uncropped 256x256 image -- this project's frozen canonicalization size, deliberately not SDXL's native
        1024px training resolution (see REVIEWER_VALIDATION_PLAN.md for why the input size is matched across
        probes rather than each probe's own native resolution)."""
        ids = torch.tensor([[256, 256, 0, 0, 256, 256]], device=self.device, dtype=self.dtype)
        return ids.expand(batch, -1)

    @torch.inference_mode()
    def embeds(self, prompt: str, mode: str) -> tuple[dict, dict]:
        if mode == "null": prompt = ""
        pe, npe, pooled, npooled = self.pipe.encode_prompt(prompt=prompt, device=self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt="")
        return {"embeds": pe, "pooled": pooled}, {"embeds": npe, "pooled": npooled}

    @torch.inference_mode()
    def _prediction_parts(self, z, t, p, n, guidance: float):
        model_input = torch.cat([z, z])
        emb = torch.cat([n["embeds"], p["embeds"]])
        add_te = torch.cat([n["pooled"], p["pooled"]])
        add_tid = self._add_time_ids(model_input.shape[0])
        out = self.pipe.unet(model_input, t.reshape(-1)[0].expand(model_input.shape[0]), encoder_hidden_states=emb,
            added_cond_kwargs={"text_embeds": add_te, "time_ids": add_tid}).sample
        eu, ec = out.chunk(2); gap = ec - eu
        return eu + guidance * gap, ec, eu, gap

    @torch.inference_mode()
    def _eps(self, z, t, p, n, guidance: float):
        guided, _, _, gap = self._prediction_parts(z, t, p, n, guidance)
        return guided, gap

    @torch.inference_mode()
    def invert_reconstruct(self, image: np.ndarray, prompt: str, mode: str = "caption", guidance: float = 1.0,
                            capture_predictions: bool = False) -> dict:
        z0 = self.encode_image(image); z = z0.clone(); p, n = self.embeds(prompt, mode)
        forward = [z.float().cpu().numpy()[0]]; cond_scores = []; uncond_scores = []
        for t in self.inverse_scheduler.timesteps:
            tt = t[None]; eps, ec, eu, gap = self._prediction_parts(z, tt, p, n, guidance)
            z = self.inverse_scheduler.step(eps, t, z).prev_sample
            forward.append(z.float().cpu().numpy()[0])
            if capture_predictions:
                cond_scores.append(ec.float().cpu().numpy()[0]); uncond_scores.append(eu.float().cpu().numpy()[0])
        for t in self.pipe.scheduler.timesteps:
            eps, _ = self._eps(z, t[None], p, n, guidance)
            z = self.pipe.scheduler.step(eps, t, z, eta=0.0).prev_sample
        recon = self.decode_latent(z)
        return {"z0": z0.float().cpu().numpy()[0], "forward": np.stack(forward), "reconstruction": recon,
                **({"cond_scores": np.stack(cond_scores), "uncond_scores": np.stack(uncond_scores)} if capture_predictions else {})}
