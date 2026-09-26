"""Conditioned latent-DDIM inversion probe for Stable Diffusion v1.5."""
from __future__ import annotations
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

class SD15Probe:
    def __init__(self, steps: int = 8, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, clip_sample=False)
        self.pipe = self.pipe.to(self.device)
        # Keep peak unified-memory use modest on Apple Silicon.
        self.pipe.enable_attention_slicing("max")
        self.pipe.enable_vae_slicing()
        self.pipe.unet.eval(); self.pipe.vae.eval()
        self.inverse_scheduler=DDIMInverseScheduler.from_config(self.pipe.scheduler.config, clip_sample=False)
        self.alphas=self.pipe.scheduler.alphas_cumprod.to(self.device, dtype=dtype)
        self.dtype=dtype
        self.set_steps(steps)

    def set_steps(self, steps: int) -> None:
        """Reconfigure schedulers without reloading the pipeline."""
        self.steps=steps
        self.pipe.scheduler.set_timesteps(steps, device=self.device)
        self.inverse_scheduler.set_timesteps(steps, device=self.device)
        self.ts=self.pipe.scheduler.timesteps.detach().cpu().numpy().astype(int)

    @torch.inference_mode()
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """HWC RGB [-1,1] -> correctly scaled SD latent."""
        x=torch.from_numpy(image.transpose(2,0,1)[None]).to(self.device,self.dtype)
        return self.pipe.vae.encode(x).latent_dist.mean * self.pipe.vae.config.scaling_factor

    @torch.inference_mode()
    def decode_latent(self, z: torch.Tensor) -> np.ndarray:
        x=self.pipe.vae.decode(z/self.pipe.vae.config.scaling_factor).sample
        return x.float().cpu().numpy()[0]

    @torch.inference_mode()
    def encode_images(self, images: list[np.ndarray]) -> torch.Tensor:
        x=torch.from_numpy(np.stack([image.transpose(2,0,1) for image in images])).to(self.device,self.dtype)
        return self.pipe.vae.encode(x).latent_dist.mean * self.pipe.vae.config.scaling_factor

    @torch.inference_mode()
    def decode_latents(self, z: torch.Tensor) -> np.ndarray:
        return self.pipe.vae.decode(z/self.pipe.vae.config.scaling_factor).sample.float().cpu().numpy()

    @torch.inference_mode()
    def embeds(self, prompt: str, mode: str) -> tuple[torch.Tensor,torch.Tensor]:
        if mode == "null": prompt=""
        p,n=self.pipe.encode_prompt(prompt=[prompt], device=self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=[""])
        return p,n

    @torch.inference_mode()
    def _prediction_parts(self,z,t,p,n,guidance:float):
        model_input=torch.cat([z,z]); emb=torch.cat([n,p])
        out=self.pipe.unet(model_input,t.reshape(-1)[0].expand(model_input.shape[0]),encoder_hidden_states=emb).sample
        eu,ec=out.chunk(2); gap=ec-eu
        return eu+guidance*gap,ec,eu,gap

    @torch.inference_mode()
    def _eps(self,z,t,p,n,guidance:float):
        guided,_,_,gap=self._prediction_parts(z,t,p,n,guidance)
        return guided,gap

    @torch.inference_mode()
    def embeds_batch(self, prompts: list[str]) -> tuple[torch.Tensor,torch.Tensor]:
        p,n=self.pipe.encode_prompt(prompt=prompts, device=self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=[""]*len(prompts))
        return p,n

    @torch.inference_mode()
    def invert_reconstruct_batch(self, images: list[np.ndarray], prompts: list[str], guidance: float=1.0, capture_predictions: bool=False) -> list[dict]:
        """Batch equivalent of caption-conditioned inversion; no partial rounds."""
        if len(images)!=len(prompts) or not images: raise ValueError("non-empty matching images/prompts required")
        z0=self.encode_images(images); z=z0.clone(); p,n=self.embeds_batch(prompts); batch=len(images)
        forward=[z.float().cpu().numpy()]; eps_norm=[]; cond_gap=[]; cond_scores=[]; uncond_scores=[]
        for t in self.inverse_scheduler.timesteps:
            eps,ec,eu,gap=self._prediction_parts(z,t.expand(batch),p,n,guidance)
            z=self.inverse_scheduler.step(eps,t,z).prev_sample
            forward.append(z.float().cpu().numpy())
            eps_norm.append(eps.float().square().mean(dim=(1,2,3)).sqrt().cpu().numpy())
            cond_gap.append(gap.float().square().mean(dim=(1,2,3)).sqrt().cpu().numpy())
            if capture_predictions:
                cond_scores.append(ec.float().cpu().numpy()); uncond_scores.append(eu.float().cpu().numpy())
        endpoint=z; reverse=[z.float().cpu().numpy()]
        for t in self.pipe.scheduler.timesteps:
            eps,_=self._eps(z,t.expand(batch),p,n,guidance)
            z=self.pipe.scheduler.step(eps,t,z,eta=0.0).prev_sample
            reverse.append(z.float().cpu().numpy())
        reconstruction=self.decode_latents(z)
        return [{"z0":z0.float().cpu().numpy()[i],"forward":np.stack([step[i] for step in forward]),"reverse":np.stack([step[i] for step in reverse]),"reconstruction":reconstruction[i],"eps_norm":[float(step[i]) for step in eps_norm],"cond_gap":[float(step[i]) for step in cond_gap],"partial_roundtrips":{},**({"cond_scores":np.stack([step[i] for step in cond_scores]),"uncond_scores":np.stack([step[i] for step in uncond_scores])} if capture_predictions else {})} for i in range(batch)]

    def _eps_grad(self,z,t,p,n,guidance:float):
        model_input=torch.cat([z,z]); emb=torch.cat([n,p])
        out=self.pipe.unet(model_input,t.reshape(-1)[0].expand(model_input.shape[0]),encoder_hidden_states=emb).sample
        eu,ec=out.chunk(2); return eu+guidance*(ec-eu)

    def optimize_conditioning(self, image: np.ndarray, prompt: str, iterations: int=2, lr: float=2e-3, reg: float=1e-2) -> dict:
        """Small, regularized continuous-text inversion; weights remain frozen."""
        for module in (self.pipe.unet, self.pipe.text_encoder):
            for param in module.parameters(): param.requires_grad_(False)
        with torch.no_grad():
            raw_z0=self.encode_image(image); raw_p0,raw_n=self.embeds(prompt,'caption')
        # `inference_mode` tensors cannot be retained by autograd; copy them
        # into ordinary tensors while keeping the models frozen.
        z0=torch.empty_like(raw_z0).copy_(raw_z0)
        p0=torch.empty_like(raw_p0).copy_(raw_p0)
        n=torch.empty_like(raw_n).copy_(raw_n)
        p=torch.nn.Parameter(p0.clone()); opt=torch.optim.Adam([p],lr=lr); losses=[]
        for _ in range(iterations):
            z=z0
            for t in self.inverse_scheduler.timesteps:
                z=self.inverse_scheduler.step(self._eps_grad(z,t[None],p,n,1.0),t,z).prev_sample
            for t in self.pipe.scheduler.timesteps:
                z=self.pipe.scheduler.step(self._eps_grad(z,t[None],p,n,1.0),t,z,eta=0.0).prev_sample
            loss=torch.nn.functional.mse_loss(z,z0)+reg*torch.nn.functional.mse_loss(p,p0)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        return {'prompt_embeds':p.detach(),'negative_embeds':n.detach(),'initial_loss':losses[0],'final_loss':losses[-1],
                'conditioning_displacement':float((p.detach()-p0).float().square().mean().sqrt()),'losses':losses}

    @torch.inference_mode()
    def invert_reconstruct(self, image: np.ndarray, prompt: str, mode: str="caption", guidance: float=1.0, embeddings=None,
                           partial_indices: tuple[int, ...] = (), capture_predictions: bool=False) -> dict:
        z0=self.encode_image(image); z=z0.clone(); p,n=embeddings if embeddings is not None else self.embeds(prompt,mode)
        forward=[z.float().cpu().numpy()[0]]; eps_norm=[]; cond_gap=[]; cond_scores=[]; uncond_scores=[]
        # The input latent is x_0, not x at the scheduler's first retained
        # timestep.  Include t=0 before moving to retained noise levels.
        for t in self.inverse_scheduler.timesteps:
            tt=t[None]; eps,ec,eu,gap=self._prediction_parts(z,tt,p,n,guidance)
            z=self.inverse_scheduler.step(eps,t,z).prev_sample
            forward.append(z.float().cpu().numpy()[0]); eps_norm.append(float(eps.float().square().mean().sqrt())); cond_gap.append(float(gap.float().square().mean().sqrt()))
            if capture_predictions: cond_scores.append(ec.float().cpu().numpy()[0]); uncond_scores.append(eu.float().cpu().numpy()[0])
        endpoint=z; reverse=[z.float().cpu().numpy()[0]]
        for t in self.pipe.scheduler.timesteps:
            eps,_=self._eps(z,t[None],p,n,guidance)
            z=self.pipe.scheduler.step(eps,t,z,eta=0.0).prev_sample
            reverse.append(z.float().cpu().numpy()[0])
        recon=self.decode_latent(z)
        partial={}
        # After inverse step j, DDIMInverseScheduler has reached the next
        # retained noise level.  The compatible reverse suffix begins one
        # scheduler entry earlier: e.g. 6 steps, z_1 (~t=167) -> [167, 1].
        # This is deliberately a diagnostic, not a new inversion algorithm.
        for k in partial_indices:
            if not 1 <= k <= self.steps:
                raise ValueError(f"partial index must be in [1,{self.steps}], got {k}")
            if k == self.steps:
                zr=z; xr=recon
            else:
                zr=torch.from_numpy(forward[k][None]).to(self.device,self.dtype)
                for t in self.pipe.scheduler.timesteps[self.steps-k-1:]:
                    e,_=self._eps(zr,t[None],p,n,guidance)
                    zr=self.pipe.scheduler.step(e,t,zr,eta=0.0).prev_sample
                xr=self.decode_latent(zr)
            partial[str(k)]={"latent_mse":float((zr-z0).float().square().mean()),
                             "pixel_mse":float(np.mean((xr-image.transpose(2,0,1))**2))}
        return {"z0":z0.float().cpu().numpy()[0],"forward":np.stack(forward),"reverse":np.stack(reverse),"reconstruction":recon,"eps_norm":eps_norm,"cond_gap":cond_gap,"partial_roundtrips":partial,**({"cond_scores":np.stack(cond_scores),"uncond_scores":np.stack(uncond_scores)} if capture_predictions else {})}
