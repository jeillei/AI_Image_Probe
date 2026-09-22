"""SynthImage v2 stage panel: 10 literature-anchored scalar features, computed from tensors the frozen SD1.5
probe already produces (or one small additional single-timestep UNet call for LaRE).  See
LITERATURE_FEATURE_PANEL.md for the exact formula, citation and reproduced-vs-adapted status of every feature.
No feature here was invented post hoc; this module is the single source of truth for the v2 panel's math."""
from __future__ import annotations
import numpy as np
import torch

LPIPS_LAYER = 1  # 0-indexed; AEROBLADE's "LPIPS2" (1-indexed second VGG16 layer) -> index 1 here
LARE_TIMESTEP = 200  # fixed by LaRE^2 paper, pre-declared, not tuned
LARE_ENSEMBLE = 4    # e=4, fixed by LaRE^2 paper


def lpips_layer_score(lpips_net, x_hwc: np.ndarray, y_hwc: np.ndarray, device, dtype) -> float:
    """LPIPS layer-2 distance between two HWC images in [-1,1] (AEROBLADE's chosen layer). x_hwc/y_hwc: (H,W,C)."""
    a = torch.from_numpy(x_hwc.transpose(2, 0, 1)[None]).to(device, dtype)
    b = torch.from_numpy(y_hwc.transpose(2, 0, 1)[None]).to(device, dtype)
    with torch.no_grad():
        _, per_layer = lpips_net(a.float(), b.float(), retPerLayer=True)
    return float(per_layer[LPIPS_LAYER].squeeze().item())


def vae_only(probe, image_hwc: np.ndarray, lpips_net) -> dict:
    """Stage 1: S1.1 lpips_ae, S1.2 pixel_mse_ae, S1.3 latent_mse_ae. One extra encode+decode+re-encode; no UNet."""
    z0 = probe.encode_image(image_hwc)                          # (1,4,32,32)
    x_recon = probe.decode_latent(z0)                            # (C,H,W) pixel space, matches image_hwc.transpose
    x_recon_hwc = x_recon.transpose(1, 2, 0)
    z0_reenc = probe.encode_image(x_recon_hwc)
    pixel_mse = float(np.mean((x_recon_hwc - image_hwc) ** 2))
    latent_mse = float(torch.mean((z0 - z0_reenc) ** 2).item())
    lpips_ae = lpips_layer_score(lpips_net, image_hwc, x_recon_hwc, probe.device, probe.dtype)
    return {"lpips_ae": lpips_ae, "pixel_mse_ae": pixel_mse, "latent_mse_ae": latent_mse}


def lare_t200(probe, z0: torch.Tensor, p_embed: torch.Tensor, seed: int) -> float:
    """Stage 2: S2.1 -- LaRE^2's single-step known-noise prediction error at fixed t=200, e=4 ensemble.
    z0: (1,4,32,32) latent (already encoded). p_embed: (1,77,768) conditional text embedding."""
    alpha_t = probe.alphas[LARE_TIMESTEP].to(probe.device, probe.dtype)
    t_tensor = torch.tensor([LARE_TIMESTEP], device=probe.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    errs = []
    for i in range(LARE_ENSEMBLE):
        eps_true = torch.randn(z0.shape, generator=gen, dtype=torch.float32).to(probe.device, probe.dtype)
        noisy = alpha_t.sqrt() * z0 + (1 - alpha_t).sqrt() * eps_true
        with torch.inference_mode():
            eps_pred = probe.pipe.unet(noisy, t_tensor, encoder_hidden_states=p_embed).sample
        errs.append(float(torch.mean((eps_true - eps_pred) ** 2).item()))
    return float(np.mean(errs))


def score_norm_step0(cond_scores: np.ndarray) -> float:
    """Stage 2: S2.2 -- RMS of the conditional score at the first (least-noised) inversion step. Reads cached
    tensor directly, no extra compute. cond_scores: (6,4,32,32)."""
    return float(np.sqrt(np.mean(cond_scores[0] ** 2)))


def diffpath_curvature(cond_scores: np.ndarray) -> float:
    """Stage 3: S3.1 -- DiffPath-1D: sum_t ||eps_theta(x_t,t) - eps_theta(x_{t+1},t+1)||^2 over consecutive
    inversion steps (finite difference of the conditional score). cond_scores: (6,4,32,32)."""
    diffs = np.diff(cond_scores.astype(np.float64), axis=0)  # (5,4,32,32)
    return float(np.sum(np.mean(diffs ** 2, axis=(1, 2, 3))))


def path_length(z: np.ndarray) -> float:
    """Stage 3: S3.2 -- simple documented control (RMS-per-dim step size, summed). z: (7,4,32,32)."""
    d = np.diff(z.astype(np.float64), axis=0)
    return float(np.sum(np.sqrt(np.mean(d ** 2, axis=(1, 2, 3)))))


def roundtrip(image_hwc: np.ndarray, reconstruction_chw: np.ndarray, z0: np.ndarray, z_reverse_final: np.ndarray,
              lpips_net, probe) -> dict:
    """Stage 4: S4.1 pixel_l1_roundtrip (DIRE-adapted), S4.2 lpips_roundtrip, S4.3 latent_mse_roundtrip."""
    recon_hwc = reconstruction_chw.transpose(1, 2, 0)
    pixel_l1 = float(np.mean(np.abs(recon_hwc - image_hwc)))
    lpips_rt = lpips_layer_score(lpips_net, image_hwc, recon_hwc, probe.device, probe.dtype)
    latent_mse = float(np.mean((z0.astype(np.float64) - z_reverse_final.astype(np.float64)) ** 2))
    return {"pixel_l1_roundtrip": pixel_l1, "lpips_roundtrip": lpips_rt, "latent_mse_roundtrip": latent_mse}


CORE_FEATURE_NAMES = ["lpips_ae", "pixel_mse_ae", "latent_mse_ae", "lare_t200", "score_norm_step0",
                       "diffpath_curvature", "path_length", "pixel_l1_roundtrip", "lpips_roundtrip", "latent_mse_roundtrip"]
STAGE_OF = {"lpips_ae": "vae", "pixel_mse_ae": "vae", "latent_mse_ae": "vae", "lare_t200": "score",
            "score_norm_step0": "score", "diffpath_curvature": "trajectory", "path_length": "trajectory",
            "pixel_l1_roundtrip": "roundtrip", "lpips_roundtrip": "roundtrip", "latent_mse_roundtrip": "roundtrip"}
STAGE_ORDER = ["vae", "score", "trajectory", "roundtrip"]
