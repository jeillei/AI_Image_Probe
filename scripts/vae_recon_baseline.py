"""E77 (exploratory control): pure VAE auto-encoding error as a single-scalar baseline on the content-matched set.
SD1.5 outputs are decoder outputs; a latent-decoder fixed-point effect (AEROBLADE-style) predicts LOWER VAE reconstruction error for SD1.5 fakes.
No diffusion/UNet involved.  Measured at the probe's canonical 256 px and at the generator-native 512 px."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from diffusers import AutoencoderKL
dev = "cpu"; dt = torch.float32   # MPS fp16 VAE at 512 px hung in testing; CPU fp32 is robust and cheap for a VAE-only pass
vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae", torch_dtype=dt, local_files_only=True).to(dev).eval()
m = pd.read_csv("data/content_matched/manifest.csv"); rows = []
@torch.inference_mode()
def err(im, size):
    x = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 127.5 - 1; t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(dev, dt)
    z = vae.encode(t).latent_dist.mean; r = vae.decode(z).sample.float().cpu().numpy()[0]; return float(np.mean((r - x.transpose(2, 0, 1)) ** 2)), float(np.mean(np.abs(r - x.transpose(2, 0, 1))))
for _, r in m.iterrows():
    im = Image.open(r.path); e256 = err(im, 256); e512 = err(im, 512); rows.append({"path": r.path, "label": r.label, "content_id": r.content_id, "mse256": e256[0], "mae256": e256[1], "mse512": e512[0], "mae512": e512[1]})
d = pd.DataFrame(rows); d.to_csv("results/content_matched/analysis/vae_recon_errors.csv", index=False); out = {}
rng = np.random.default_rng(0); ids = d.content_id.unique()
for c in ["mse256", "mae256", "mse512", "mae512"]:
    a = roc_auc_score(d.label, -d[c]); by = {i: np.where(d.content_id == i)[0] for i in ids}; bs = [roc_auc_score(d.label.values[ix], -d[c].values[ix]) for ix in (np.concatenate([by[i] for i in rng.choice(ids, len(ids))]) for _ in range(500))]
    pair = d.pivot_table(index="content_id", columns="label", values=c); out[c] = {"auroc_fake_has_lower_error": float(a), "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "frac_pairs_fake_lower": float((pair[1] < pair[0]).mean())}
json.dump(out, open("results/content_matched/analysis/vae_recon_baseline.json", "w"), indent=1); print(json.dumps(out, indent=1))
