"""Extract the v2 literature-anchored stage panel (LITERATURE_FEATURE_PANEL.md) for the 180 caption-matched
images.  Reuses the frozen SD1.5 v1 protocol (256px squash-resize, 6-step DDIM inversion, human-caption
conditioning) -- no protocol change.  Resumable per-image (skips cached ids); one MPS process.

PASS 1 (this script): everything that needs the SD1.5 UNet (lare_t200, score_norm_step0,
diffpath_curvature, path_length, pixel_l1_roundtrip, latent_mse_roundtrip) plus the cheap VAE-only pixel/latent
MSE (pixel_mse_ae, latent_mse_ae).  It also caches the round-trip reconstruction pixels to a local .npz (not
committed) so scripts/compute_lpips_panel.py (PASS 2) never needs to touch the UNet again.

Why two passes: empirically, calling the LPIPS VGG16 network (float32) and the SD1.5 UNet (float16) in the same
MPS process, in either interleaving order once an LPIPS call has happened, corrupts every subsequent UNet
forward pass to NaN for the rest of the process -- reproduced and isolated during Phase-2 smoke testing (a
known category of MPS float16/float32 state-corruption bug, not specific to this project's code). Splitting into
two single-purpose passes (this one never imports lpips; PASS 2 never touches the UNet) avoids it entirely
rather than working around it with per-image process restarts."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np, torch
from synthimage.data.loading import load
from synthimage.probes.sd15 import SD15Probe
from synthimage.features.panel_v2 import lare_t200, score_norm_step0, diffpath_curvature, path_length

def rowkey(r):
    return (r["generator"], r["content_id"], r.get("transform") or "clean", str(r.get("transform_value") or ""))


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

PROTOCOL_VERSION = "synthimage_v2_stage_panel_1.0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/caption_matched/manifest.csv")
    p.add_argument("--output", default="results/stage_decomposition/panel_features_pass1.json")
    p.add_argument("--recon-cache", default="data/stage_panel_cache")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    recon_cache = Path(a.recon_cache); recon_cache.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    have = {rowkey(r) for r in done}
    rows = list(csv.DictReader(open(a.manifest)))[: a.limit]
    pending = [r for r in rows if rowkey(r) not in have]
    print(f"{len(rows) - len(pending)}/{len(rows)} already done; {len(pending)} to extract")
    if not pending:
        return
    probe = SD15Probe(a.steps, device=pick_device())
    for i, r in enumerate(pending, 1):
        t0 = time.time()
        seed17 = int(r.get("transform_seed") or 17)
        x = load(r["path"], a.size, seed17, transform=r.get("transform") or None, transform_value=r.get("transform_value") or None)
        caption = r["caption"]
        res = probe.invert_reconstruct(x, caption, mode="caption", guidance=1.0, capture_predictions=True)
        z0 = probe.encode_image(x)
        x_ae_recon = probe.decode_latent(z0)  # VAE-only reconstruction, no UNet -- pixel/latent MSE here, LPIPS in pass 2
        pixel_mse_ae = float(np.mean((x_ae_recon.transpose(1, 2, 0) - x) ** 2))
        z0_reenc = probe.encode_image(x_ae_recon.transpose(1, 2, 0))
        latent_mse_ae = float(torch.mean((z0 - z0_reenc) ** 2).item())
        seed = int(hashlib.sha256(f"{r['generator']}:{r['content_id']}".encode()).hexdigest()[:8], 16)
        p_embed, _ = probe.embeds(caption, "caption")
        feats = {
            "pixel_mse_ae": pixel_mse_ae, "latent_mse_ae": latent_mse_ae,
            "lare_t200": lare_t200(probe, z0, p_embed, seed),
            "score_norm_step0": score_norm_step0(res["cond_scores"]),
            "diffpath_curvature": diffpath_curvature(res["cond_scores"]),
            "path_length": path_length(res["forward"]),
            "pixel_l1_roundtrip": float(np.mean(np.abs(res["reconstruction"].transpose(1, 2, 0) - x))),
            "latent_mse_roundtrip": float(np.mean((res["z0"].astype(np.float64) - res["reverse"][-1].astype(np.float64)) ** 2)),
        }
        key = f"{r['generator']}__{r['content_id']}__{r.get('transform') or 'clean'}__{r.get('transform_value') or ''}"
        np.savez_compressed(recon_cache / f"{key}.npz", image=x.astype(np.float32),
                            ae_recon=x_ae_recon.astype(np.float32), roundtrip_recon=res["reconstruction"].astype(np.float32))
        done.append({"path": r["path"], "label": int(r["label"]), "generator": r["generator"],
                     "content_id": r["content_id"], "caption": caption, "transform": r.get("transform") or "clean",
                     "transform_value": r.get("transform_value") or "", "protocol_version": PROTOCOL_VERSION,
                     "features": feats})
        out.write_text(json.dumps(done, indent=1))
        assert not any(v != v for v in feats.values()), f"NaN produced for {key}: {feats}"  # never silently keep a corrupted row
        print(i, "/", len(pending), r["generator"], r["content_id"], r.get("transform") or "clean", round(time.time() - t0, 2), "s", flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()


if __name__ == "__main__":
    main()
