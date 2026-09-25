"""Extract SynthImage's frozen diffusion-pipeline forensic measurements from one image.

    uv run python scripts/reproduce/analyze_image.py photo.jpg --caption "a dog on a beach"

EXPERIMENTAL / RESEARCH ONLY. This project did not establish a calibrated, generator-independent "is this image
AI-generated" detector -- see docs/FINAL_RESULTS.md for exactly what was and was not established, and its
Limitations section in particular. This command reports the same ten literature-grounded scalar measurements
used throughout the project's analysis (see docs/METHODS.md), not a REAL/AI verdict. Interpreting the raw
numbers requires a reference cohort (e.g. the ones already committed under results/) -- a single number in
isolation is not meaningful.

Requires the Stable Diffusion 1.5 weights (fetched automatically on first use, ~5GB; see
docs/REPRODUCIBILITY.md). Runs on CPU, CUDA, or Apple Silicon MPS -- pick automatically or with --device."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, torch
from PIL import Image, ImageOps, UnidentifiedImageError


def pick_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path: Path, size: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 127.5 - 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract SynthImage's frozen v2 forensic feature panel from one image (research measurement, not a detector).",
        epilog="See docs/METHODS.md for what each feature measures and docs/FINAL_RESULTS.md for what was (and was not) established with them.")
    ap.add_argument("image", type=Path, help="path to a PNG/JPEG image")
    ap.add_argument("--caption", required=True,
                    help="a plausible content description of the image. This conditions the probe's diffusion "
                         "steps (matching how every image in this project was measured) -- it is not a "
                         "classifier input and does not need to be exact.")
    ap.add_argument("--steps", type=int, default=6, help="DDIM inversion steps (default: 6, the frozen project protocol)")
    ap.add_argument("--size", type=int, default=256, help="canonicalization size in pixels (default: 256, the frozen project protocol)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="compute device (default: auto-detect)")
    ap.add_argument("--skip-lpips", action="store_true", help="skip the two LPIPS-based features (lpips_ae, lpips_roundtrip) -- faster, one fewer dependency at runtime")
    ap.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    a = ap.parse_args()

    if not a.image.is_file():
        ap.error(f"not a readable file: {a.image}")
    try:
        x = load_image(a.image, a.size)
    except UnidentifiedImageError:
        ap.error(f"not a readable image (unsupported or corrupt file): {a.image}")

    device = pick_device(a.device)
    print(f"Loading SD1.5 probe on {device}...", file=sys.stderr)
    from src.probes.sd15 import SD15Probe
    from src.features.panel_v2 import lare_t200, score_norm_step0, diffpath_curvature, path_length

    probe = SD15Probe(a.steps, device=device)
    seed = 17  # fixed, matching this project's frozen convention for the (rare) stochastic sub-computation (lare_t200)

    print("Running DDIM inversion + round trip...", file=sys.stderr)
    result = probe.invert_reconstruct(x, a.caption, mode="caption", guidance=1.0, capture_predictions=True)
    z0 = probe.encode_image(x)
    x_ae_recon = probe.decode_latent(z0)
    z0_reenc = probe.encode_image(x_ae_recon.transpose(1, 2, 0))
    p_embed, _ = probe.embeds(a.caption, "caption")

    features = {
        "pixel_mse_ae": float(np.mean((x_ae_recon.transpose(1, 2, 0) - x) ** 2)),
        "latent_mse_ae": float(torch.mean((z0 - z0_reenc) ** 2).item()),
        "lare_t200": lare_t200(probe, z0, p_embed, seed),
        "score_norm_step0": score_norm_step0(result["cond_scores"]),
        "diffpath_curvature": diffpath_curvature(result["cond_scores"]),
        "path_length": path_length(result["forward"]),
        "pixel_l1_roundtrip": float(np.mean(np.abs(result["reconstruction"].transpose(1, 2, 0) - x))),
        "latent_mse_roundtrip": float(np.mean((result["z0"].astype(np.float64) - result["reverse"][-1].astype(np.float64)) ** 2)),
    }

    if not a.skip_lpips:
        # Computed last, after every UNet forward pass above: interleaving LPIPS (float32) and the SD1.5 UNet
        # (float16 on MPS) calls in one process has been observed to corrupt *subsequent* UNet outputs on Apple
        # Silicon (see docs/METHODS.md) -- doing all UNet work first and LPIPS last avoids that ordering entirely.
        print("Computing LPIPS features...", file=sys.stderr)
        import lpips
        from src.features.panel_v2 import lpips_layer_score
        net = lpips.LPIPS(net="vgg").to(device).eval()
        features["lpips_ae"] = lpips_layer_score(net, x, x_ae_recon.transpose(1, 2, 0), device, torch.float32)
        features["lpips_roundtrip"] = lpips_layer_score(net, x, result["reconstruction"].transpose(1, 2, 0), device, torch.float32)

    out = {
        "_framing": "EXPERIMENTAL / RESEARCH ONLY -- forensic measurements, not a REAL/AI verdict. See docs/FINAL_RESULTS.md.",
        "image": str(a.image), "caption": a.caption, "steps": a.steps, "size": a.size, "device": device,
        "features": features,
    }
    text = json.dumps(out, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
