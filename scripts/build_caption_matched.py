"""Content-matched real/synthetic pairs (Phase 8 design).  Removes semantic, caption, size and container confounds by construction:

  real:  COCO photo -> centre-crop square -> LANCZOS 512 -> PNG
  fake:  generator(prompt = the SAME human COCO caption) -> 512x512 PNG   (generator seed = hash(content_id), fixed)
  probe conditioning for BOTH classes = that human caption (manifest `caption` column; BLIP is bypassed).

Currently implemented generator: SD1.5 (local weights; also the SD1.5 probe's own family => positive-control/self-generation row).
Adding another (SDXL-turbo, PixArt, ...) = add an entry to GENERATORS; do NOT interpret SD1.5-only results as provenance evidence.
Run order: `--stage real` (network, no GPU) then `--stage fake` (GPU).  Both are resumable.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, sys
from pathlib import Path
from urllib.request import urlopen
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--annotations", default="data/coco/coco_karpathy_val.json"); ap.add_argument("--out", default="data/caption_matched"); ap.add_argument("--count", type=int, default=60)
ap.add_argument("--stage", choices=["real", "fake", "manifest"], default="real"); ap.add_argument("--generator", default="sd15"); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--steps", type=int, default=25); ap.add_argument("--guidance", type=float, default=7.5); a = ap.parse_args()
out = Path(a.out); (out / "real").mkdir(parents=True, exist_ok=True); items = json.loads(Path(a.annotations).read_text()); random.Random(a.seed).shuffle(items)
sel = []; seen = set()
for x in items:   # deterministic; label/feature blind; skip near-duplicate image names
    n = Path(x["image"]).name
    if n in seen or not x["caption"]: continue
    seen.add(n); sel.append({"content_id": "coco2014_" + n.rsplit(".", 1)[0], "name": n, "caption": x["caption"][0].strip().rstrip(".")})
    if len(sel) >= a.count: break
def sq512(im: Image.Image) -> Image.Image:
    w, h = im.size; k = min(w, h); l = (w - k) // 2; t = (h - k) // 2; return im.convert("RGB").crop((l, t, l + k, t + k)).resize((512, 512), Image.Resampling.LANCZOS)
if a.stage == "real":
    for s in sel:
        dst = out / "real" / f"{s['content_id']}.png"
        if dst.exists(): continue
        tmp = out / "real" / f"{s['content_id']}.orig.jpg"; tmp.write_bytes(urlopen(f"http://images.cocodataset.org/val2014/{s['name']}", timeout=60).read()); sq512(Image.open(tmp)).save(dst); tmp.unlink(); print("real", dst.name, flush=True)
elif a.stage == "fake":
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if a.generator == "amused":   # masked-token model, VQGAN decoder: no shared VAE/denoiser with SD1.5 (Addendum 1)
        from diffusers import AmusedPipeline
        pipe = AmusedPipeline.from_pretrained("amused/amused-512", variant="fp16", torch_dtype=torch.float16 if dev == "mps" else torch.float32).to(dev); (out / "amused").mkdir(exist_ok=True)
        for s in sel:
            dst = out / "amused" / f"{s['content_id']}.png"
            if dst.exists(): continue
            g = torch.Generator("cpu").manual_seed(int(hashlib.sha256(s["content_id"].encode()).hexdigest()[:8], 16)); im = pipe(s["caption"], generator=g, height=512, width=512).images[0]; im.convert("RGB").save(dst); print("fake", dst.name, flush=True)
        sys.exit(0)
    if a.generator != "sd15": sys.exit("only sd15/amused implemented; see docstring")
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16 if dev == "mps" else torch.float32, safety_checker=None, requires_safety_checker=False, local_files_only=True).to(dev)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config); pipe.enable_attention_slicing("max"); (out / a.generator).mkdir(exist_ok=True)
    for s in sel:
        dst = out / a.generator / f"{s['content_id']}.png"
        if dst.exists(): continue
        g = torch.Generator("cpu").manual_seed(int(hashlib.sha256(s["content_id"].encode()).hexdigest()[:8], 16)); im = pipe(s["caption"], num_inference_steps=a.steps, guidance_scale=a.guidance, height=512, width=512, generator=g).images[0]; im.save(dst); print("fake", dst.name, flush=True)
else:
    rows = []
    for s in sel:
        r = out / "real" / f"{s['content_id']}.png"
        if r.exists(): rows.append({"path": str(r), "label": 0, "generator": "real", "content_id": s["content_id"], "split": "", "caption": s["caption"], "real_source": "coco_val2014", "transform_seed": 17})
        for gdir in sorted(p.name for p in out.iterdir() if p.is_dir() and p.name not in ("real",)):
            f = out / gdir / f"{s['content_id']}.png"
            if f.exists(): rows.append({"path": str(f), "label": 1, "generator": gdir, "content_id": s["content_id"], "split": "", "caption": s["caption"], "real_source": "", "transform_seed": 17})
    with open(out / "manifest.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows"); 
