"""Small, serial controlled-content trajectory discovery sweep.

The manifest supplies the *same human caption* to each COCO photo and its
synthetic counterpart.  This script intentionally stores per-step curves so
feature selection can happen after inspection, rather than inside a detector.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image, ImageOps
from src.probes.sd15 import SD15Probe

def load_image(path: Path, size: int) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    # Canonical pixels, independent of native dimensions/metadata/encoding.
    return np.asarray(im.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 127.5 - 1

def moments(z: np.ndarray) -> dict[str, float]:
    a=z.astype(np.float64).ravel(); mean=float(a.mean()); var=float(a.var()); sd=np.sqrt(var+1e-12)
    centered=(a-mean)/sd
    return {"norm_per_dim": float(np.sqrt(np.mean(a*a))), "mean":mean, "var":var,
            "skew":float(np.mean(centered**3)), "kurtosis":float(np.mean(centered**4)-3),
            "extreme_fraction":float(np.mean(np.abs(a)>3.0))}

def row_features(result: dict, x: np.ndarray) -> dict:
    f=result["forward"].astype(np.float64); r=result["reverse"].astype(np.float64); d=f[0].size
    speed=np.sqrt(np.mean(np.diff(f,axis=0)**2,axis=(1,2,3)))
    accel=np.diff(speed)
    eps=np.asarray(result["eps_norm"],dtype=float); gap=np.asarray(result["cond_gap"],dtype=float)
    ep=moments(f[-1]); curves={"endpoint_norm":[],"endpoint_var":[],"endpoint_kurtosis":[]}
    for z in f:
        m=moments(z); curves["endpoint_norm"].append(m["norm_per_dim"]); curves["endpoint_var"].append(m["var"]); curves["endpoint_kurtosis"].append(m["kurtosis"])
    return {
      "pixel_roundtrip_mse":float(np.mean((result["reconstruction"]-x.transpose(2,0,1))**2)),
      "latent_roundtrip_mse":float(np.mean((r[-1]-f[0])**2)),
      "path_length":float(speed.sum()), "speed_mean":float(speed.mean()), "speed_max":float(speed.max()),
      "acceleration_abs_mean":float(np.abs(accel).mean()) if len(accel) else 0.0,
      "eps_mean":float(eps.mean()), "eps_max":float(eps.max()),
      "guidance_mean":float(gap.mean()), "guidance_max":float(gap.max()),
      **{f"endpoint_{k}":v for k,v in ep.items()},
      "curves": {"speed":speed.tolist(), "acceleration":accel.tolist(), "eps_norm":eps.tolist(), "guidance":gap.tolist(), **curves},
    }

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",default="data/controlled_coco/manifest.csv")
    ap.add_argument("--synthetic-dir",default="data/controlled_coco/synthetic/openai_imagegen")
    ap.add_argument("--groups",type=int,default=3); ap.add_argument("--depths",nargs="+",type=int,default=[6,12])
    ap.add_argument("--size",type=int,default=256); ap.add_argument("--partial-indices",nargs="*",type=int,default=[])
    ap.add_argument("--output",default="results/controlled_coco/depth_sweep_stage_a.json")
    a=ap.parse_args(); manifest=list(csv.DictReader(open(a.manifest))); selected=manifest[:a.groups]
    records=[]; output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): records=json.loads(output.read_text())
    done={(r["content_id"],r["provenance"],r["steps"]) for r in records}
    examples=[]
    for m in selected:
        stem=m["content_id"].replace(".jpg","")
        syn=Path(a.synthetic_dir)/(stem+".png")
        if not syn.exists():
            print("missing counterpart",syn); continue
        examples += [(m["content_id"],"real",Path(m["real_path"]),m["caption"]),(m["content_id"],"ai_openai_imagegen",syn,m["caption"])]
    print({"pairs":len(examples)//2,"depths":a.depths,"cached":len(done)})
    probe=SD15Probe(a.depths[0])
    for depth in a.depths:
      probe.set_steps(depth)
      for content_id, provenance, path, caption in examples:
        if (content_id,provenance,depth) in done: continue
        started=time.time(); x=load_image(path,a.size)
        partial=tuple(k for k in a.partial_indices if k <= depth)
        out=probe.invert_reconstruct(x,caption,mode="caption",guidance=1.0,partial_indices=partial)
        rec={"content_id":content_id,"provenance":provenance,"source_generator":"openai_imagegen" if provenance.startswith("ai_") else "real_coco","path":str(path),"caption":caption,"steps":depth,"size":a.size,"seconds":round(time.time()-started,2),**row_features(out,x),"partial_roundtrips":out["partial_roundtrips"]}
        records.append(rec); output.write_text(json.dumps(records,indent=2)); print(depth,provenance,content_id, {k:round(rec[k],5) for k in ("pixel_roundtrip_mse","endpoint_norm_per_dim","path_length","eps_mean","guidance_mean")})
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    output.write_text(json.dumps(records,indent=2))

if __name__ == "__main__": main()
