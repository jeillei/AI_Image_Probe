"""Extract the current SD1.5 generative-compatibility measurements for one image.

This is analysis only: the project does not yet have a calibrated, general AI
detector threshold.  Use the JSON output to compare against a reference cohort.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageOps
from src.probes.sd15 import SD15Probe
from scripts.controlled_depth_sweep import row_features

def load(path: Path, size: int) -> np.ndarray:
    image=ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image.resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32)/127.5-1

def main() -> None:
    p=argparse.ArgumentParser(description="Analyze one image with the SD1.5 trajectory probe.")
    p.add_argument("image",type=Path); p.add_argument("--caption",required=True,help="Plausible content caption; it is probe conditioning, not a classifier feature.")
    p.add_argument("--steps",type=int,default=6); p.add_argument("--size",type=int,default=256)
    p.add_argument("--partial-indices",nargs="*",type=int,default=[]); p.add_argument("--output",type=Path)
    a=p.parse_args();
    if not a.image.is_file(): p.error(f"not a readable image: {a.image}")
    probe=SD15Probe(a.steps); x=load(a.image,a.size)
    result=probe.invert_reconstruct(x,a.caption,mode="caption",guidance=1.0,
                                    partial_indices=tuple(k for k in a.partial_indices if k<=a.steps))
    features=row_features(result,x)
    out={"image":str(a.image),"caption":a.caption,"steps":a.steps,"size":a.size,
         **{k:v for k,v in features.items() if k!="curves"},"partial_roundtrips":result["partial_roundtrips"],"curves":features["curves"]}
    text=json.dumps(out,indent=2)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text); print(a.output)
    else: print(text)
if __name__=="__main__": main()
