"""Numerically validate optional batch-2 SD extraction against serial output."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps
from src.probes.sd15 import SD15Probe
def load(path):return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB').resize((256,256)),dtype=np.float32)/127.5-1
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',default='results/features/stage_a_chunk000.json');p.add_argument('--output',default='results/scaling/batch2_benchmark.json');a=p.parse_args()
 rows=json.load(open(a.features))[:2];xs=[load(r['path']) for r in rows]; captions=[r['caption'] for r in rows];probe=SD15Probe(6)
 t=time.time();one=[probe.invert_reconstruct(x,c,mode='caption') for x,c in zip(xs,captions)];serial=time.time()-t
 t=time.time();two=probe.invert_reconstruct_batch(xs,captions);batched=time.time()-t
 out={'n':2,'serial_seconds':serial,'batch2_seconds':batched,'speedup':serial/batched,'max_forward_abs_error':[float(np.max(np.abs(a['forward']-b['forward']))) for a,b in zip(one,two)],'max_recon_abs_error':[float(np.max(np.abs(a['reconstruction']-b['reconstruction']))) for a,b in zip(one,two)]}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
