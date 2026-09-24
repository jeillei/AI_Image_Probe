"""One-image, memory-bounded continuous conditioning optimization diagnostic."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageOps
from src.probes.sd15 import SD15Probe

p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--steps',type=int,default=4);p.add_argument('--iterations',type=int,default=2);p.add_argument('--size',type=int,default=256);p.add_argument('--prompt',default='a natural photograph');p.add_argument('--output',default='results/sd15_optimized_one');a=p.parse_args()
x=np.asarray(ImageOps.exif_transpose(Image.open(a.image)).convert('RGB').resize((a.size,a.size)),dtype=np.float32)/127.5-1
probe=SD15Probe(a.steps); opt=probe.optimize_conditioning(x,a.prompt,a.iterations)
r=probe.invert_reconstruct(x,a.prompt,embeddings=(opt['prompt_embeds'],opt['negative_embeds']))
row={k:v for k,v in opt.items() if k not in {'prompt_embeds','negative_embeds'}}
row['latent_roundtrip_mse']=float(np.mean((r['reverse'][-1]-r['z0'])**2)); row['pixel_roundtrip_mse']=float(np.mean((r['reconstruction']-x.transpose(2,0,1))**2))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'optimization.json').write_text(json.dumps(row,indent=2));print(row)
