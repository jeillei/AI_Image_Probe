"""Small balanced multi-update optimized-conditioning replication, serial/cached."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps
from src.probes.sd15 import SD15Probe

p=argparse.ArgumentParser();p.add_argument('--captions',default='results/sd15_conditioning/captions.json');p.add_argument('--per-class',type=int,default=2);p.add_argument('--steps',type=int,default=2);p.add_argument('--iterations',type=int,default=3);p.add_argument('--size',type=int,default=256);p.add_argument('--output',default='results/sd15_conditioning/optimized_subset.json');a=p.parse_args()
caps=json.loads(Path(a.captions).read_text());out=Path(a.output);rows=json.loads(out.read_text()) if out.exists() else []; done={r['path'] for r in rows}
sel=[]
for label in [0,1]: sel += [(k,v) for k,v in caps.items() if v['label']==label][:a.per_class]
probe=SD15Probe(a.steps)
for i,(path,m) in enumerate(sel,1):
 if path in done: continue
 x=np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB').resize((a.size,a.size)),dtype=np.float32)/127.5-1; st=time.time()
 cap=probe.invert_reconstruct(x,m['caption'],mode='caption'); opt=probe.optimize_conditioning(x,m['caption'],a.iterations); best=probe.invert_reconstruct(x,m['caption'],embeddings=(opt['prompt_embeds'],opt['negative_embeds']))
 c_mse=float(np.mean((cap['reconstruction']-x.transpose(2,0,1))**2));o_mse=float(np.mean((best['reconstruction']-x.transpose(2,0,1))**2)); diff=np.linalg.norm((cap['forward']-best['forward']).reshape(len(cap['forward']),-1),axis=1)/np.sqrt(cap['forward'][0].size)
 row={'path':path,'label':m['label'],'caption':m['caption'],'caption_mse':c_mse,'optimized_mse':o_mse,'optimization_benefit':c_mse-o_mse,'optimization_benefit_norm':(c_mse-o_mse)/(c_mse+1e-8),'optimization_losses':opt['losses'],'conditioning_displacement':opt['conditioning_displacement'],'caption_to_opt_trajectory_displacement':float(diff.mean()),'seconds':round(time.time()-st,3)};rows.append(row);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2));print(i,row)
 if torch.backends.mps.is_available():torch.mps.empty_cache()
