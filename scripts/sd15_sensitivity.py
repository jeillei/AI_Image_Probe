"""Serial, cache-resumable null→caption sensitivity extraction."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image,ImageOps
from src.probes.sd15 import SD15Probe

def image(path,size): return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB').resize((size,size)),dtype=np.float32)/127.5-1
def mse(r,x): return float(np.mean((r['reconstruction']-x.transpose(2,0,1))**2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--captions',default='results/sd15_conditioning/captions.json');p.add_argument('--per-class',type=int,default=4);p.add_argument('--steps',type=int,default=6);p.add_argument('--size',type=int,default=256);p.add_argument('--output',default='results/sd15_conditioning/sensitivity_null_caption.json');a=p.parse_args()
 caps=json.loads(Path(a.captions).read_text()); out=Path(a.output); rows=json.loads(out.read_text()) if out.exists() else [] ; done={r['path'] for r in rows}
 selected=[]
 for label in [0,1]: selected += [(path,v) for path,v in caps.items() if v['label']==label][:a.per_class]
 print({'selected':len(selected),'cached':len(done)}); probe=SD15Probe(a.steps)
 for i,(path,meta) in enumerate(selected,1):
  if path in done: continue
  x=image(path,a.size); start=time.time(); null=probe.invert_reconstruct(x,'',mode='null'); cap=probe.invert_reconstruct(x,meta['caption'],mode='caption')
  f0,f1=null['forward'],cap['forward']; diff=np.linalg.norm((f0-f1).reshape(len(f0),-1),axis=1)/np.sqrt(f0[0].size)
  l0,l1=mse(null,x),mse(cap,x); row={'path':path,'label':meta['label'],'caption':meta['caption'],'steps':a.steps,'size':a.size,'null_mse':l0,'caption_mse':l1,'caption_benefit':l0-l1,'caption_benefit_norm':(l0-l1)/(l0+1e-8),'trajectory_displacement_mean':float(diff.mean()),'trajectory_displacement_max':float(diff.max()),'endpoint_displacement':float(diff[-1]),'guidance_response_mean':float(np.mean(cap['cond_gap'])),'guidance_response_max':float(np.max(cap['cond_gap'])),'seconds':round(time.time()-start,3)}
  rows.append(row); out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2));print(i,row['label'],Path(path).name,{k:round(row[k],4) for k in ['null_mse','caption_mse','caption_benefit_norm','trajectory_displacement_mean','guidance_response_mean']})
  if torch.backends.mps.is_available(): torch.mps.empty_cache()
if __name__=='__main__': main()
