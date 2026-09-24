from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageOps
from src.probes.sd15 import SD15Probe

def load(p, size):
    a=np.asarray(ImageOps.exif_transpose(Image.open(p)).convert('RGB').resize((size,size)),dtype=np.float32)/127.5-1
    return a
def main():
    ap=argparse.ArgumentParser();ap.add_argument('images',nargs='+');ap.add_argument('--output',default='results/sd15_validation');ap.add_argument('--steps',type=int,default=6);ap.add_argument('--size',type=int,default=256);ap.add_argument('--mode',choices=['null','caption'],default='caption');ap.add_argument('--prompt',default='a natural photograph');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);probe=SD15Probe(a.steps); rows=[]
    for i,path in enumerate(a.images):
        x=load(path,a.size); r=probe.invert_reconstruct(x,a.prompt,mode=a.mode,guidance=1.0)
        zerr=float(np.mean((r['reverse'][-1]-r['z0'])**2)); pix=float(np.mean((r['reconstruction']-x.transpose(2,0,1))**2))
        vae_recon=probe.decode_latent(probe.encode_image(x)); vae_pix=float(np.mean((vae_recon-x.transpose(2,0,1))**2))
        Image.fromarray(np.clip((r['reconstruction'].transpose(1,2,0)+1)*127.5,0,255).astype('uint8')).save(out/f'recon_{i}.png')
        rows.append({'image':path,'mode':a.mode,'prompt':a.prompt,'vae_pixel_mse':vae_pix,'latent_roundtrip_mse':zerr,'pixel_roundtrip_mse':pix,'endpoint_norm':float(np.linalg.norm(r['forward'][-1])/np.sqrt(r['forward'][-1].size)),'path_steps':len(r['forward'])})
        print(rows[-1])
    (out/'validation.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
