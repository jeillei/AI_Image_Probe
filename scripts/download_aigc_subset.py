"""Download a small, provenance-labelled public benchmark subset by API rows.

The source is evaluation-only; this script creates an explicitly small pilot,
not a replacement for a future training corpus.  It stores only pixels and a
local manifest—no image dimensions/URLs are used as detector features.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
GENERATORS={1:'ADM',2:'BigGAN',4:'DALLE2',6:'GLIDE',7:'Midjourney',9:'SD14',10:'SD15',11:'SDXL',15:'VQDM',17:'Wukong'}
FAMILIES={'ADM':'diffusion','BigGAN':'gan','DALLE2':'diffusion','GLIDE':'diffusion','Midjourney':'diffusion','SD14':'latent_diffusion','SD15':'latent_diffusion','SDXL':'latent_diffusion','VQDM':'diffusion','Wukong':'diffusion'}
def fetch(offset,length):
 url='https://datasets-server.huggingface.co/rows?'+urlencode({'dataset':'TheKernel01/AIGC-Detection-Benchmark','config':'default','split':'test','offset':offset,'length':length})
 for attempt in range(4):
  try:return json.load(urlopen(url,timeout=90))['rows']
  except Exception as exc:
   if attempt==3:raise RuntimeError(f'row API failed at offset {offset}') from exc
   time.sleep(2**attempt)
def download(url,dest):
 for attempt in range(4):
  try:
   with urlopen(url,timeout=90) as r: dest.write_bytes(r.read())
   return
  except Exception:
   if dest.exists():dest.unlink()
   if attempt==3:raise
   time.sleep(2**attempt)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/aigc_pilot');p.add_argument('--per-generator',type=int,default=4);p.add_argument('--max-rows',type=int,default=500);p.add_argument('--start-offset',type=int,default=0);p.add_argument('--append',action='store_true');p.add_argument('--generators',nargs='+',default=['ADM','BigGAN','GLIDE','DALLE2']);a=p.parse_args()
    wanted={k:v for k,v in GENERATORS.items() if v in a.generators}; found={v:0 for v in wanted.values()}; reals=[]; fake=[]
    for offset in range(a.start_offset,a.start_offset+a.max_rows,100):
        for item in fetch(offset,min(100,a.max_rows-offset)):
            row=item['row']; gen=row['generator']; label=row['label']
            if label==1 and gen in wanted and found[wanted[gen]]<a.per_generator:
                fake.append((item['row_idx'],row,wanted[gen]));found[wanted[gen]]+=1
            elif label==0 and len(reals)<a.per_generator*len(wanted):
                reals.append((item['row_idx'],row,'real'))
        print({'scanned_to':offset+100,'found':found,'real':len(reals)},flush=True)
        if all(v>=a.per_generator for v in found.values()) and len(reals)>=a.per_generator*len(wanted): break
    if not all(v>=a.per_generator for v in found.values()): raise SystemExit(f'insufficient rows: {found}')
    out=Path(a.out);rows=[]
    for idx,row,gen in reals+fake:
        label=int(row['label']); suffix='.png' if row['image']['src'].split('?')[0].endswith('.png') else '.jpg'; dest=out/('real' if label==0 else gen)/f'{idx:06d}{suffix}';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): download(row['image']['src'],dest)
        bucket=int(hashlib.sha256(f'aigc_{idx}'.encode()).hexdigest()[:8],16)%10; split='train' if bucket<7 else 'val' if bucket<9 else 'test'
        rows.append({'image_id':f'aigc_{idx}','path':str(dest),'label':label,'provenance':'synthetic' if label else 'real','generator':gen,'generator_family':FAMILIES.get(gen,'real') if label else 'real','real_source':'aigc_benchmark_real' if not label else '','content_id':f'aigc_{idx}','caption':'','caption_status':'pending','split':split,'source':'TheKernel01/AIGC-Detection-Benchmark','original_width':row['image'].get('width'),'original_height':row['image'].get('height'),'original_format':suffix[1:],'feature_status':'pending'})
    manifest=out/'manifest.csv'
    if a.append and manifest.exists():
        existing=list(csv.DictReader(open(manifest))); known={r['image_id'] for r in existing}; rows=existing+[r for r in rows if r['image_id'] not in known]
    with open(manifest,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print({'rows':len(rows),'new_generator_rows':found,'manifest':str(manifest)})
if __name__=='__main__':main()
