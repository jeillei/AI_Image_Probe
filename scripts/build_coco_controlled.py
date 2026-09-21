"""Build a small, reproducible COCO-captioned real side of the controlled set."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
from urllib.request import urlopen

ap=argparse.ArgumentParser();ap.add_argument('--annotations',default='data/coco/annotations/captions_val2017.json');ap.add_argument('--out',default='data/controlled_coco');ap.add_argument('--count',type=int,default=10);ap.add_argument('--seed',type=int,default=41);a=ap.parse_args()
ann=json.loads(Path(a.annotations).read_text()); imgs={x['id']:x for x in ann['images']}; by={}
for x in ann['annotations']: by.setdefault(x['image_id'],[]).append(x['caption'])
# Coarse content strata prevent an all-people convenience sample.
keys=['person','dog','cat','bird','car','train','bus','boat','pizza','cake','building','street','beach','elephant','horse','kitchen','table']
rng=random.Random(a.seed); candidates=list(by);rng.shuffle(candidates); picked=[];used=set()
for key in keys:
 for iid in candidates:
  caption=by[iid][0].lower()
  if iid not in used and key in caption:
   picked.append(iid);used.add(iid);break
for iid in candidates:
 if len(picked)>=a.count:break
 if iid not in used:picked.append(iid);used.add(iid)
out=Path(a.out); real=out/'real';real.mkdir(parents=True,exist_ok=True); rows=[]
for n,iid in enumerate(picked[:a.count]):
 im=imgs[iid]; dest=real/f'{n:03d}_{iid}.jpg'
 if not dest.exists():
  with urlopen(im['coco_url'],timeout=60) as r:dest.write_bytes(r.read())
 rows.append({'content_id':f'coco_{iid}','real_path':str(dest),'caption':by[iid][0],'all_captions':json.dumps(by[iid]),'source':'COCO 2017 val','image_id':iid})
with open(out/'manifest.csv','w',newline='') as f:csv.DictWriter(f,fieldnames=rows[0]).writeheader();csv.DictWriter(f,fieldnames=rows[0]).writerows(rows)
print(json.dumps(rows,indent=2))
