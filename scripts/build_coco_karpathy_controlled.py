"""Build controlled real/caption pairs from compact COCO Karpathy annotations."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
from urllib.request import urlopen
p=argparse.ArgumentParser();p.add_argument('--annotations',default='data/coco/coco_karpathy_val.json');p.add_argument('--out',default='data/controlled_coco');p.add_argument('--count',type=int,default=10);p.add_argument('--seed',type=int,default=41);a=p.parse_args()
items=json.loads(Path(a.annotations).read_text());rng=random.Random(a.seed);rng.shuffle(items)
keys=['person','dog','cat','bird','car','train','bus','boat','pizza','cake','building','street','beach','elephant','horse','kitchen','table'];picked=[];seen=set()
for key in keys:
 for x in items:
  text=x['caption'][0].lower()
  if key in text and x['image'] not in seen:picked.append(x);seen.add(x['image']);break
for x in items:
 if len(picked)>=a.count:break
 if x['image'] not in seen:picked.append(x);seen.add(x['image'])
out=Path(a.out);(out/'real').mkdir(parents=True,exist_ok=True);rows=[]
for n,x in enumerate(picked[:a.count]):
 name=Path(x['image']).name;url=f'http://images.cocodataset.org/val2014/{name}';dest=out/'real'/f'{n:03d}_{name}'
 if not dest.exists():
  with urlopen(url,timeout=60) as r:dest.write_bytes(r.read())
 rows.append({'content_id':f'coco2014_{name}','real_path':str(dest),'caption':x['caption'][0],'source':'COCO 2014 Karpathy validation','image_name':name})
with open(out/'manifest.csv','w',newline='') as f:csv.DictWriter(f,fieldnames=rows[0]).writeheader();csv.DictWriter(f,fieldnames=rows[0]).writerows(rows)
print(json.dumps(rows,indent=2))
