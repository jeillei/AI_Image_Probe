from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.captioning import BlipCaptioner,load_cache,save_cache

ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/rr_pilot');ap.add_argument('--per-class',type=int,default=10);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--cache',default='results/sd15_conditioning/captions.json');a=ap.parse_args()
rng=random.Random(a.seed); root=Path(a.data); cache_path=Path(a.cache); cache=load_cache(cache_path)
items=[]
for label,name in [(0,'real'),(1,'synthetic')]:
    paths=sorted((root/name/'val').glob('*')); rng.shuffle(paths)
    items += [(str(p),label) for p in paths[:a.per_class]]
todo=[(p,y) for p,y in items if p not in cache]
print({'selected':len(items),'cached':len(items)-len(todo),'to_caption':len(todo)})
if todo:
    cap=BlipCaptioner()
    for i,(p,y) in enumerate(todo,1):
        t=time.time(); text=cap.caption(p); cache[p]={'caption':text,'label':y,'seconds':round(time.time()-t,3)}; save_cache(cache_path,cache)
        print(i, y, Path(p).name, text, cache[p]['seconds'])
print('cache',cache_path)
