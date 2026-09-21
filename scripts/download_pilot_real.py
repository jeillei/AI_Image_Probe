"""Download independent natural-photo pilot samples; URLs are recorded as provenance."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
from urllib.request import urlopen

p=argparse.ArgumentParser(); p.add_argument('--count',type=int,default=40); p.add_argument('--out',default='data/real/picsum')
a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(a.count):
    url=f'https://picsum.photos/seed/synthimage-real-{i}/512/512.jpg'
    try:
        with urlopen(url,timeout=30) as r: (out/f'{i:03d}.jpg').write_bytes(r.read())
        rows.append({'path':str(out/f'{i:03d}.jpg'),'url':url,'label':'real','source':'picsum'})
        print(i, 'ok')
    except Exception as e: print(i, 'failed',e)
    time.sleep(.05)
with open(out/'provenance.csv','w',newline='') as f: csv.DictWriter(f,fieldnames=rows[0].keys()).writeheader(); csv.DictWriter(f,fieldnames=rows[0].keys()).writerows(rows)
