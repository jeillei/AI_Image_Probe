"""Materialize detector rows from the controlled COCO manifest without IDs as features."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/controlled_coco/manifest.csv');p.add_argument('--synthetic-dir',default='data/controlled_coco/synthetic/openai_imagegen');p.add_argument('--output',default='data/controlled_coco/detector_manifest.csv');a=p.parse_args()
 rows=[]
 for idx,m in enumerate(csv.DictReader(open(a.manifest))):
  stem=m['content_id'].replace('.jpg',''); syn=Path(a.synthetic_dir)/(stem+'.png')
  if not syn.exists(): continue
  # Fixed content-group split for smoke plumbing only. One source means this
  # manifest must never be called generator-held-out evaluation.
  split='train' if idx<4 else 'val' if idx<6 else 'test'
  rows += [dict(path=m['real_path'],label=0,generator='real_coco',content_id=m['content_id'],caption=m['caption'],split=split),dict(path=str(syn),label=1,generator='openai_imagegen',content_id=m['content_id'],caption=m['caption'],split=split)]
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=['path','label','generator','content_id','caption','split']);w.writeheader();w.writerows(rows)
 print({'rows':len(rows),'groups':len(rows)//2,'output':a.output})
if __name__=='__main__':main()
