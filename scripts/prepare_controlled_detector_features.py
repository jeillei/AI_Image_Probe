"""Convert cached controlled trajectories into classifier-safe feature rows."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.features.compatibility import compatibility_features
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='results/controlled_coco/depth_sweep_stage_a.json');p.add_argument('--manifest',default='data/controlled_coco/detector_manifest.csv');p.add_argument('--output',default='results/detector/controlled_features.json');a=p.parse_args()
 m={r['path']:r for r in csv.DictReader(open(a.manifest))}; rows=[]
 for r in json.loads(Path(a.input).read_text()):
  if r['steps']!=6 or r['path'] not in m: continue
  x=m[r['path']]; rows.append({'path':r['path'],'label':int(x['label']),'generator':x['generator'],'content_id':x['content_id'],'split':x['split'],'severity':0,'features':compatibility_features(r)})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(rows,indent=2));print({'rows':len(rows),'groups':len(rows)//2,'output':a.output})
if __name__=='__main__':main()
