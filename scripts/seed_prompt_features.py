"""Reuse prompt-0 rich features from the completed single-caption pilot."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--prompts',required=True);p.add_argument('--single-features',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 prompts=pd.read_parquet(a.prompts);base=json.load(open(a.single_features));by={r['path']:r for r in base}; seeded=[]
 for _,r in prompts[prompts.caption_id.eq(0)].iterrows():
  x=dict(by[r.path]);x['caption_id']=0;x['caption_method']=r.caption_method;x['caption']=r.caption;seeded.append(x)
 o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(seeded,indent=2));print({'seeded':len(seeded),'remaining':len(prompts)-len(seeded)})
if __name__=='__main__':main()
