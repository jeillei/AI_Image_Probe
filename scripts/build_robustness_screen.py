"""Freeze the paired 120-original, 15-condition robustness screen."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.corruption.robustness_suite import CONDITIONS,condition_id
def seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/stage_b_cross_source/manifest.csv');p.add_argument('--output',default='data/robustness_screen/manifest.csv');a=p.parse_args();d=pd.read_csv(a.manifest);syn=[]
 for g,x in d[d.label.eq(1)].groupby('generator',sort=True):syn.append(x.sort_values('path').iloc[::max(1,len(x)//10)].head(10))
 real=[]
 for s,x in d[d.label.eq(0)].groupby('real_source',sort=True):real.append(x.sort_values('path').iloc[::max(1,len(x)//10)].head(10))
 base=pd.concat(syn+real,ignore_index=True); assert len(base)==120
 rows=[]
 for _,r in base.iterrows():
  oid=str(r.image_id)
  for name,value in CONDITIONS:
   rows.append({**r.to_dict(),'original_id':oid,'condition_id':condition_id(name,value),'transform':name,'transform_value':'' if value is None else value,'transform_seed':seed(f'{oid}:{name}:{value}'),'extractor_protocol_version':'sd15_blip_256_ddim6_rich_v1'})
 out=pd.DataFrame(rows);o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);out.to_csv(o,index=False);Path(str(o)+'.json').write_text(json.dumps({'originals':base.image_id.tolist(),'conditions':[condition_id(n,v) for n,v in CONDITIONS]},indent=2));print({'originals':len(base),'rows':len(out),'by_generator':base[base.label.eq(1)].generator.value_counts().to_dict(),'by_real_source':base[base.label.eq(0)].real_source.value_counts().to_dict()})
if __name__=='__main__':main()
