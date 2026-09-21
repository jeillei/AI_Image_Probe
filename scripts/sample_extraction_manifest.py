"""Make deterministic balanced, resumable extraction chunks from the master manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--master',default='data/master_manifest.parquet');p.add_argument('--per-generator',type=int,default=5);p.add_argument('--real-per-generator',type=int,default=5);p.add_argument('--offset',type=int,default=0);p.add_argument('--seed',type=int,default=17);p.add_argument('--output',required=True);a=p.parse_args()
 df=pd.read_parquet(a.master); fake=df[df.label.astype(int)==1];gens=sorted(fake.generator.unique()); pieces=[]
 for g in gens:
  rows=fake[fake.generator==g].sample(frac=1,random_state=a.seed).iloc[a.offset:a.offset+a.per_generator];pieces.append(rows)
 real=df[df.label.astype(int)==0].sample(frac=1,random_state=a.seed).iloc[a.offset*len(gens):a.offset*len(gens)+a.real_per_generator*len(gens)];pieces.append(real)
 out=pd.concat(pieces).sample(frac=1,random_state=a.seed).copy();out['caption']='';Path(a.output).parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False);print({'rows':len(out),'generators':gens,'real':int((out.label.astype(int)==0).sum()),'output':a.output})
if __name__=='__main__':main()
