"""Extend Stage-A with only genuinely new, labelled synthetic sources."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage-a',default='data/aigc_stage_a/manifest.csv');p.add_argument('--extra',default='data/aigc_stage_b_extra/manifest.csv');p.add_argument('--output',default='data/aigc_stage_b/manifest.csv');a=p.parse_args()
 base=pd.read_csv(a.stage_a);extra=pd.read_csv(a.extra);extra=extra[extra.label.astype(int).eq(1)].copy();both=pd.concat([base,extra],ignore_index=True)
 if both.image_id.duplicated().any() or both.path.duplicated().any() or not both.path.map(lambda x:Path(x).is_file()).all():raise SystemExit('invalid Stage-B manifest')
 o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);both.to_csv(o,index=False);print({'rows':len(both),'real':int((both.label==0).sum()),'synthetic':int((both.label==1).sum()),'generators':both[both.label.eq(1)].generator.value_counts().to_dict()})
if __name__=='__main__':main()
