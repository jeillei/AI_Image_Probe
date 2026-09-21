"""Merge source manifests into one auditable master Parquet manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
REQUIRED=['image_id','path','label','provenance','generator','generator_family','real_source','content_id','caption','caption_status','split','source','original_width','original_height','original_format','feature_status']
def main():
 p=argparse.ArgumentParser();p.add_argument('manifests',nargs='+');p.add_argument('--output',default='data/master_manifest.parquet');a=p.parse_args()
 frames=[]
 for source in a.manifests:
  x=pd.read_csv(source)
  for col in REQUIRED:
   if col not in x: x[col]=''
  frames.append(x[REQUIRED])
 df=pd.concat(frames,ignore_index=True)
 if df.image_id.duplicated().any(): raise SystemExit('duplicate image_id: namespace source manifests before merging')
 if not df.path.map(lambda value: Path(value).is_file()).all(): raise SystemExit('master manifest has missing local paths')
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);df.to_parquet(a.output,index=False);df.to_csv(Path(a.output).with_suffix('.csv'),index=False)
 print({'rows':len(df),'synthetic_generators':sorted(df.loc[df.label.astype(int)==1,'generator'].unique()),'real_sources':sorted(df.loc[df.label.astype(int)==0,'real_source'].unique()),'output':a.output})
if __name__=='__main__':main()
