"""Create a cached, conservative five-prompt conditioning ensemble per image."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

TEMPLATES=(
 ('blip_base','{caption}'),
 ('depicting','an image depicting {caption}'),
 ('scene','a scene showing {caption}'),
 ('detailed','a detailed view of {caption}'),
 ('featuring','an image featuring {caption}'),
)
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',default='results/prompts/stage_a_pilot_prompt_ensemble.parquet');a=p.parse_args()
 rows=json.load(open(a.features));out=[]
 for r in rows:
  cap=' '.join(r['caption'].strip().split())
  for i,(method,t) in enumerate(TEMPLATES): out.append({'image_id':r['path'],'path':r['path'],'label':r['label'],'generator':r['generator'],'content_id':r.get('content_id',''),'caption_id':i,'caption':t.format(caption=cap),'caption_method':method,'seed':None})
 df=pd.DataFrame(out);o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(o,index=False);df.to_csv(o.with_suffix('.csv'),index=False);print({'images':df.image_id.nunique(),'prompt_rows':len(df),'captions_per_image':df.groupby('image_id').size().unique().tolist(),'output':str(o)})
if __name__=='__main__':main()
