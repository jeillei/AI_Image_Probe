"""Append duplicate-audited RR real photographs to the existing Stage-B manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='data/aigc_stage_b/manifest.csv');p.add_argument('--audit',default='results/stage_b/rr_real_duplicate_audit.json');p.add_argument('--output',default='data/stage_b_cross_source/manifest.csv');a=p.parse_args()
 base=pd.read_csv(a.base);paths=[Path(x) for x in json.loads(Path(a.audit).read_text())['kept']];rows=[]
 for i,path in enumerate(paths):
  rows.append({'image_id':f'rrreal_{path.stem}','path':str(path),'label':0,'provenance':'real','generator':'real','generator_family':'real','real_source':'rrdataset_real','content_id':f'rrreal_{path.stem}','caption':'','caption_status':'pending','split':'pending','source':'RRDataset_original_train_val','original_width':'','original_height':'','original_format':path.suffix.lstrip('.'),'feature_status':'pending'})
 out=pd.concat([base,pd.DataFrame(rows)],ignore_index=True)
 if out.image_id.duplicated().any() or out.path.duplicated().any() or not out.path.map(lambda p:Path(p).is_file()).all():raise SystemExit('invalid extended manifest')
 o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);out.to_csv(o,index=False);print({'rows':len(out),'real_sources':out[out.label.eq(0)].real_source.value_counts().to_dict()})
if __name__=='__main__':main()
