"""Paired, descriptive analysis for controlled trajectory sweeps."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

FEATURES=["pixel_roundtrip_mse","latent_roundtrip_mse","path_length","speed_mean","acceleration_abs_mean","eps_mean","guidance_mean","endpoint_norm_per_dim","endpoint_var","endpoint_kurtosis"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",default="results/controlled_coco/depth_sweep_stage_a.json");p.add_argument("--output",default="results/controlled_coco/depth_sweep_stage_a_summary.json");a=p.parse_args()
 rows=json.loads(Path(a.input).read_text()); summary={}
 for depth in sorted({x['steps'] for x in rows}):
  rs=[x for x in rows if x['steps']==depth]; by={}
  ids=sorted({x['content_id'] for x in rs})
  for feat in FEATURES:
   paired=[]
   for cid in ids:
    real=[x for x in rs if x['content_id']==cid and x['provenance']=='real']
    fake=[x for x in rs if x['content_id']==cid and x['provenance'].startswith('ai_')]
    if real and fake: paired.append(fake[0][feat]-real[0][feat])
   vals=np.asarray(paired,float)
   by[feat]={"n_pairs":len(vals),"ai_minus_real_mean":float(vals.mean()) if len(vals) else None,"ai_minus_real_median":float(np.median(vals)) if len(vals) else None,"sign_consistency":float(np.mean(vals>0)) if len(vals) else None,"values":vals.tolist()}
  summary[str(depth)]=by
 Path(a.output).write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
