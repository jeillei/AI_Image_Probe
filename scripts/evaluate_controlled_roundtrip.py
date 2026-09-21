"""Pre-specified paired evaluation of the shallow round-trip candidate."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

FEATURES=("pixel_roundtrip_mse","latent_roundtrip_mse","eps_mean","path_length","guidance_mean")
def bootstrap_mean(x, seed=17, n=5000):
 r=np.random.default_rng(seed); x=np.asarray(x,float)
 if not len(x): return [None,None]
 b=np.array([r.choice(x,len(x),replace=True).mean() for _ in range(n)])
 return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def one_split(rows, content_ids):
 pairs=[]
 for cid in content_ids:
  rs=[r for r in rows if r['content_id']==cid and r['provenance']=='real']
  fs=[r for r in rows if r['content_id']==cid and r['provenance'].startswith('ai_')]
  if rs and fs: pairs.append((cid,rs[0],fs[0]))
 answer={"n_pairs":len(pairs),"content_ids":[p[0] for p in pairs],"features":{}}
 for feature in FEATURES:
  diff=np.array([f[feature]-r[feature] for _,r,f in pairs])
  vals=np.array([r[feature] for _,r,_ in pairs]+[f[feature] for _,_,f in pairs])
  labels=np.array([0]*len(pairs)+[1]*len(pairs))
  # Either direction can be discriminative in discovery; preserve both the
  # direction-free AUROC and the prespecified lower-roundtrip orientation.
  auc=float(roc_auc_score(labels,-vals)) if len(set(labels))==2 else None
  answer['features'][feature]={"ai_minus_real":diff.tolist(),"mean":float(diff.mean()),"median":float(np.median(diff)),"ci95":bootstrap_mean(diff),"ai_lower_fraction":float(np.mean(diff<0)),"auroc_ai_lower":auc}
 return answer
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='results/controlled_coco/depth_sweep_stage_a.json');p.add_argument('--manifest',default='data/controlled_coco/manifest.csv');p.add_argument('--discovery-groups',type=int,default=3);p.add_argument('--output',default='results/controlled_coco/roundtrip_held_content.json');p.add_argument('--plot',default='results/controlled_coco/roundtrip_held_content.png');a=p.parse_args()
 rows=[r for r in json.loads(Path(a.input).read_text()) if r['steps']==6]
 available={r['content_id'] for r in rows}
 # Manifest order was frozen before feature inspection.  Do not sort IDs:
 # alphabetical ordering could silently turn validation contents into discovery.
 ids=[r['content_id'] for r in csv.DictReader(open(a.manifest)) if r['content_id'] in available]
 disc=ids[:a.discovery_groups]; valid=ids[a.discovery_groups:]
 out={'discovery':one_split(rows,disc),'held_content_validation':one_split(rows,valid)}
 Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
 fig,ax=plt.subplots(figsize=(7,4));
 for name, ids_ in [('discovery',disc),('held content',valid)]:
  pairs=one_split(rows,ids_)['features']['pixel_roundtrip_mse']['ai_minus_real']
  ax.plot(range(1,len(pairs)+1),pairs,'o-',label=name)
 ax.axhline(0,color='black',lw=.8); ax.set(xlabel='paired content group (within split)',ylabel='AI − real pixel round-trip MSE',title='Controlled six-step round-trip differences'); ax.legend();fig.tight_layout();fig.savefig(a.plot,dpi=160)
if __name__=='__main__':main()
