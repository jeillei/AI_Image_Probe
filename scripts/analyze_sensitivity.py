from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',default='results/sd15_conditioning/sensitivity_null_caption.json');p.add_argument('--output',default='results/sd15_conditioning/analysis');p.add_argument('--seed',type=int,default=31);a=p.parse_args()
df=pd.DataFrame(json.loads(Path(a.input).read_text())); out=Path(a.output);out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'sensitivity.csv',index=False)
rng=np.random.default_rng(a.seed); feats=['null_mse','caption_mse','caption_benefit','caption_benefit_norm','trajectory_displacement_mean','endpoint_displacement','guidance_response_mean']
rows=[]
for f in feats:
 r=df[df.label==0][f].to_numpy(); s=df[df.label==1][f].to_numpy(); dif=r.mean()-s.mean(); boots=[]
 for _ in range(2000): boots.append(rng.choice(r,len(r),True).mean()-rng.choice(s,len(s),True).mean())
 pooled=np.sqrt((r.var(ddof=1)+s.var(ddof=1))/2); d=float(dif/(pooled+1e-12)); auc=float(roc_auc_score(df.label,df[f])); auc=max(auc,1-auc)
 rows.append({'feature':f,'real_mean':float(r.mean()),'synthetic_mean':float(s.mean()),'real_median':float(np.median(r)),'synthetic_median':float(np.median(s)),'mean_difference':float(dif),'bootstrap_95_ci':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],'cohen_d':d,'univariate_auroc_orientation_free':auc})
(out/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
fig,axs=plt.subplots(1,3,figsize=(11,3));
for ax,f in zip(axs,['caption_benefit_norm','trajectory_displacement_mean','guidance_response_mean']):
 ax.scatter(np.full((df.label==0).sum(),0),df[df.label==0][f],alpha=.8,label='real');ax.scatter(np.full((df.label==1).sum(),1),df[df.label==1][f],alpha=.8,label='AI');ax.set_xticks([0,1],['real','AI']);ax.set_title(f)
fig.tight_layout();fig.savefig(out/'distributions.png',dpi=160)
