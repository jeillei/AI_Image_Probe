"""Generator-aware real-source and crossed provenance evaluation from cached rich features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,balanced_accuracy_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
C=[.001,.01,.1,1.,10.]
def fit(x,y,c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,class_weight='balanced',random_state=17)).fit(x,y)
def metrics(y,p,thr):return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'tpr_at_1pct_fpr':float(np.mean(p[y==1]>=thr[.01])),'tpr_at_5pct_fpr':float(np.mean(p[y==1]>=thr[.05]))}
def thresholds(y,p):
 neg=np.sort(p[y==0]);return {q:float(neg[max(0,int(np.ceil((1-q)*len(neg)))-1)]) for q in [.01,.05]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output-dir',default='results/stage_b_scaled');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 rows=json.load(open(a.features));cols=sorted(k for k in rows[0]['features'] if k.startswith('rich_'));d=pd.DataFrame([{'path':r['path'],'label':r['label'],'generator':r['generator'],'real_source':'rrdataset_real' if 'rr_stage_b_real' in r['path'] else ('aigc_benchmark_real' if r['label']==0 else ''),**{k:r['features'][k] for k in cols}} for r in rows]);gens=sorted(d[d.label.eq(1)].generator.unique());sources=sorted(d[d.label.eq(0)].real_source.unique());res=[]
 for source in sources:
  for held in gens:
   test=d[(d.generator.eq(held))|((d.label.eq(0))&d.real_source.eq(source))];pool=d.drop(test.index);allowed=sorted(pool[pool.label.eq(1)].generator.unique());best=[]
   for c in C:
    vs=[];vy=[];vp=[]
    for val in allowed:
     syn=pool[pool.generator.eq(val)];real=pool[pool.label.eq(0)].sort_values('path').iloc[:len(syn)];va=pd.concat([syn,real]);tr=pool.drop(va.index);m=fit(tr[cols],tr.label,c);pr=m.predict_proba(va[cols])[:,1];vs.append(roc_auc_score(va.label,pr));vy.extend(va.label);vp.extend(pr)
    best.append((np.mean(vs),c,thresholds(np.asarray(vy),np.asarray(vp))))
   _,c,thr=max(best,key=lambda z:z[0]);m=fit(pool[cols],pool.label,c);pr=m.predict_proba(test[cols])[:,1];res.append({'held_generator':held,'held_real_source':source,'selected_c':c,'n_train':len(pool),'n_test':len(test),**metrics(test.label.to_numpy(),pr,thr)})
 pd.DataFrame(res).to_csv(out/'cross_source_metrics.csv',index=False)
 realout=[]
 for source in sources:
  te=d[(d.label.eq(0))&d.real_source.eq(source)];tr=d.drop(te.index);m=fit(tr[cols],tr.label,.1);pr=m.predict_proba(te[cols])[:,1];realout.append({'held_real_source':source,'n':len(te),'mean_score':pr.mean(),'fraction_score_ge_0_5':float(np.mean(pr>=.5))})
 pd.DataFrame(realout).to_csv(out/'real_source_holdout.csv',index=False);print(pd.DataFrame(res).groupby('held_real_source').auroc.agg(['median','mean','min']).round(3))
if __name__=='__main__':main()
