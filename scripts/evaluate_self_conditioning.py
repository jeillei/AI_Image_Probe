"""Nested pilot comparison of null, automatic, and conditioning-response features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
C=[.001,.01,.1,1.,10.]
def auc(y,p): return float(roc_auc_score(y,p))
def fit(x,y,c): return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,class_weight='balanced',random_state=17)).fit(x,y)
def main():
 p=argparse.ArgumentParser();p.add_argument('--null',required=True);p.add_argument('--auto',required=True);p.add_argument('--output-dir',default='results/self_conditioning');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 n=json.load(open(a.null));u=json.load(open(a.auto));cols=sorted(k for k in n[0]['features'] if k.startswith('rich_'));N={r['path']:r for r in n};U={r['path']:r for r in u};paths=sorted(N)
 meta=pd.DataFrame([{'path':x,'label':N[x]['label'],'generator':N[x]['generator']} for x in paths]).set_index('path');xn=pd.DataFrame({x:N[x]['features'] for x in paths}).T[cols];xa=pd.DataFrame({x:U[x]['features'] for x in paths}).T[cols];resp=xa-xn
 pd.concat([meta,xn.add_prefix('null_'),xa.add_prefix('auto_'),resp.add_prefix('response_')],axis=1).reset_index().to_parquet(out/'conditioning_response.parquet',index=False)
 reps={'null_only':xn,'auto':xa,'null_auto':pd.concat([xn.add_prefix('null_'),xa.add_prefix('auto_')],axis=1),'response':resp,'null_auto_response':pd.concat([xn.add_prefix('null_'),xa.add_prefix('auto_'),resp.add_prefix('response_')],axis=1)}
 gens=sorted(meta[meta.label.eq(1)].generator.unique());real=meta[meta.label.eq(0)].sort_index();rf={g:set(real.index[i*5:(i+1)*5]) for i,g in enumerate(gens)};res=[]
 for held in gens:
  teids=set(meta[meta.generator.eq(held)].index)|rf[held];te=meta.loc[list(teids)];pool=meta.drop(list(teids))
  for name,x in reps.items():
   best=[]
   for c in C:
    vaus=[]
    for val in gens:
     if val==held:continue
     ids=set(pool[pool.generator.eq(val)].index)|rf[val];va=pool.loc[list(ids)];tr=pool.drop(list(ids));vaus.append(auc(va.label,fit(x.loc[tr.index],tr.label,c).predict_proba(x.loc[va.index])[:,1]))
    best.append((np.mean(vaus),c))
   _,c=max(best);pr=fit(x.loc[pool.index],pool.label,c).predict_proba(x.loc[te.index])[:,1];res.append({'held_generator':held,'representation':name,'selected_c':c,'n_features':x.shape[1],'auroc':auc(te.label,pr)})
 pd.DataFrame(res).to_csv(out/'pilot_logo_metrics.csv',index=False);print(pd.DataFrame(res).pivot(index='held_generator',columns='representation',values='auroc').round(3))
if __name__=='__main__':main()
