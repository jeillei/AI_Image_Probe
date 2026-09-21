"""Pre-specified learning/anomaly/identity diagnostics on cached Stage-A features."""
from __future__ import annotations
import argparse,json
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def au(y,p): return roc_auc_score(y,p) if len(set(y))==2 else np.nan
def clf(c=1): return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,class_weight='balanced',random_state=17))
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--metrics',required=True);p.add_argument('--output-dir',default='results/stage_a_full');a=p.parse_args()
 rows=json.load(open(a.features)); cols=sorted(rows[0]['features']);df=pd.DataFrame([{**{k:v for k,v in r.items() if k in {'path','label','generator'}},**r['features']} for r in rows]);gens=sorted(df[df.label.eq(1)].generator.unique());real=df[df.label.eq(0)].sort_values('path').reset_index(drop=True);rf={g:set(real.iloc[i*25:(i+1)*25].path) for i,g in enumerate(gens)};out=Path(a.output_dir); met=pd.read_csv(a.metrics)
 reps={'legacy':[c for c in cols if not c.startswith('rich_')],'rich':[c for c in cols if c.startswith('rich_')]}
 curve=[]
 for held in gens:
  test=df[(df.generator.eq(held))|((df.label.eq(0))&df.path.isin(rf[held]))];pool=df.drop(test.index)
  for rep,names in reps.items():
   c=float(met[(met.held_generator.eq(held))&(met.representation.eq(rep))&(met.model.eq('logistic'))].selected_c.iloc[0])
   for n in [5,10,15,20,25]:
    syn=pd.concat([pool[pool.generator.eq(g)].sort_values('path').head(n) for g in gens if g!=held]);rr=pool[pool.label.eq(0)].sort_values('path').head(min(len(pool[pool.label.eq(0)]),7*n))
    tr=pd.concat([syn,rr]);curve.append({'held_generator':held,'representation':rep,'images_per_generator':n,'auroc':au(test.label,clf(c).fit(tr[names],tr.label).predict_proba(test[names])[:,1])})
 pd.DataFrame(curve).to_csv(out/'image_count_learning_curve.csv',index=False)
 gc=[]
 for held in gens:
  allowed=[g for g in gens if g!=held];test=df[(df.generator.eq(held))|((df.label.eq(0))&df.path.isin(rf[held]))];pool=df.drop(test.index);names=reps['rich'];c=float(met[(met.held_generator.eq(held))&(met.representation.eq('rich'))&(met.model.eq('logistic'))].selected_c.iloc[0])
  for n in range(1,8):
   combos=list(combinations(allowed,n)); combos=combos[:min(7,len(combos))]
   for combo in combos:
    tr=pd.concat([pool[pool.generator.isin(combo)],pool[pool.label.eq(0)].sort_values('path').head(25*n)]);gc.append({'held_generator':held,'generator_count':n,'combination':'+'.join(combo),'auroc':au(test.label,clf(c).fit(tr[names],tr.label).predict_proba(test[names])[:,1])})
 pd.DataFrame(gc).to_csv(out/'generator_count_learning_curve.csv',index=False)
 # Generator identity diagnostic (synthetic only), five stratified folds.
 syn=df[df.label.eq(1)];x=syn[reps['rich']].to_numpy();y=syn.generator.to_numpy();pred=np.empty(len(y),object)
 for tr,te in StratifiedKFold(5,shuffle=True,random_state=17).split(x,y): pred[te]=clf(1).fit(x[tr],y[tr]).predict(x[te])
 pd.DataFrame([{'n':len(y),'classes':len(gens),'cv_accuracy':accuracy_score(y,pred),'chance':1/len(gens)}]).to_csv(out/'generator_identity_diagnostic.csv',index=False)
 # Real-only anomaly baselines, outer held-generator test sets.
 anomaly=[]
 for held in gens:
  te=df[(df.generator.eq(held)) | ((df.label.eq(0)) & df.path.isin(rf[held]))];tr=df[(df.label.eq(0))&~df.path.isin(rf[held])];xtr=tr[reps['rich']].to_numpy();xte=te[reps['rich']].to_numpy();sc=StandardScaler().fit(xtr);z=sc.transform(xte);diag=np.sum(z*z,1);lw=LedoitWolf().fit(sc.transform(xtr));mah=lw.mahalanobis(z);pca=PCA(n_components=20,random_state=17).fit(sc.transform(xtr));rec=np.mean((z-pca.inverse_transform(pca.transform(z)))**2,1)
  for name,v in [('diagonal_distance',diag),('shrinkage_mahalanobis',mah),('pca_reconstruction',rec)]: anomaly.append({'held_generator':held,'method':name,'auroc':au(te.label,v)})
 pd.DataFrame(anomaly).to_csv(out/'real_manifold_baseline.csv',index=False)
 # Fixed descriptive plots, not used for selection.
 for file,xlab in [('image_count_learning_curve.csv','images per training generator'),('generator_count_learning_curve.csv','training generators')]:
  q=pd.read_csv(out/file);fig,ax=plt.subplots(figsize=(6,4));key='images_per_generator' if 'image' in file else 'generator_count'
  for rep,g in q.groupby('representation') if 'representation' in q else [('rich',q)]: ax.plot(g.groupby(key).auroc.mean().index,g.groupby(key).auroc.mean().values,marker='o',label=rep)
  ax.set(xlabel=xlab,ylabel='mean held-generator AUROC',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/(file.replace('.csv','.png')),dpi=160);plt.close(fig)
if __name__=='__main__':main()
