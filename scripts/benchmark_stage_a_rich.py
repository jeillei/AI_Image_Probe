"""Nested generator-held-out Stage-A benchmark from cached trajectory features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,balanced_accuracy_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEGACY=lambda cols:[c for c in cols if not c.startswith('rich_')]
RICH=lambda cols:[c for c in cols if c.startswith('rich_')]
def core_rich(cols):
    """Frozen before Stage-B: noise magnitude + score dynamics + spatial score."""
    return [c for c in cols if c.startswith('eps_') or c.startswith(('rich_score_norm_','rich_score_change_','rich_score_consecutive_','rich_score_angle_','rich_score_relchange_','rich_score_secondchange_','rich_score_crosscos_')) or (c.startswith('rich_score_t') and any(q in c for q in ('channel_','autocorr','_fft_','_var','_absmean','_rms','_skew','_kurtosis')))]
FAMILIES={
 'noise_magnitude':lambda c:[x for x in c if x.startswith('eps_')],
 'score_dynamics':lambda c:[x for x in c if x.startswith(('rich_score_norm_','rich_score_change_','rich_score_consecutive_','rich_score_angle_','rich_score_relchange_','rich_score_secondchange_','rich_score_crosscos_'))],
 'latent_geometry':lambda c:[x for x in c if x.startswith(('rich_latent_step_','rich_latent_turn_','rich_latent_accel_','rich_latent_path_','rich_latent_endpoint_','rich_latent_cumulative_','rich_dz_crosscos_'))],
 'alignment':lambda c:[x for x in c if x.startswith(('rich_cond_dz_','rich_uncond_dz_','rich_guidance_dz_'))],
 'conditional_geometry':lambda c:[x for x in c if x.startswith(('rich_cond_uncond_','rich_guidance_cond_'))],
 'spatial_score':lambda c:[x for x in c if x.startswith('rich_score_t') and any(q in x for q in ('channel_','autocorr'))],
 'frequency_score':lambda c:[x for x in c if x.startswith('rich_score_t') and '_fft_' in x],
 'latent_typicality':lambda c:[x for x in c if x.startswith('rich_latent_t')],
 'cross_timestep':lambda c:[x for x in c if x.startswith(('rich_score_crosscos_','rich_dz_crosscos_'))],
 'roundtrip_endpoint':lambda c:[x for x in c if x.startswith(('pixel_','latent_','endpoint_','roundtrip_'))],
}
C_GRID=[.01,.1,1.,10.,100.]
PCA_GRID=[5,10,20,40,80]

def auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
def boot(y,p,n=400):
 rng=np.random.default_rng(17); vals=[]
 for _ in range(n):
  i=rng.integers(0,len(y),len(y));
  if len(np.unique(y[i]))==2: vals.append(roc_auc_score(y[i],p[i]))
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def metric(y,p,threshold):
 return {'auroc':auc(y,p),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'tpr_at_1pct_fpr':float(np.mean(p[y==1]>=threshold[.01])),'tpr_at_5pct_fpr':float(np.mean(p[y==1]>=threshold[.05])),'auroc_ci95':boot(y,p)}
def model(kind,c,pca=None):
 if kind=='boosted': return make_pipeline(StandardScaler(),HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=8,l2_regularization=2.,learning_rate=.06,random_state=17))
 steps=[StandardScaler()]
 if pca: steps.append(PCA(n_components=pca,random_state=17))
 steps.append(LogisticRegression(C=c,max_iter=3000,class_weight='balanced',random_state=17))
 return make_pipeline(*steps)
def threshold(y,p):
 n=np.sort(p[y==0]);return {q:float(n[max(0,int(np.ceil((1-q)*len(n)))-1)]) for q in (.01,.05)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output-dir',default='results/stage_a_full');a=ap.parse_args()
 rows=json.loads(Path(a.features).read_text());meta={'path','label','generator','content_id','split','severity','caption','seconds'}
 cols=sorted(set(rows[0]['features']));df=pd.DataFrame([{**{k:v for k,v in r.items() if k in meta},**r['features']} for r in rows])
 if len(df)<400 or df.path.nunique()!=len(df) or not np.isfinite(df[cols].to_numpy(float)).all(): raise SystemExit('requires at least 400 unique finite feature rows')
 gens=sorted(df.loc[df.label.eq(1),'generator'].unique());real=df[df.label.eq(0)].sort_values('path').reset_index(drop=True);fold_size=len(real)//len(gens);real_folds={g:set(real.iloc[i*fold_size:(i+1)*fold_size].path) for i,g in enumerate(gens)}
 reps={'legacy':LEGACY(cols),'rich':RICH(cols),'core_rich':core_rich(cols),'combined':cols}
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);allmetrics=[]
 def fit_eval(tr,te,names,kind='logistic',pca=None,c=1.):
  m=model(kind,c,pca);m.fit(tr[names],tr.label);return m.predict_proba(te[names])[:,1]
 for held in gens:
  test=df[(df.generator.eq(held)) | ((df.label.eq(0)) & df.path.isin(real_folds[held]))]; pool=df.drop(test.index); validators=[g for g in gens if g!=held]
  for rep,names in reps.items():
   for kind,pca_mode in [('logistic',False),('pca_logistic',True),('boosted',False)]:
    choices=[]
    configs=[(c,d) for c in C_GRID for d in (PCA_GRID if pca_mode else [None])] if kind!='boosted' else [(1.,None)]
    for c,d in configs:
     vals=[];py=[];pp=[]
     for val in validators:
      vr=real_folds[val];va=pool[(pool.generator.eq(val)) | ((pool.label.eq(0))&pool.path.isin(vr))];tr=pool.drop(va.index)
      maxd=min(tr.shape[0]-1,len(names),d or 9999)
      if d and d>maxd: continue
      p=fit_eval(tr,va,names,'boosted' if kind=='boosted' else 'logistic',d,c);vals.append(auc(va.label.to_numpy(),p));py.extend(va.label);pp.extend(p)
     if vals: choices.append((float(np.mean(vals)),c,d,threshold(np.array(py),np.array(pp))))
    best=max(choices,key=lambda z:z[0]);p=fit_eval(pool,test,names,'boosted' if kind=='boosted' else 'logistic',best[2],best[1]);r={'held_generator':held,'representation':rep,'model':kind,'n_features':len(names),'inner_auroc':best[0],'selected_c':best[1],'selected_pca':best[2],**metric(test.label.to_numpy(),p,best[3])};allmetrics.append(r)
 # Rich family ablations with nested raw logistic.
 for family,pick in FAMILIES.items():
  names=pick(cols)
  for held in gens:
   test=df[(df.generator.eq(held)) | ((df.label.eq(0))&df.path.isin(real_folds[held]))];pool=df.drop(test.index); vals=[]
   for c in C_GRID:
    v=[]
    for g in gens:
     if g==held:continue
     va=pool[(pool.generator.eq(g))|((pool.label.eq(0))&pool.path.isin(real_folds[g]))];tr=pool.drop(va.index);v.append(auc(va.label,fit_eval(tr,va,names,'logistic',None,c)))
    vals.append((np.mean(v),c))
   c=max(vals)[1];p=fit_eval(pool,test,names,'logistic',None,c);allmetrics.append({'held_generator':held,'representation':family,'model':'family_logistic','n_features':len(names),'selected_c':c,**metric(test.label.to_numpy(),p,{.01:.5,.05:.5})})
 pd.DataFrame(allmetrics).to_csv(out/'logo_metrics.csv',index=False)
 # Family median summary plus alignment curves.
 fam=pd.DataFrame([x for x in allmetrics if x['model']=='family_logistic']);fam.groupby('representation').auroc.agg(['median','mean','min']).reset_index().to_csv(out/'feature_family_ablation.csv',index=False)
 curve=[]
 for group in ['real']+gens:
  q=df[df.label.eq(0)] if group=='real' else df[df.generator.eq(group)]
  for t in range(6):
   col=f'rich_cond_dz_align_t{t}'
   if col in q: curve.append({'group':group,'step':t,'n':len(q),'mean':q[col].mean(),'std':q[col].std()})
 pd.DataFrame(curve).to_csv(out/'score_alignment_curves.csv',index=False)
 print(pd.DataFrame(allmetrics).groupby(['representation','model']).auroc.agg(['median','mean','min']).round(3))
if __name__=='__main__':main()
