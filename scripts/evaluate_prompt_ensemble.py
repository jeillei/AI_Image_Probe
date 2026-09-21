"""Aggregate cached prompt ensembles and compare prompt-aware rich signatures."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID=[.001,.01,.1,1.,10.]
def auc(y,p): return float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan
def fit(x,y,c): return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,class_weight='balanced',random_state=17)).fit(x,y)
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--detector',default='results/detector/sd15_rich_stage_a_experimental.joblib');p.add_argument('--output-dir',default='results/prompt_ensemble');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 rows=json.load(open(a.features));rich=sorted(k for k in rows[0]['features'] if k.startswith('rich_'));flat=[]
 for r in rows: flat.append({'path':r['path'],'label':r['label'],'generator':r['generator'],'caption_id':r['caption_id'],**{k:r['features'][k] for k in rich}})
 d=pd.DataFrame(flat); assert d.groupby('path').size().eq(5).all()
 # Detector-score stability is diagnostic only; model is not used to choose representations.
 art=joblib.load(a.detector);score=art['model'].predict_proba(d[art['feature_names']])[:,1];d['detector_score']=score
 stability=d.groupby('path').agg(label=('label','first'),generator=('generator','first'),score_mean=('detector_score','mean'),score_std=('detector_score','std'),score_min=('detector_score','min'),score_max=('detector_score','max')).reset_index();stability['score_range']=stability.score_max-stability.score_min;stability['classification_flip']=d.groupby('path').detector_score.apply(lambda x: int((x>=.5).nunique()>1)).to_numpy();stability.to_csv(out/'prompt_score_stability.csv',index=False)
 # Aggregate rich features.  Mean and SD are both numerical trajectory features.
 means=d.groupby('path')[rich].mean().add_prefix('mean_');stds=d.groupby('path')[rich].std(ddof=1).add_prefix('std_');meta=d.groupby('path').agg(label=('label','first'),generator=('generator','first')).reset_index().set_index('path');agg=pd.concat([meta,means,stds],axis=1).reset_index();agg.to_parquet(out/'image_prompt_aggregates.parquet',index=False)
 # Per-family prompt variance (pre-specified groups).
 fam={'noise_curve':['rich_score_norm_'],'score_dynamics':['rich_score_change_','rich_score_consecutive_','rich_score_angle_'],'spatial_score':['rich_score_t'],'frequency_score':['_fft_'],'latent_typicality':['rich_latent_t']};fv=[]
 for name,prefix in fam.items():
  names=[c for c in rich if any(q in c for q in prefix)]
  if name=='spatial_score': names=[c for c in names if any(q in c for q in ('channel_','autocorr','_var','_absmean'))]
  sens=d.groupby('path')[names].std().mean(axis=1);q=meta.copy();q['sensitivity']=sens
  for g,gg in q.groupby('generator'): fv.append({'family':name,'generator':g,'n':len(gg),'mean_prompt_std':gg.sensitivity.mean(),'median_prompt_std':gg.sensitivity.median()})
 pd.DataFrame(fv).to_csv(out/'prompt_variance_by_family.csv',index=False)
 # Same eight outer generator folds; each gets 5 disjoint real test images.
 gens=sorted(agg[agg.label.eq(1)].generator.unique());real=agg[agg.label.eq(0)].sort_values('path').reset_index(drop=True);rf={g:set(real.iloc[i*5:(i+1)*5].path) for i,g in enumerate(gens)}
 single=d[d.caption_id.eq(0)].set_index('path')[rich];single.index.name='path';reps={'single':single,'prompt_mean':means,'prompt_mean_sensitivity':pd.concat([means,stds],axis=1)};metrics=[]
 for held in gens:
  test_ids=set(agg[agg.generator.eq(held)].path)|rf[held];test=agg[agg.path.isin(test_ids)];pool=agg[~agg.path.isin(test_ids)]
  for rep,xall in reps.items():
   best=[]
   for c in C_GRID:
    vals=[]
    for val in gens:
     if val==held:continue
     vaids=set(pool[pool.generator.eq(val)].path)|rf[val];va=pool[pool.path.isin(vaids)];tr=pool[~pool.path.isin(vaids)];m=fit(xall.loc[tr.path],tr.label,c);vals.append(auc(va.label,m.predict_proba(xall.loc[va.path])[:,1]))
    best.append((np.nanmean(vals),c))
   _,c=max(best);m=fit(xall.loc[pool.path],pool.label,c);pr=m.predict_proba(xall.loc[test.path])[:,1];metrics.append({'held_generator':held,'representation':rep,'selected_c':c,'n_features':xall.shape[1],'auroc':auc(test.label,pr)})
 pd.DataFrame(metrics).to_csv(out/'logo_prompt_comparison.csv',index=False)
 # Timestep-level prompt variance for E/G curves.
 curves=[]
 for base in ['eps_t','guidance_t','rich_score_norm_t']:
  names=[c for c in d if c.startswith(base)]
  for t,n in enumerate(names):
   q=d.groupby('path')[n].var().rename('variance').to_frame().join(meta)
   for g,gg in q.groupby('generator'): curves.append({'curve':base,'step':t,'generator':g,'mean_prompt_variance':gg.variance.mean(),'n':len(gg)})
 pd.DataFrame(curves).to_csv(out/'prompt_timestep_variance.csv',index=False)
 print(pd.DataFrame(metrics).pivot(index='held_generator',columns='representation',values='auroc').round(3));print(stability[['score_std','score_range','classification_flip']].mean().to_dict())
if __name__=='__main__':main()
