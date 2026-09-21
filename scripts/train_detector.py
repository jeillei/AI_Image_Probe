"""Fit interpretable detector baselines from trajectory features only."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np
from sklearn.metrics import average_precision_score,balanced_accuracy_score,roc_auc_score
from src.classifiers.detector import fit_boosted,fit_logistic,score
from src.features.compatibility import family_names

RICH_PREFIXES = {
 'rich_score_dynamics': ('rich_score_norm_', 'rich_score_change_', 'rich_score_consecutive_cos_', 'rich_score_angle_', 'rich_score_relchange_', 'rich_score_secondchange_', 'rich_score_crosscos_'),
 'rich_alignment': ('rich_cond_dz_', 'rich_uncond_dz_', 'rich_guidance_dz_', 'rich_cond_uncond_', 'rich_guidance_cond_'),
 'rich_geometry': ('rich_latent_step_', 'rich_latent_turn_', 'rich_latent_accel_', 'rich_latent_path_', 'rich_latent_endpoint_', 'rich_latent_cumulative_', 'rich_dz_crosscos_'),
 'rich_spatial': ('rich_score_t',),
 'rich_typicality': ('rich_latent_t',),
}

def rich_names(rows, family):
    columns = sorted({key for row in rows for key in row['features']})
    if family == 'rich': return [key for key in columns if key.startswith('rich_')]
    prefixes = RICH_PREFIXES[family]
    selected = [key for key in columns if key.startswith(prefixes)]
    # Spatial and typicality prefixes overlap score/latent distribution fields;
    # retain the intended mechanism only.
    if family == 'rich_spatial': selected = [key for key in selected if any(x in key for x in ('channel_', 'autocorr', 'fft_'))]
    if family == 'rich_typicality': selected = [key for key in selected if any(x in key for x in ('_mean','_var','_skew','_kurtosis','_extreme3','_pr','_maxabs'))]
    return selected

def thresholds_at_fpr(y,p):
    neg=np.sort(p[y==0]); return {str(q):float(neg[max(0,int(np.ceil((1-q)*len(neg)))-1)]) if len(neg) else 1. for q in (.01,.05)}
def metrics(y,p,thresholds=None):
 ans={'n':len(y),'auroc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'auprc':float(average_precision_score(y,p)) if len(set(y))==2 else None,'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}
 for q in (.01,.05):
  threshold=(thresholds or thresholds_at_fpr(y,p))[str(q)]
  ans[f'tpr_at_{int(q*100)}pct_fpr']=float(np.mean(p[y==1]>=threshold)) if np.any(y==1) else None
 return ans
def bootstrap_auroc(y,p,n=1000):
    rng=np.random.default_rng(17);y=np.asarray(y);p=np.asarray(p); vals=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))==2: vals.append(roc_auc_score(y[idx],p[idx]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--family',choices=['noise','guidance','roundtrip','endpoint','geometry','noise_guidance','full','rich','rich_score_dynamics','rich_alignment','rich_geometry','rich_spatial','rich_typicality'],default='noise');p.add_argument('--model',choices=['logistic','boosted'],default='logistic');p.add_argument('--train-split',default='train');p.add_argument('--validation-split',default='val');p.add_argument('--test-split',default='test');p.add_argument('--output',default='results/detector/sd15_detector.joblib');p.add_argument('--report',default='results/detector/report.json');a=p.parse_args()
 rows=json.loads(Path(a.features).read_text()); names=rich_names(rows,a.family) if a.family.startswith('rich') else family_names(a.family)
 def pack(rs): return np.array([[r['features'][n] for n in names] for r in rs],float),np.array([r['label'] for r in rs],int)
 train=[r for r in rows if r['split']==a.train_split]; val=[r for r in rows if r['split']==a.validation_split]; test=[r for r in rows if r['split']==a.test_split]
 x,y=pack(train)
 if len(train)<4 or len(set(y))<2: raise SystemExit('training split needs both classes and at least four rows')
 model=fit_logistic(x,y) if a.model=='logistic' else fit_boosted(x,y)
 result={'feature_family':a.family,'feature_names':names,'model':a.model,'n_train':len(train),'train':metrics(y,score(model,x))}
 frozen_thresholds=None
 for name,rs in [('validation',val),('test',test)]:
  if rs:
   xx,yy=pack(rs); pp=score(model,xx); result[name]=metrics(yy,pp, frozen_thresholds); result[name]['auroc_bootstrap_ci95']=bootstrap_auroc(yy,pp); result[name]['by_generator']={}
   if name=='validation': frozen_thresholds=thresholds_at_fpr(yy,pp); result['frozen_validation_thresholds']=frozen_thresholds
   for g in sorted({r['generator'] for r in rs}):
    gg=[r for r in rs if r['label']==0 or r['generator']==g]
    gx,gy=pack(gg); gp=score(model,gx); result[name]['by_generator'][g]=metrics(gy,gp,frozen_thresholds if name=='test' else None) if len(set(gy))==2 else {'n':len(gy)}
    if len(set(gy))==2: result[name]['by_generator'][g]['auroc_bootstrap_ci95']=bootstrap_auroc(gy,gp)
 artifact={'model':model,'feature_names':names,'family':a.family,'probe':'stable-diffusion-v1-5','steps':6,'size':256,'status':'experimental_single_source' if len({r['generator'] for r in rows if r['label']==1})<2 else 'experimental'}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,a.output);Path(a.report).parent.mkdir(parents=True,exist_ok=True);Path(a.report).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
