"""Fit the explicitly experimental all-Stage-A rich logistic deployment model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',default='results/features/stage_a_full_rich.json');p.add_argument('--output',default='results/detector/sd15_rich_stage_a_experimental.joblib');p.add_argument('--c',type=float,default=.1);a=p.parse_args()
 rows=json.loads(Path(a.features).read_text());names=sorted(k for k in rows[0]['features'] if k.startswith('rich_'))
 x=np.array([[r['features'][n] for n in names] for r in rows],float);y=np.array([r['label'] for r in rows],int)
 model=make_pipeline(StandardScaler(),LogisticRegression(C=a.c,max_iter=3000,class_weight='balanced',random_state=17)).fit(x,y)
 artifact={'model':model,'feature_names':names,'family':'rich_trajectory','probe':'stable-diffusion-v1-5','steps':6,'size':256,'capture_rich':True,'status':'experimental_stage_a_not_generator_general','warning':'Stage-A median held-generator AUROC 0.722, but DALL-E 2 AUROC 0.421. This is an experimental score, not proof of provenance.'}
 o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,o);print({'output':str(o),'rows':len(rows),'features':len(names),'c':a.c})
if __name__=='__main__':main()
