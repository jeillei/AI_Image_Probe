"""Run the real-DDPM trajectory pilot, with matched corruption and no metadata features."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import sys
# Permit `python scripts/run_pilot.py` as documented without requiring install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.corruption import apply_chain
from src.probes import CifarDDPMProbe
from src.features import extract_features

IMG={'.jpg','.jpeg','.png','.webp'}
def collect(root):
    rows=[]
    for label,name in [(0,'real'),(1,'synthetic')]:
        base=root/name
        for p in base.rglob('*') if base.exists() else []:
            if p.suffix.lower() in IMG:
                rows.append({'path':p,'label':label,'group':p.parent.name})
    return rows
def tensor(img):
    x=np.asarray(img.resize((32,32),Image.Resampling.LANCZOS).convert('RGB'),dtype=np.float32)/127.5-1
    return x.transpose(2,0,1)
def lowfreq_embedding(x):
    # Generic visual baseline: 8x8 RGB thumbnail, no trajectory/model access.
    return x[:,::4,::4].ravel()
def metric(y,s):
    if len(set(y))<2:return {}
    fpr,tpr,_=roc_curve(y,s); at=lambda q:float(np.interp(q,fpr,tpr))
    return {'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s)), 'tpr_at_1pct_fpr':at(.01),'tpr_at_5pct_fpr':at(.05)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); ap.add_argument('--output',default='results/pilot'); ap.add_argument('--steps',type=int,default=12); ap.add_argument('--max-per-group',type=int,default=8); ap.add_argument('--severity',type=int,default=0); ap.add_argument('--seed',type=int,default=7)
    a=ap.parse_args(); random.seed(a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    samples=collect(Path(a.data)); assert samples, 'No images found in data/{real,synthetic}/<group>/'
    # Prevent any file encoding/size shortcut: load pixels, strip metadata, canonicalize then corrupt.
    selected=[]; counts={}
    for s in samples:
        k=(s['label'],s['group']); counts[k]=counts.get(k,0)
        if counts[k]<a.max_per_group: selected.append(s); counts[k]+=1
    probe=CifarDDPMProbe(a.steps)
    records=[]; embeds=[]
    for i,s in enumerate(selected):
        im=ImageOps.exif_transpose(Image.open(s['path'])).convert('RGB')
        im=apply_chain(im,a.severity,a.seed+i,base_size=256)
        x=tensor(im); res=probe.invert_and_reconstruct(x); feat=extract_features(res,x)
        feat.update({'path':str(s['path']),'label':s['label'],'group':s['group'],'severity':a.severity})
        records.append(feat); embeds.append(lowfreq_embedding(x))
        print(f'[{i+1}/{len(selected)}] {s["label"]} {s["group"]}')
    df=pd.DataFrame(records); df.to_csv(out/'features.csv',index=False); np.save(out/'embeddings.npy',np.array(embeds))
    feature_cols=[c for c in df if c not in {'path','label','group','severity'}]
    summary={'n':len(df),'class_counts':df.label.value_counts().to_dict(),'groups':df.groupby('label').group.unique().map(list).to_dict(),'severity':a.severity,'steps':a.steps,'status':'smoke_only'}
    # Strict held-group test: never fit/evaluate a group on both sides.  Require two groups per class.
    groups=df.groupby('label').group.unique().map(list).to_dict()
    # A group split with one image per class can return an arbitrary AUROC (even
    # exactly 0); refuse to turn that numerical artifact into a research claim.
    if all(len(groups.get(c,[]))>=2 for c in [0,1]):
        test_groups={c:sorted(groups[c])[-1] for c in [0,1]}; test=((df.label==0)&(df.group==test_groups[0]))|((df.label==1)&(df.group==test_groups[1])); train=~test
        min_n=min(int((train & (df.label==c)).sum()) for c in [0,1]); min_test=min(int((test & (df.label==c)).sum()) for c in [0,1])
        if min_n >= 5 and min_test >= 5:
            for name,cols in {'reconstruction':['recon_mse','recon_mae'],'endpoint':[c for c in feature_cols if c.startswith('endpoint')],'trajectory':[c for c in feature_cols if c.startswith(('path','speed','acceleration','direction'))],'full':feature_cols}.items():
                model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=.2)); model.fit(df.loc[train,cols],df.loc[train,'label']); prob=model.predict_proba(df.loc[test,cols])[:,1]; summary[name]=metric(df.loc[test,'label'],prob); summary[name]['balanced_accuracy']=float(balanced_accuracy_score(df.loc[test,'label'],prob>=.5))
            summary['status']='held_group_evaluated'; summary['test_groups']=test_groups
            # Generic image-only baseline: RGB 8x8 thumbnail.  It establishes
            # whether the probe adds value over a deliberately simple visual model.
            e=np.asarray(embeds)
            base=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=.2)); base.fit(e[train],df.loc[train,'label']); prob=base.predict_proba(e[test])[:,1]
            summary['visual_thumbnail']=metric(df.loc[test,'label'],prob); summary['visual_thumbnail']['balanced_accuracy']=float(balanced_accuracy_score(df.loc[test,'label'],prob>=.5))
        else:
            summary['status']='insufficient_group_samples'; summary['required']='>=5 train and >=5 test images per class after group split'; summary['observed']={'min_train_per_class':min_n,'min_test_per_class':min_test}
    # Strong shortcut sanity check: dimensions and metadata absent by construction; frequency-only remains diagnostic.
    summary['shortcut_controls']={'metadata_used':False,'filename_used':False,'canonical_rgb_size':32,'matched_corruption_seed_schedule':True}
    with open(out/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
