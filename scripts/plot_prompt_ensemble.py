"""Generate fixed diagnostic plots for the cached prompt-ensemble pilot."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--dir',default='results/prompt_ensemble');a=p.parse_args();d=Path(a.dir)
 m=pd.read_csv(d/'logo_prompt_comparison.csv');q=m.pivot(index='held_generator',columns='representation',values='auroc');ax=q.plot(kind='bar',ylim=(0,1),figsize=(8,4));ax.set(ylabel='held-generator AUROC');ax.figure.tight_layout();ax.figure.savefig(d/'prompt_representation_auroc.png',dpi=160);plt.close(ax.figure)
 s=pd.read_csv(d/'prompt_score_stability.csv');fig,ax=plt.subplots(figsize=(6,4));s.boxplot(column='score_std',by='generator',ax=ax,rot=35);ax.set(title='detector-score SD across five prompts',ylabel='score SD');fig.suptitle('');fig.tight_layout();fig.savefig(d/'prompt_score_stability.png',dpi=160);plt.close(fig)
 t=pd.read_csv(d/'prompt_timestep_variance.csv');fig,ax=plt.subplots(figsize=(7,4));
 for (curve,g),x in t.groupby(['curve','generator']):
  if g=='real': ax.plot(x.step,x.mean_prompt_variance,label=curve)
 ax.set(xlabel='inverse step',ylabel='mean prompt variance',title='Real-image prompt variance');ax.legend();fig.tight_layout();fig.savefig(d/'prompt_variance_timestep.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
