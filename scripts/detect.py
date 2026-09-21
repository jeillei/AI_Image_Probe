"""Experimental one-image detector using a trained generative-feature model."""
from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,torch
from src.data.captioning import BlipCaptioner
from src.features.compatibility import compatibility_features
from src.features.rich_trajectory import rich_features
from src.probes.sd15 import SD15Probe
from scripts.analyze_image import load
from scripts.controlled_depth_sweep import row_features
def main():
 p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--model',default='results/detector/sd15_detector.joblib');p.add_argument('--caption');p.add_argument('--json',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.image.is_file():p.error(f'not a readable image: {a.image}')
 artifact=joblib.load(a.model); caption=a.caption; conditioning='caption_override' if caption else 'automatically_inferred_blip'
 if not caption:
  cap=BlipCaptioner();caption=cap.caption(str(a.image));del cap;gc.collect()
  if torch.backends.mps.is_available():torch.mps.empty_cache()
 probe=SD15Probe(artifact['steps']);x=load(a.image,artifact['size']); raw=probe.invert_reconstruct(x,caption,mode='caption',guidance=1.0,capture_predictions=artifact.get('capture_rich',False));features=compatibility_features(row_features(raw,x))
 if artifact.get('capture_rich',False): features.update(rich_features(raw))
 values=[[features[n] for n in artifact['feature_names']]]; prob=float(artifact['model'].predict_proba(values)[0,1])
 out={'ai_score':prob,'prediction':'likely_ai' if prob>=.5 else 'likely_real','status':artifact['status'],'warning':artifact.get('warning'),'probe':'sd15','conditioning':conditioning,'caption_used':caption,'feature_family':artifact['family']}
 if a.output: a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2) if a.json else f"SynthImage experimental score: {prob:.2f}\nPrediction: {out['prediction']}\nProbe: SD1.5\nCaption: {caption}\nStatus: {out['status']}")
if __name__=='__main__':main()
