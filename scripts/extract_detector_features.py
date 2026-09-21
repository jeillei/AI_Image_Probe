"""Serial, resumable trajectory feature extraction from a neutral image manifest."""
from __future__ import annotations
import argparse,csv,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image,ImageOps
from src.corruption.chains import apply_chain
from src.corruption.robustness_suite import apply_condition
from src.data.captioning import BlipCaptioner,load_cache,save_cache
from src.features.compatibility import compatibility_features
from src.features.rich_trajectory import rich_features
from src.probes.sd15 import SD15Probe
from scripts.controlled_depth_sweep import row_features

def load(path,size,severity,seed,transform=None,transform_value=None):
 im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
 if transform: im=apply_condition(im,transform,float(transform_value) if transform_value not in (None,'') else None,seed)
 elif severity: im=apply_chain(im,severity,seed,base_size=size)
 else: im=im.resize((size,size),Image.Resampling.LANCZOS)
 return np.asarray(im,dtype=np.float32)/127.5-1
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--caption-cache',default='results/captions/detector_captions.json');p.add_argument('--steps',type=int,default=6);p.add_argument('--size',type=int,default=256);p.add_argument('--severity',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,choices=[1,2,3],default=1);p.add_argument('--rich',action='store_true');p.add_argument('--null-conditioning',action='store_true');a=p.parse_args()
 rows=list(csv.DictReader(open(a.manifest))); rows=rows[:a.limit] if a.limit else rows; out=Path(a.output); done=json.loads(out.read_text()) if out.exists() else []; keys={(x['path'],x['severity'],str(x.get('caption_id','')),str(x.get('transform','')),str(x.get('transform_value','')),str(x.get('extractor_protocol_version',''))) for x in done}
 cache0=load_cache(Path(a.caption_cache))
 # A transform manifest contains many rows per original.  Caption each image
 # once, never once per condition.
 missing=[] if a.null_conditioning else list({r['path']:r for r in rows if not r.get('caption') and r['path'] not in cache0}.values())
 if missing:
  cap=BlipCaptioner(); cache=load_cache(Path(a.caption_cache))
  for r in missing: cache[r['path']]=cap.caption(r['path']); save_cache(Path(a.caption_cache),cache); print('captioned',Path(r['path']).name)
  del cap;gc.collect();
  if torch.backends.mps.is_available():torch.mps.empty_cache()
 cache=load_cache(Path(a.caption_cache)); probe=SD15Probe(a.steps)
 pending=[(i,r) for i,r in enumerate(rows,1) if (r['path'],a.severity,r.get('caption_id',''),r.get('transform',''),r.get('transform_value',''),r.get('extractor_protocol_version','')) not in keys]
 for start in range(0,len(pending),a.batch_size):
  group=pending[start:start+a.batch_size];t=time.time();caps=['' for _ in group] if a.null_conditioning else [r.get('caption') or cache[r['path']] for _,r in group];xs=[load(r['path'],a.size,a.severity,seed=int(r.get('transform_seed',17+i)),transform=r.get('transform') or None,transform_value=r.get('transform_value')) for i,r in group]
  results=probe.invert_reconstruct_batch(xs,caps,capture_predictions=a.rich) if len(group)>1 else [probe.invert_reconstruct(xs[0],caps[0],mode='caption',guidance=1.0,capture_predictions=a.rich)]
  elapsed=(time.time()-t)/len(group)
  for (i,r),caption,x,result in zip(group,caps,xs,results):
   full=row_features(result,x); features=compatibility_features(full)
   if a.rich: features.update(rich_features(result))
   record={'path':r['path'],'label':int(r['label']),'generator':r['generator'],'content_id':r.get('content_id',''),'split':r.get('split',''), 'severity':a.severity,'caption':caption,'seconds':round(elapsed,2),'features':features}
   if 'caption_id' in r: record['caption_id']=int(r['caption_id']);record['caption_method']=r.get('caption_method','')
   if 'transform' in r: record.update({'original_id':r.get('original_id',r['path']),'transform':r['transform'],'transform_value':r.get('transform_value'),'transform_seed':r.get('transform_seed'),'extractor_protocol_version':r.get('extractor_protocol_version','')})
   done.append(record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(done,indent=2));print(i,record['label'],record['generator'],round(features['eps_mean'],4),record['seconds'])
  if torch.backends.mps.is_available():torch.mps.empty_cache()
if __name__=='__main__':main()
