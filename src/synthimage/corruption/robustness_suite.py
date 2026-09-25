"""Deterministic, auditable real-world image conditions applied before canonicalization."""
from __future__ import annotations
import io, random
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter,ImageOps

CONDITIONS=(
 ('clean',None),('jpeg',90),('jpeg',70),('jpeg',50),('jpeg',30),
 ('blur',.5),('blur',1.),('blur',2.),('resize',.5),('resize',.25),
 ('noise',.02),('noise',.05),('noise',.10),('color_jitter',.2),('center_crop',.8),
)
def condition_id(name:str,value:object)->str:
 return name if value is None else f'{name}_{str(value).replace(".","p")}'
def apply_condition(image:Image.Image,name:str,value:float|int|None,seed:int)->Image.Image:
 """Transform decoded RGB pixels; output preserves the pre-canonical image size."""
 img=ImageOps.exif_transpose(image).convert('RGB')
 if name=='clean': return img.copy()
 if name=='jpeg':
  b=io.BytesIO();img.save(b,format='JPEG',quality=int(value),subsampling=0);return Image.open(io.BytesIO(b.getvalue())).convert('RGB').copy()
 if name=='blur': return img.filter(ImageFilter.GaussianBlur(float(value)))
 if name=='resize':
  w,h=img.size;small=(max(1,round(w*float(value))),max(1,round(h*float(value))))
  return img.resize(small,Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS)
 if name=='noise':
  rng=np.random.default_rng(seed);x=np.asarray(img,dtype=np.float32)/255.;x=np.clip(x+rng.normal(0,float(value),x.shape),0,1);return Image.fromarray(np.rint(x*255).astype(np.uint8),'RGB')
 if name=='color_jitter':
  rng=random.Random(seed);lo,hi=1-float(value),1+float(value)
  img=ImageEnhance.Brightness(img).enhance(rng.uniform(lo,hi));img=ImageEnhance.Contrast(img).enhance(rng.uniform(lo,hi));return ImageEnhance.Color(img).enhance(rng.uniform(lo,hi))
 if name=='center_crop':
  w,h=img.size;k=float(value);cw,ch=max(1,round(w*k)),max(1,round(h*k));left=(w-cw)//2;top=(h-ch)//2
  return img.crop((left,top,left+cw,top+ch)).resize((w,h),Image.Resampling.LANCZOS)
 raise ValueError(f'unknown condition {name}')
