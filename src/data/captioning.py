"""Cached BLIP caption inference, intentionally separate from the SD probe."""
from __future__ import annotations
from pathlib import Path
import json, time
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID="Salesforce/blip-image-captioning-base"

class BlipCaptioner:
    def __init__(self):
        self.device="mps" if torch.backends.mps.is_available() else "cpu"
        dtype=torch.float16 if self.device=="mps" else torch.float32
        self.processor=BlipProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        self.model=BlipForConditionalGeneration.from_pretrained(MODEL_ID,torch_dtype=dtype,local_files_only=True,use_safetensors=False).to(self.device).eval()
    @torch.inference_mode()
    def caption(self,path: str)->str:
        image=ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        inputs=self.processor(images=image,return_tensors="pt").to(self.device,self.model.dtype)
        ids=self.model.generate(**inputs,max_new_tokens=30)
        return self.processor.decode(ids[0],skip_special_tokens=True).strip()

def load_cache(path: Path)->dict:
    return json.loads(path.read_text()) if path.exists() else {}
def save_cache(path: Path,cache:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(cache,indent=2,sort_keys=True))
