import numpy as np
from PIL import Image
from src.corruption.robustness_suite import CONDITIONS,apply_condition
def test_conditions_are_deterministic_and_modify_gradient():
 rng=np.random.default_rng(7);im=Image.fromarray(rng.integers(0,256,(80,96,3),dtype=np.uint8),'RGB')
 for name,value in CONDITIONS:
  a=apply_condition(im,name,value,17);b=apply_condition(im,name,value,17)
  assert a.size==im.size and a.tobytes()==b.tobytes()
  if name!='clean': assert a.tobytes()!=im.tobytes()
def test_jpeg_and_crop_are_real_pixel_operations():
 im=Image.effect_noise((96,80),80).convert('RGB')
 assert apply_condition(im,'jpeg',30,1).tobytes()!=im.tobytes()
 assert apply_condition(im,'center_crop',.8,1).size==im.size
