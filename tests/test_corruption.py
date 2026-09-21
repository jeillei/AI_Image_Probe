from PIL import Image
from src.corruption import apply_chain
def test_chain_is_reproducible_and_canonical():
    x=Image.new('RGB',(93,51),(120,40,10))
    a=apply_chain(x,4,9); b=apply_chain(x,4,9)
    assert a.size==(256,256) and a.tobytes()==b.tobytes()
