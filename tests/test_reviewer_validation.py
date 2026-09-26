"""Scientific-correctness tests for the reviewer-validation extension (conditioning ablation + probe swap).
No GPU/model weights needed -- these test the derangement logic and the cross-probe interface contract, not
feature extraction itself. See docs/research_history/reviewer_validation/REVIEWER_VALIDATION_PLAN.md."""
import pandas as pd
from scripts.reviewer_validation.conditioning_ablation import deranged_captions
from synthimage.probes.sd15 import SD15Probe
from synthimage.probes.sdxl import SDXLProbe


def _fake_manifest(n=60):
    return pd.DataFrame({"content_id": [f"c{i:03d}" for i in range(n)], "caption": [f"caption {i}" for i in range(n)]})


def test_derangement_is_a_bijection_with_no_fixed_point():
    m = _fake_manifest()
    shuffled = deranged_captions(m)
    caption_of = m.set_index("content_id").caption.to_dict()
    assert set(shuffled) == set(caption_of)  # every content id gets an assignment
    assert sorted(shuffled.values()) == sorted(caption_of.values())  # bijection: same multiset of captions
    for cid, cap in shuffled.items():
        assert cap != caption_of[cid], f"{cid} mapped to its own caption"


def test_derangement_is_deterministic_across_calls_and_row_order():
    m = _fake_manifest()
    a = deranged_captions(m)
    b = deranged_captions(m.sample(frac=1, random_state=7).reset_index(drop=True))  # shuffled row order
    assert a == b


def test_sd15_and_sdxl_probes_expose_the_same_measurement_interface():
    """path_length/diffpath_curvature/score_norm_step0/vae_only in panel_v2.py are written against one probe
    interface; a cross-probe comparison is only meaningful if both probes satisfy it identically."""
    required = {"encode_image", "decode_latent", "embeds", "invert_reconstruct"}
    assert required <= set(dir(SD15Probe))
    assert required <= set(dir(SDXLProbe))
