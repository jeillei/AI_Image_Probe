"""Multi-probe abstraction (deliberately thin).

A probe maps a canonical image (HWC float32 in [-1,1]) to a `result` dict with the SAME keys the SD1.5 probe
already emits: forward [K+1,...], reverse [K+1,...], reconstruction [C,H,W], eps_norm [K], cond_gap [K]
(zeros/absent when the probe is unconditional), optional cond_scores/uncond_scores.  Everything downstream
(row_features, compatibility_features, rich_features) then works unchanged.

Two feature classes are kept strictly separate:
  A. probe-specific features  - any representation defined on that probe's own tensors (e.g. SD1.5 rich v1 608).
  B. cross-probe quantities   - the CROSS_PROBE_COLUMNS below: defined by identical code, in per-dimension RMS
                                units, on quantities every probe has (eps curve, latent path, endpoint, round trip).
                                These are the ONLY columns that may be compared/concatenated across probes.
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
from ..features.compatibility import curve_summary, FAMILIES

CROSS_PROBE_COLUMNS = FAMILIES["noise"] + FAMILIES["roundtrip"] + FAMILIES["endpoint"] + FAMILIES["geometry"]  # 44 legacy - 14 guidance = 30

class Probe(Protocol):
    name: str
    protocol_version: str
    canonical_size: int
    def preprocess(self, path: str) -> np.ndarray: ...                        # decode -> canonical HWC [-1,1]
    def measure(self, images: list[np.ndarray], conditioning: list[str] | None = None) -> list[dict]: ...  # inverse + reverse

def cross_probe_features(result: dict, x_hwc: np.ndarray) -> dict[str, float]:
    """Class-B quantities from a probe result.  Same code path as the SD1.5 legacy signature."""
    from scripts.controlled_depth_sweep import row_features
    row = row_features({**result, "cond_gap": result.get("cond_gap") or [0.0] * len(result["eps_norm"])}, x_hwc)
    eps = row["curves"]["eps_norm"]; out = curve_summary(eps, "eps", include_median=True)
    out.update({f"eps_t{i}": float(v) for i, v in enumerate(eps)})
    for k in FAMILIES["roundtrip"] + FAMILIES["endpoint"] + FAMILIES["geometry"]: out[k] = float(row[k])
    return {k: out[k] for k in CROSS_PROBE_COLUMNS}
