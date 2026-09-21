"""Frozen, low-dimensional SD1.5 generative-compatibility signature."""
from __future__ import annotations
import numpy as np

FAMILIES={
 "noise": [*[f"eps_t{i}" for i in range(6)],"eps_mean","eps_median","eps_min","eps_max","eps_std","eps_early","eps_middle","eps_late","eps_first_last","eps_slope","eps_auc","eps_late_early"],
 "guidance": [*[f"guidance_t{i}" for i in range(6)],"guidance_mean","guidance_max","guidance_std","guidance_early","guidance_middle","guidance_late","guidance_slope","guidance_auc"],
 "roundtrip": ["pixel_roundtrip_mse","latent_roundtrip_mse"],
 "endpoint": ["endpoint_norm_per_dim","endpoint_mean","endpoint_var","endpoint_skew","endpoint_kurtosis","endpoint_extreme_fraction"],
 "geometry": ["path_length","speed_mean","speed_max","acceleration_abs_mean"],
}

def curve_summary(curve, prefix: str, include_median: bool=False) -> dict[str,float]:
    x=np.asarray(curve,dtype=float); n=len(x); thirds=np.array_split(x,3)
    ans={f"{prefix}_mean":float(x.mean()),f"{prefix}_min":float(x.min()),f"{prefix}_max":float(x.max()),f"{prefix}_std":float(x.std()),
         f"{prefix}_early":float(thirds[0].mean()),f"{prefix}_middle":float(thirds[1].mean()),f"{prefix}_late":float(thirds[2].mean()),
         f"{prefix}_first_last":float(x[-1]-x[0]),f"{prefix}_slope":float(np.polyfit(np.arange(n),x,1)[0]) if n>1 else 0.,f"{prefix}_auc":float(np.trapezoid(x,dx=1)/max(n-1,1)),
         f"{prefix}_late_early":float(thirds[-1].mean()/(thirds[0].mean()+1e-8))}
    if include_median: ans[f"{prefix}_median"]=float(np.median(x))
    return ans

def compatibility_features(row: dict) -> dict[str,float]:
    """Feature-only mapping; excludes caption, path, labels and provenance."""
    curves=row["curves"]
    ans=curve_summary(curves["eps_norm"],"eps",include_median=True)
    ans.update(curve_summary(curves["guidance"],"guidance"))
    for i,value in enumerate(curves["eps_norm"]): ans[f"eps_t{i}"]=float(value)
    for i,value in enumerate(curves["guidance"]): ans[f"guidance_t{i}"]=float(value)
    # The guidance family specification intentionally omits the ratio.
    ans.pop("guidance_min"); ans.pop("guidance_first_last"); ans.pop("guidance_late_early")
    for key in FAMILIES["roundtrip"]+FAMILIES["endpoint"]+FAMILIES["geometry"]:
        ans[key]=float(row[key])
    return ans

def family_names(family: str) -> list[str]:
    if family=="noise_guidance": return FAMILIES["noise"]+FAMILIES["guidance"]
    if family=="full": return sum(FAMILIES.values(),[])
    return FAMILIES[family]
