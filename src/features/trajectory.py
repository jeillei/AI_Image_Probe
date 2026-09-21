from __future__ import annotations
import numpy as np

def _moments(x: np.ndarray) -> tuple[float,float,float]:
    z=x.ravel(); m=z.mean(); s=z.std()+1e-8
    return float(np.linalg.norm(z)/np.sqrt(z.size)), float(np.mean(((z-m)/s)**3)), float(np.mean(((z-m)/s)**4))

def extract_features(result: dict, original: np.ndarray) -> dict[str,float]:
    p=result["forward"].reshape(len(result["forward"]),-1)
    v=np.diff(p,axis=0); speed=np.linalg.norm(v,axis=1)/np.sqrt(v.shape[1]); a=np.diff(v,axis=0)
    cos=(v[1:]*v[:-1]).sum(1)/(np.linalg.norm(v[1:],axis=1)*np.linalg.norm(v[:-1],axis=1)+1e-8)
    rec=result["reconstruction"]
    endpoint=result["forward"][-1]
    norm, skew, kurt=_moments(endpoint)
    f={"recon_mse":float(np.mean((rec-original)**2)),"recon_mae":float(np.mean(abs(rec-original))),
       "path_length":float(speed.sum()),"speed_mean":float(speed.mean()),"speed_std":float(speed.std()),
       "acceleration":float(np.linalg.norm(a,axis=1).mean()/np.sqrt(v.shape[1])) if len(a) else 0.,
       "direction_change":float(1-cos.mean()) if len(cos) else 0.,"endpoint_norm":norm,"endpoint_skew":skew,"endpoint_kurtosis":kurt,
       "endpoint_absmean":float(abs(endpoint).mean())}
    # Small smoke schedules can have fewer than four intervals.  Preserve a
    # fixed schema without manufacturing NaNs for an empty final bin.
    for i,q in enumerate(np.array_split(speed,4)):
        f[f"speed_q{i}"] = float(q.mean()) if len(q) else 0.0
    return f
