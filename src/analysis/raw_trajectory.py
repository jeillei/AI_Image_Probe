"""Load/pair/normalize the raw SD1.5 forward-trajectory tensors cached by scripts/extract_raw_trajectory.py,
and small linear-algebra helpers (economy SVD, principal angles, subspace projection energy) used throughout
RAW_TRAJECTORY_ANALYSIS.md.  No model fitting here beyond ordinary SVD/PCA.

Flattening order is fixed and documented: a raw tensor of shape (T, C, H, W) is ravelled in C order, i.e.
timestep-major, then channel, then spatial (row-major).  Every image goes through the identical 256px
canonicalization and identical 6-step DDIM-inverse schedule, so this ordering is consistent across images —
but it is only *grid*-aligned, not *semantically* aligned (a fixed latent pixel does not correspond to the
same scene content across different photographs).  This limitation is discussed in RAW_TRAJECTORY_ANALYSIS.md
Phase 0 and is not hidden by any preprocessing step here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

EPS = 1e-8
TENSOR_KEYS = ("z", "cond", "uncond", "guidance", "dz")


def load_cache(cache_dir: str | Path) -> dict[tuple[str, str], dict]:
    """{(generator, content_id): {'z':.., 'cond':.., 'uncond':.., 'path':.., 'label':..}}"""
    cache_dir = Path(cache_dir)
    out = {}
    for f in sorted(cache_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        gen, cid = str(d["generator"]), str(d["content_id"])
        out[(gen, cid)] = {"z": d["z"], "cond": d["cond"], "uncond": d["uncond"],
                            "path": str(d["path"]), "label": int(d["label"])}
    return out


def assert_triplets_complete(cache: dict, content_ids: list[str], generators=("real", "sd15", "amused")) -> None:
    """Never silently drop a malformed/missing sample: raise loudly instead."""
    missing = [(g, c) for c in content_ids for g in generators if (g, c) not in cache]
    if missing:
        raise ValueError(f"{len(missing)} missing (generator, content_id) pairs, e.g. {missing[:5]}")
    shapes = {k: {cache[(g, c)][k].shape for g in generators for c in content_ids} for k in ("z", "cond", "uncond")}
    for k, s in shapes.items():
        if len(s) != 1:
            raise ValueError(f"inconsistent shapes for tensor '{k}': {s}")


def derive(entry: dict, key: str) -> np.ndarray:
    if key in ("z", "cond", "uncond"):
        return entry[key]
    if key == "guidance":
        return entry["cond"] - entry["uncond"]
    if key == "dz":
        return np.diff(entry["z"], axis=0)  # (6,4,32,32): dz[k] = z[k+1]-z[k], aligned with cond[k]/uncond[k]
    raise ValueError(key)


def stack_tensor(cache: dict, generator: str, content_ids: list[str], key: str) -> np.ndarray:
    """(n_content, flattened_dim) matrix, rows in the given content_id order (fixed across generators)."""
    return np.stack([derive(cache[(generator, c)], key).astype(np.float64).ravel() for c in content_ids])


# ---------- normalizations (Phase 2) ----------
def norm_raw(fake: np.ndarray, real: np.ndarray) -> np.ndarray:
    """A. raw difference: T(fake) - T(real). Preserves absolute magnitude of change."""
    return fake - real


def norm_sample(fake: np.ndarray, real: np.ndarray) -> np.ndarray:
    """B. per-sample normalized difference: each raw tensor divided by its OWN L2 norm before differencing.
    Removes each image's own overall trajectory 'energy' so only directional/shape change remains."""
    fn = fake / (np.linalg.norm(fake, axis=1, keepdims=True) + EPS)
    rn = real / (np.linalg.norm(real, axis=1, keepdims=True) + EPS)
    return fn - rn


def norm_relative(fake: np.ndarray, real: np.ndarray) -> np.ndarray:
    """C. relative difference: (T(fake)-T(real)) / scale(T(real)), scale = per-sample RMS of the real tensor.
    Preserves the magnitude of the change but expresses it relative to that content's own baseline scale."""
    scale = np.sqrt(np.mean(real ** 2, axis=1, keepdims=True)) + EPS
    return (fake - real) / scale


NORMALIZATIONS = {"raw": norm_raw, "sample_normalized": norm_sample, "relative": norm_relative}


# ---------- linear algebra ----------
def economy_svd(X: np.ndarray):
    """X: (n, p), n << p. Returns U (n,r), S (r,), Vt (r,p) with r=min(n,p)."""
    return np.linalg.svd(X, full_matrices=False)


def effective_rank(S: np.ndarray) -> float:
    """Participation-ratio effective rank: (sum s_i^2)^2 / sum s_i^4."""
    s2 = S ** 2
    return float((s2.sum() ** 2) / (np.square(s2).sum() + EPS))


def k_for_variance(S: np.ndarray, frac: float = 0.95) -> int:
    s2 = S ** 2
    cum = np.cumsum(s2) / (s2.sum() + EPS)
    return int(np.searchsorted(cum, frac) + 1)


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """V1: (k1,p), V2: (k2,p), each row an orthonormal basis vector (top rows of Vt from economy_svd on
    CENTERED data). Returns principal angles in degrees, length min(k1,k2), ascending."""
    M = V1 @ V2.T
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def projection_energy(X: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of X's total (centered) energy captured by projecting each row onto `basis` (k,p) orthonormal
    rows. Aggregate energy ratio, not per-row average (rows with larger norm count more, matching how the
    basis' own explained-variance is defined)."""
    proj = X @ basis.T  # (n,k)
    return float(np.square(proj).sum() / (np.square(X).sum() + EPS))
