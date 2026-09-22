"""Phase 7: is cross-generator structure special to the inverse trajectory, or equally present in generic
low-level image representations?  Three cheap baselines, no new giant baseline zoo:
  (a) raw downsampled RGB pixels (32x32x3, from the identical 256px-canonicalized image)
  (b) the VAE latent alone, before any inversion step (= z[0] from the already-extracted cache, zero extra
      compute -- this is the single most direct "does it need the trajectory at all" control)
  (c) 2-D FFT magnitude of the canonical grayscale image (32x32, cheap, directly relevant given aMUSEd's
      FFT-heavy features in FEATURE_AUDIT.md)
  (d) the already-computed, already-frozen CIFAR-32 Class-B 30-feature vectors (reused, not recomputed)
Same subspace statistics (principal angles k=5, projection energy) + the pair-breaking null, on the SAME
60 matched triplets, so the comparison to the raw-trajectory numbers is apples-to-apples."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from PIL import Image
from src.analysis.raw_trajectory import load_cache, assert_triplets_complete, economy_svd, principal_angles, projection_energy, norm_raw

OUT = Path("results/raw_trajectory/baselines"); OUT.mkdir(parents=True, exist_ok=True)
K = [1, 2, 3, 5, 10]
N_PAIRBREAK = 300
rng = np.random.default_rng(2)
cache = load_cache("data/raw_trajectory_cache")
cids = sorted({cid for (g, cid) in cache if all((gg, cid) in cache for gg in ("real", "sd15", "amused"))})
assert_triplets_complete(cache, cids)
manifest = pd.read_csv("data/content_matched/manifest.csv")
path_of = {(r.generator, r.content_id): r.path for r in manifest.itertuples()}

def rgb32(cid, gen):
    im = Image.open(path_of[(gen, cid)]).convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    return (np.asarray(im, dtype=np.float64) / 127.5 - 1).ravel()

def fft32(cid, gen):
    im = Image.open(path_of[(gen, cid)]).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    return np.abs(np.fft.fft2(np.asarray(im, dtype=np.float64) / 255.0)).ravel()

def vae_latent(cid, gen):
    return cache[(gen, cid)]["z"][0].astype(np.float64).ravel()  # z0: before any inversion step

def cifar32_classb(cid, gen):
    return None  # filled below from the cached JSON

reps = {"rgb32": rgb32, "vae_latent_z0": vae_latent, "fft32": fft32}
stack = lambda fn, gen: np.stack([fn(cid, gen) for cid in cids])

# CIFAR-32 Class-B (30 features), reused from results/crossprobe/cifar32_content_matched.json
cif = {(r["generator"], __import__("re").sub(r"\.png$", "", Path(r["path"]).stem)): r["features"]
       for r in json.load(open("results/crossprobe/cifar32_content_matched.json"))}
cid_of_path = {r.path: r.content_id for r in manifest.itertuples()}
cif2 = {}
for r in json.load(open("results/crossprobe/cifar32_content_matched.json")):
    cif2[(r["generator"], cid_of_path[r["path"]])] = np.array(list(r["features"].values()), dtype=np.float64)
reps["cifar32_classB"] = None  # handled specially

rows = []; null_rows = []
for name in ("rgb32", "vae_latent_z0", "fft32", "cifar32_classB"):
    if name == "cifar32_classB":
        real = np.stack([cif2[("real", c)] for c in cids]); sd = np.stack([cif2[("sd15", c)] for c in cids]); am = np.stack([cif2[("amused", c)] for c in cids])
    else:
        fn = reps[name]; real = stack(fn, "real"); sd = stack(fn, "sd15"); am = stack(fn, "amused")
    Dsd = norm_raw(sd, real); Dam = norm_raw(am, real)
    Dsd_c = Dsd - Dsd.mean(0); Dam_c = Dam - Dam.mean(0)
    Usd, Ssd, Vsd = economy_svd(Dsd_c); Uam, Sam, Vam = economy_svd(Dam_c)
    for k in K:
        kk = min(k, Vsd.shape[0], Vam.shape[0], Dsd.shape[1])
        ang = principal_angles(Vsd[:kk], Vam[:kk])
        rows.append({"representation": name, "k": kk, "mean_principal_angle_deg": float(ang.mean()),
                    "projection_energy_AM_on_SD": projection_energy(Dam_c, Vsd[:kk]),
                    "projection_energy_SD_on_AM": projection_energy(Dsd_c, Vam[:kk])})
    # pair-breaking null at k=5
    obs5 = [r for r in rows if r["representation"] == name and r["k"] == min(5, Vsd.shape[0])][0]["mean_principal_angle_deg"]
    null = []
    for b in range(N_PAIRBREAK):
        perm_sd = rng.permutation(len(cids)); perm_am = rng.permutation(len(cids))
        Dsd_n = sd - real[perm_sd]; Dam_n = am - real[perm_am]
        Usd_n, Ssd_n, Vsd_n = economy_svd(Dsd_n - Dsd_n.mean(0)); Uam_n, Sam_n, Vam_n = economy_svd(Dam_n - Dam_n.mean(0))
        kk = min(5, Vsd_n.shape[0], Vam_n.shape[0])
        null.append(float(principal_angles(Vsd_n[:kk], Vam_n[:kk]).mean()))
    arr = np.array(null)
    null_rows.append({"representation": name, "observed_mean_angle_deg_k5": obs5, "null_mean_deg": float(arr.mean()),
                      "null_sd_deg": float(arr.std()), "empirical_p_more_aligned_than_null": float(np.mean(arr <= obs5))})
    print(name, "done:", rows[-len(K):][-1] if len(K) else None, null_rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUT / "baseline_cross_generator_subspace.csv", index=False)
pd.DataFrame(null_rows).to_csv(OUT / "baseline_pairbreaking_null.csv", index=False)
print("\nDONE"); print(pd.DataFrame(rows)[pd.DataFrame(rows).k == 5].to_string(index=False)); print(pd.DataFrame(null_rows).to_string(index=False))

# ---- supplementary: paired per-content cosine (same statistic as raw_trajectory_paired_cosine.py) + mean-vector cosine ----
def cosine_rows(A, B):
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8); b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(1)

paired_rows = []
for name in ("rgb32", "vae_latent_z0", "fft32", "cifar32_classB"):
    if name == "cifar32_classB":
        real = np.stack([cif2[("real", c)] for c in cids]); sd = np.stack([cif2[("sd15", c)] for c in cids]); am = np.stack([cif2[("amused", c)] for c in cids])
    else:
        fn = reps[name]; real = stack(fn, "real"); sd = stack(fn, "sd15"); am = stack(fn, "amused")
    Dsd = norm_raw(sd, real); Dam = norm_raw(am, real)
    obs = cosine_rows(Dsd, Dam)
    rng2 = np.random.default_rng(3)
    null_means = np.array([cosine_rows(Dsd, Dam[rng2.permutation(len(cids))]).mean() for _ in range(2000)])
    mean_cos = float(np.dot(Dsd.mean(0), Dam.mean(0)) / (np.linalg.norm(Dsd.mean(0)) * np.linalg.norm(Dam.mean(0)) + 1e-8))
    paired_rows.append({"representation": name, "mean_paired_cosine": float(obs.mean()), "frac_positive": float((obs > 0).mean()),
                        "null_mean": float(null_means.mean()), "null_sd": float(null_means.std()),
                        "empirical_p": float(np.mean(np.abs(null_means - null_means.mean()) >= abs(obs.mean() - null_means.mean()))),
                        "mean_vector_cosine": mean_cos})
    print(name, paired_rows[-1])
pd.DataFrame(paired_rows).to_csv(OUT / "baseline_paired_cosine.csv", index=False)
print("\nPAIRED COSINE (baselines):"); print(pd.DataFrame(paired_rows).to_string(index=False))

# ---- supplementary: random-subspace null for baselines too (fair, dimension-normalized comparison to the
# raw-trajectory numbers -- absolute principal-angle magnitude is NOT comparable across representations of
# very different ambient dimension p; only the p-value relative to each representation's OWN random-subspace
# null is comparable). ----
def random_orthonormal(p, k, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(p, k)))
    return Q.T

randsub_rows = []
for name in ("rgb32", "vae_latent_z0", "fft32", "cifar32_classB"):
    if name == "cifar32_classB":
        real = np.stack([cif2[("real", c)] for c in cids]); sd = np.stack([cif2[("sd15", c)] for c in cids]); am = np.stack([cif2[("amused", c)] for c in cids])
    else:
        fn = reps[name]; real = stack(fn, "real"); sd = stack(fn, "sd15"); am = stack(fn, "amused")
    Dsd = norm_raw(sd, real); Dam = norm_raw(am, real)
    Dsd_c = Dsd - Dsd.mean(0); Dam_c = Dam - Dam.mean(0)
    Usd, Ssd, Vsd = economy_svd(Dsd_c)
    p = Dsd.shape[1]; rng3 = np.random.default_rng(4)
    for k in (5,):
        kk = min(k, Vsd.shape[0], p)
        obs_pe = [r for r in rows if r["representation"] == name and r["k"] == kk][0]["projection_energy_AM_on_SD"]
        rand_pe = [projection_energy(Dam_c, random_orthonormal(p, kk, rng3)) for _ in range(300)]
        arr = np.array(rand_pe)
        randsub_rows.append({"representation": name, "k": kk, "p_ambient": p, "observed_projection_energy_AM_on_SD": obs_pe,
                             "random_subspace_mean": float(arr.mean()), "random_subspace_sd": float(arr.std()),
                             "empirical_p_exceeds_random": float(np.mean(arr >= obs_pe))})
        print(name, randsub_rows[-1])
pd.DataFrame(randsub_rows).to_csv(OUT / "baseline_randomsubspace_null.csv", index=False)
print("\nRANDOM-SUBSPACE NULL (baselines):"); print(pd.DataFrame(randsub_rows).to_string(index=False))
