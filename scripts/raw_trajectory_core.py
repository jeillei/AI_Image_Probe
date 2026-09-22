"""Phases 3-6: within/cross-generator subspace structure on the primary normalization ('raw', chosen in
scripts/raw_trajectory_normcompare.py's Phase-2 comparison -- see RAW_TRAJECTORY_ANALYSIS.md for the reasoning),
for every raw tensor type; permutation nulls; the strict fit-on-one-freeze-project-other test; content-grouped
bootstrap CIs.  k is always drawn from the pre-specified set (1,2,3,5,10); never selected post hoc."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.analysis.raw_trajectory import (load_cache, assert_triplets_complete, stack_tensor, TENSOR_KEYS,
    NORMALIZATIONS, economy_svd, effective_rank, k_for_variance, principal_angles, projection_energy)

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--normalization", default="raw", choices=["raw", "sample_normalized", "relative"]); _args = _ap.parse_args()
TAG = _args.normalization
OUT = Path("results/raw_trajectory"); (OUT / "svd").mkdir(parents=True, exist_ok=True); (OUT / "nulls").mkdir(exist_ok=True); (OUT / "projection").mkdir(exist_ok=True)
K = [1, 2, 3, 5, 10]
N_PAIRBREAK = 300
N_RANDSUB = 1000
N_CONTENTSHUFFLE = 2000
N_BOOT = 300
rng_global = np.random.default_rng(0)

cache = load_cache("data/raw_trajectory_cache")
cids = sorted({cid for (g, cid) in cache if all((gg, cid) in cache for gg in ("real", "sd15", "amused"))})
assert_triplets_complete(cache, cids)
print("complete triplets:", len(cids))
if len(cids) < 60:
    print(f"WARNING: expected 60 complete triplets, found {len(cids)}; continuing on the available subset")

spectra_rows = []; rank_rows = []; cross_rows = []; pairbreak_rows = []; randsub_rows = []; boot_rows = []; meancos_rows = []

NORM_FN = NORMALIZATIONS[_args.normalization]
def make_delta(key, gen):
    real = stack_tensor(cache, "real", cids, key)
    fake = stack_tensor(cache, gen, cids, key)
    return NORM_FN(fake, real)

def random_orthonormal(p, k, rng):
    A = rng.normal(size=(p, k))
    Q, _ = np.linalg.qr(A)
    return Q.T  # (k,p)

for key in TENSOR_KEYS:
    D = {gen: make_delta(key, gen) for gen in ("sd15", "amused")}
    bases = {}
    for gen in ("sd15", "amused"):
        Dc = D[gen] - D[gen].mean(0)
        U, S, Vt = economy_svd(Dc)
        bases[gen] = Vt
        er = effective_rank(S)
        k95 = k_for_variance(S)
        rank_rows.append({"tensor": key, "generator": gen, "n": Dc.shape[0], "p": Dc.shape[1],
                          "effective_rank": er, "k_for_95pct_var": k95})
        for i, s in enumerate(S[:15]):
            spectra_rows.append({"tensor": key, "generator": gen, "component": i + 1, "singular_value": float(s),
                                 "cum_var_frac": float(np.cumsum(S**2)[i] / np.sum(S**2))})
    # ---- cross-generator subspace stats (Phase 3) ----
    mean_cos = float(np.dot(D["sd15"].mean(0), D["amused"].mean(0)) /
                     (np.linalg.norm(D["sd15"].mean(0)) * np.linalg.norm(D["amused"].mean(0)) + 1e-8))
    meancos_rows.append({"tensor": key, "mean_vector_cosine": mean_cos})
    for k in K:
        k = min(k, bases["sd15"].shape[0], bases["amused"].shape[0])
        ang = principal_angles(bases["sd15"][:k], bases["amused"][:k])
        pe_am_on_sd = projection_energy(D["amused"] - D["amused"].mean(0), bases["sd15"][:k])
        pe_sd_on_am = projection_energy(D["sd15"] - D["sd15"].mean(0), bases["amused"][:k])
        cross_rows.append({"tensor": key, "k": k, "mean_principal_angle_deg": float(ang.mean()),
                           "min_principal_angle_deg": float(ang.min()), "max_principal_angle_deg": float(ang.max()),
                           "projection_energy_AM_on_SD_basis": pe_am_on_sd, "projection_energy_SD_on_AM_basis": pe_sd_on_am})
    print(key, "cross-generator k=5:", [r for r in cross_rows if r["tensor"] == key and r["k"] == 5])

    # ---- Null A: pair-breaking (Phase 4) ----
    real_mat = stack_tensor(cache, "real", cids, key)
    fake_mats = {gen: stack_tensor(cache, gen, cids, key) for gen in ("sd15", "amused")}
    null_stats = {k: [] for k in K}
    for b in range(N_PAIRBREAK):
        perm_sd = rng_global.permutation(len(cids)); perm_am = rng_global.permutation(len(cids))
        Dsd_n = fake_mats["sd15"] - real_mat[perm_sd]; Dam_n = fake_mats["amused"] - real_mat[perm_am]
        Usd, Ssd, Vsd = economy_svd(Dsd_n - Dsd_n.mean(0)); Uam, Sam, Vam = economy_svd(Dam_n - Dam_n.mean(0))
        for k in K:
            kk = min(k, Vsd.shape[0], Vam.shape[0])
            ang = principal_angles(Vsd[:kk], Vam[:kk])
            null_stats[k].append(float(ang.mean()))
    for k in K:
        obs = [r for r in cross_rows if r["tensor"] == key and r["k"] == k][0]["mean_principal_angle_deg"]
        arr = np.array(null_stats[k])
        p_lower = float(np.mean(arr <= obs))  # smaller angle = more aligned; test if observed is unusually SMALL (aligned)
        pairbreak_rows.append({"tensor": key, "k": k, "observed_mean_angle_deg": obs, "null_mean_deg": float(arr.mean()),
                               "null_sd_deg": float(arr.std()), "empirical_p_more_aligned_than_null": p_lower})
    print(key, "pair-break null done")

    # ---- Null C: random-subspace (Phase 4) ----
    p = D["sd15"].shape[1]
    for k in K:
        kk = min(k, p)
        rand_pe = []
        for b in range(min(N_RANDSUB, 300)):
            basis = random_orthonormal(p, kk, rng_global)
            rand_pe.append(projection_energy(D["amused"] - D["amused"].mean(0), basis))
        obs_pe = [r for r in cross_rows if r["tensor"] == key and r["k"] == k][0]["projection_energy_AM_on_SD_basis"]
        arr = np.array(rand_pe)
        randsub_rows.append({"tensor": key, "k": kk, "observed_projection_energy_AM_on_SD": obs_pe,
                             "random_subspace_mean": float(arr.mean()), "random_subspace_sd": float(arr.std()),
                             "empirical_p_exceeds_random": float(np.mean(arr >= obs_pe))})
    print(key, "random-subspace null done")

    # ---- Phase 5: strict cross-generator projection test + Phase 6 content-grouped bootstrap ----
    for k in [5]:  # headline k, pre-specified (not searched)
        boot_pe_am = []; boot_pe_sd = []; boot_ang = []
        ids_arr = np.arange(len(cids))
        for b in range(N_BOOT):
            samp = rng_global.choice(ids_arr, len(ids_arr), replace=True)
            Dsd_b = D["sd15"][samp]; Dam_b = D["amused"][samp]
            Dsd_bc = Dsd_b - Dsd_b.mean(0); Dam_bc = Dam_b - Dam_b.mean(0)
            Usd, Ssd, Vsd_b = economy_svd(Dsd_bc); Uam, Sam, Vam_b = economy_svd(Dam_bc)
            kk = min(k, Vsd_b.shape[0], Vam_b.shape[0])
            boot_pe_am.append(projection_energy(Dam_bc, Vsd_b[:kk]))
            boot_pe_sd.append(projection_energy(Dsd_bc, Vam_b[:kk]))
            boot_ang.append(float(principal_angles(Vsd_b[:kk], Vam_b[:kk]).mean()))
        boot_rows.append({"tensor": key, "k": k, "metric": "projection_energy_AM_on_SD",
                          "point": [r for r in cross_rows if r["tensor"] == key and r["k"] == k][0]["projection_energy_AM_on_SD_basis"],
                          "ci_lo": float(np.percentile(boot_pe_am, 2.5)), "ci_hi": float(np.percentile(boot_pe_am, 97.5))})
        boot_rows.append({"tensor": key, "k": k, "metric": "projection_energy_SD_on_AM",
                          "point": [r for r in cross_rows if r["tensor"] == key and r["k"] == k][0]["projection_energy_SD_on_AM_basis"],
                          "ci_lo": float(np.percentile(boot_pe_sd, 2.5)), "ci_hi": float(np.percentile(boot_pe_sd, 97.5))})
        boot_rows.append({"tensor": key, "k": k, "metric": "mean_principal_angle_deg",
                          "point": [r for r in cross_rows if r["tensor"] == key and r["k"] == k][0]["mean_principal_angle_deg"],
                          "ci_lo": float(np.percentile(boot_ang, 2.5)), "ci_hi": float(np.percentile(boot_ang, 97.5))})
    print(key, "bootstrap done", flush=True)

pd.DataFrame(spectra_rows).to_csv(OUT / "svd" / f"singular_value_spectra_{TAG}.csv", index=False)
pd.DataFrame(rank_rows).to_csv(OUT / "svd" / f"effective_rank_{TAG}.csv", index=False)
pd.DataFrame(cross_rows).to_csv(OUT / "svd" / f"cross_generator_subspace_{TAG}.csv", index=False)
pd.DataFrame(meancos_rows).to_csv(OUT / "svd" / f"mean_vector_cosine_{TAG}.csv", index=False)
pd.DataFrame(pairbreak_rows).to_csv(OUT / "nulls" / f"pairbreaking_null_{TAG}.csv", index=False)
pd.DataFrame(randsub_rows).to_csv(OUT / "nulls" / f"randomsubspace_null_{TAG}.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "projection" / f"bootstrap_ci_k5_{TAG}.csv", index=False)
json.dump({"n_complete_triplets": len(cids), "content_ids": cids}, open(OUT / f"content_ids_used_{TAG}.json", "w"), indent=1)
print("\nDONE.")
print(pd.DataFrame(cross_rows)[pd.DataFrame(cross_rows).k == 5].to_string(index=False))
print(pd.DataFrame(pairbreak_rows)[pd.DataFrame(pairbreak_rows).k == 5].to_string(index=False))
print(pd.DataFrame(randsub_rows)[pd.DataFrame(randsub_rows).k == 5].to_string(index=False))
print(pd.DataFrame(boot_rows).to_string(index=False))
