"""Phase 2: compare the three principled normalizations on ONE representative tensor type (guidance -- the
family E88 identified as SD1.5's dominant signal) before committing a normalization for the main Phase 3-6
analysis.  Reports per-sample-norm heterogeneity (does A vs B/C matter) and the top-line cross-generator
principal angle under each, so the choice is visible and not picked because it "worked better"."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.analysis.raw_trajectory import load_cache, stack_tensor, NORMALIZATIONS, economy_svd, principal_angles, projection_energy

OUT = Path("results/raw_trajectory/svd"); OUT.mkdir(parents=True, exist_ok=True)
cache = load_cache("data/raw_trajectory_cache")
cids = sorted({cid for (g, cid) in cache if all((gg, cid) in cache for gg in ("real", "sd15", "amused"))})
print("complete triplets:", len(cids))

rows = []
key = "guidance"
real = stack_tensor(cache, "real", cids, key)
for gen in ("sd15", "amused"):
    fake = stack_tensor(cache, gen, cids, key)
    # per-sample L2 norm heterogeneity: does the real tensor's own scale vary a lot across content?
    real_norms = np.linalg.norm(real, axis=1); fake_norms = np.linalg.norm(fake, axis=1)
    rows.append({"generator": gen, "quantity": "real_norm_cv", "value": float(real_norms.std() / (real_norms.mean() + 1e-8))})
    rows.append({"generator": gen, "quantity": "fake_norm_cv", "value": float(fake_norms.std() / (fake_norms.mean() + 1e-8))})

cross = []
for name, fn in NORMALIZATIONS.items():
    D_sd = fn(stack_tensor(cache, "sd15", cids, key), real)
    D_am = fn(stack_tensor(cache, "amused", cids, key), real)
    Usd, Ssd, Vsd = economy_svd(D_sd - D_sd.mean(0)); Uam, Sam, Vam = economy_svd(D_am - D_am.mean(0))
    k = 5
    ang = principal_angles(Vsd[:k], Vam[:k])
    pe_am_on_sd = projection_energy(D_am - D_am.mean(0), Vsd[:k])
    pe_sd_on_am = projection_energy(D_sd - D_sd.mean(0), Vam[:k])
    mean_cos = float(np.dot(D_sd.mean(0), D_am.mean(0)) / (np.linalg.norm(D_sd.mean(0)) * np.linalg.norm(D_am.mean(0)) + 1e-8))
    cross.append({"normalization": name, "top5_principal_angles_deg": [round(float(a), 1) for a in ang],
                  "mean_top5_angle": float(ang.mean()), "projection_energy_AM_on_SD_top5": pe_am_on_sd,
                  "projection_energy_SD_on_AM_top5": pe_sd_on_am, "mean_vector_cosine": mean_cos})
pd.DataFrame(rows).to_csv(OUT / "normalization_scale_heterogeneity.csv", index=False)
json.dump(cross, open(OUT / "normalization_comparison_guidance.csv.json", "w"), indent=1)
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(cross, indent=1))
