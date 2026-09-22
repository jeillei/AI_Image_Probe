import numpy as np
import pytest
from src.analysis.raw_trajectory import (assert_triplets_complete, derive, load_cache, stack_tensor,
    norm_raw, norm_sample, norm_relative, economy_svd, principal_angles, projection_energy, effective_rank)

def _fake_entry(rng, seed=0):
    r = np.random.default_rng(seed)
    z = r.normal(size=(7, 4, 3, 3))
    cond = r.normal(size=(6, 4, 3, 3))
    uncond = r.normal(size=(6, 4, 3, 3))
    return {"z": z, "cond": cond, "uncond": uncond}

def _write_cache(tmp_path, triplets):
    """triplets: list of (generator, content_id, seed)."""
    for gen, cid, seed in triplets:
        e = _fake_entry(None, seed)
        np.savez_compressed(tmp_path / f"{gen}__{cid}.npz", z=e["z"], cond=e["cond"], uncond=e["uncond"],
                             path=f"data/{gen}/{cid}.png", content_id=cid, generator=gen, label=0 if gen == "real" else 1)
    return load_cache(tmp_path)

def test_cache_key_disambiguates_generator_for_same_content_id(tmp_path):
    cache = _write_cache(tmp_path, [("real", "c1", 1), ("sd15", "c1", 2), ("amused", "c1", 3)])
    assert len(cache) == 3  # would be 1 if the cache key collided across generators (the original bug)
    assert not np.allclose(cache[("real", "c1")]["z"], cache[("sd15", "c1")]["z"])

def test_assert_triplets_complete_raises_on_missing(tmp_path):
    cache = _write_cache(tmp_path, [("real", "c1", 1), ("sd15", "c1", 2)])  # amused missing
    with pytest.raises(ValueError, match="missing"):
        assert_triplets_complete(cache, ["c1"])

def test_assert_triplets_complete_raises_on_shape_mismatch(tmp_path):
    cache = _write_cache(tmp_path, [("real", "c1", 1), ("sd15", "c1", 2), ("amused", "c1", 3)])
    cache[("amused", "c1")]["z"] = cache[("amused", "c1")]["z"][:, :, :2, :2]  # corrupt shape
    with pytest.raises(ValueError, match="inconsistent shapes"):
        assert_triplets_complete(cache, ["c1"])

def test_derive_guidance_and_dz_alignment(tmp_path):
    cache = _write_cache(tmp_path, [("real", "c1", 1)])
    e = cache[("real", "c1")]
    g = derive(e, "guidance"); dz = derive(e, "dz")
    assert g.shape == e["cond"].shape == (6, 4, 3, 3)
    assert dz.shape == (6, 4, 3, 3)  # z has 7 timesteps -> 6 differences, same index alignment as cond/uncond
    assert np.allclose(g, e["cond"] - e["uncond"])
    assert np.allclose(dz, e["z"][1:] - e["z"][:-1])

def test_stack_tensor_preserves_content_id_row_order(tmp_path):
    cache = _write_cache(tmp_path, [("real", "cB", 1), ("real", "cA", 2)])
    m = stack_tensor(cache, "real", ["cA", "cB"], "z")
    assert m.shape[0] == 2
    assert np.allclose(m[0], cache[("real", "cA")]["z"].ravel())
    assert np.allclose(m[1], cache[("real", "cB")]["z"].ravel())

def test_normalizations_are_distinct_and_zero_for_identical_pair():
    rng = np.random.default_rng(0)
    fake = rng.normal(size=(5, 20)); real = rng.normal(size=(5, 20))
    for fn in (norm_raw, norm_sample, norm_relative):
        assert np.allclose(fn(real, real), 0.0, atol=1e-8)  # fake==real -> zero difference under every normalization
    a = norm_raw(fake, real); b = norm_sample(fake, real); c = norm_relative(fake, real)
    assert not np.allclose(a, b) and not np.allclose(a, c)

def test_svd_and_principal_angles_identical_subspace_is_zero_degrees():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 50))
    U, S, Vt = economy_svd(X - X.mean(0))
    angles = principal_angles(Vt[:3], Vt[:3])
    assert np.allclose(angles, 0.0, atol=1e-6)

def test_principal_angles_orthogonal_subspaces_are_90_degrees():
    V1 = np.eye(10)[:3]; V2 = np.eye(10)[3:6]
    angles = principal_angles(V1, V2)
    assert np.allclose(angles, 90.0, atol=1e-6)

def test_projection_energy_full_basis_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 20))
    U, S, Vt = economy_svd(X)
    assert projection_energy(X, Vt) == pytest.approx(1.0, abs=1e-8)  # full-rank basis captures all energy

def test_effective_rank_bounds():
    assert effective_rank(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4.0)  # equal singular values -> full rank
    assert effective_rank(np.array([10.0, 0.001, 0.001])) < 1.5  # one dominant value -> near rank 1
