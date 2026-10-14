"""CPU-side tests for the terrain, ca_creatures, exoplanet_atmosphere renderers.

Pure numpy / pure Python parts only; the Taichi engine classes are
exercised on Colab through the notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- Exoplanet topology registry --------------------------------------

def test_exoplanet_topology_id_registry() -> None:
    from render.exoplanet_atmosphere import TOPOLOGY_IDS, topology_id

    assert set(TOPOLOGY_IDS.keys()) == {"hot_jupiter", "mini_neptune", "brown_dwarf"}
    assert sorted(TOPOLOGY_IDS.values()) == [0, 1, 2]
    assert topology_id("hot_jupiter") == 0
    try:
        topology_id("unknown")
    except ValueError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError("expected ValueError")
    print("ok: test_exoplanet_topology_id_registry")


# ---- Lenia helpers ---------------------------------------------------

def test_lenia_kernel_normalised_and_annular() -> None:
    from render.ca_creatures import _lenia_kernel

    K = _lenia_kernel(grid_size=64, R=12.0, mu=0.5, sigma=0.15)
    assert K.shape == (64, 64)
    assert K.dtype == np.float32
    # Normalised.
    assert abs(K.sum() - 1.0) < 1e-4
    # Annular: center should be near zero (outside the radius), and the
    # max should sit roughly at R * mu away from the center.
    cy, cx = K.shape[0] // 2, K.shape[1] // 2
    assert K[cy, cx] < K.max()
    # Outside R should be zero.
    assert K[0, 0] == 0.0
    print("ok: test_lenia_kernel_normalised_and_annular")


def test_lenia_growth_function_peaks_at_mu() -> None:
    from render.ca_creatures import _gaussian_growth

    mu, sigma = 0.15, 0.015
    at_mu = _gaussian_growth(mu, mu, sigma)
    away = _gaussian_growth(mu + 5.0 * sigma, mu, sigma)
    assert at_mu > away
    # Peak value: 2 * 1 - 1 = 1
    assert abs(at_mu - 1.0) < 1e-6
    print("ok: test_lenia_growth_function_peaks_at_mu")


def test_lenia_seed_state_correct_shape_and_range() -> None:
    from render.ca_creatures import _seed_state

    state = _seed_state(grid_size=64, seed=0, n_seeds=4, seed_radius=8)
    assert state.shape == (64, 64)
    assert state.dtype == np.float32
    assert state.min() >= 0.0
    assert state.max() <= 1.0
    # Some non-zero cells.
    assert (state > 0).sum() > 0
    print("ok: test_lenia_seed_state_correct_shape_and_range")


# ---- Manifest sanity ------------------------------------------------

def test_terrain_manifest_loads() -> None:
    from scene.manifest import load_manifest

    p = ROOT / "scene" / "examples" / "terrain_journey.json"
    if not p.exists():
        return  # manifest not yet created — skip
    g = load_manifest(p)
    assert len(g.shots) >= 1
    assert all(s.renderer == "terrain" for s in g.shots)
    print("ok: test_terrain_manifest_loads")


def test_ca_creatures_manifest_loads() -> None:
    from scene.manifest import load_manifest

    p = ROOT / "scene" / "examples" / "ca_creatures.json"
    if not p.exists():
        return
    g = load_manifest(p)
    assert len(g.shots) >= 1
    assert all(s.renderer == "ca_creatures" for s in g.shots)
    print("ok: test_ca_creatures_manifest_loads")


def test_exoplanet_manifest_loads_three_topologies() -> None:
    from scene.manifest import load_manifest
    from render.exoplanet_atmosphere import TOPOLOGY_IDS

    p = ROOT / "scene" / "examples" / "exoplanet_trinity.json"
    g = load_manifest(p)
    assert len(g.shots) == 3
    for shot in g.shots:
        assert shot.renderer == "exoplanet_atmosphere"
        assert shot.params["topology"] in TOPOLOGY_IDS
    print("ok: test_exoplanet_manifest_loads_three_topologies")


def main() -> int:
    test_exoplanet_topology_id_registry()
    test_lenia_kernel_normalised_and_annular()
    test_lenia_growth_function_peaks_at_mu()
    test_lenia_seed_state_correct_shape_and_range()
    test_terrain_manifest_loads()
    test_ca_creatures_manifest_loads()
    test_exoplanet_manifest_loads_three_topologies()
    print("\nall new-renderer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
