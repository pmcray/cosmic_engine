"""CPU-side tests for the diffuse_nebula renderer.

Covers the topology id registry, the numpy CPU references for the
shared density helpers (torus, shell, pillar erosion), and the
example manifest's structural correctness across all three shots.
The Taichi ray-march kernel itself is exercised on Colab through
the notebook.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_topology_id_registry() -> None:
    from render.diffuse_nebula import TOPOLOGY_IDS, topology_id

    expected = {"pillars", "crab", "helix", "veil", "orion", "pleiades"}
    assert set(TOPOLOGY_IDS.keys()) == expected
    # Ids are sequential and start at 0 so the kernel dispatch is simple.
    assert sorted(TOPOLOGY_IDS.values()) == list(range(len(expected)))
    assert topology_id("pillars") == TOPOLOGY_IDS["pillars"]
    try:
        topology_id("not_a_real_nebula")
    except ValueError as e:
        assert "not_a_real_nebula" in str(e)
    else:
        raise AssertionError("expected ValueError on unknown topology")
    print("ok: test_topology_id_registry")


def test_torus_density_peaks_at_major_radius() -> None:
    from render.diffuse_nebula import _np_torus_density

    R, r_min = 0.7, 0.18
    on_ring = _np_torus_density((R, 0.0, 0.0), R, r_min)
    off_inside = _np_torus_density((0.1, 0.0, 0.0), R, r_min)
    off_outside = _np_torus_density((1.5, 0.0, 0.0), R, r_min)
    above_ring = _np_torus_density((R, 0.5, 0.0), R, r_min)
    assert on_ring > off_inside
    assert on_ring > off_outside
    assert on_ring > above_ring
    print("ok: test_torus_density_peaks_at_major_radius")


def test_shell_density_peaks_at_radius() -> None:
    from render.diffuse_nebula import _np_shell_density

    R = 1.0
    at_shell = _np_shell_density((R, 0.0, 0.0), R, 0.1)
    interior = _np_shell_density((0.3, 0.0, 0.0), R, 0.1)
    far_out = _np_shell_density((2.0, 0.0, 0.0), R, 0.1)
    assert at_shell > interior
    assert at_shell > far_out
    print("ok: test_shell_density_peaks_at_radius")


def test_pillar_erosion_kills_density_above_threshold() -> None:
    from render.diffuse_nebula import _np_pillar_erosion

    h = 0.2
    base = _np_pillar_erosion((0.0, h, 0.0), erosion_height=h, decay=2.5)
    well_above = _np_pillar_erosion((0.0, h + 0.5, 0.0), erosion_height=h, decay=2.5)
    below = _np_pillar_erosion((0.0, h - 0.2, 0.0), erosion_height=h, decay=2.5)
    assert base == 1.0  # exactly at threshold
    assert well_above == 0.0  # eroded away
    assert below == 1.0  # below threshold, full density
    print("ok: test_pillar_erosion_kills_density_above_threshold")


def test_example_manifest_loads_three_shots() -> None:
    from scene.manifest import load_manifest

    p = ROOT / "scene" / "examples" / "nebula_trinity.json"
    g = load_manifest(p)
    assert len(g.shots) == 3
    assert all(s.renderer == "diffuse_nebula" for s in g.shots)
    assert [s.id for s in g.shots] == [
        "pillars_of_creation", "crab_remnant", "helix_planetary",
    ]
    assert len(g.transitions) == 2
    print("ok: test_example_manifest_loads_three_shots")


def test_each_shot_picks_a_valid_topology() -> None:
    from render.diffuse_nebula import TOPOLOGY_IDS
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "nebula_trinity.json")
    expected = {
        "pillars_of_creation": "pillars",
        "crab_remnant": "crab",
        "helix_planetary": "helix",
    }
    for shot in g.shots:
        topo = shot.params.get("topology")
        assert topo in TOPOLOGY_IDS, (shot.id, topo)
        assert expected[shot.id] == topo
    print("ok: test_each_shot_picks_a_valid_topology")


def test_shots_have_sensible_march_steps() -> None:
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "nebula_trinity.json")
    for s in g.shots:
        n = s.params.get("march_steps", 80)
        assert 32 <= n <= 256, (s.id, n)
    print("ok: test_shots_have_sensible_march_steps")


def main() -> int:
    test_topology_id_registry()
    test_torus_density_peaks_at_major_radius()
    test_shell_density_peaks_at_radius()
    test_pillar_erosion_kills_density_above_threshold()
    test_example_manifest_loads_three_shots()
    test_each_shot_picks_a_valid_topology()
    test_shots_have_sensible_march_steps()
    print("\nall diffuse_nebula tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
