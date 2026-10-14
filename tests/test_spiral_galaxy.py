"""CPU-side tests for the spiral_galaxy renderer.

Tests cover the numpy CPU references for the density-field helpers,
the example manifest's structural correctness, and a face-on /
isolated-vs-companion check on the three example shots. The Taichi
ray-march kernel itself is exercised on Colab through the notebook.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_sersic_profile_monotone_decreasing() -> None:
    from render.spiral_galaxy import _np_sersic

    xs = np.linspace(0.05, 4.0, 40)
    ys = _np_sersic(xs, 0.25)
    diffs = np.diff(ys)
    assert (diffs <= 1e-6).all(), "Sersic profile must be monotone decreasing"
    print("ok: test_sersic_profile_monotone_decreasing")


def test_bulge_brighter_at_center() -> None:
    from render.spiral_galaxy import _np_bulge

    center = _np_bulge((0.0, 0.0, 0.0), R_b=0.18, q=0.85)
    edge = _np_bulge((0.5, 0.0, 0.0), R_b=0.18, q=0.85)
    far = _np_bulge((1.5, 0.0, 0.0), R_b=0.18, q=0.85)
    assert center > edge * 10.0, (center, edge)
    assert edge > far
    print("ok: test_bulge_brighter_at_center")


def test_disk_falls_off_with_radius_and_height() -> None:
    from render.spiral_galaxy import _np_disk

    r0 = _np_disk((0.0, 0.0, 0.0), R_disk=1.0, h_disk=0.05)
    r_mid = _np_disk((0.5, 0.0, 0.0), R_disk=1.0, h_disk=0.05)
    r_far = _np_disk((1.5, 0.0, 0.0), R_disk=1.0, h_disk=0.05)
    assert r0 > r_mid > r_far

    z0 = _np_disk((0.5, 0.0, 0.0), R_disk=1.0, h_disk=0.05)
    z_mid = _np_disk((0.5, 0.0, 0.05), R_disk=1.0, h_disk=0.05)
    z_far = _np_disk((0.5, 0.0, 0.20), R_disk=1.0, h_disk=0.05)
    assert z0 > z_mid > z_far
    print("ok: test_disk_falls_off_with_radius_and_height")


def test_arm_modulation_periodic_in_theta() -> None:
    from render.spiral_galaxy import _np_arm_modulation

    n_arms = 2
    for r in (0.3, 0.6, 1.0):
        for theta in np.linspace(-3.0, 3.0, 9):
            a = _np_arm_modulation(r, theta, n_arms=n_arms)
            b = _np_arm_modulation(r, theta + 2.0 * math.pi / n_arms, n_arms=n_arms)
            assert abs(a - b) < 1e-4, (r, theta, a, b)
    print("ok: test_arm_modulation_periodic_in_theta")


def test_arm_modulation_amplitude_within_band() -> None:
    from render.spiral_galaxy import _np_arm_modulation

    strength = 0.85
    rs = np.linspace(0.05, 1.5, 30)
    thetas = np.linspace(-math.pi, math.pi, 80)
    r2, t2 = np.meshgrid(rs, thetas, indexing="ij")
    vals = _np_arm_modulation(
        r2, t2, n_arms=2, pitch_deg=12.0, strength=strength, width=0.45
    )
    lo, hi = 1.0 - strength - 0.01, 1.0 + strength + 0.01
    assert float(vals.min()) >= lo, vals.min()
    assert float(vals.max()) <= hi, vals.max()
    print("ok: test_arm_modulation_amplitude_within_band")


def test_companion_bridge_higher_along_segment() -> None:
    from render.spiral_galaxy import _np_companion_bridge

    comp = (1.05, 0.05, 0.35)
    midpoint = tuple(c / 2.0 for c in comp)
    off_axis = (midpoint[0], midpoint[1] + 0.5, midpoint[2])
    on_seg = _np_companion_bridge(midpoint, comp, 0.5, 0.7)
    off_seg = _np_companion_bridge(off_axis, comp, 0.5, 0.7)
    assert on_seg > off_seg + 0.05, (on_seg, off_seg)
    print("ok: test_companion_bridge_higher_along_segment")


def test_companion_zero_mass_yields_isolated() -> None:
    from render.spiral_galaxy import _np_companion_bridge

    comp = (1.05, 0.05, 0.35)
    samples = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.1), comp, (-0.3, 0.4, 0.0)]
    for s in samples:
        v = _np_companion_bridge(s, comp, mass_ratio=0.0, bridge_strength=0.0)
        assert v == 0.0, (s, v)
    print("ok: test_companion_zero_mass_yields_isolated")


def test_example_manifest_loads_three_shots() -> None:
    from scene.manifest import load_manifest

    p = ROOT / "scene" / "examples" / "m51_whirlpool.json"
    g = load_manifest(p)
    assert len(g.shots) == 3
    assert all(s.renderer == "spiral_galaxy" for s in g.shots)
    ids = [s.id for s in g.shots]
    assert ids == ["m51_wide", "m51_arm_closeup", "m51_companion_bridge"]
    assert len(g.transitions) == 2
    print("ok: test_example_manifest_loads_three_shots")


def test_first_two_shots_isolated_third_has_companion() -> None:
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "m51_whirlpool.json")
    wide, closeup, pair = g.shots
    assert wide.params.get("companion_mass_ratio", 0.5) == 0.0
    assert wide.params.get("bridge_strength", 0.5) == 0.0
    assert closeup.params.get("companion_mass_ratio", 0.5) == 0.0
    assert pair.params.get("companion_mass_ratio") == 0.5
    assert pair.params.get("bridge_strength") > 0.0
    print("ok: test_first_two_shots_isolated_third_has_companion")


def test_wide_shot_near_face_on_pitch_in_range() -> None:
    """The wide shot must be near face-on, and every shot's pitch_deg
    must lie in a physically plausible range for grand-design spirals."""
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "m51_whirlpool.json")
    wide = g.shots[0]
    assert wide.params.get("inclination_deg", 0.0) <= 15.0
    for s in g.shots:
        pitch = s.params.get("pitch_deg", 12.0)
        assert 5.0 <= pitch <= 25.0, (s.id, pitch)
    print("ok: test_wide_shot_near_face_on_pitch_in_range")


def main() -> int:
    test_sersic_profile_monotone_decreasing()
    test_bulge_brighter_at_center()
    test_disk_falls_off_with_radius_and_height()
    test_arm_modulation_periodic_in_theta()
    test_arm_modulation_amplitude_within_band()
    test_companion_bridge_higher_along_segment()
    test_companion_zero_mass_yields_isolated()
    test_example_manifest_loads_three_shots()
    test_first_two_shots_isolated_third_has_companion()
    test_wide_shot_near_face_on_pitch_in_range()
    print("\nall spiral_galaxy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
