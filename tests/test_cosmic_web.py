"""CPU-side tests for the cosmic_web renderer and DESI pipeline.

Covers the w0wa cosmology, RA/Dec/z -> Cartesian conversion, tracer
classification, the procedural synthesizer's structure, the manifest
load, and the numpy splat-to-grid helper.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_comoving_distance_zero_at_z_zero() -> None:
    from render.cosmic_pipeline import comoving_distance

    D = comoving_distance(0.0)
    assert abs(D) < 1e-6
    print("ok: test_comoving_distance_zero_at_z_zero")


def test_comoving_distance_monotone_in_z() -> None:
    from render.cosmic_pipeline import comoving_distance

    zs = np.array([0.05, 0.2, 0.5, 1.0, 1.5, 2.0])
    Ds = comoving_distance(zs)
    assert np.all(np.diff(Ds) > 0.0)
    # Order of magnitude sanity: at z = 1, D_C ~ 3000-3500 Mpc for
    # Planck-ish cosmology.
    assert 2500.0 < Ds[3] < 4500.0, Ds[3]
    print("ok: test_comoving_distance_monotone_in_z")


def test_E_w0wa_matches_lcdm_when_w0_neg_one_wa_zero() -> None:
    """For w0 = -1, wa = 0 the w0wa parameterisation reduces to
    LambdaCDM. E(z) should land at standard values."""
    from render.cosmic_pipeline import E_w0wa

    E0 = E_w0wa(0.0, Omega_m=0.3, w0=-1.0, wa=0.0)
    assert abs(E0 - 1.0) < 1e-6, E0
    E1 = E_w0wa(1.0, Omega_m=0.3, w0=-1.0, wa=0.0)
    # E(1)^2 = 0.3 * 8 + 0.7 = 3.1, E(1) ~ 1.7607
    assert abs(E1 - math.sqrt(3.1)) < 1e-5, E1
    print("ok: test_E_w0wa_matches_lcdm_when_w0_neg_one_wa_zero")


def test_radec_z_to_cartesian_north_pole_is_z_axis() -> None:
    from render.cosmic_pipeline import radec_z_to_cartesian

    # Dec = +90, the celestial north pole, regardless of RA, must land
    # purely on the +Z axis.
    p = radec_z_to_cartesian(45.0, 90.0, 0.5)
    assert abs(p[0]) < 1e-3
    assert abs(p[1]) < 1e-3
    assert p[2] > 0.0
    print("ok: test_radec_z_to_cartesian_north_pole_is_z_axis")


def test_radec_z_to_cartesian_origin_alignment() -> None:
    """RA=0, Dec=0 must land along +X with the right magnitude."""
    from render.cosmic_pipeline import comoving_distance, radec_z_to_cartesian

    p = radec_z_to_cartesian(0.0, 0.0, 0.8)
    D = comoving_distance(0.8)
    assert abs(p[0] - D) < 1e-3
    assert abs(p[1]) < 1e-3
    assert abs(p[2]) < 1e-3
    print("ok: test_radec_z_to_cartesian_origin_alignment")


def test_tracer_classification_covers_full_z_range() -> None:
    from render.cosmic_pipeline import classify_by_redshift, TRACER_NAMES

    z = np.array([0.05, 0.3, 0.6, 1.0, 1.4, 1.8, 2.1])
    t = classify_by_redshift(z)
    assert t.shape == z.shape
    assert t.min() >= 0 and t.max() < len(TRACER_NAMES)
    # BGS dominates at low z, QSO at high z.
    assert t[0] == 0  # BGS
    assert t[-1] == 3  # QSO
    print("ok: test_tracer_classification_covers_full_z_range")


def test_tracer_colors_are_four_rgb_triplets() -> None:
    from render.cosmic_pipeline import TRACER_COLORS, TRACER_NAMES

    assert len(TRACER_COLORS) == len(TRACER_NAMES) == 4
    for c in TRACER_COLORS:
        assert len(c) == 3
        for v in c:
            assert 0.0 <= v <= 1.5
    print("ok: test_tracer_colors_are_four_rgb_triplets")


def test_synthesizer_produces_expected_shapes() -> None:
    from render.cosmic_pipeline import synthesize_cosmic_web

    pos, tracer, mag, z = synthesize_cosmic_web(
        n_particles=2000, n_clusters=20, seed=3, box_size_mpc=1000.0,
    )
    assert pos.shape == (2000, 3)
    assert tracer.shape == (2000,)
    assert mag.shape == (2000,)
    assert z.shape == (2000,)
    assert pos.dtype == np.float32
    assert tracer.dtype == np.int32
    assert mag.dtype == np.float32
    # Clipped to box half-extent.
    assert pos.min() >= -500.0
    assert pos.max() <= 500.0
    print("ok: test_synthesizer_produces_expected_shapes")


def test_synthesizer_produces_clustered_structure() -> None:
    """A plausible cosmic web is more clustered than uniform random."""
    from render.cosmic_pipeline import synthesize_cosmic_web

    pos, _, _, _ = synthesize_cosmic_web(
        n_particles=5000, n_clusters=50, seed=7, box_size_mpc=1000.0,
    )
    # Build a coarse 16^3 grid, count occupancy variance.
    half = 500.0
    g = ((pos + half) / (1000.0 / 16.0)).astype(np.int32)
    g = np.clip(g, 0, 15)
    counts = np.zeros((16, 16, 16), dtype=np.int32)
    np.add.at(counts, (g[:, 0], g[:, 1], g[:, 2]), 1)
    # Coefficient of variation should be > 1 (way above the ~1/sqrt(mean)
    # expected from a Poisson uniform field).
    mean = float(counts.mean())
    std = float(counts.std())
    poisson_cv = 1.0 / math.sqrt(mean) if mean > 0 else 0.0
    cv = std / max(mean, 1.0)
    assert cv > poisson_cv * 3.0, (
        f"synthesised field is too uniform: cv={cv:.3f} vs poisson={poisson_cv:.3f}"
    )
    print("ok: test_synthesizer_produces_clustered_structure")


def test_splat_to_grid_returns_correct_shapes_and_finite_values() -> None:
    from render.cosmic_pipeline import synthesize_cosmic_web
    from render.cosmic_web import splat_particles_to_grid

    pos, tracer, mag, _ = synthesize_cosmic_web(
        n_particles=3000, n_clusters=20, seed=1, box_size_mpc=1000.0,
    )
    em, den = splat_particles_to_grid(pos, tracer, mag,
                                       box_size_mpc=1000.0, grid_dim=32)
    assert em.shape == (32, 32, 32, 3)
    assert den.shape == (32, 32, 32)
    assert em.dtype == np.float32
    assert den.dtype == np.float32
    assert np.isfinite(em).all()
    assert np.isfinite(den).all()
    # Some density landed somewhere.
    assert float(den.max()) > 0.0
    print("ok: test_splat_to_grid_returns_correct_shapes_and_finite_values")


def test_manifest_loads_three_shots() -> None:
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "cosmic_web_flythrough.json")
    assert len(g.shots) == 3
    assert all(s.renderer == "cosmic_web" for s in g.shots)
    assert [s.id for s in g.shots] == [
        "wide_field", "filament_skim", "deep_redshift",
    ]
    print("ok: test_manifest_loads_three_shots")


def main() -> int:
    test_comoving_distance_zero_at_z_zero()
    test_comoving_distance_monotone_in_z()
    test_E_w0wa_matches_lcdm_when_w0_neg_one_wa_zero()
    test_radec_z_to_cartesian_north_pole_is_z_axis()
    test_radec_z_to_cartesian_origin_alignment()
    test_tracer_classification_covers_full_z_range()
    test_tracer_colors_are_four_rgb_triplets()
    test_synthesizer_produces_expected_shapes()
    test_synthesizer_produces_clustered_structure()
    test_splat_to_grid_returns_correct_shapes_and_finite_values()
    test_manifest_loads_three_shots()
    print("\nall cosmic_web tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
