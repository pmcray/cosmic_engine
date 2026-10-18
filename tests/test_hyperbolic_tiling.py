"""Tests for `render.hyperbolic_tiling` ({7,3}, {3,7}, and 7-fold quasicrystal
topologies, coupled spatiotemporal graph dynamics, SU(1,1) Mobius camera
isometries, MetaPost vector export, adapter, and voyage composer integration).
"""
from __future__ import annotations

import cmath
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- 1. Mathematical & Geometric Topologies --------------------------

def test_poincare_7_3_topology_and_klein_invariants() -> None:
    from render.hyperbolic_tiling import build_tiling, poincare_circumradius

    g = build_tiling(mode="poincare_7_3", layers=4)
    assert g.mode == "poincare_7_3"
    assert g.p == 7 and g.q == 3
    assert 150 <= len(g) <= 600, f"expected ~150-600 tiles at layers=4, got {len(g)}"

    # Central tile sits at origin with exact circumradius r0
    r0 = poincare_circumradius(7, 3)
    assert abs(g.tiles[0].center) < 1e-12
    for v in g.tiles[0].vertices:
        assert abs(abs(v) - r0) < 1e-12

    # All vertices and centers lie strictly inside the Poincare unit disk |z| < 1
    assert np.all(np.abs(g.vertices) < 1.0)
    assert np.all(np.abs(g.centers) < 1.0)

    # Central heptagon and layer-1 heptagons have exact degree 7
    assert g.degrees[0] == 7
    for tid in g.tiles[0].neighbors:
        assert g.degrees[tid] == 7

    # Adjacency is symmetric and deduplicated
    for i, nbrs in enumerate(g.adjacency):
        assert len(nbrs) == len(set(nbrs))
        for j in nbrs:
            assert i in g.adjacency[j]

    # At most q=3 heptagons meet at any vertex, and all vertices of the central
    # heptagon have exact valence 3
    v_counts = np.zeros(len(g.vertices), dtype=np.int32)
    for tile in g.tiles:
        assert len(tile.vertices) == 7
        for vid in tile.vertex_ids:
            v_counts[vid] += 1
    assert v_counts.max() == 3
    for vid in g.tiles[0].vertex_ids:
        assert v_counts[vid] == 3


def test_poincare_3_7_dual_topology() -> None:
    from render.hyperbolic_tiling import build_tiling

    g = build_tiling(mode="poincare_3_7", layers=4)
    assert g.mode == "poincare_3_7"
    assert g.p == 3 and g.q == 7
    assert 150 <= len(g) <= 600, f"expected ~150-600 tiles at layers=4, got {len(g)}"

    assert np.all(np.abs(g.vertices) < 1.0)
    assert np.all(np.abs(g.centers) < 1.0)

    # Central triangle has 3 vertices and 3 neighbors
    assert len(g.tiles[0].vertices) == 3
    assert g.degrees[0] == 3
    assert g.degrees.max() == 3

    # Interior vertices have valence q=7 (7 triangles meeting at a vertex)
    v_counts = np.zeros(len(g.vertices), dtype=np.int32)
    for tile in g.tiles:
        assert len(tile.vertices) == 3
        for vid in tile.vertex_ids:
            v_counts[vid] += 1
    assert v_counts.max() == 7
    for vid in g.tiles[0].vertex_ids:
        assert v_counts[vid] == 7


def test_quasicrystal_7_de_bruijn_topology() -> None:
    from render.hyperbolic_tiling import build_tiling

    g = build_tiling(mode="quasicrystal_7", layers=4)
    assert g.mode == "quasicrystal_7"
    assert g.p == 4 and g.q == 7
    assert 150 <= len(g) <= 600, f"expected ~150-600 rhombi at layers=4, got {len(g)}"

    # Every tile is a 4-vertex rhombus with <= 4 edge neighbors, and interior tiles have degree 4
    assert all(len(t.vertices) == 4 for t in g.tiles)
    assert g.degrees.max() == 4
    assert (g.degrees == 4).mean() > 0.80

    # All 3 distinct 7-fold Penrose/De Bruijn prototile rhombus shapes (pi/7, 2pi/7, 3pi/7) appear
    kinds = {t.kind for t in g.tiles}
    assert kinds == {0, 1, 2}

    # Check 7-fold orientational symmetry of rhombus edges (angles mod pi/7)
    edge_angles = []
    for t in g.tiles[:50]:
        for k in range(4):
            dz = t.vertices[(k + 1) % 4] - t.vertices[k]
            ang = math.atan2(dz.imag, dz.real) % (math.pi / 7.0)
            edge_angles.append(min(ang, math.pi / 7.0 - ang))
    assert max(edge_angles) < 1e-9


def test_parquet_deformation_hofstadter_spatial_morph_and_convexity() -> None:
    from render.hyperbolic_tiling import build_tiling

    g = build_tiling(mode="parquet_deformation", layers=4)
    assert g.mode == "parquet_deformation"
    assert g.p == 6 and g.q == 3
    assert 150 <= len(g) <= 400

    # Every tile has 6 vertices, <= 6 neighbors, and interior tiles have degree 6
    assert all(len(t.vertices) == 6 for t in g.tiles)
    assert g.degrees.max() == 6
    assert (g.degrees == 6).mean() > 0.70

    # Compare edge-length variance on the left (tau=0, regular hexagons d-1=6)
    # vs on the right (tau=1, frustrated 7-fold chiral prototiles d=7)
    left_rel_stds = []
    right_rel_stds = []
    for t in g.tiles:
        vs = t.vertices
        edges = [abs(vs[(k + 1) % 6] - vs[k]) for k in range(6)]
        rel_std = float(np.std(edges) / np.mean(edges))
        if t.center.real < -1.25:
            left_rel_stds.append(rel_std)
        elif t.center.real > 1.25:
            right_rel_stds.append(rel_std)

        # Verify strict convexity (all CCW cross products at consecutive edges > 0)
        for k in range(6):
            e1 = vs[(k + 1) % 6] - vs[k]
            e2 = vs[(k + 2) % 6] - vs[(k + 1) % 6]
            cross = e1.real * e2.imag - e1.imag * e2.real
            assert cross > 0.0, f"tile {t.index} became non-convex at vertex {k}"

    assert left_rel_stds and right_rel_stds
    # Left side is regular Euclidean hexagons (relative edge std ~ 0)
    assert max(left_rel_stds) < 1e-3
    # Right side has strong chiral / 7-fold edge differentiation
    assert np.mean(right_rel_stds) > 0.14


# ---- 2. Spatiotemporal Dynamics on the Tile Graph --------------------

def test_cml_dynamics_exact_recurrence_and_evolution() -> None:
    from render.hyperbolic_tiling import (
        build_tiling,
        compute_tile_states,
        logistic_map,
        step_cml,
    )

    g = build_tiling(mode="poincare_7_3", layers=3)
    u0 = np.linspace(0.1, 0.9, len(g))
    eps, r = 0.30, 3.80
    u1 = step_cml(u0, g.adjacency, epsilon=eps, r=r)

    # Verify exact mathematical formula on tile 0 and tile 5:
    # u_i(t+1) = (1 - eps) f(u_i(t)) + (eps / deg(i)) sum_{j in N(i)} f(u_j(t))
    fu0 = logistic_map(u0, r=r)
    for idx in (0, 5):
        nbrs = g.adjacency[idx]
        expected = (1.0 - eps) * fu0[idx] + (eps / len(nbrs)) * sum(fu0[j] for j in nbrs)
        assert abs(u1[idx] - expected) < 1e-12

    s_a = compute_tile_states(g, frame_idx=0, t=0.0, dynamics="cml", seed=42)
    s_b = compute_tile_states(g, frame_idx=12, t=0.5, dynamics="cml", seed=42)
    assert s_a.shape == (len(g),) and s_b.shape == (len(g),)
    assert np.all((s_a >= 0.0) & (s_a <= 1.0))
    assert np.all((s_b >= 0.0) & (s_b <= 1.0))
    assert s_a.std() > 0.05 and s_b.std() > 0.05
    assert not np.allclose(s_a, s_b)
    # Deterministic
    assert np.array_equal(s_a, compute_tile_states(g, frame_idx=0, t=0.0, dynamics="cml", seed=42))


def test_life_like_dynamics_evolution() -> None:
    from render.hyperbolic_tiling import build_tiling, compute_tile_states

    for mode in ("poincare_7_3", "poincare_3_7", "quasicrystal_7", "parquet_deformation"):
        g = build_tiling(mode=mode, layers=3)
        s0 = compute_tile_states(g, frame_idx=0, t=0.0, dynamics="life_like", seed=7)
        s1 = compute_tile_states(g, frame_idx=10, t=0.4, dynamics="life_like", seed=7)
        assert s0.shape == (len(g),)
        assert np.all((s0 >= 0.0) & (s0 <= 1.0))
        assert np.all((s1 >= 0.0) & (s1 <= 1.0))
        assert s0.std() > 0.05
        assert not np.allclose(s0, s1)


# ---- 3. Non-Euclidean Camera Motion (SU(1,1) Mobius Isometries) ------

def test_mobius_su11_isometry_preserves_hyperbolic_distance_and_boundary() -> None:
    from render.hyperbolic_tiling import (
        hyperbolic_distance,
        mobius_inverse,
        mobius_transform,
    )

    z1 = 0.25 - 0.18j
    z2 = -0.42 + 0.51j
    a = 0.37 + 0.29j
    theta = 1.15

    w1 = mobius_transform(z1, a=a, theta=theta)
    w2 = mobius_transform(z2, a=a, theta=theta)
    assert abs(w1) < 1.0 and abs(w2) < 1.0

    # Hyperbolic metric invariance: d_H(T(z1), T(z2)) == d_H(z1, z2)
    d_orig = hyperbolic_distance(z1, z2)
    d_trans = hyperbolic_distance(w1, w2)
    assert abs(d_orig - d_trans) < 1e-12

    # Exact inverse recovery
    assert abs(mobius_inverse(w1, a=a, theta=theta) - z1) < 1e-12
    assert abs(mobius_inverse(w2, a=a, theta=theta) - z2) < 1e-12

    # Unit circle boundary |z| = 1 maps onto |z| = 1
    boundary_pts = np.exp(1j * np.linspace(0.0, 2.0 * math.pi, 32))
    mapped_boundary = mobius_transform(boundary_pts, a=a, theta=theta)
    assert np.allclose(np.abs(mapped_boundary), 1.0, atol=1e-12)


# ---- 4. Native MetaPost Vector Exporter ------------------------------

def test_export_metapost_generates_valid_mp_and_compiles(tmp_path: Path) -> None:
    from render.hyperbolic_tiling import (
        build_tiling,
        compute_tile_states,
        export_metapost,
    )

    for mode in ("poincare_7_3", "poincare_3_7", "quasicrystal_7", "parquet_deformation"):
        g = build_tiling(mode=mode, layers=2)
        states = compute_tile_states(g, frame_idx=6, t=0.25, dynamics="cml", seed=42)
        mp_file = tmp_path / f"test_{mode}.mp"
        content = export_metapost(g, states, "trumbull_2001", mp_file, scale_bp=150.0)
        assert mp_file.exists()
        assert "beginfig(1);" in content
        assert "endfig;" in content
        assert "withcolor" in content
        if mode.startswith("poincare"):
            assert ".." in content  # circular geodesic arc splines

    mpost_bin = shutil.which("mpost")
    if mpost_bin:
        for mp_name in ("test_poincare_7_3.mp", "test_parquet_deformation.mp"):
            res = subprocess.run(
                [mpost_bin, "-interaction=nonstopmode", mp_name],
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"mpost failed on {mp_name}:\n{res.stdout}\n{res.stderr}"


# ---- 5. Rasterization, Adapter, Composer & CLI -----------------------

def test_render_tiling_numpy_and_adapter_frames() -> None:
    import render.adapters  # noqa: F401
    from render.core import get_renderer, list_renderers
    from scene.manifest import Shot

    assert "hyperbolic_tiling" in list_renderers()

    for mode in ("poincare_7_3", "poincare_3_7", "quasicrystal_7", "parquet_deformation"):
        shot = Shot(
            id=f"test_{mode}",
            renderer="hyperbolic_tiling",
            params={
                "mode": mode,
                "layers": 3,
                "dynamics": "cml",
                "samples": 1,
                "engine": "numpy",
            },
            duration_frames=12,
            resolution=(96, 54),
        )
        renderer = get_renderer("hyperbolic_tiling")(shot)
        f0 = renderer.render_frame(0, 0.0, shot.camera.at(0.0))
        f1 = renderer.render_frame(11, 1.0, shot.camera.at(1.0))

        assert f0.shape == (54, 96, 3) and f0.dtype == np.float32
        assert np.isfinite(f0).all() and np.isfinite(f1).all()
        assert f0.min() >= 0.0 and f0.max() <= 1.0
        assert not np.array_equal(f0, f1), f"{mode} frame did not advance"
        # Bit-for-bit deterministic re-render
        assert np.array_equal(f0, renderer.render_frame(0, 0.0, shot.camera.at(0.0)))


def test_composer_places_hyperbolic_tiling() -> None:
    from studio.voyage_composer import compose_voyage

    seen = set()
    for seed in range(12):
        g = compose_voyage(target_seconds=600.0, seed=seed)
        seen |= {s.renderer for s in g.shots}
    assert "hyperbolic_tiling" in seen


def test_cli_still_and_metapost_export(tmp_path: Path) -> None:
    from render.hyperbolic_tiling import main

    out_png = tmp_path / "still.png"
    out_mp = tmp_path / "still.mp"
    rc = main([
        "--mode", "poincare_7_3",
        "--layers", "3",
        "--dynamics", "cml",
        "--width", "128",
        "--height", "72",
        "--out", str(out_png),
        "--export-metapost", str(out_mp),
    ])
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 500
    assert out_mp.exists() and out_mp.stat().st_size > 1000

