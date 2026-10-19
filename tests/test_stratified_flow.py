"""Tests for `render.stratified_flow` (`kelvin_helmholtz` rocking two-liquid
wave machine and `sand_strata` invertible falling-sand sedimentation, detrng
determinism, frame-addressable stepping, MetaPost contour/horizon export,
adapter, and voyage composer integration).
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- 1. Mode 1: Kelvin-Helmholtz Rocking Wave Machine ----------------

def test_rocking_gravity_harmonic_vector() -> None:
    from render.stratified_flow import rocking_gravity

    for t in (0.0, 0.4, 1.2, 2.7):
        theta, (gx, gy) = rocking_gravity(t, g0=1.5, theta_max_deg=15.0, omega=1.35, phi0=0.25)
        assert abs(math.hypot(gx, gy) - 1.5) < 1e-12
        assert gy < 0.0
        assert abs(gx - 1.5 * math.sin(theta)) < 1e-12
        assert abs(gy - (-1.5 * math.cos(theta))) < 1e-12


def test_kelvin_helmholtz_baroclinic_rollup_and_overturning() -> None:
    from render.stratified_flow import StratifiedFlowSimulator, render_stratified_frame

    sim = StratifiedFlowSimulator(mode="kelvin_helmholtz", seed=42, warmup_steps=8, substeps_per_frame=2)
    s0 = sim.state_at(0)
    s36 = sim.state_at(36)

    # Interface billow amplitude grows and develops overturning spirals (non-monotonic x along contour)
    span0 = float(s0.kh_y.max() - s0.kh_y.min())
    span36 = float(s36.kh_y.max() - s36.kh_y.min())
    assert span36 > span0 * 1.25
    assert not np.allclose(s0.kh_gamma, s36.kh_gamma)

    # Check overturning (dx < 0 segments where billow crests curl back over troughs)
    dx = np.diff(s36.kh_x)
    dx = np.where(dx < -0.5, dx + 1.0, np.where(dx > 0.5, dx - 1.0, dx))
    assert (dx < 0.0).any(), "expected Kelvin-Helmholtz contour to overturn in billow spirals"

    img = render_stratified_frame(s36, 128, 72, palette="trumbull_2001")
    assert img.shape == (72, 128, 3) and img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.std() > 0.08


# ---- 2. Mode 2: Sand Strata Falling-Sand Picture ---------------------

def test_sand_strata_species_plumes_and_repose_stratification() -> None:
    from render.stratified_flow import (
        DEFAULT_GRAIN_SPECIES,
        StratifiedFlowSimulator,
        _relax_angle_of_repose,
        render_stratified_frame,
    )

    # Verify 4 distinct grain classes ordered by density and terminal velocity
    assert len(DEFAULT_GRAIN_SPECIES) == 4
    densities = [sp.density for sp in DEFAULT_GRAIN_SPECIES]
    velocities = [sp.settling_velocity for sp in DEFAULT_GRAIN_SPECIES]
    assert densities == sorted(densities, reverse=True)
    assert velocities == sorted(velocities, reverse=True)

    # Verify angle-of-repose SOC avalanche relaxation bounds local slope
    n = 101
    dx = 1.0 / (n - 1)
    base = np.zeros(n, dtype=np.float64)
    spike = np.zeros(n, dtype=np.float64)
    spike[50] = 0.50
    relaxed = _relax_angle_of_repose(spike, base, dx=dx, repose_deg=30.0, iterations=200)
    assert abs(relaxed.sum() - spike.sum()) < 1e-9  # mass conservation
    assert relaxed.max() < spike.max() * 0.4

    # Simulate falling-sand stratification over 36 frames
    sim = StratifiedFlowSimulator(mode="sand_strata", seed=42, warmup_steps=10, substeps_per_frame=2)
    s0 = sim.state_at(0)
    s36 = sim.state_at(36)

    assert len(s36.horizons) > len(s0.horizons)
    assert len(s36.plumes) >= 4
    assert len(s36.bubbles) >= 10
    # Stacked sedimentary horizons never cross: h_0(x) <= h_1(x) <= ... <= h_K(x)
    for k in range(1, len(s36.horizons)):
        assert np.all(s36.horizons[k] >= s36.horizons[k - 1] - 1e-12)
    # Total accumulated sediment volume increases while upper reservoir depletes
    assert float(s36.horizons[-1].sum()) > float(s0.horizons[-1].sum())
    assert float(s36.reservoir_top.sum()) < float(s0.reservoir_top.sum())

    img = render_stratified_frame(s36, 128, 72, palette="trumbull_2001")
    assert img.shape == (72, 128, 3) and img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.std() > 0.06


# ---- 3. Determinism & Frame-Addressability ---------------------------

def test_detrng_parity_and_frame_addressable_reproducibility() -> None:
    from render.detrng import py_rand01
    from render.stratified_flow import StratifiedFlowSimulator, _det_hash_2d

    # Vectorized 2D PCG hash matches render.detrng.py_rand01 bit-for-bit
    ix = np.array([[0, 1, 7], [12, 25, 99]])
    iy = np.array([[3, 4, 8], [15, 31, 42]])
    vec_vals = _det_hash_2d(ix, iy, seed=42, salt=12345)
    for r in range(2):
        for c in range(3):
            ref = py_rand01(int(ix[r, c]), int(iy[r, c]), 42, 12345)
            assert abs(float(vec_vals[r, c]) - ref) < 1e-12

    # Sequential stepping vs direct jump vs out-of-order rewind must match bit-for-bit
    for mode in ("kelvin_helmholtz", "sand_strata", "sensitive_chaos"):
        sim_seq = StratifiedFlowSimulator(mode=mode, seed=99)
        for f in range(15):
            _ = sim_seq.state_at(f)
        st_seq = sim_seq.state_at(15)

        sim_jump = StratifiedFlowSimulator(mode=mode, seed=99)
        st_jump = sim_jump.state_at(15)
        st_rewind = sim_jump.state_at(4)
        st_jump_again = sim_jump.state_at(15)

        if mode == "kelvin_helmholtz":
            assert np.array_equal(st_seq.kh_x, st_jump.kh_x)
            assert np.array_equal(st_seq.kh_y, st_jump.kh_y)
            assert np.array_equal(st_seq.kh_x, st_jump_again.kh_x)
            assert not np.array_equal(st_rewind.kh_x, st_jump.kh_x)
        elif mode == "sand_strata":
            assert np.array_equal(st_seq.horizons[-1], st_jump.horizons[-1])
            assert np.array_equal(st_seq.horizons[-1], st_jump_again.horizons[-1])
            assert not np.array_equal(st_rewind.horizons[-1], st_jump.horizons[-1])
        else:
            assert abs(st_seq.sc_strokes - st_jump.sc_strokes) < 1e-12
            assert np.array_equal(st_seq.sc_filaments[0], st_jump.sc_filaments[0])
            assert np.array_equal(st_seq.sc_filaments[0], st_jump_again.sc_filaments[0])
            assert not np.array_equal(st_rewind.sc_filaments[0], st_jump.sc_filaments[0])


# ---- 4. Mode 3: Sensitive Chaos & 2x2 Quartet Plate ------------------

def test_sensitive_chaos_von_karman_stretching_and_folding() -> None:
    from render.stratified_flow import (
        StratifiedFlowSimulator,
        filament_complexity_metrics,
        render_stratified_frame,
    )

    # Verify alternating Bénard-von Kármán vortex street and monotonic
    # Lagrangian stretching & folding across strokes 1 -> 2 -> 4 -> 7
    metrics_by_stroke = {}
    for s_val in (1.0, 2.2, 4.4, 7.2):
        sim = StratifiedFlowSimulator(mode="sensitive_chaos", seed=42, strokes=s_val)
        st = sim.state_at(0)
        assert abs(st.sc_strokes - s_val) < 1e-9
        assert len(st.sc_vortices) >= 8
        assert len(st.sc_filaments) > 20
        # Alternating signs in primary pass vortex street
        pass0_signs = [v["circ"] for v in st.sc_vortices[:8]]
        for idx in range(1, len(pass0_signs)):
            assert pass0_signs[idx] * pass0_signs[idx - 1] < 0.0
        metrics_by_stroke[s_val] = filament_complexity_metrics(st)

    # Total arc length and direction reversals (folds) increase strictly with strokes
    assert metrics_by_stroke[2.2]["total_arclength"] > metrics_by_stroke[1.0]["total_arclength"] * 1.15
    assert metrics_by_stroke[4.4]["total_arclength"] > metrics_by_stroke[2.2]["total_arclength"] * 1.20
    assert metrics_by_stroke[7.2]["total_arclength"] > metrics_by_stroke[4.4]["total_arclength"] * 1.15
    assert metrics_by_stroke[7.2]["reversals"] > metrics_by_stroke[1.0]["reversals"]

    # Render single sensitive_chaos frame
    sim_single = StratifiedFlowSimulator(mode="sensitive_chaos", seed=42, strokes=4.0)
    img = render_stratified_frame(sim_single.state_at(10), 96, 128, palette="trumbull_2001")
    assert img.shape == (128, 96, 3) and img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.std() > 0.08


def test_sensitive_chaos_quartet_plate_and_metapost(tmp_path: Path) -> None:
    from render.stratified_flow import (
        export_metapost_quartet,
        main,
        render_sensitive_chaos_quartet,
    )

    plate = render_sensitive_chaos_quartet(
        seed=42, width=160, height=220, palette="trumbull_2001"
    )
    assert plate.shape == (220, 160, 3) and plate.dtype == np.float32
    assert np.isfinite(plate).all()
    assert plate.min() >= 0.0 and plate.max() <= 1.0
    # Verify crisp white border lines and non-identical quadrants
    q_tl = plate[15:95, 15:70]
    q_tr = plate[15:95, 90:145]
    q_bl = plate[125:205, 15:70]
    q_br = plate[125:205, 90:145]
    assert not np.allclose(q_tl, q_tr)
    assert not np.allclose(q_bl, q_br)
    assert not np.allclose(q_tl, q_br)

    mp_quartet = tmp_path / "quartet.mp"
    content = export_metapost_quartet(mp_quartet, seed=42, scale_bp=160.0)
    assert mp_quartet.exists()
    assert "Panel 1" in content and "Panel 4" in content

    mpost_bin = shutil.which("mpost")
    if mpost_bin:
        res = subprocess.run(
            [mpost_bin, "-interaction=nonstopmode", mp_quartet.name],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"mpost failed on quartet:\n{res.stdout}\n{res.stderr}"
        assert (tmp_path / "quartet-1.mps").exists()

    # Verify --quartet CLI option with --export-metapost
    out_q_png = tmp_path / "cli_quartet.png"
    out_q_mp = tmp_path / "cli_quartet.mp"
    rc = main([
        "--quartet", str(out_q_png),
        "--width", "160",
        "--height", "220",
        "--export-metapost", str(out_q_mp),
    ])
    assert rc == 0
    assert out_q_png.exists() and out_q_png.stat().st_size > 1000
    assert out_q_mp.exists() and out_q_mp.stat().st_size > 1000


# ---- 5. Native MetaPost Contour & Horizon Exporter -------------------

def test_export_metapost_all_modes_and_mpost_compile(tmp_path: Path) -> None:
    from render.stratified_flow import StratifiedFlowSimulator, export_metapost

    mpost_bin = shutil.which("mpost")
    for mode in ("kelvin_helmholtz", "sand_strata", "sensitive_chaos"):
        sim = StratifiedFlowSimulator(mode=mode, seed=42)
        state = sim.state_at(24)
        mp_file = tmp_path / f"test_{mode}.mp"
        content = export_metapost(state, mp_file, scale_bp=180.0, palette="trumbull_2001")
        assert mp_file.exists()
        assert "prologues := 3;" in content
        assert "beginfig(1);" in content
        assert "endfig;" in content
        assert "withcolor" in content

        if mpost_bin:
            res = subprocess.run(
                [mpost_bin, "-interaction=nonstopmode", mp_file.name],
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"mpost failed on {mode}:\n{res.stdout}\n{res.stderr}"
            assert (tmp_path / f"test_{mode}-1.mps").exists()


# ---- 6. Adapter, Voyage Composer & CLI -------------------------------

def test_stratified_flow_adapter_and_composer_wiring(tmp_path: Path) -> None:
    import render.adapters  # noqa: F401
    from render.core import get_renderer, list_renderers
    from render.stratified_flow import main
    from scene.manifest import Shot
    from studio.voyage_composer import compose_voyage

    assert "stratified_flow" in list_renderers()

    for mode in ("kelvin_helmholtz", "sand_strata", "sensitive_chaos"):
        shot = Shot(
            id=f"shot_{mode}",
            renderer="stratified_flow",
            params={"mode": mode, "samples": 1, "engine": "numpy"},
            duration_frames=10,
            resolution=(96, 54),
            seed=42,
        )
        r = get_renderer("stratified_flow")(shot)
        f0 = r.render_frame(0, 0.0, shot.camera.at(0.0))
        f9 = r.render_frame(9, 1.0, shot.camera.at(1.0))
        assert f0.shape == (54, 96, 3) and f0.dtype == np.float32
        assert np.isfinite(f0).all() and np.isfinite(f9).all()
        assert not np.array_equal(f0, f9)
        assert np.array_equal(f0, r.render_frame(0, 0.0, shot.camera.at(0.0)))

    seen = set()
    for seed in range(16):
        g = compose_voyage(target_seconds=600.0, seed=seed)
        seen |= {s.renderer for s in g.shots}
    assert "stratified_flow" in seen

    out_png = tmp_path / "cli_kh.png"
    out_mp = tmp_path / "cli_kh.mp"
    rc = main([
        "--mode", "kelvin_helmholtz",
        "--frame", "20",
        "--width", "128",
        "--height", "72",
        "--out", str(out_png),
        "--export-metapost", str(out_mp),
    ])
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 500
    assert out_mp.exists() and out_mp.stat().st_size > 500

