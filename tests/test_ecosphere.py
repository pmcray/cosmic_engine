"""Tests for `render.ecosphere` (Sealed Glass Ecosphere & W. Ross Ashby
Ultrastable Homeostat, circadian light cycle, 4-variable coupled biosphere ODE,
2D Gray-Scott reaction-diffusion, L-system gorgonian frond, swimming
Halocaridina shrimp trajectories, MetaPost exporter, adapter, and composer).
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


def test_circadian_light_vector_and_gorgonian_lsystem() -> None:
    from render.ecosphere import build_gorgonian_lsystem, circadian_light_vector

    intensities = []
    for t in (0.0, 0.45, 0.90, 1.35, 1.80):
        I_t, (lx, ly, lz) = circadian_light_vector(t, period_sec=1.8)
        assert 0.05 <= I_t <= 1.0001
        assert abs(math.sqrt(lx * lx + ly * ly + lz * lz) - 1.0) < 1e-12
        assert ly > 0.0
        intensities.append(I_t)

    assert max(intensities) - min(intensities) > 0.70

    segs = build_gorgonian_lsystem(seed=42, iterations=4)
    assert len(segs) >= 20
    for x0, y0, x1, y1, depth in segs:
        assert math.hypot(x0, y0) < 0.85
        assert math.hypot(x1, y1) < 0.85
        assert depth >= 0

    # Deterministic across calls with same seed
    assert segs == build_gorgonian_lsystem(seed=42, iterations=4)
    assert segs != build_gorgonian_lsystem(seed=99, iterations=4)


def test_ashby_ultrastable_homeostat_step_reconfiguration_and_recovery() -> None:
    from render.ecosphere import EcosphereSimulator

    sim = EcosphereSimulator(
        seed=42,
        rd_size=48,
        n_shrimp=7,
        warmup_steps=10,
        substeps_per_frame=4,
        dt=0.042,
        O2_star=0.62,
        eta_crit=0.14,
        perturbation_frame=16,
        perturbation_strength=0.21,
    )

    s0 = sim.state_at(0)
    assert s0.uniselector_steps == 0
    assert s0.eta < s0.eta_crit

    # At or immediately after perturbation_frame=16, eta breaches eta_crit and
    # triggers an Ashby uniselector step-reconfiguration of M and (F, k)
    s20 = sim.state_at(20)
    assert s20.uniselector_steps >= 1
    assert len(s20.step_events) >= 1
    assert not np.allclose(s0.coupling_matrix, s20.coupling_matrix)
    assert (s0.F, s0.k) != (s20.F, s20.k)

    # By frame 90, the reconfigured negative-feedback matrix restores ultrastable
    # equilibrium with eta(t) = |O2(t) - O2*| well below eta_crit
    s90 = sim.state_at(90)
    assert s90.eta < s90.eta_crit * 0.65
    assert 0.05 <= s90.N <= 0.98
    assert 0.05 <= s90.A <= 0.98
    assert 0.05 <= s90.S <= 0.98
    assert 0.05 <= s90.O2 <= 0.98
    assert np.isfinite(s90.rd_U).all() and np.isfinite(s90.rd_V).all()
    assert float(s90.rd_V.std()) > 0.01


def test_frame_addressable_determinism_and_refractive_renderer() -> None:
    from render.ecosphere import EcosphereSimulator, render_ecosphere_frame

    sim_seq = EcosphereSimulator(seed=77, rd_size=48)
    for f in range(18):
        _ = sim_seq.state_at(f)
    st_seq = sim_seq.state_at(18)

    sim_jump = EcosphereSimulator(seed=77, rd_size=48)
    st_jump = sim_jump.state_at(18)
    st_rewind = sim_jump.state_at(5)
    st_jump_again = sim_jump.state_at(18)

    assert abs(st_seq.O2 - st_jump.O2) < 1e-12
    assert np.array_equal(st_seq.rd_V, st_jump.rd_V)
    assert np.array_equal(st_seq.shrimp_positions, st_jump.shrimp_positions)
    assert np.array_equal(st_seq.rd_V, st_jump_again.rd_V)
    assert not np.array_equal(st_rewind.rd_V, st_jump.rd_V)

    img = render_ecosphere_frame(st_jump, 160, 90, palette="trumbull_2001")
    assert img.shape == (90, 160, 3) and img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.std() > 0.08


def test_export_metapost_and_mpost_compile(tmp_path: Path) -> None:
    from render.ecosphere import EcosphereSimulator, export_metapost

    sim = EcosphereSimulator(seed=42, rd_size=48)
    state = sim.state_at(32)
    mp_file = tmp_path / "test_ecosphere.mp"
    content = export_metapost(state, mp_file, scale_bp=200.0, palette="trumbull_2001")

    assert mp_file.exists()
    assert "prologues := 3;" in content
    assert "beginfig(1);" in content
    assert "orb_circle" in content
    assert "homeostat_ring" in content
    assert "endfig;" in content

    mpost_bin = shutil.which("mpost")
    if mpost_bin:
        res = subprocess.run(
            [mpost_bin, "-interaction=nonstopmode", mp_file.name],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"mpost failed:\n{res.stdout}\n{res.stderr}"
        assert (tmp_path / "test_ecosphere-1.mps").exists()


def test_ecosphere_adapter_composer_and_cli(tmp_path: Path) -> None:
    import render.adapters  # noqa: F401
    from render.core import get_renderer, list_renderers
    from render.ecosphere import main
    from scene.manifest import Shot
    from studio.voyage_composer import compose_voyage

    assert "ecosphere" in list_renderers()

    shot = Shot(
        id="shot_ecosphere",
        renderer="ecosphere",
        params={"rd_size": 48, "samples": 1, "engine": "numpy"},
        duration_frames=12,
        resolution=(96, 54),
        seed=42,
    )
    r = get_renderer("ecosphere")(shot)
    f0 = r.render_frame(0, 0.0, shot.camera.at(0.0))
    f11 = r.render_frame(11, 1.0, shot.camera.at(1.0))
    assert f0.shape == (54, 96, 3) and f0.dtype == np.float32
    assert np.isfinite(f0).all() and np.isfinite(f11).all()
    assert not np.array_equal(f0, f11)
    assert np.array_equal(f0, r.render_frame(0, 0.0, shot.camera.at(0.0)))

    seen = set()
    for seed in range(24):
        g = compose_voyage(target_seconds=600.0, seed=seed)
        seen |= {s.renderer for s in g.shots}
    assert "ecosphere" in seen

    out_png = tmp_path / "cli_eco.png"
    out_mp = tmp_path / "cli_eco.mp"
    rc = main([
        "--frame", "24",
        "--rd-size", "48",
        "--width", "128",
        "--height", "72",
        "--out", str(out_png),
        "--export-metapost", str(out_mp),
    ])
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 500
    assert out_mp.exists() and out_mp.stat().st_size > 500
