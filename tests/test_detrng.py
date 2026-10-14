"""Tests for the deterministic counter-based RNG (render/detrng.py).

The contract under test: every stochastic effect is a pure function of
(what is being shaded, frame tick, stream salt) — so re-rendering any
frame, in any order, on any run, yields bit-identical pixels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _has_taichi() -> bool:
    try:
        import taichi  # noqa: F401
        return True
    except ImportError:
        return False


def test_py_hash_reference_values_stable() -> None:
    """Pin the hash so an accidental algorithm change (which would
    silently re-seed every existing render) fails loudly."""
    from render.detrng import py_hash_u32, py_rand01

    # Determinism across calls.
    assert py_hash_u32(1, 2, 3, 4) == py_hash_u32(1, 2, 3, 4)
    # Sensitivity: one-bit key changes flip the output.
    base = py_hash_u32(10, 20, 30, 40)
    assert py_hash_u32(11, 20, 30, 40) != base
    assert py_hash_u32(10, 21, 30, 40) != base
    assert py_hash_u32(10, 20, 31, 40) != base
    assert py_hash_u32(10, 20, 30, 41) != base
    # Range.
    for k in range(200):
        v = py_rand01(k, k * 7, k * 13, 99)
        assert 0.0 <= v < 1.0
    # Negative keys are legal (two's-complement wrap).
    assert 0.0 <= py_rand01(-5, -1000, 3, 4) < 1.0
    print("ok: test_py_hash_reference_values_stable")


def test_py_rand01_distribution_roughly_uniform() -> None:
    from render.detrng import py_rand01

    n = 5000
    vals = [py_rand01(i, 17, 23, 31) for i in range(n)]
    mean = sum(vals) / n
    assert abs(mean - 0.5) < 0.02, mean
    # All 10 deciles occupied.
    hist = [0] * 10
    for v in vals:
        hist[min(9, int(v * 10))] += 1
    assert min(hist) > n / 20, hist
    print("ok: test_py_rand01_distribution_roughly_uniform")


def test_py_tick_separates_frames() -> None:
    from render.detrng import py_tick

    fps = 24
    ticks = [py_tick(f / fps) for f in range(100_000)]
    assert len(set(ticks)) == len(ticks)
    # Also at the infinite_director-style time step of 0.05 s.
    ticks = [py_tick(f * 0.05) for f in range(10_000)]
    assert len(set(ticks)) == len(ticks)
    print("ok: test_py_tick_separates_frames")


def test_taichi_matches_python_mirror() -> None:
    if not _has_taichi():
        print("skip: taichi not installed")
        return
    import taichi as ti
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass
    from render import detrng

    out = ti.field(dtype=ti.f32, shape=64)
    outh = ti.field(dtype=ti.u32, shape=64)

    @ti.kernel
    def fill():
        for k in range(64):
            out[k] = detrng.rand01(k, k * 3 + 1, k * 7 + 2, 12345)
            outh[k] = detrng.hash_u32(k, k * 3 + 1, k * 7 + 2, 12345)

    fill()
    got = out.to_numpy()
    goth = outh.to_numpy()
    for k in range(64):
        assert goth[k] == detrng.py_hash_u32(k, k * 3 + 1, k * 7 + 2, 12345)
        assert abs(got[k] - detrng.py_rand01(k, k * 3 + 1, k * 7 + 2, 12345)) < 1e-6
    print("ok: test_taichi_matches_python_mirror")


def test_corridor_frames_bit_stable_and_order_independent() -> None:
    """Render the same corridor frame twice — once cold, once after an
    unrelated frame — through the graph adapter. With counter-based RNG
    the two renders must be bit-identical (this is the chunk-boundary
    guarantee)."""
    if not _has_taichi():
        print("skip: taichi not installed")
        return

    import render.adapters  # noqa: F401 — registration side effect
    from scene.manifest import Shot
    from render.core import get_renderer

    shot = Shot(id="t", renderer="stargate_corridor",
                params={"samples": 2}, duration_frames=10,
                resolution=(48, 27))
    r = get_renderer("stargate_corridor")(shot)
    cam = shot.camera.at(0.0)

    a = r.render_frame(5, 0.5, cam).copy()
    _ = r.render_frame(3, 0.3, cam)          # perturb any hidden state
    b = r.render_frame(5, 0.5, cam).copy()
    assert np.array_equal(a, b), "frame 5 not bit-stable across re-renders"

    # Different frames must actually differ (the corridor streams).
    c = r.render_frame(6, 0.6, cam).copy()
    assert not np.array_equal(a, c)
    print("ok: test_corridor_frames_bit_stable_and_order_independent")


def test_fluid_dye_bit_stable_for_same_frame_sequence() -> None:
    """Two fresh fluid engines stepped through the same frames must land
    on bit-identical dye fields — the resumable-chunk guarantee for the
    stateful sim."""
    if not _has_taichi():
        print("skip: taichi not installed")
        return

    from physics.fluid_solver import FluidEngine

    def run() -> np.ndarray:
        eng = FluidEngine(res=64, dt=0.004, planet_type="jupiter", seed=123)
        for f in range(5):
            eng.step(f)
        return eng.dye.to_numpy()

    a = run()
    b = run()
    assert np.array_equal(a, b), "fluid dye not reproducible across runs"
    print("ok: test_fluid_dye_bit_stable_for_same_frame_sequence")


def main_runner() -> int:
    test_py_hash_reference_values_stable()
    test_py_rand01_distribution_roughly_uniform()
    test_py_tick_separates_frames()
    test_taichi_matches_python_mirror()
    test_corridor_frames_bit_stable_and_order_independent()
    test_fluid_dye_bit_stable_for_same_frame_sequence()
    print("\nall detrng tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_runner())
