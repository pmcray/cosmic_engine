"""Tests for the procedural detail synthesis toolkit.

The numpy primitives are tested directly. The Taichi @ti.func helpers
are only exercised at GPU compile time inside renderers (since they
are not standalone callable from Python); these tests assert that the
symbols exist and have the right kind.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_np_fbm_2d_is_finite_zero_mean() -> None:
    from render.noise import np_fbm_2d

    n = np_fbm_2d((96, 96), octaves=5, seed=0)
    assert n.shape == (96, 96)
    assert n.dtype == np.float32
    assert np.isfinite(n).all()
    assert abs(float(n.mean())) < 1e-4
    assert 0.5 < float(n.std()) < 2.0


def test_np_fbm_2d_seed_is_deterministic() -> None:
    from render.noise import np_fbm_2d

    a = np_fbm_2d((32, 32), octaves=4, seed=7)
    b = np_fbm_2d((32, 32), octaves=4, seed=7)
    c = np_fbm_2d((32, 32), octaves=4, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_np_fbm_2d_octave_count_changes_spectrum() -> None:
    """More octaves means finer detail, so the standard deviation of
    the row-to-row differences should grow with octave count."""
    from render.noise import np_fbm_2d

    low = np_fbm_2d((64, 64), octaves=2, seed=3)
    high = np_fbm_2d((64, 64), octaves=6, seed=3)
    low_d = float(np.diff(low, axis=0).std())
    high_d = float(np.diff(high, axis=0).std())
    assert high_d > low_d


def test_taichi_symbols_exist() -> None:
    """Every Taichi primitive must be importable, even from a CPU host.
    On CPU the symbol is a stub that raises when called, but the import
    itself must succeed so renderer modules can be imported and have
    their kernels JIT-compiled later on a GPU host."""
    from render import noise

    for name in (
        "hash3", "value_noise_3d", "fbm_3d", "fbm_3d_footprint",
        "ridged_fbm_3d", "domain_warp_fbm", "worley_3d", "curl_noise_3d",
    ):
        assert hasattr(noise, name), f"toolkit missing {name}"


def test_cpu_stub_raises() -> None:
    """When taichi isn't loaded, calling a stub must raise clearly."""
    from render import noise

    if not noise._HAS_TAICHI:
        try:
            noise.value_noise_3d((0.0, 0.0, 0.0))
        except RuntimeError as e:
            assert "Taichi" in str(e)
        else:
            raise AssertionError("expected RuntimeError on CPU stub call")


def test_volumetric_gas_giant_uses_toolkit() -> None:
    """Refactor sanity check: the volumetric_gas_giant renderer is the
    first user of the toolkit. Its module must import the helpers and
    not re-define them locally."""
    src = (ROOT / "render" / "volumetric_gas_giant.py").read_text()
    assert "from render.noise import" in src
    assert "domain_warp_fbm" in src
    # The old per-class helpers must be gone.
    assert "def _hash3(" not in src
    assert "def _value_noise_3d(" not in src
    assert "def _detail_fbm(" not in src
    assert "def _surface_detail(" not in src


def main() -> int:
    test_np_fbm_2d_is_finite_zero_mean()
    print("ok: test_np_fbm_2d_is_finite_zero_mean")
    test_np_fbm_2d_seed_is_deterministic()
    print("ok: test_np_fbm_2d_seed_is_deterministic")
    test_np_fbm_2d_octave_count_changes_spectrum()
    print("ok: test_np_fbm_2d_octave_count_changes_spectrum")
    test_taichi_symbols_exist()
    print("ok: test_taichi_symbols_exist")
    test_cpu_stub_raises()
    print("ok: test_cpu_stub_raises")
    test_volumetric_gas_giant_uses_toolkit()
    print("ok: test_volumetric_gas_giant_uses_toolkit")
    print("\nall noise toolkit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
