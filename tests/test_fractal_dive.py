"""CPU-side tests for the Mandelbrot / Multibrot perturbation dive.

The load-bearing claim is that perturbation reproduces plain escape-time
iteration where plain iteration still works (shallow zooms), and then
keeps producing structure far past where float64 gives up.
"""
from __future__ import annotations

import math
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


# ---- Precision / budget helpers --------------------------------------

def test_required_precision_grows_with_depth() -> None:
    from render.fractal_dive import required_precision

    assert required_precision(1.0) >= 30
    shallow = required_precision(1e-6)
    deep = required_precision(1e-40)
    assert deep > shallow
    # Enough digits to resolve pixel-scale offsets, with guard digits.
    assert deep >= 40 + 15
    try:
        required_precision(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive scale")
    print("ok: test_required_precision_grows_with_depth")


def test_iter_budget_grows_with_depth() -> None:
    from render.fractal_dive import iter_budget

    assert iter_budget(1.0) < iter_budget(1e-10) < iter_budget(1e-30)
    print("ok: test_iter_budget_grows_with_depth")


def test_dive_scale_is_exponential() -> None:
    from render.fractal_dive import dive_scale

    a, b = 2.0, 2e-20
    assert dive_scale(0.0, a, b) == a
    assert abs(dive_scale(1.0, a, b) - b) < 1e-30
    # Equal time steps give equal ratios (constant apparent zoom speed).
    r1 = dive_scale(0.25, a, b) / dive_scale(0.0, a, b)
    r2 = dive_scale(0.75, a, b) / dive_scale(0.5, a, b)
    assert abs(r1 - r2) / r1 < 1e-9
    print("ok: test_dive_scale_is_exponential")


# ---- Reference orbit -------------------------------------------------

def test_reference_orbit_starts_at_zero_and_rejects_bad_power() -> None:
    from render.fractal_dive import ReferenceOrbit

    orb = ReferenceOrbit(power=2, max_iter=64, precision=40)
    assert orb.z[0] == 0
    assert len(orb) <= 65
    try:
        ReferenceOrbit(power=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for power < 2")
    print("ok: test_reference_orbit_starts_at_zero_and_rejects_bad_power")


def test_reference_orbit_escapes_for_exterior_centre() -> None:
    from render.fractal_dive import ReferenceOrbit

    # (2, 2) is far outside the set: the orbit must bail out early.
    orb = ReferenceOrbit("2.0", "2.0", power=2, max_iter=500, precision=30)
    assert len(orb) < 20
    print("ok: test_reference_orbit_escapes_for_exterior_centre")


# ---- Perturbation vs direct iteration --------------------------------

def _compare(power: int, cre: str, cim: str, scale: float,
             w: int = 96, h: int = 54, max_iter: int = 800):
    from render.fractal_dive import (ReferenceOrbit, perturbation_grid,
                                     direct_grid, pixel_deltas,
                                     required_precision)

    orbit = ReferenceOrbit(cre, cim, power=power, max_iter=max_iter,
                           precision=required_precision(scale))
    dc = pixel_deltas(w, h, scale)
    mu_p = perturbation_grid(orbit, dc, max_iter)
    mu_d = direct_grid(complex(float(cre), float(cim)) + dc, power, max_iter)
    return mu_p, mu_d


def test_perturbation_matches_direct_iteration_mandelbrot() -> None:
    from render.fractal_dive import DEFAULT_CENTER_RE, DEFAULT_CENTER_IM

    mu_p, mu_d = _compare(2, DEFAULT_CENTER_RE, DEFAULT_CENTER_IM, 0.002)
    # Interior/exterior classification agrees almost everywhere (a few
    # pixels sit exactly on the boundary and land either side).
    agree = ((mu_p < 0) == (mu_d < 0)).mean()
    assert agree > 0.999, agree
    both = (mu_p >= 0) & (mu_d >= 0)
    diff = np.abs(mu_p[both] - mu_d[both])
    # Smooth counts agree to well under a hundredth of an iteration for
    # the overwhelming majority; boundary pixels can differ by ~1.
    assert np.percentile(diff, 99) < 0.01, np.percentile(diff, 99)
    assert (diff > 1.5).mean() < 0.005, (diff > 1.5).mean()
    print("ok: test_perturbation_matches_direct_iteration_mandelbrot")


def test_perturbation_matches_direct_iteration_multibrot_d3() -> None:
    mu_p, mu_d = _compare(3, "-0.12", "0.75", 0.01, w=64, h=36, max_iter=500)
    agree = ((mu_p < 0) == (mu_d < 0)).mean()
    assert agree > 0.999, agree
    both = (mu_p >= 0) & (mu_d >= 0)
    diff = np.abs(mu_p[both] - mu_d[both])
    assert np.percentile(diff, 99) < 0.01, np.percentile(diff, 99)
    print("ok: test_perturbation_matches_direct_iteration_multibrot_d3")


def test_deep_zoom_beyond_float64_still_resolves_structure() -> None:
    """At 1e-20 the pixel spacing is far below float64 resolution of the
    centre, so direct iteration collapses to a single flat value. The
    perturbation grid must still vary across the frame."""
    from render.fractal_dive import (ReferenceOrbit, perturbation_grid,
                                     direct_grid, pixel_deltas,
                                     required_precision, iter_budget,
                                     DEFAULT_CENTER_RE, DEFAULT_CENTER_IM)

    scale = 1e-20
    budget = iter_budget(scale)
    orbit = ReferenceOrbit(DEFAULT_CENTER_RE, DEFAULT_CENTER_IM, power=2,
                           max_iter=budget,
                           precision=required_precision(scale))
    dc = pixel_deltas(24, 16, scale)
    mu = perturbation_grid(orbit, dc, budget)

    escaped = mu >= 0
    assert escaped.any(), "deep dive escaped nowhere — budget too small?"
    spread = mu[escaped].max() - mu[escaped].min()
    assert spread > 10.0, f"deep frame is flat (spread {spread})"

    # The float64 comparison: adding dc to a float64 centre is a no-op at
    # this scale, so every pixel gets the identical value.
    c = complex(float(DEFAULT_CENTER_RE), float(DEFAULT_CENTER_IM)) + dc
    assert len(np.unique(c)) == 1, "float64 unexpectedly resolved this scale"
    mu_direct = direct_grid(c, 2, 400)
    assert len(np.unique(mu_direct)) == 1
    print("ok: test_deep_zoom_beyond_float64_still_resolves_structure")


def test_perturbation_is_deterministic() -> None:
    from render.fractal_dive import (ReferenceOrbit, perturbation_grid,
                                     pixel_deltas)

    orbit = ReferenceOrbit(power=2, max_iter=400, precision=40)
    dc = pixel_deltas(48, 27, 0.01)
    a = perturbation_grid(orbit, dc, 400)
    b = perturbation_grid(orbit, dc, 400)
    assert np.array_equal(a, b)
    print("ok: test_perturbation_is_deterministic")


# ---- Geometry / colour -----------------------------------------------

def test_pixel_deltas_shape_orientation_and_rotation() -> None:
    from render.fractal_dive import pixel_deltas

    dc = pixel_deltas(80, 40, 1.0)
    assert dc.shape == (40, 80)
    # Row 0 is the top of frame => positive imaginary part.
    assert dc[0, 0].imag > 0 and dc[-1, 0].imag < 0
    # Aspect: horizontal extent is wider than vertical for a 2:1 frame.
    assert abs(dc[0, -1].real) > abs(dc[0, 0].imag)
    # A quarter turn maps the real axis onto the imaginary axis.
    straight = pixel_deltas(64, 64, 1.0)
    turned = pixel_deltas(64, 64, 1.0, rotation=math.pi / 2.0)
    assert abs(turned[32, 40].imag - straight[32, 40].real) < 1e-12
    print("ok: test_pixel_deltas_shape_orientation_and_rotation")


def test_supersample_offsets_are_fixed_and_centred() -> None:
    from render.fractal_dive import supersample_offsets

    offs = supersample_offsets(2)
    assert len(offs) == 4
    assert offs == supersample_offsets(2)  # deterministic, no RNG
    mx = sum(o[0] for o in offs) / len(offs)
    my = sum(o[1] for o in offs) / len(offs)
    assert abs(mx) < 1e-12 and abs(my) < 1e-12
    assert all(-0.5 <= o[0] < 0.5 and -0.5 <= o[1] < 0.5 for o in offs)
    assert len(supersample_offsets(1)) == 1
    print("ok: test_supersample_offsets_are_fixed_and_centred")


def test_colorize_shape_range_and_interior() -> None:
    from render.fractal_dive import colorize

    anchors = ((0.0, 0.0, 0.1), (0.9, 0.2, 0.1), (1.0, 0.9, 0.3))
    mu = np.array([[-1.0, 5.0], [50.0, 500.0]])
    rgb = colorize(mu, anchors, interior_color=(0.02, 0.0, 0.05))
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.float32
    assert np.isfinite(rgb).all()
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    # Interior pixel takes the interior colour exactly.
    assert np.allclose(rgb[0, 0], (0.02, 0.0, 0.05))
    # Different depths give different colours.
    assert not np.allclose(rgb[0, 1], rgb[1, 1])
    try:
        colorize(mu, ((1.0, 1.0, 1.0),))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a single anchor")
    print("ok: test_colorize_shape_range_and_interior")


# ---- Registration / composer wiring ----------------------------------

def test_fractal_dive_renderer_registered() -> None:
    import render.adapters  # noqa: F401 — registration side effect
    from render.core import list_renderers

    assert "fractal_dive" in list_renderers()
    print("ok: test_fractal_dive_renderer_registered")


def test_composer_places_fractal_dives() -> None:
    from studio.voyage_composer import compose_voyage

    seen = set()
    for seed in range(8):
        g = compose_voyage(target_seconds=600.0, seed=seed)
        seen |= {s.renderer for s in g.shots}
    assert "fractal_dive" in seen
    print("ok: test_composer_places_fractal_dives")


def test_fractal_locations_are_valid_and_parse() -> None:
    from render.fractal_dive import ReferenceOrbit
    from studio.voyage_composer import FRACTAL_LOCATIONS

    assert len(FRACTAL_LOCATIONS) >= 5
    for name, cre, cim, power in FRACTAL_LOCATIONS:
        float(cre), float(cim)          # parseable as decimals
        assert power >= 2
        # Each location should be in (or very near) the set: a short
        # orbit must not escape immediately.
        orb = ReferenceOrbit(cre, cim, power=power, max_iter=200,
                             precision=40)
        assert len(orb) > 50, f"{name} escapes too early ({len(orb)})"
    print("ok: test_fractal_locations_are_valid_and_parse")


# ---- Taichi engine parity --------------------------------------------

def test_taichi_engine_matches_numpy_engine() -> None:
    if not _has_taichi():
        print("skip: taichi not installed")
        return
    import taichi as ti
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass
    from render.fractal_dive import (ReferenceOrbit, perturbation_grid,
                                     pixel_deltas, FractalDiveEngine)

    scale, max_iter = 0.002, 800
    orbit = ReferenceOrbit(power=2, max_iter=max_iter, precision=40)
    mu_np = perturbation_grid(orbit, pixel_deltas(96, 54, scale), max_iter)
    eng = FractalDiveEngine(96, 54, orbit)
    mu_ti = eng.compute_mu(scale, 0.0, max_iter).astype(np.float64)

    assert ((mu_ti < 0) == (mu_np < 0)).mean() > 0.999
    both = (mu_ti >= 0) & (mu_np >= 0)
    diff = np.abs(mu_ti[both] - mu_np[both])
    assert np.percentile(diff, 99) < 0.01, np.percentile(diff, 99)
    # Bit-stable across repeated calls (no RNG anywhere in the path).
    assert np.array_equal(eng.compute_mu(scale, 0.0, max_iter),
                          eng.compute_mu(scale, 0.0, max_iter))
    print("ok: test_taichi_engine_matches_numpy_engine")


def test_fractal_dive_adapter_frames() -> None:
    """Two frames through the graph adapter: right shape, finite, and the
    dive actually moves (later frame differs from the first)."""
    import render.adapters  # noqa: F401
    from scene.manifest import Shot
    from render.core import get_renderer

    shot = Shot(
        id="t", renderer="fractal_dive",
        params={"scale_start": 0.05, "scale_end": 1e-9, "samples": 1,
                "iter_base": 200, "iter_per_decade": 100,
                "engine": "numpy"},
        duration_frames=10, resolution=(48, 27),
    )
    r = get_renderer("fractal_dive")(shot)
    a = r.render_frame(0, 0.0, shot.camera.at(0.0))
    b = r.render_frame(9, 1.0, shot.camera.at(1.0))
    assert a.shape == (27, 48, 3) and a.dtype == np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.array_equal(a, b), "dive did not advance"
    # Deterministic re-render.
    assert np.array_equal(a, r.render_frame(0, 0.0, shot.camera.at(0.0)))
    print("ok: test_fractal_dive_adapter_frames")


def test_heptabrot_locations_survive_5000_iters_and_resolve_1e18() -> None:
    """Option A validation: at least two power=7 Heptabrot coordinates
    survive >5,000 iterations and exhibit rich structure down to 1e-18."""
    from render.fractal_dive import (
        HEPTABROT_LOCATIONS,
        ReferenceOrbit,
        iter_budget,
        perturbation_grid,
        pixel_deltas,
        required_precision,
    )
    from studio.voyage_composer import FRACTAL_LOCATIONS

    p7_in_fractal = [loc for loc in FRACTAL_LOCATIONS if loc[3] == 7]
    assert len(HEPTABROT_LOCATIONS) >= 2
    assert len(p7_in_fractal) >= 2

    for name, cre, cim, power in HEPTABROT_LOCATIONS:
        assert power == 7
        # 1. Must survive > 5,000 iterations at high precision
        orb = ReferenceOrbit(cre, cim, power=7, max_iter=5100, precision=65)
        assert len(orb) > 5000, f"{name} escaped at {len(orb)} <= 5000"

        # 2. Must resolve rich filament structure down to 1e-18
        scale = 1e-18
        budget = iter_budget(scale)
        deep_orb = ReferenceOrbit(
            cre, cim, power=7, max_iter=budget,
            precision=required_precision(scale),
        )
        dc = pixel_deltas(16, 12, scale)
        mu = perturbation_grid(deep_orb, dc, budget)
        escaped = mu >= 0
        assert escaped.any(), f"{name} did not escape anywhere at 1e-18"
        spread = float(mu[escaped].max() - mu[escaped].min())
        assert spread > 50.0, f"{name} flat at 1e-18 (spread={spread})"
    print("ok: test_heptabrot_locations_survive_5000_iters_and_resolve_1e18")


def test_bisect_boundary_utility() -> None:
    """High-precision bisection utility locates a power=7 boundary point."""
    from render.fractal_dive import ReferenceOrbit, bisect_boundary

    # (0, 0) is strictly interior for z^7 + c; (0.9, 0.0) escapes immediately.
    cre, cim = bisect_boundary("0.0", "0.0", "0.9", "0.0",
                               power=7, min_iter=400, steps=28, precision=40)
    orb = ReferenceOrbit(cre, cim, power=7, max_iter=400, precision=40)
    assert len(orb) > 400
    print("ok: test_bisect_boundary_utility")


def test_fractional_power_direct_de_moivre() -> None:
    """Option B validation: continuous fractional exponents (e.g. p=2.5, 7.35)
    route through polar De Moivre in both NumPy and Taichi f64."""
    from render.fractal_dive import direct_grid, pixel_deltas

    dc = pixel_deltas(48, 32, 0.8)
    c = (-0.5 + 0.0j) + dc
    mu_25 = direct_grid(c, 2.5, max_iter=120)
    mu_735 = direct_grid(c, 7.35, max_iter=120)

    assert mu_25.shape == (32, 48) and mu_735.shape == (32, 48)
    assert (mu_25 < 0).any() and (mu_25 >= 0).any()
    assert (mu_735 < 0).any() and (mu_735 >= 0).any()
    assert not np.allclose(mu_25, mu_735)

    if _has_taichi():
        from render.fractal_dive import FractalDiveEngine
        eng = FractalDiveEngine(48, 32, orbit=None, power=2.5,
                                center_re=-0.5, center_im=0.0, direct=True)
        mu_ti = eng.compute_mu_direct(-0.5, 0.0, 2.5, 0.8, 0.0, 120).astype(np.float64)
        assert ((mu_ti < 0) == (mu_25 < 0)).mean() > 0.999
        both = (mu_ti >= 0) & (mu_25 >= 0)
        assert np.percentile(np.abs(mu_ti[both] - mu_25[both]), 99) < 0.01
    print("ok: test_fractional_power_direct_de_moivre")


def test_fractional_power_morphing_adapter() -> None:
    """Animated power_start -> power_end morphing in FractalDiveRenderer."""
    import render.adapters  # noqa: F401
    from render.core import get_renderer
    from scene.manifest import Shot

    shot = Shot(
        id="morph",
        renderer="fractal_dive",
        params={
            "center_re": "-0.75",
            "center_im": "0.02",
            "power_start": 2.0,
            "power_end": 7.0,
            "scale_start": 1.6,
            "scale_end": 0.05,
            "samples": 1,
            "iter_base": 120,
            "iter_per_decade": 100,
        },
        duration_frames=5,
        resolution=(48, 27),
    )
    r = get_renderer("fractal_dive")(shot)
    f0 = r.render_frame(0, 0.0, shot.camera.at(0.0))
    f_mid = r.render_frame(2, 0.5, shot.camera.at(0.5))
    f_end = r.render_frame(4, 1.0, shot.camera.at(1.0))
    assert f0.shape == (27, 48, 3)
    assert np.isfinite(f0).all() and np.isfinite(f_mid).all() and np.isfinite(f_end).all()
    assert not np.array_equal(f0, f_mid) and not np.array_equal(f_mid, f_end)
    print("ok: test_fractional_power_morphing_adapter")


def main_runner() -> int:
    test_required_precision_grows_with_depth()
    test_iter_budget_grows_with_depth()
    test_dive_scale_is_exponential()
    test_reference_orbit_starts_at_zero_and_rejects_bad_power()
    test_reference_orbit_escapes_for_exterior_centre()
    test_perturbation_matches_direct_iteration_mandelbrot()
    test_perturbation_matches_direct_iteration_multibrot_d3()
    test_deep_zoom_beyond_float64_still_resolves_structure()
    test_perturbation_is_deterministic()
    test_pixel_deltas_shape_orientation_and_rotation()
    test_supersample_offsets_are_fixed_and_centred()
    test_colorize_shape_range_and_interior()
    test_fractal_dive_renderer_registered()
    test_composer_places_fractal_dives()
    test_fractal_locations_are_valid_and_parse()
    test_taichi_engine_matches_numpy_engine()
    test_fractal_dive_adapter_frames()
    test_heptabrot_locations_survive_5000_iters_and_resolve_1e18()
    test_bisect_boundary_utility()
    test_fractional_power_direct_de_moivre()
    test_fractional_power_morphing_adapter()
    print("\nall fractal dive tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_runner())
