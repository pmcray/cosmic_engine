"""CPU-side tests for the stellar_surface renderer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_planck_color_sun_is_white() -> None:
    from render.stellar_surface import _planck_color

    r, g, b = _planck_color(5800.0)
    # All channels in the same ballpark for sun-like temperatures.
    avg = (r + g + b) / 3.0
    for c in (r, g, b):
        assert 0.5 < c / avg < 1.6, (r, g, b)
    print("ok: test_planck_color_sun_is_white")


def test_planck_color_cool_red_hot_blue() -> None:
    from render.stellar_surface import _planck_color

    # Cool M-dwarf is reddish (R > B).
    r_m, _, b_m = _planck_color(3400.0)
    assert r_m > b_m
    # Hot O-star is blue-tinted (B >= R).
    r_o, _, b_o = _planck_color(32000.0)
    assert b_o >= r_o * 0.9
    print("ok: test_planck_color_cool_red_hot_blue")


def test_limb_darkening_dims_toward_limb() -> None:
    from render.stellar_surface import _np_limb_darkening

    center = _np_limb_darkening(1.0, u=0.6)
    mid = _np_limb_darkening(0.5, u=0.6)
    limb = _np_limb_darkening(0.0, u=0.6)
    assert center > mid > limb
    assert abs(center - 1.0) < 1e-5
    assert abs(limb - 0.4) < 1e-5  # 1 - u
    print("ok: test_limb_darkening_dims_toward_limb")


def test_sunspot_mask_peaks_at_center() -> None:
    from render.stellar_surface import _np_sunspot_mask

    at_center = _np_sunspot_mask(20.0, 50.0, 20.0, 50.0, r_deg=6.0)
    nearby = _np_sunspot_mask(20.0, 55.0, 20.0, 50.0, r_deg=6.0)
    far = _np_sunspot_mask(20.0, 100.0, 20.0, 50.0, r_deg=6.0)
    assert at_center > nearby > far
    assert abs(at_center - 1.0) < 1e-5
    print("ok: test_sunspot_mask_peaks_at_center")


def test_stellar_manifest_loads_three_shots() -> None:
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "stellar_close.json")
    assert len(g.shots) == 3
    assert all(s.renderer == "stellar_surface" for s in g.shots)
    assert [s.id for s in g.shots] == [
        "sun_active_region", "m_dwarf", "hot_o_star",
    ]
    # Spectral ordering by T_eff.
    Ts = [s.params["T_eff"] for s in g.shots]
    assert Ts[1] < Ts[0] < Ts[2], Ts
    # Sunspots present on the sun-like shot.
    assert len(g.shots[0].params["sunspots"]) >= 1
    print("ok: test_stellar_manifest_loads_three_shots")


def test_stellar_each_shot_has_valid_blackbody_temperature() -> None:
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "stellar_close.json")
    for s in g.shots:
        T = s.params.get("T_eff", 5800.0)
        # M dwarf to hot O star plausibility band.
        assert 1500.0 <= T <= 60000.0, (s.id, T)
    print("ok: test_stellar_each_shot_has_valid_blackbody_temperature")


def main() -> int:
    test_planck_color_sun_is_white()
    test_planck_color_cool_red_hot_blue()
    test_limb_darkening_dims_toward_limb()
    test_sunspot_mask_peaks_at_center()
    test_stellar_manifest_loads_three_shots()
    test_stellar_each_shot_has_valid_blackbody_temperature()
    print("\nall stellar_surface tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
