"""CPU-side tests for the volumetric gas-giant bake pipeline.

Validates the numpy half of the renderer (band table consistency,
texture shape, NaN/Inf hygiene, storm placement, drift speeds) without
needing Taichi or a GPU. The Taichi kernel itself is exercised on
Colab through the notebook.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_band_table_covers_full_range_no_gaps() -> None:
    from render.volumetric_gas_giant import JOVIAN_BANDS

    sorted_bands = sorted(JOVIAN_BANDS, key=lambda b: b[0])
    assert sorted_bands[0][0] == -90, "bands must start at south pole"
    assert sorted_bands[-1][1] == 90, "bands must end at north pole"
    for a, b in zip(sorted_bands[:-1], sorted_bands[1:]):
        assert a[1] == b[0], f"gap between {a[2]} and {b[2]} at lat {a[1]}/{b[0]}"
    # Equatorial Zone must be the fastest eastward jet.
    by_name = {b[2]: b for b in JOVIAN_BANDS}
    assert by_name["EZ"][5] >= 80.0, "EZ jet must be at least 80 m/s prograde"
    print("ok: test_band_table_covers_full_range_no_gaps")


def test_fbm_is_finite_and_zero_centered() -> None:
    from render.volumetric_gas_giant import _fbm_2d

    n = _fbm_2d((96, 96), octaves=4, seed=3)
    assert n.shape == (96, 96)
    assert np.isfinite(n).all()
    assert abs(float(n.mean())) < 1e-4
    assert 0.5 < float(n.std()) < 2.0
    print("ok: test_fbm_is_finite_and_zero_centered")


def test_bake_returns_well_shaped_arrays() -> None:
    from render.volumetric_gas_giant import bake_textures

    H, W = 96, 192
    surface, layers, omega = bake_textures(height=H, width=W, seed=0)
    assert surface.shape == (H, W, 3) and surface.dtype == np.float32
    assert layers.shape == (H, W, 3) and layers.dtype == np.float32
    assert omega.shape == (H,) and omega.dtype == np.float32
    assert np.isfinite(surface).all()
    assert np.isfinite(layers).all()
    assert np.isfinite(omega).all()
    # Surface color should land in a reasonable scene-linear range.
    assert 0.0 <= float(surface.min()) <= 0.5
    assert 0.4 <= float(surface.max()) <= 1.5
    # Layer densities are non-negative.
    assert float(layers.min()) >= -1e-6
    print("ok: test_bake_returns_well_shaped_arrays")


def test_great_red_spot_appears_in_correct_band() -> None:
    """A targeted region around (lat=-22.5, lon=100) should be visibly
    redder than the surrounding SEB after baking."""
    from render.volumetric_gas_giant import bake_textures

    H, W = 180, 360
    surface, _, _ = bake_textures(height=H, width=W, seed=11, grs_lon=100.0)

    def latlon_to_idx(lat, lon):
        i = int(round((lat + 90.0) / 180.0 * (H - 1)))
        j = int(round((lon + 180.0) / 360.0 * (W - 1)))
        return i, j

    grs_i, grs_j = latlon_to_idx(-22.5, 100.0)
    grs_box = surface[max(0, grs_i - 3) : grs_i + 4, max(0, grs_j - 8) : grs_j + 9]
    # Neutral SEB sample 180 degrees away in longitude.
    seb_i, seb_j = latlon_to_idx(-22.5, -80.0)
    seb_box = surface[max(0, seb_i - 3) : seb_i + 4, max(0, seb_j - 8) : seb_j + 9]

    grs_redness = float(grs_box[..., 0].mean() - grs_box[..., 2].mean())
    seb_redness = float(seb_box[..., 0].mean() - seb_box[..., 2].mean())
    assert grs_redness > seb_redness + 0.05, (grs_redness, seb_redness)
    print("ok: test_great_red_spot_appears_in_correct_band")


def test_polar_cyclones_brighten_high_latitudes() -> None:
    from render.volumetric_gas_giant import bake_textures

    H, W = 180, 360
    surface, _, _ = bake_textures(height=H, width=W, seed=0)
    npr_row = int(round((83.0 + 90.0) / 180.0 * (H - 1)))
    spr_row = int(round((-83.0 + 90.0) / 180.0 * (H - 1)))
    # Bare NPR/SPR baseline color is dark brown-gray (R~0.42, sum ~1.0). Cyclones
    # overlay a brighter tan, so the row mean luma must exceed the bare baseline.
    base_npr_luma = 0.42 + 0.32 + 0.26
    base_spr_luma = 0.45 + 0.35 + 0.28
    npr_luma = float(surface[npr_row].sum(axis=-1).mean())
    spr_luma = float(surface[spr_row].sum(axis=-1).mean())
    assert npr_luma > base_npr_luma + 0.1, npr_luma
    assert spr_luma > base_spr_luma + 0.1, spr_luma
    print("ok: test_polar_cyclones_brighten_high_latitudes")


def test_equatorial_jet_drift_dominates() -> None:
    """The EZ row's per-second longitude drift should be the largest in
    magnitude among prograde bands, matching the equatorial super-jet."""
    from render.volumetric_gas_giant import bake_textures

    H, W = 180, 360
    _, _, omega = bake_textures(height=H, width=W, seed=0)
    ez_row = H // 2  # lat ~ 0
    ez_omega = abs(float(omega[ez_row]))
    poles = max(abs(float(omega[2])), abs(float(omega[-3])))
    assert ez_omega > poles, (ez_omega, poles)
    # The drift in degrees/sec is omega * 180/pi.
    drift_deg_per_sec = ez_omega * 180.0 / math.pi
    assert 1.0 < drift_deg_per_sec < 120.0, drift_deg_per_sec
    print("ok: test_equatorial_jet_drift_dominates")


def test_example_manifest_loads() -> None:
    from scene.manifest import load_manifest

    p = ROOT / "scene" / "examples" / "jupiter_juno_approach.json"
    g = load_manifest(p)
    assert len(g.shots) == 3
    assert all(s.renderer == "volumetric_gas_giant" for s in g.shots)
    assert g.shots[0].id == "perijove_approach"
    assert g.shots[2].id == "north_pole_cyclones"
    print("ok: test_example_manifest_loads")


def test_sun_dir_illuminates_camera_facing_hemisphere() -> None:
    """For shots whose camera looks toward +Z (camera at -Z), the
    sun_dir must have a non-positive Z component so the day side of
    the planet faces the camera. The opposite sign puts us behind
    Jupiter looking at the night side -- which is exactly the bug
    that produced the thin-crescent-only render.

    The polar shot is exempt (camera is above, looking down)."""
    from scene.manifest import load_manifest

    for manifest in ("jupiter_to_stargate.json", "jupiter_juno_approach.json"):
        g = load_manifest(ROOT / "scene" / "examples" / manifest)
        for shot in g.shots:
            if shot.renderer != "volumetric_gas_giant":
                continue
            cam = shot.camera
            sd = shot.params.get("sun_dir")
            if sd is None:
                continue
            # Pole shots are special-cased: camera is above the planet
            # so the lit-side check on Z is replaced by a Y check.
            if "polar" in (shot.motion_hint or "") or "pole" in shot.id:
                assert sd[1] > 0.3, (
                    f"{manifest} shot {shot.id}: polar shot needs sun overhead"
                )
                continue
            # General case: camera looks toward +Z, so sun_dir.z must be
            # negative for the visible hemisphere to be lit.
            cam_fwd_z = cam.target[2] - cam.position[2]
            if cam_fwd_z > 0:
                assert sd[2] < 0, (
                    f"{manifest} shot {shot.id}: sun_dir.z={sd[2]} would "
                    f"light the back of the planet"
                )
    print("ok: test_sun_dir_illuminates_camera_facing_hemisphere")


def main() -> int:
    test_band_table_covers_full_range_no_gaps()
    test_fbm_is_finite_and_zero_centered()
    test_bake_returns_well_shaped_arrays()
    test_great_red_spot_appears_in_correct_band()
    test_polar_cyclones_brighten_high_latitudes()
    test_equatorial_jet_drift_dominates()
    test_example_manifest_loads()
    test_sun_dir_illuminates_camera_facing_hemisphere()
    print("\nall volumetric gas-giant tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
