"""CPU-side tests for the saturn_class renderer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_band_table_covers_full_range_no_gaps() -> None:
    from render.saturn_class import SATURN_BANDS

    sorted_bands = sorted(SATURN_BANDS, key=lambda b: b[0])
    assert sorted_bands[0][0] == -90
    assert sorted_bands[-1][1] == 90
    for a, b in zip(sorted_bands[:-1], sorted_bands[1:]):
        assert a[1] == b[0], f"gap between {a[2]} and {b[2]}"
    print("ok: test_band_table_covers_full_range_no_gaps")


def test_saturn_equatorial_jet_is_fastest() -> None:
    """Saturn's EZ jet (~470 m/s) is dramatically faster than the
    polar bands -- much more extreme than Jupiter's."""
    from render.saturn_class import SATURN_BANDS

    by_name = {b[2]: b for b in SATURN_BANDS}
    ez_wind = by_name["EZ"][5]
    other_winds = [b[5] for b in SATURN_BANDS if b[2] != "EZ"]
    assert ez_wind >= 200.0, "Saturn EZ must be at least 200 m/s prograde"
    assert all(ez_wind > 2.0 * abs(w) for w in other_winds), (
        "Saturn EZ should be >2x faster than any other band"
    )
    print("ok: test_saturn_equatorial_jet_is_fastest")


def test_ring_segments_ordered_no_overlap() -> None:
    from render.saturn_class import RING_SEGMENTS

    rs = sorted(RING_SEGMENTS, key=lambda r: r[0])
    for r in rs:
        assert r[0] < r[1], f"ring {r[4]} inner >= outer"
    for a, b in zip(rs[:-1], rs[1:]):
        assert a[1] <= b[0] + 1e-3, f"overlap: {a[4]} -> {b[4]}"
    print("ok: test_ring_segments_ordered_no_overlap")


def test_ring_density_lookup() -> None:
    from render.saturn_class import _np_ring_density

    # Inside the B ring (densest).
    d_B, _ = _np_ring_density(1.75)
    # In the Cassini division (very sparse).
    d_cass, _ = _np_ring_density(1.99)
    # Outside all rings.
    d_out, _ = _np_ring_density(3.0)
    assert d_B > 0.7
    assert d_cass < 0.05
    assert d_out == 0.0
    print("ok: test_ring_density_lookup")


def test_saturn_band_color_picks_right_band() -> None:
    from render.saturn_class import _np_saturn_band_color

    eq = _np_saturn_band_color(0.0)
    pol_n = _np_saturn_band_color(85.0)
    assert eq[0] > pol_n[0]  # equatorial brighter
    print("ok: test_saturn_band_color_picks_right_band")


def test_hex_mask_six_fold_symmetric() -> None:
    """At the hex latitude, the mask should peak at six longitudes
    spaced 60 deg apart."""
    from render.saturn_class import _np_hex_mask

    hex_lat = 78.0
    samples = []
    for lon in range(0, 360, 5):
        samples.append((lon, _np_hex_mask(hex_lat, lon - 180, hex_lat, 0.45)))
    # Identify peaks (local maxima).
    peaks = []
    for i in range(1, len(samples) - 1):
        if samples[i][1] > samples[i - 1][1] and samples[i][1] > samples[i + 1][1]:
            peaks.append(samples[i][0])
    assert len(peaks) >= 5, f"expected ~6 hexagon peaks, got {peaks}"
    print("ok: test_hex_mask_six_fold_symmetric")


def test_hex_mask_falls_off_away_from_pole() -> None:
    from render.saturn_class import _np_hex_mask

    near_hex = _np_hex_mask(78.0, 0.0, 78.0, 0.45)
    equator = _np_hex_mask(0.0, 0.0, 78.0, 0.45)
    assert near_hex > 0.2
    assert equator == 0.0
    print("ok: test_hex_mask_falls_off_away_from_pole")


def test_saturn_manifest_loads_three_shots() -> None:
    from scene.manifest import load_manifest

    g = load_manifest(ROOT / "scene" / "examples" / "saturn_voyage.json")
    assert len(g.shots) == 3
    assert all(s.renderer == "saturn_class" for s in g.shots)
    assert [s.id for s in g.shots] == [
        "rings_oblique", "ring_skim", "north_hexagon",
    ]
    # The hex shot should have hex_enable on.
    hex_shot = g.shots[2]
    assert hex_shot.params.get("hex_enable") == 1
    # The early shots have hex disabled.
    for shot in g.shots[:2]:
        assert shot.params.get("hex_enable", 1) == 0
    print("ok: test_saturn_manifest_loads_three_shots")


def main() -> int:
    test_band_table_covers_full_range_no_gaps()
    test_saturn_equatorial_jet_is_fastest()
    test_ring_segments_ordered_no_overlap()
    test_ring_density_lookup()
    test_saturn_band_color_picks_right_band()
    test_hex_mask_six_fold_symmetric()
    test_hex_mask_falls_off_away_from_pole()
    test_saturn_manifest_loads_three_shots()
    print("\nall saturn_class tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
