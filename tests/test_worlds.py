"""CPU-side tests for the worlds package.

Covers UWP parsing and its physical interpretation, the procedural
provider's Traveller legality, sector determinism, and the adapter's
mapping from world codes to renderer parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- UWP --------------------------------------------------------------

def test_uwp_parses_and_round_trips() -> None:
    from worlds.uwp import UWP

    u = UWP.parse("A867949-C")
    assert u.starport == "A"
    assert (u.size, u.atmosphere, u.hydrographics) == (8, 6, 7)
    assert (u.population, u.government, u.law_level) == (9, 4, 9)
    assert u.tech_level == 12
    assert str(u) == "A867949-C"
    # Without the dash, and lower case.
    assert UWP.parse("a867949c") == u
    print("ok: test_uwp_parses_and_round_trips")


def test_uwp_rejects_malformed() -> None:
    from worlds.uwp import UWP, UWPError

    for bad in ("", "nonsense", "Z867949-C", "A86794-C", "A867949", "AK67949-C", "AB67949-C"):
        try:
            UWP.parse(bad)
        except UWPError:
            continue
        raise AssertionError(f"expected UWPError for {bad!r}")
    print("ok: test_uwp_rejects_malformed")


def test_uwp_pop_zero_forces_canonical() -> None:
    from worlds.uwp import UWP

    # Any UWP with Pop == 0 must force Starport X, Gov 0, Law 0, TL 0
    u = UWP.parse("A867049-C")
    assert u.population == 0
    assert u.starport == "X"
    assert u.government == 0
    assert u.law_level == 0
    assert u.tech_level == 0
    assert str(u) == "X867000-0"

    u_direct = UWP(starport="B", size=7, atmosphere=5, hydrographics=4,
                   population=0, government=3, law_level=2, tech_level=8)
    assert u_direct.starport == "X"
    assert u_direct.government == 0
    assert u_direct.law_level == 0
    assert u_direct.tech_level == 0
    assert str(u_direct) == "X754000-0"
    print("ok: test_uwp_pop_zero_forces_canonical")


def test_ehex_skips_i_and_o() -> None:
    from worlds.uwp import UWPError, ehex_to_int, int_to_ehex

    assert ehex_to_int("9") == 9
    assert ehex_to_int("A") == 10
    # H is 17, then J (I is skipped).
    assert int_to_ehex(17) == "H"
    assert int_to_ehex(18) == "J"
    assert "I" not in [int_to_ehex(i) for i in range(34)]
    assert "O" not in [int_to_ehex(i) for i in range(34)]
    for bad in ("I", "O", "?"):
        try:
            ehex_to_int(bad)
        except UWPError:
            continue
        raise AssertionError(f"expected UWPError for {bad!r}")
    print("ok: test_ehex_skips_i_and_o")


def test_uwp_physical_interpretation() -> None:
    from worlds.uwp import UWP

    earth = UWP.parse("A867949-C")          # size 8, atm 6, hyd 7
    assert 12000 < earth.diameter_km < 13500
    assert abs(earth.gravity_g - 1.0) < 0.01
    assert earth.breathable
    assert abs(earth.pressure_atm - 1.0) < 0.01
    assert abs(earth.water_fraction - 0.7) < 1e-9
    assert not earth.is_vacuum

    vac = UWP.parse("X000000-0")
    assert vac.is_vacuum and not vac.breathable
    assert vac.pressure_atm == 0.0
    assert vac.water_fraction == 0.0
    assert not vac.is_inhabited

    water_world = UWP.parse("A86A949-C")    # hydrographics A = 10
    assert water_world.water_fraction == 1.0
    print("ok: test_uwp_physical_interpretation")


# ---- Model ------------------------------------------------------------

def test_hex_label_and_subsector_mapping() -> None:
    from worlds.model import World

    assert World(name="x", hex_x=7, hex_y=3).hex_label == "0703"
    assert World(name="x", hex_x=17, hex_y=21).hex_label == "1721"
    # Subsectors: 4 columns x 4 rows, lettered A-P across then down.
    assert World(name="x", hex_x=1, hex_y=1).subsector == "A"
    assert World(name="x", hex_x=9, hex_y=1).subsector == "B"
    assert World(name="x", hex_x=32, hex_y=1).subsector == "D"
    assert World(name="x", hex_x=1, hex_y=11).subsector == "E"
    assert World(name="x", hex_x=32, hex_y=40).subsector == "P"
    print("ok: test_hex_label_and_subsector_mapping")


def test_liquid_water_requires_temperature_band() -> None:
    from worlds.model import World
    from worlds.uwp import UWP

    wet = UWP.parse("A867949-C")
    assert World(name="w", uwp=wet, mean_temp_k=288.0).has_liquid_water
    assert not World(name="w", uwp=wet, mean_temp_k=120.0).has_liquid_water
    assert not World(name="w", uwp=wet, mean_temp_k=800.0).has_liquid_water
    assert World(name="w", uwp=wet, mean_temp_k=120.0).is_frozen
    # A dry world has no liquid water at any temperature.
    dry = UWP.parse("A860949-C")
    assert not World(name="d", uwp=dry, mean_temp_k=288.0).has_liquid_water
    print("ok: test_liquid_water_requires_temperature_band")


def test_star_temperature_ordering() -> None:
    from worlds.model import Star

    temps = [Star(spectral_class=c).temperature_k
             for c in ("O", "B", "A", "F", "G", "K", "M")]
    assert temps == sorted(temps, reverse=True), temps
    # The Sun is a G2 V, ~5700 K.
    assert 5300 < Star("G", 2).temperature_k < 5900
    # Giants are far more luminous than main-sequence stars of the class.
    assert Star("K", 0, "III").luminosity_sol > Star("K", 0, "V").luminosity_sol
    print("ok: test_star_temperature_ordering")


# ---- Procedural provider ----------------------------------------------

def test_procedural_provider_is_deterministic() -> None:
    from worlds.providers import ProceduralProvider

    p = ProceduralProvider()
    a = p.generate(seed=42, count=5)
    b = p.generate(seed=42, count=5)
    assert [w.name for w in a] == [w.name for w in b]
    assert [str(w.uwp) for w in a] == [str(w.uwp) for w in b]
    # Asking for more worlds does not disturb the earlier ones.
    more = p.generate(seed=42, count=12)
    assert [w.name for w in more[:5]] == [w.name for w in a]
    # A different seed diverges.
    other = p.generate(seed=43, count=5)
    assert [w.name for w in other] != [w.name for w in a]
    print("ok: test_procedural_provider_is_deterministic")


def test_rolled_uwps_are_traveller_legal() -> None:
    import random

    from worlds.providers import roll_uwp
    from worlds.uwp import UWP

    for seed in range(300):
        u = roll_uwp(random.Random(seed))
        assert u.starport in "ABCDEX"
        assert 0 <= u.size <= 10
        assert 0 <= u.atmosphere <= 15
        assert 0 <= u.hydrographics <= 10
        assert 0 <= u.population <= 10
        # Dependency chain: a size-0 world has no atmosphere and no sea.
        if u.size == 0:
            assert u.atmosphere == 0 and u.hydrographics == 0
        # An uninhabited world has no government, law or tech.
        if u.population == 0:
            assert u.government == 0 and u.law_level == 0 and u.tech_level == 0
        # Round-trips through its own string form.
        assert UWP.parse(str(u)) == u
    print("ok: test_rolled_uwps_are_traveller_legal")


def test_giants_get_giant_profiles() -> None:
    from worlds.providers import ProceduralProvider

    worlds = ProceduralProvider().generate(seed=7, count=400)
    giants = [w for w in worlds if w.is_giant]
    assert giants, "no giants generated in 400 worlds"
    for g in giants:
        # Canonical rule: mainworld UWP strictly describes terrestrial mainworld / satellite,
        # never a gas giant with Size K. Size must be 0-A (0-10).
        assert 0 <= g.uwp.size <= 10, g.summary()
        assert g.has_gas_giant is True, g.summary()
        assert g.gas_giant_count >= 1, g.summary()
        if g.uwp.population == 0:
            assert g.uwp.starport == "X"
            assert g.uwp.government == 0 and g.uwp.law_level == 0 and g.uwp.tech_level == 0
    print("ok: test_giants_get_giant_profiles")


def test_canonical_trade_codes() -> None:
    from worlds.uwp import UWP, compute_trade_codes

    # Ba: Barren (Pop 0) -> forces X...000-0
    u_ba = UWP(size=5, atmosphere=5, hydrographics=5, population=0)
    tc_ba = compute_trade_codes(u_ba)
    assert "Ba" in tc_ba
    assert "Ag" not in tc_ba and "Ni" not in tc_ba and "Ri" not in tc_ba

    # As: Asteroid (Size 0, Atm 0, Hyd 0)
    u_as = UWP(starport="C", size=0, atmosphere=0, hydrographics=0, population=5, government=4, law_level=3, tech_level=8)
    assert "As" in compute_trade_codes(u_as)
    assert "Va" in compute_trade_codes(u_as)

    # Va: Vacuum (Atm 0)
    u_va = UWP(starport="C", size=5, atmosphere=0, hydrographics=0, population=5, government=4, law_level=3, tech_level=8)
    assert "Va" in compute_trade_codes(u_va)

    # De: Desert (Atm >= 2, Hyd 0)
    u_de = UWP(starport="C", size=5, atmosphere=5, hydrographics=0, population=5, government=4, law_level=3, tech_level=8)
    assert "De" in compute_trade_codes(u_de)

    # Ic: Ice-Capped (Atm 0-1, Hyd 1+)
    u_ic = UWP(starport="C", size=5, atmosphere=1, hydrographics=5, population=5, government=4, law_level=3, tech_level=8)
    assert "Ic" in compute_trade_codes(u_ic)

    # Fl: Fluid Oceans (Atm 10+, Hyd 1+)
    u_fl = UWP(starport="C", size=5, atmosphere=10, hydrographics=5, population=5, government=4, law_level=3, tech_level=8)
    assert "Fl" in compute_trade_codes(u_fl)

    # Hi: High Population (Pop 9+)
    u_hi = UWP(starport="A", size=8, atmosphere=6, hydrographics=7, population=9, government=5, law_level=4, tech_level=12)
    assert "Hi" in compute_trade_codes(u_hi)

    # In: Industrial (Atm 0, 1, 2, 4, 7, 9; Pop 9+)
    u_in = UWP(starport="A", size=8, atmosphere=2, hydrographics=0, population=9, government=5, law_level=4, tech_level=12)
    assert "In" in compute_trade_codes(u_in)

    # Ag: Agricultural (Atm 4-9, Hyd 4-8, Pop 5-7)
    u_ag = UWP(starport="C", size=6, atmosphere=6, hydrographics=6, population=6, government=4, law_level=3, tech_level=8)
    assert "Ag" in compute_trade_codes(u_ag)

    # Na: Non-Agricultural (Atm 0-3, Hyd 0-3, Pop 6+)
    u_na = UWP(starport="C", size=6, atmosphere=2, hydrographics=2, population=6, government=4, law_level=3, tech_level=8)
    assert "Na" in compute_trade_codes(u_na)

    # Ni: Non-Industrial (Pop 4-6)
    u_ni = UWP(starport="C", size=6, atmosphere=3, hydrographics=5, population=5, government=4, law_level=3, tech_level=8)
    assert "Ni" in compute_trade_codes(u_ni)

    # Po: Poor (Atm 2-5, Hyd 0-3)
    u_po = UWP(starport="C", size=6, atmosphere=3, hydrographics=2, population=5, government=4, law_level=3, tech_level=8)
    assert "Po" in compute_trade_codes(u_po)

    # Ri: Rich (Atm 6 or 8, Pop 6-8, Gov 4-9)
    u_ri = UWP(starport="A", size=8, atmosphere=6, hydrographics=5, population=7, government=5, law_level=3, tech_level=10)
    assert "Ri" in compute_trade_codes(u_ri)
    print("ok: test_canonical_trade_codes")


def test_gas_giant_and_satellite_shots() -> None:
    from worlds.adapter import params_for, renderer_for, shot_for_world
    from worlds.model import World
    from worlds.uwp import UWP

    canonical_uwp = UWP.parse("A867949-C")
    # System with gas giants where body_type targets the giant
    w_giant = World(
        name="Calandron",
        uwp=canonical_uwp,
        body_type="gas_giant",
        has_gas_giant=True,
        gas_giant_count=2,
    )
    assert renderer_for(w_giant) == "volumetric_gas_giant"
    shot = shot_for_world(w_giant)
    assert shot.renderer == "volumetric_gas_giant"
    assert shot.params["world"]["uwp"] == "A867949-C"
    assert shot.params["world"]["has_gas_giant"] is True
    assert shot.params["world"]["gas_giant_count"] == 2

    # Ringed giant
    w_ringed = World(
        name="Saturnia",
        uwp=canonical_uwp,
        body_type="ringed_giant",
        has_gas_giant=True,
        gas_giant_count=1,
    )
    assert renderer_for(w_ringed) == "saturn_class"
    shot_r = shot_for_world(w_ringed)
    assert shot_r.renderer == "saturn_class"
    assert shot_r.params["rings_enable"] == 1

    # Terrestrial mainworld orbiting a gas giant as a moon
    w_moon = World(
        name="Europa Prime",
        uwp=canonical_uwp,
        body_type="terrestrial",
        has_gas_giant=True,
        gas_giant_count=3,
        parent_body="Jupiter Major",
        parent_body_type="gas_giant",
    )
    # Default is surface terrain
    assert renderer_for(w_moon) == "terrain"
    # When targeting the system's giant or requesting flyby:
    assert renderer_for(w_moon, target="gas_giant") == "volumetric_gas_giant"
    assert renderer_for(w_moon, target="satellite_flyby") == "volumetric_gas_giant"
    flyby_shot = shot_for_world(w_moon, target="satellite_flyby")
    assert flyby_shot.renderer == "volumetric_gas_giant"
    assert flyby_shot.params["world"]["uwp"] == "A867949-C"
    assert flyby_shot.params["world"]["parent_body"] == "Jupiter Major"
    print("ok: test_gas_giant_and_satellite_shots")


def test_worldmaker_provider_clean_mapping() -> None:
    from worlds.providers import WorldmakerProvider

    wm = WorldmakerProvider()
    assert wm.available(), "worldmaker should be detected as available"
    worlds = wm.generate(seed=1977, count=5)
    assert len(worlds) == 5
    for w in worlds:
        assert 0 <= w.uwp.size <= 10, f"size out of bounds: {w.uwp}"
        if w.uwp.population == 0:
            assert w.uwp.starport == "X"
            assert w.uwp.government == 0 and w.uwp.law_level == 0 and w.uwp.tech_level == 0
        assert w.provider == "worldmaker"
        assert w.star.spectral_class in ("O", "B", "A", "F", "G", "K", "M")
    print("ok: test_worldmaker_provider_clean_mapping")


def test_get_provider_falls_back_to_procedural() -> None:
    from worlds.providers import (ENV_WORLDMAKER, ProceduralProvider,
                                  ProviderUnavailable, get_provider)

    old = os.environ.pop(ENV_WORLDMAKER, None)
    try:
        # With no worldmaker checkout reachable, auto-selection must
        # still return something that works.
        p = get_provider()
        assert p.available()
        assert p.generate(1, 1)
        assert isinstance(get_provider("procedural"), ProceduralProvider)
        # Explicitly demanding worldmaker is an error, not a silent
        # substitution.
        try:
            get_provider("worldmaker")
        except ProviderUnavailable:
            pass
        else:
            # If a checkout genuinely is importable here, that is fine.
            pass
    finally:
        if old is not None:
            os.environ[ENV_WORLDMAKER] = old
    print("ok: test_get_provider_falls_back_to_procedural")


def test_provider_import_does_not_leak_sys_path() -> None:
    from worlds.providers import WorldmakerProvider

    before = list(sys.path)
    WorldmakerProvider(path="/nonexistent/worldmaker").available()
    assert sys.path == before, "provider left its path on sys.path"
    print("ok: test_provider_import_does_not_leak_sys_path")


def test_provider_status_reports_all_four() -> None:
    from worlds.providers import provider_status

    st = provider_status()
    assert set(st) == {"procedural", "worldmaker", "weorold", "erith"}
    assert st["procedural"] is True
    print("ok: test_provider_status_reports_all_four")


# ---- Sector -----------------------------------------------------------

def test_sector_is_deterministic_and_populated() -> None:
    from worlds.sector import generate_sector

    a = generate_sector(seed=1977)
    b = generate_sector(seed=1977)
    assert len(a) == len(b)
    assert [w.name for w in a.worlds] == [w.name for w in b.worlds]
    assert [w.hex_label for w in a.worlds] == [w.hex_label for w in b.worlds]
    # Roughly a third of 32x40 hexes, with generous slack.
    assert 250 < len(a) < 600, len(a)
    # Hexes are unique and in range.
    labels = [w.hex_label for w in a.worlds]
    assert len(set(labels)) == len(labels)
    for w in a.worlds:
        assert 1 <= w.hex_x <= 32 and 1 <= w.hex_y <= 40
        assert w.sector == "Foreven"
    print("ok: test_sector_is_deterministic_and_populated")


def test_world_identity_depends_only_on_hex_and_seed() -> None:
    """A world's identity must not shift when the sector's density or
    other hexes change — 'Foreven 1721' is a fixed address."""
    from worlds.sector import generate_sector

    full = generate_sector(seed=1977, density=1.0)
    sparse = generate_sector(seed=1977, density=0.5)
    shared = {w.hex_label for w in full.worlds} & {w.hex_label
                                                  for w in sparse.worlds}
    assert shared, "no overlapping hexes to compare"
    for label in list(shared)[:40]:
        assert full.by_hex(label).name == sparse.by_hex(label).name
        assert str(full.by_hex(label).uwp) == str(sparse.by_hex(label).uwp)
    print("ok: test_world_identity_depends_only_on_hex_and_seed")


def test_sector_lookups_and_filters() -> None:
    from worlds.sector import generate_sector

    s = generate_sector(seed=1977)
    w = s.worlds[0]
    assert s.by_hex(w.hex_label) is w
    try:
        s.by_hex("9999")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for an empty hex")

    all_ids = {id(x) for x in s.worlds}
    assert {id(x) for x in s.terrestrials()} <= all_ids
    assert all(x.body_type == "terrestrial" for x in s.terrestrials())
    assert all(x.is_giant for x in s.giants())
    for h in s.habitable():
        assert h.uwp.breathable and h.has_liquid_water
    # Subsector partition covers everything exactly once.
    total = sum(len(s.in_subsector(c)) for c in "ABCDEFGHIJKLMNOP")
    assert total == len(s)
    print("ok: test_sector_lookups_and_filters")


def test_pick_worlds_spreads_across_subsectors() -> None:
    from worlds.sector import generate_sector, pick_worlds

    s = generate_sector(seed=1977)
    picked = pick_worlds(s, 12, seed=5)
    assert len(picked) == 12
    assert len({w.hex_label for w in picked}) == 12
    # A round-robin across subsectors should touch several of them.
    assert len({w.subsector for w in picked}) >= 6
    # Deterministic.
    assert [w.hex_label for w in pick_worlds(s, 12, seed=5)] == \
           [w.hex_label for w in picked]
    # Kind filtering.
    giants = pick_worlds(s, 5, seed=1, kinds=("gas_giant", "ringed_giant"))
    assert all(w.body_type in ("gas_giant", "ringed_giant") for w in giants)
    print("ok: test_pick_worlds_spreads_across_subsectors")


# ---- Adapter ----------------------------------------------------------

def _world(uwp_str: str, temp: float = 288.0, body: str = "terrestrial"):
    from worlds.model import World
    from worlds.uwp import UWP
    return World(name="Test", uwp=UWP.parse(uwp_str), body_type=body,
                 mean_temp_k=temp, seed=12345)


def test_renderer_selection_by_body_type() -> None:
    from worlds.adapter import renderer_for

    assert renderer_for(_world("A867949-C")) == "terrain"
    assert renderer_for(_world("X000000-0", body="asteroid")) == "terrain"
    assert renderer_for(_world("X000000-0", body="gas_giant")) \
        == "volumetric_gas_giant"
    assert renderer_for(_world("X000000-0", body="ringed_giant")) \
        == "saturn_class"
    for b in ("hot_jupiter", "mini_neptune", "brown_dwarf"):
        assert renderer_for(_world("X000000-0", body=b)) \
            == "exoplanet_atmosphere"
    print("ok: test_renderer_selection_by_body_type")


def test_hydrographics_drives_water_level() -> None:
    from worlds.adapter import terrain_params

    dry = terrain_params(_world("A860949-C"))       # hydrographics 0
    mid = terrain_params(_world("A865949-C"))       # 5
    wet = terrain_params(_world("A86A949-C"))       # 10
    assert dry["water_enable"] == 0
    assert mid["water_enable"] == 1 and wet["water_enable"] == 1
    # More water means the sea rises.
    assert wet["water_level"] > mid["water_level"]
    # A frozen world has no liquid sea.
    frozen = terrain_params(_world("A868949-C", temp=100.0))
    assert frozen["water_enable"] == 0
    print("ok: test_hydrographics_drives_water_level")


def test_atmosphere_drives_haze_and_erosion() -> None:
    from worlds.adapter import terrain_params

    vacuum = terrain_params(_world("A807949-C"))     # atmosphere 0
    standard = terrain_params(_world("A867949-C"))   # 6
    dense = terrain_params(_world("A8D7949-C"))      # atmosphere D = dense, high
    assert vacuum["haze_strength"] == 0.0
    assert standard["haze_strength"] > 0.0
    assert dense["haze_strength"] > standard["haze_strength"]
    # Thicker air erodes: fewer sharp octaves.
    assert dense["octaves"] <= standard["octaves"] <= vacuum["octaves"]
    print("ok: test_atmosphere_drives_haze_and_erosion")


def test_gravity_drives_relief() -> None:
    from worlds.adapter import terrain_params

    # Low gravity holds up taller mountains than high gravity.
    low = terrain_params(_world("A267949-C"))    # size 2 -> 0.15 g
    high = terrain_params(_world("A967949-C"))   # size 9 -> 1.25 g
    assert low["terrain_amplitude"] > high["terrain_amplitude"]
    print("ok: test_gravity_drives_relief")


def test_temperature_drives_snow_line() -> None:
    from worlds.adapter import terrain_params

    cold = terrain_params(_world("A867949-C", temp=250.0))
    temperate = terrain_params(_world("A867949-C", temp=288.0))
    hot = terrain_params(_world("A867949-C", temp=340.0))
    # Warmer worlds push the snow line higher up the mountains.
    assert cold["snow_line"] < temperate["snow_line"] < hot["snow_line"]
    print("ok: test_temperature_drives_snow_line")


def test_params_are_json_safe_and_deterministic() -> None:
    import json

    from worlds.adapter import params_for
    from worlds.sector import generate_sector

    sector = generate_sector(seed=1977)
    for w in sector.worlds[:40]:
        p = params_for(w)
        json.dumps(p)                       # must serialise into a manifest
        assert params_for(w) == p           # same world, same params
        assert p["world"]["uwp"] == str(w.uwp)
        assert p["world"]["hex"] == w.hex_label
    print("ok: test_params_are_json_safe_and_deterministic")


def test_shot_for_world_round_trips_through_a_manifest() -> None:
    import tempfile

    from scene.manifest import ShotGraph, dump_manifest, load_manifest
    from worlds.adapter import shot_for_world
    from worlds.sector import generate_sector

    sector = generate_sector(seed=1977)
    shots = [shot_for_world(w) for w in sector.worlds[:6]]
    graph = ShotGraph(shots=shots, title="test")
    graph.validate()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.json"
        dump_manifest(graph, path)
        back = load_manifest(path)
    assert [s.id for s in back.shots] == [s.id for s in shots]
    assert [s.renderer for s in back.shots] == [s.renderer for s in shots]
    assert back.shots[0].params["world"] == shots[0].params["world"]
    print("ok: test_shot_for_world_round_trips_through_a_manifest")


def test_adapted_shots_name_registered_renderers() -> None:
    import render.adapters  # noqa: F401 — registration side effect
    from render.core import list_renderers
    from worlds.adapter import renderer_for
    from worlds.sector import generate_sector

    known = set(list_renderers())
    for w in generate_sector(seed=1977).worlds:
        assert renderer_for(w) in known, (w.summary(), renderer_for(w))
    print("ok: test_adapted_shots_name_registered_renderers")


def test_star_light_colour_by_class() -> None:
    from worlds.adapter import star_light_rgb
    from worlds.model import Star, World

    hot = star_light_rgb(World(name="a", star=Star("O")))
    cool = star_light_rgb(World(name="b", star=Star("M")))
    # Hot stars are blue-weighted, cool ones red-weighted.
    assert hot[2] > hot[0]
    assert cool[0] > cool[2]
    print("ok: test_star_light_colour_by_class")


# ---- Composer wiring --------------------------------------------------

def test_composer_places_traveller_worlds() -> None:
    from studio.voyage_composer import compose_voyage

    found = []
    for seed in range(10):
        g = compose_voyage(target_seconds=600.0, seed=seed)
        found += [s for s in g.shots if "world" in s.params]
    assert found, "no world-derived shots across ten voyages"
    for s in found:
        assert s.params["world"]["sector"] == "Foreven"
        assert s.params["world"]["uwp"]
    print("ok: test_composer_places_traveller_worlds")


# ---- CLI --------------------------------------------------------------

def test_cli_manifest_writes_renderable_graph() -> None:
    import tempfile

    from scene.manifest import load_manifest
    from worlds.cli import main

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tour.json"
        rc = main(["--seed", "1977", "manifest", "--count", "4",
                   "--resolution", "640x360", "--out", str(out)])
        assert rc == 0
        g = load_manifest(out)
        assert len(g.shots) == 4
        assert len(g.transitions) == 3
        assert all(s.resolution == (640, 360) for s in g.shots)
    print("ok: test_cli_manifest_writes_renderable_graph")


def test_cli_status_and_world_run() -> None:
    from worlds.cli import main
    from worlds.sector import generate_sector

    assert main(["status"]) == 0
    hex_label = generate_sector(seed=1977).worlds[0].hex_label
    assert main(["--seed", "1977", "world", "--hex", hex_label]) == 0
    print("ok: test_cli_status_and_world_run")


def main_runner() -> int:
    test_uwp_parses_and_round_trips()
    test_uwp_rejects_malformed()
    test_uwp_pop_zero_forces_canonical()
    test_ehex_skips_i_and_o()
    test_uwp_physical_interpretation()
    test_hex_label_and_subsector_mapping()
    test_liquid_water_requires_temperature_band()
    test_star_temperature_ordering()
    test_procedural_provider_is_deterministic()
    test_rolled_uwps_are_traveller_legal()
    test_canonical_trade_codes()
    test_giants_get_giant_profiles()
    test_gas_giant_and_satellite_shots()
    test_worldmaker_provider_clean_mapping()
    test_get_provider_falls_back_to_procedural()
    test_provider_import_does_not_leak_sys_path()
    test_provider_status_reports_all_four()
    test_sector_is_deterministic_and_populated()
    test_world_identity_depends_only_on_hex_and_seed()
    test_sector_lookups_and_filters()
    test_pick_worlds_spreads_across_subsectors()
    test_renderer_selection_by_body_type()
    test_hydrographics_drives_water_level()
    test_atmosphere_drives_haze_and_erosion()
    test_gravity_drives_relief()
    test_temperature_drives_snow_line()
    test_params_are_json_safe_and_deterministic()
    test_shot_for_world_round_trips_through_a_manifest()
    test_adapted_shots_name_registered_renderers()
    test_star_light_colour_by_class()
    test_composer_places_traveller_worlds()
    test_cli_manifest_writes_renderable_graph()
    test_cli_status_and_world_run()
    print("\nall worlds tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_runner())
