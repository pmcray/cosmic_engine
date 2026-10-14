"""Turn a `World` into shot parameters for the render pipeline.

This is where Traveller's abstract codes become physical rendering
knobs: hydrographics picks the water level, atmosphere sets haze and
scale height, temperature places the snow line, the primary's spectral
class colours the sunlight. The mapping is deterministic — the same
world always yields the same shot — and everything it emits is plain
JSON, so an adapted world can be dumped into a manifest and rendered on
a machine that has never heard of worldmaker.

Renderer selection:

    terrestrial / asteroid  -> "terrain"           (surface, on the ground)
    gas_giant               -> "volumetric_gas_giant"
    ringed_giant            -> "saturn_class"
    hot_jupiter, mini_neptune, brown_dwarf
                            -> "exoplanet_atmosphere"
"""
from __future__ import annotations

import math
import random

from scene.manifest import Camera, PaletteRef, Shot

from .model import World

# Approximate scene-linear RGB of a blackbody at each spectral class,
# normalised so the brightest channel is 1.0 — the sunlight tint.
STAR_LIGHT_RGB = {
    "O": (0.62, 0.72, 1.00),
    "B": (0.72, 0.80, 1.00),
    "A": (0.88, 0.92, 1.00),
    "F": (1.00, 0.98, 0.96),
    "G": (1.00, 0.94, 0.84),
    "K": (1.00, 0.84, 0.66),
    "M": (1.00, 0.68, 0.46),
}


def star_light_rgb(world: World) -> tuple:
    return STAR_LIGHT_RGB.get(world.star.spectral_class.upper(),
                              STAR_LIGHT_RGB["G"])


def renderer_for(world: World) -> str:
    if world.body_type == "gas_giant":
        return "volumetric_gas_giant"
    if world.body_type == "ringed_giant":
        return "saturn_class"
    if world.body_type in ("hot_jupiter", "mini_neptune", "brown_dwarf"):
        return "exoplanet_atmosphere"
    return "terrain"


def palette_for(world: World) -> str:
    """Pick a look that suits the world rather than the shot's neighbours;
    the continuity engine still gets the final say."""
    if world.is_giant:
        return "trumbull_2001"
    if world.uwp.is_vacuum:
        return "trumbull_2001"
    if world.star.spectral_class.upper() in ("O", "B", "A"):
        return "jwst_nircam"
    if world.uwp.atmosphere in (10, 11, 12, 15):   # exotic chemistries
        return "hubble_sii_ha_oiii"
    return "trumbull_2001"


def _sun_direction(rng: random.Random) -> list:
    """A low-ish sun: long shadows read better than noon flatness."""
    az = rng.uniform(0.0, 2.0 * math.pi)
    elev = rng.uniform(0.25, 0.75)          # radians above the horizon
    return [math.cos(az) * math.cos(elev), math.sin(elev),
            -abs(math.sin(az) * math.cos(elev))]


def terrain_params(world: World) -> dict:
    """Physical mapping for a solid surface.

    The terrain engine works in a normalised height field roughly in
    [-1, 1], so `water_level` is the height below which the surface is
    flooded and `snow_line` the height above which it freezes.
    """
    rng = random.Random(world.seed ^ 0x7E44A1)
    uwp = world.uwp

    # Water level from hydrographics: 0 tenths -> no ocean at all,
    # 10 tenths -> everything but the highest peaks is drowned.
    wet = uwp.water_fraction
    has_water = world.has_liquid_water
    water_level = -1.5 + 2.4 * wet if has_water else -10.0

    # Relief scales inversely with gravity — low-gravity worlds hold up
    # far taller mountains (Olympus Mons on 0.38 g).
    g = max(uwp.gravity_g, 0.05)
    amplitude = float(min(3.2, 1.25 / (g ** 0.55)))

    # Thicker air erodes: high-pressure worlds get smoother, rounder
    # landforms and fewer sharp octaves.
    pressure = uwp.pressure_atm
    octaves = 8 if pressure < 0.5 else 7 if pressure < 1.6 else 6

    # Haze comes from the atmosphere; vacuum worlds have knife-edge
    # horizons.
    haze = 0.0 if uwp.is_vacuum else min(1.4, 0.22 + 0.55 * pressure)

    # Snow line: at freezing everything is white, in the tropics nothing
    # is. Expressed as a height in the same normalised units.
    t = world.mean_temp_k
    if not has_water and not world.is_frozen:
        snow_line = 5.0                       # unreachable: no snow
    elif t < 240.0:
        snow_line = -2.0                      # unreachable: all snow
    else:
        snow_line = float(min(2.5, (t - 240.0) / 45.0 - 0.2))

    return {
        "terrain_amplitude": amplitude,
        "terrain_freq": rng.uniform(0.035, 0.085),
        "octaves": octaves,
        "snow_line": snow_line,
        "water_level": water_level,
        "water_enable": 1 if has_water else 0,
        "sun_dir": _sun_direction(rng),
        "haze_strength": haze,
        "max_dist": rng.uniform(70.0, 115.0),
        "march_steps": rng.choice([130, 150]),
        "samples": 1,
        "seed": world.seed & 0xFFFFF,
    }


def giant_params(world: World) -> dict:
    """Bands and storms for a gas giant. Colder giants are paler and
    more banded (Saturn); hotter ones are darker and more turbulent."""
    rng = random.Random(world.seed ^ 0x5C1A77)
    t = world.mean_temp_k
    warm = max(0.0, min(1.0, (t - 90.0) / 200.0))
    params = {
        "atm_thickness": rng.uniform(0.038, 0.068),
        "march_steps": rng.choice([24, 28, 32]),
        "samples": 2,
        "sun_dir": _sun_direction(rng),
        "grs_lon": rng.uniform(-140.0, 140.0),
        "seed": world.seed & 0xFFFFF,
        "push_in_factor": rng.uniform(1.15, 1.5),
    }
    if world.body_type == "ringed_giant":
        params.update({
            "rings_enable": 1,
            "ring_brightness": rng.uniform(0.8, 1.15),
            "ring_shadows_enable": 1,
            "hex_enable": 1 if rng.random() < 0.5 else 0,
            "hex_lat": 78.0,
            "hex_amp": rng.uniform(0.35, 0.55),
        })
        params.pop("grs_lon", None)
    else:
        # Warmer giants get deeper, more contrasted belts.
        params["atm_thickness"] *= 1.0 + 0.25 * warm
    return params


def exoplanet_params(world: World) -> dict:
    rng = random.Random(world.seed ^ 0x3E0B1A)
    topology = {
        "hot_jupiter": "hot_jupiter",
        "mini_neptune": "mini_neptune",
        "brown_dwarf": "brown_dwarf",
    }.get(world.body_type, "hot_jupiter")
    return {
        "topology": topology,
        "atm_thickness": rng.uniform(0.06, 0.13),
        "march_steps": rng.choice([28, 32]),
        "samples": 2,
        "sun_dir": _sun_direction(rng),
        "seed": world.seed & 0xFFFFF,
        "push_in_factor": rng.uniform(1.1, 1.4),
    }


def world_provenance(world: World) -> dict:
    """The record carried in a shot's params so a rendered frame can be
    traced back to the world that produced it."""
    return {
        "name": world.name,
        "uwp": str(world.uwp),
        "sector": world.sector,
        "hex": world.hex_label,
        "subsector": world.subsector,
        "body_type": world.body_type,
        "mean_temp_k": round(world.mean_temp_k, 1),
        "star": str(world.star),
        "provider": world.provider,
    }


def params_for(world: World, provenance: bool = True) -> dict:
    renderer = renderer_for(world)
    if renderer == "terrain":
        params = terrain_params(world)
    elif renderer in ("volumetric_gas_giant", "saturn_class"):
        params = giant_params(world)
    else:
        params = exoplanet_params(world)
    if provenance:
        params["world"] = world_provenance(world)
    return params


def camera_for(world: World, rng: random.Random | None = None) -> Camera:
    """A surface world gets a low fly-through; a giant gets a slow
    approach from outside."""
    rng = rng or random.Random(world.seed ^ 0x0CA3E1)
    if renderer_for(world) == "terrain":
        return Camera(
            position=(rng.uniform(-5.0, -1.5), rng.uniform(0.9, 1.4),
                      rng.uniform(-6.0, -1.5)),
            target=(rng.uniform(2.0, 5.0), rng.uniform(0.4, 0.8),
                    rng.uniform(3.0, 5.0)),
            up=(0.0, 1.0, 0.0),
            fov_deg=rng.uniform(48.0, 58.0),
            path_to=Camera(
                position=(rng.uniform(-1.0, 2.0), rng.uniform(1.0, 1.4),
                          rng.uniform(-3.5, -0.5)),
                target=(rng.uniform(3.0, 5.0), rng.uniform(0.4, 0.8),
                        rng.uniform(4.0, 5.5)),
                up=(0.0, 1.0, 0.0),
                fov_deg=rng.uniform(45.0, 55.0),
            ),
        )
    return Camera(
        position=(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15),
                  rng.uniform(-4.4, -3.2)),
        target=(0.0, rng.uniform(-0.1, 0.1), 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=rng.uniform(30.0, 38.0),
        path_to=Camera(
            position=(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15),
                      rng.uniform(-2.6, -1.8)),
            target=(0.0, rng.uniform(-0.12, 0.12), 0.0),
            up=(0.0, 1.0, 0.0),
            fov_deg=rng.uniform(28.0, 33.0),
        ),
    )


def motion_hint_for(world: World) -> str:
    return "fly_through" if renderer_for(world) == "terrain" else "push_in"


def shot_for_world(world: World,
                   shot_id: str | None = None,
                   duration_frames: int = 120,
                   fps: int = 24,
                   resolution: tuple = (1920, 1080),
                   palette: str | None = None) -> Shot:
    """Build a complete, renderable Shot for one world."""
    renderer = renderer_for(world)
    params = params_for(world)
    safe = "".join(c if c.isalnum() else "_" for c in world.name).strip("_")
    return Shot(
        id=shot_id or f"world_{world.hex_label}_{safe or 'unnamed'}",
        renderer=renderer,
        params=params,
        camera=camera_for(world),
        palette=PaletteRef(name=palette or palette_for(world)),
        duration_frames=duration_frames,
        fps=fps,
        resolution=resolution,
        seed=world.seed & 0xFFFFF,
        motion_hint=motion_hint_for(world),
    )
