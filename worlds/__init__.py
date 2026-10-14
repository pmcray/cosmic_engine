"""Traveller worlds as Cosmic Engine shots.

Bridges the sister projects — worldmaker (system generation), weorold
and Erith (terrestrial surfaces) — into the manifest-driven render
pipeline, and supplies a self-contained procedural generator so nothing
in the pipeline depends on those repos being present.

    from worlds import generate_sector, shot_for_world

    foreven = generate_sector(seed=1977)          # ~420 worlds
    shot = shot_for_world(foreven.by_hex("1721"))

Point the wrappers at local checkouts with COSMIC_WORLDMAKER_PATH,
COSMIC_WEOROLD_PATH and COSMIC_ERITH_PATH; `worlds.providers.
provider_status()` reports what is reachable.
"""
from .adapter import (
    camera_for,
    palette_for,
    params_for,
    renderer_for,
    shot_for_world,
    star_light_rgb,
)
from .model import Star, World
from .providers import (
    ErithProvider,
    ProceduralProvider,
    ProviderUnavailable,
    WeoroldProvider,
    WorldmakerProvider,
    get_provider,
    provider_status,
    roll_uwp,
)
from .sector import Sector, generate_sector, pick_worlds
from .uwp import UWP, UWPError, ehex_to_int, int_to_ehex

__all__ = [
    "UWP", "UWPError", "ehex_to_int", "int_to_ehex",
    "World", "Star",
    "ProceduralProvider", "WorldmakerProvider", "WeoroldProvider",
    "ErithProvider", "ProviderUnavailable", "get_provider",
    "provider_status", "roll_uwp",
    "Sector", "generate_sector", "pick_worlds",
    "shot_for_world", "renderer_for", "params_for", "camera_for",
    "palette_for", "star_light_rgb",
]
