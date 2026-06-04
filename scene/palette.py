"""Named palette / look profiles.

A Palette carries the aesthetic identity of a sequence: a small set of
RGB anchor colors, a contrast/exposure curve hint, and an optional film
emulation tag. The continuity engine reads these to match adjacent
shots; the AI Factory uses them to seed prompts and IP-Adapter style.

Values are scene-linear (pre-tonemap). RGB tuples in 0..1.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Palette:
    name: str
    anchors: tuple[tuple[float, float, float], ...]
    exposure: float = 0.0  # stops
    contrast: float = 1.0
    saturation: float = 1.0
    film_emulation: str | None = None
    prompt_tags: tuple[str, ...] = ()


PALETTES: dict[str, Palette] = {
    "neutral": Palette(
        name="neutral",
        anchors=((0.02, 0.02, 0.03), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ),
    "trumbull_2001": Palette(
        name="trumbull_2001",
        anchors=(
            (0.02, 0.0, 0.06),  # deep violet void
            (0.95, 0.15, 0.55),  # magenta plate
            (0.15, 0.65, 0.95),  # cyan/blue band
            (1.0, 0.85, 0.2),  # solar yellow
            (1.0, 1.0, 1.0),  # blown-out highlight
        ),
        exposure=0.5,
        contrast=1.25,
        saturation=1.35,
        film_emulation="kodak_2383",
        prompt_tags=(
            "saturated dye-transfer film",
            "anamorphic flare",
            "high contrast",
            "1968 Cinerama",
        ),
    ),
    "nfb_universe_1960": Palette(
        name="nfb_universe_1960",
        anchors=(
            (0.0, 0.0, 0.0),
            (0.18, 0.18, 0.20),
            (0.55, 0.55, 0.58),
            (0.95, 0.95, 0.95),
        ),
        exposure=-0.2,
        contrast=1.15,
        saturation=0.05,
        film_emulation="kodak_plus_x_bw",
        prompt_tags=(
            "black and white astronomical plate",
            "fine grain",
            "1960 NFB documentary",
            "subtle vignette",
        ),
    ),
    "jwst_nircam": Palette(
        name="jwst_nircam",
        anchors=(
            (0.02, 0.0, 0.08),
            (0.95, 0.45, 0.15),  # F444W orange
            (0.95, 0.85, 0.55),  # F356W gold
            (0.35, 0.65, 0.95),  # F200W blue
            (1.0, 0.95, 0.85),
        ),
        exposure=0.0,
        contrast=1.1,
        saturation=1.15,
        film_emulation=None,
        prompt_tags=("JWST NIRCam composite", "infrared", "diffraction spikes"),
    ),
    "hubble_sii_ha_oiii": Palette(
        name="hubble_sii_ha_oiii",
        anchors=(
            (0.0, 0.0, 0.05),
            (1.0, 0.2, 0.2),  # SII red
            (0.2, 1.0, 0.4),  # Halpha-as-green Hubble palette
            (0.2, 0.3, 1.0),  # OIII blue
            (1.0, 1.0, 1.0),
        ),
        exposure=0.0,
        contrast=1.2,
        saturation=1.3,
        prompt_tags=("Hubble narrowband palette", "SII Halpha OIII"),
    ),
}


def get_palette(name: str) -> Palette:
    if name not in PALETTES:
        raise KeyError(
            f"unknown palette '{name}'. Known: {sorted(PALETTES.keys())}"
        )
    return PALETTES[name]
