"""The neutral world model the render pipeline consumes.

`World` is deliberately provider-independent: worldmaker, weorold, Erith
and the built-in procedural generator all produce these, and the adapter
(`worlds.adapter`) only ever sees this. Adding a fifth source means
writing one more provider, not touching the renderers.

Coordinates follow Traveller convention: a sector is 32 hexes wide by 40
tall, divided into sixteen 8x10 subsectors lettered A-P.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .uwp import UWP

SECTOR_WIDTH = 32
SECTOR_HEIGHT = 40
SUBSECTOR_WIDTH = 8
SUBSECTOR_HEIGHT = 10
# Subsectors are lettered A-P across four columns and four rows. Unlike
# ehex digits, these do include I and O.
SUBSECTORS = "ABCDEFGHIJKLMNOP"


# Stellar classes, with effective temperature (K) and luminosity (solar).
STAR_TYPES = {
    "O": (35000.0, 50000.0),
    "B": (18000.0, 500.0),
    "A": (8500.0, 20.0),
    "F": (6600.0, 2.5),
    "G": (5700.0, 1.0),
    "K": (4400.0, 0.35),
    "M": (3200.0, 0.05),
}


@dataclass
class Star:
    """The primary of a system."""
    spectral_class: str = "G"
    subclass: int = 2
    luminosity_class: str = "V"

    @property
    def temperature_k(self) -> float:
        base = STAR_TYPES.get(self.spectral_class.upper(), STAR_TYPES["G"])[0]
        # Subclass 0 is the hot end of the class, 9 the cool end; step
        # toward the next cooler class.
        classes = list(STAR_TYPES)
        i = classes.index(self.spectral_class.upper())
        cooler = STAR_TYPES[classes[min(i + 1, len(classes) - 1)]][0]
        return base + (cooler - base) * (self.subclass / 10.0)

    @property
    def luminosity_sol(self) -> float:
        lum = STAR_TYPES.get(self.spectral_class.upper(), STAR_TYPES["G"])[1]
        if self.luminosity_class.upper() in ("III", "II", "I"):
            lum *= 60.0          # giants
        return lum

    def __str__(self) -> str:
        return f"{self.spectral_class}{self.subclass} {self.luminosity_class}"


@dataclass
class World:
    """One world, as the render pipeline sees it."""
    name: str
    uwp: UWP = field(default_factory=UWP)
    # "terrestrial" | "gas_giant" | "ringed_giant" | "hot_jupiter"
    # | "mini_neptune" | "brown_dwarf" | "asteroid"
    body_type: str = "terrestrial"
    mean_temp_k: float = 288.0
    star: Star = field(default_factory=Star)
    orbit_au: float = 1.0
    # Sector position: hex column 1..32, row 1..40.
    hex_x: int = 1
    hex_y: int = 1
    sector: str = "Foreven"
    # Trade / remark codes ("Ag", "In", "Va", "Wa", ...).
    trade_codes: tuple = ()
    # Deterministic per-world seed; every renderer parameter derived for
    # this world keys off it.
    seed: int = 0
    provider: str = "procedural"

    # ---- derived -----------------------------------------------------

    @property
    def hex_label(self) -> str:
        """Traveller hex address, e.g. '0703'."""
        return f"{self.hex_x:02d}{self.hex_y:02d}"

    @property
    def subsector(self) -> str:
        """Subsector letter A-P."""
        col = (self.hex_x - 1) // SUBSECTOR_WIDTH
        row = (self.hex_y - 1) // SUBSECTOR_HEIGHT
        return SUBSECTORS[row * 4 + col]

    @property
    def is_giant(self) -> bool:
        return self.body_type in ("gas_giant", "ringed_giant", "hot_jupiter",
                                  "mini_neptune", "brown_dwarf")

    @property
    def has_liquid_water(self) -> bool:
        """Water is liquid at the surface only within a temperature band
        that widens slightly with pressure."""
        if self.uwp.water_fraction <= 0.0 or self.uwp.is_vacuum:
            return False
        boiling = 373.0 + 25.0 * math.log10(max(self.uwp.pressure_atm, 0.01))
        return 258.0 < self.mean_temp_k < boiling

    @property
    def is_frozen(self) -> bool:
        return self.mean_temp_k < 258.0 and self.uwp.water_fraction > 0.0

    def summary(self) -> str:
        return (f"{self.name} ({self.sector} {self.hex_label}) "
                f"{self.uwp} — {self.body_type}, {self.mean_temp_k:.0f} K, "
                f"{self.star}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uwp"] = str(self.uwp)
        d["star"] = str(self.star)
        d["trade_codes"] = list(self.trade_codes)
        return d
