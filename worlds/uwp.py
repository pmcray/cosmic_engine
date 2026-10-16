"""Traveller Universal World Profile parsing and physical interpretation.

A UWP is the canonical Traveller shorthand for a world:

    A867949-C
    │││││││ └── tech level
    ││││││└──── law level
    │││││└───── government
    ││││└────── population (log10)
    │││└─────── hydrographics (tenths of surface water)
    ││└──────── atmosphere
    │└───────── size (thousands of miles diameter)
    └────────── starport quality

Every field is a single "pseudo-hex" digit (0-9 then A-Z, skipping I and
O, which are excluded to avoid confusion with 1 and 0). Starport is a
letter grade A-E plus X.

This module owns two things: parsing the string, and turning those
abstract codes into the physical quantities a renderer actually needs —
diameter, surface gravity, atmospheric pressure, water coverage. The
mapping follows the classic Traveller tables (Book 3 / MegaTraveller
World Builder's Handbook), simplified where the tables give ranges.

Nothing here imports the render stack, so it stays cheap and testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Pseudo-hex: I and O are skipped.
_EHEX = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

UWP_RE = re.compile(r"^([A-EX])([0-9A])([0-9A-HJ-NP-Z])([0-9A-HJ-NP-Z])"
                    r"([0-9A-HJ-NP-Z])([0-9A-HJ-NP-Z])([0-9A-HJ-NP-Z])"
                    r"-?([0-9A-HJ-NP-Z])$")


class UWPError(ValueError):
    pass


def ehex_to_int(ch: str) -> int:
    """Pseudo-hex digit -> integer."""
    idx = _EHEX.find(str(ch).upper())
    if idx < 0:
        raise UWPError(f"not a valid ehex digit: {ch!r}")
    return idx


def int_to_ehex(value: int) -> str:
    """Integer -> pseudo-hex digit."""
    if not 0 <= value < len(_EHEX):
        raise UWPError(f"value out of ehex range: {value}")
    return _EHEX[value]


# Atmosphere codes. (label, breathable, surface pressure in atm).
# Pressure is the midpoint of the code's band; 0 means vacuum.
ATMOSPHERES = {
    0:  ("vacuum", False, 0.0),
    1:  ("trace", False, 0.05),
    2:  ("very thin, tainted", False, 0.35),
    3:  ("very thin", True, 0.35),
    4:  ("thin, tainted", False, 0.7),
    5:  ("thin", True, 0.7),
    6:  ("standard", True, 1.0),
    7:  ("standard, tainted", False, 1.0),
    8:  ("dense", True, 1.7),
    9:  ("dense, tainted", False, 1.7),
    10: ("exotic", False, 1.0),
    11: ("corrosive", False, 2.5),
    12: ("insidious", False, 2.5),
    13: ("dense, high", True, 3.5),
    14: ("thin, low", True, 0.5),
    15: ("unusual", False, 1.0),
}

# Size code -> (diameter km, surface gravity in g).
SIZES = {
    0:  (800.0, 0.00),
    1:  (1600.0, 0.05),
    2:  (3200.0, 0.15),
    3:  (4800.0, 0.25),
    4:  (6400.0, 0.35),
    5:  (8000.0, 0.45),
    6:  (9600.0, 0.70),
    7:  (11200.0, 0.90),
    8:  (12800.0, 1.00),
    9:  (14400.0, 1.25),
    10: (16000.0, 1.40),
    11: (17600.0, 1.55),
    12: (19200.0, 1.70),
}


@dataclass(frozen=True)
class UWP:
    """A parsed Universal World Profile."""
    starport: str = "C"
    size: int = 8
    atmosphere: int = 6
    hydrographics: int = 7
    population: int = 5
    government: int = 4
    law_level: int = 3
    tech_level: int = 9

    def __post_init__(self) -> None:
        sp = str(self.starport).upper()
        if sp not in "ABCDEX":
            raise UWPError(f"invalid starport: {self.starport!r}")
        object.__setattr__(self, "starport", sp)

        if not (0 <= self.size <= 10):
            raise UWPError(f"canonical mainworld size must be 0-A (0-10), got {self.size}")
        if not (0 <= self.atmosphere <= 15):
            raise UWPError(f"atmosphere out of range 0-15: {self.atmosphere}")
        if not (0 <= self.hydrographics <= 10):
            raise UWPError(f"hydrographics out of range 0-10: {self.hydrographics}")
        if not (0 <= self.population <= 10):
            raise UWPError(f"population out of range 0-10: {self.population}")
        if not (0 <= self.government <= 15):
            raise UWPError(f"government out of range 0-15: {self.government}")
        if not (0 <= self.law_level <= 15):
            raise UWPError(f"law level out of range 0-15: {self.law_level}")
        if not (0 <= self.tech_level <= 33):
            raise UWPError(f"tech level out of range: {self.tech_level}")

        # Canonical rule: whenever Population == 0,
        # Government, Law Level, and Tech Level MUST be 0 (000-0) with Starport X.
        if self.population == 0:
            object.__setattr__(self, "starport", "X")
            object.__setattr__(self, "government", 0)
            object.__setattr__(self, "law_level", 0)
            object.__setattr__(self, "tech_level", 0)

    # ---- construction ------------------------------------------------

    @classmethod
    def parse(cls, text: str) -> "UWP":
        s = str(text).strip().upper().replace(" ", "")
        m = UWP_RE.match(s)
        if not m:
            raise UWPError(f"malformed UWP: {text!r}")
        sp, sz, atm, hyd, pop, gov, law, tl = m.groups()
        return cls(
            starport=sp,
            size=ehex_to_int(sz),
            atmosphere=ehex_to_int(atm),
            hydrographics=ehex_to_int(hyd),
            population=ehex_to_int(pop),
            government=ehex_to_int(gov),
            law_level=ehex_to_int(law),
            tech_level=ehex_to_int(tl),
        )

    def __str__(self) -> str:
        return (f"{self.starport}"
                f"{int_to_ehex(self.size)}"
                f"{int_to_ehex(self.atmosphere)}"
                f"{int_to_ehex(self.hydrographics)}"
                f"{int_to_ehex(self.population)}"
                f"{int_to_ehex(self.government)}"
                f"{int_to_ehex(self.law_level)}"
                f"-{int_to_ehex(self.tech_level)}")

    # ---- physical interpretation -------------------------------------

    @property
    def diameter_km(self) -> float:
        return SIZES.get(self.size, SIZES[10])[0]

    @property
    def gravity_g(self) -> float:
        return SIZES.get(self.size, SIZES[10])[1]

    @property
    def atmosphere_label(self) -> str:
        return ATMOSPHERES.get(self.atmosphere, ("unusual", False, 1.0))[0]

    @property
    def breathable(self) -> bool:
        return ATMOSPHERES.get(self.atmosphere, ("unusual", False, 1.0))[1]

    @property
    def pressure_atm(self) -> float:
        return ATMOSPHERES.get(self.atmosphere, ("unusual", False, 1.0))[2]

    @property
    def water_fraction(self) -> float:
        """Fraction of the surface under liquid, 0..1. Hydrographics is
        in tenths, so A (10) is a full water world."""
        return min(1.0, max(0.0, self.hydrographics / 10.0))

    @property
    def is_vacuum(self) -> bool:
        return self.atmosphere == 0

    @property
    def is_inhabited(self) -> bool:
        return self.population > 0

    def trade_codes(self, temp_k: float | None = None) -> tuple[str, ...]:
        return compute_trade_codes(self, temp_k)

    def describe(self) -> str:
        wet = f"{self.water_fraction * 100:.0f}% water"
        pop = ("uninhabited" if not self.is_inhabited
               else f"pop 10^{self.population}")
        return (f"{self.diameter_km:.0f} km, {self.gravity_g:.2f}g, "
                f"{self.atmosphere_label} atmosphere, {wet}, {pop}, "
                f"TL{self.tech_level}")


def compute_trade_codes(uwp: UWP, temp_k: float | None = None) -> tuple[str, ...]:
    """Compute canonical Traveller trade codes for a world.

    Computed strictly according to canonical Traveller UWP ranges
    (1981 Book 3 / Mongoose World Builder's Handbook):
    Ba, Va, De, Ic, Fl, Hi, In, Ag, Na, Ni, Po, Ri, As (plus Wa, Fr, Ho).
    """
    codes: list[str] = []

    # Ag: Agricultural (Atm 4-9, Hyd 4-8, Pop 5-7)
    if 4 <= uwp.atmosphere <= 9 and 4 <= uwp.hydrographics <= 8 and 5 <= uwp.population <= 7:
        codes.append("Ag")

    # As: Asteroid (Size 0, Atm 0, Hyd 0)
    if uwp.size == 0 and uwp.atmosphere == 0 and uwp.hydrographics == 0:
        codes.append("As")

    # Ba: Barren (Pop 0, Gov 0, Law 0)
    if uwp.population == 0:
        codes.append("Ba")

    # De: Desert (Atm 2+, Hyd 0)
    if uwp.atmosphere >= 2 and uwp.hydrographics == 0:
        codes.append("De")

    # Fl: Fluid Oceans (Atm 10+, Hyd 1+)
    if uwp.atmosphere >= 10 and uwp.hydrographics >= 1:
        codes.append("Fl")

    # Hi: High Population (Pop 9+)
    if uwp.population >= 9:
        codes.append("Hi")

    # Ic: Ice-Capped (Atm 0-1, Hyd 1+)
    if uwp.atmosphere in (0, 1) and uwp.hydrographics >= 1:
        codes.append("Ic")

    # In: Industrial (Atm 0, 1, 2, 4, 7, 9; Pop 9+)
    if uwp.atmosphere in (0, 1, 2, 4, 7, 9) and uwp.population >= 9:
        codes.append("In")

    # Na: Non-Agricultural (Atm 0-3, Hyd 0-3, Pop 6+)
    if 0 <= uwp.atmosphere <= 3 and 0 <= uwp.hydrographics <= 3 and uwp.population >= 6:
        codes.append("Na")

    # Ni: Non-Industrial (Pop 4-6)
    if 4 <= uwp.population <= 6:
        codes.append("Ni")

    # Po: Poor (Atm 2-5, Hyd 0-3)
    if 2 <= uwp.atmosphere <= 5 and 0 <= uwp.hydrographics <= 3:
        codes.append("Po")

    # Ri: Rich (Atm 6 or 8, Pop 6-8, Gov 4-9)
    if uwp.atmosphere in (6, 8) and 6 <= uwp.population <= 8 and 4 <= uwp.government <= 9:
        codes.append("Ri")

    # Va: Vacuum (Atm 0)
    if uwp.atmosphere == 0:
        codes.append("Va")

    # Wa: Water World (Hyd 10)
    if uwp.hydrographics == 10:
        codes.append("Wa")

    if temp_k is not None:
        if temp_k < 240.0:
            codes.append("Fr")
        if temp_k > 330.0:
            codes.append("Ho")

    return tuple(sorted(codes))
