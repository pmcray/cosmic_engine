"""Sector generation — the hundreds of worlds of Foreven.

A Traveller sector is a 32x40 hex grid, about a third of whose hexes
hold a system. Foreven sits spinward of the Imperium and is canonically
sparse in detail, which makes it the right place to hang procedurally
generated worlds: the sector seed fixes every world in it, so
"Foreven 1721" means the same planet on every machine, forever.

    from worlds.sector import generate_sector
    foreven = generate_sector(seed=1977)          # ~420 worlds
    print(foreven.by_hex("1721").summary())

Sector data can be written out as Traveller-style tab-separated sector
data (`to_sec_lines`) for cross-checking against other tools.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .model import SECTOR_HEIGHT, SECTOR_WIDTH, SUBSECTORS, World
from .providers import ProceduralProvider, get_provider

# Fraction of hexes that hold a system. Traveller's standard density is
# a roll of 4+ on 1d6 for "rift" through "extremely dense" space; a
# third is the ordinary-space default.
DEFAULT_DENSITY = 1.0 / 3.0


@dataclass
class Sector:
    name: str
    seed: int
    worlds: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.worlds)

    def by_hex(self, hex_label: str) -> World:
        for w in self.worlds:
            if w.hex_label == str(hex_label):
                return w
        raise KeyError(f"no world at {self.name} {hex_label}")

    def in_subsector(self, letter: str) -> list:
        letter = letter.upper()
        if letter not in SUBSECTORS:
            raise KeyError(f"no subsector {letter!r} (expected one of {SUBSECTORS})")
        return [w for w in self.worlds if w.subsector == letter]

    def terrestrials(self) -> list:
        return [w for w in self.worlds if w.body_type == "terrestrial"]

    def giants(self) -> list:
        return [w for w in self.worlds if w.is_giant]

    def habitable(self) -> list:
        """Worlds a person could stand on without a suit."""
        return [w for w in self.worlds
                if w.body_type == "terrestrial"
                and w.uwp.breathable
                and w.has_liquid_water]

    def to_sec_lines(self) -> list:
        """Traveller .sec-style rows: hex, name, UWP, trade codes."""
        rows = []
        for w in sorted(self.worlds, key=lambda x: (x.hex_x, x.hex_y)):
            rows.append(f"{w.hex_label}\t{w.name}\t{w.uwp}\t"
                        f"{' '.join(w.trade_codes)}")
        return rows


def generate_sector(seed: int = 1977,
                    name: str = "Foreven",
                    density: float = DEFAULT_DENSITY,
                    provider=None) -> Sector:
    """Populate a full 32x40 sector.

    Every occupied hex draws its world from a substream keyed by the
    hex itself, so a world's identity depends only on (sector seed, hex)
    — adding or removing worlds elsewhere never shifts it.
    """
    provider = provider or ProceduralProvider(sector=name)
    worlds = []
    for x in range(1, SECTOR_WIDTH + 1):
        for y in range(1, SECTOR_HEIGHT + 1):
            hex_seed = (seed * 1_000_003 + x * 41 + y * 1601) & 0x7FFFFFFF
            if random.Random(hex_seed ^ 0xBEEF).random() >= density:
                continue
            w = provider.generate(hex_seed, 1)[0]
            w.hex_x, w.hex_y = x, y
            w.sector = name
            worlds.append(w)
    return Sector(name=name, seed=seed, worlds=worlds)


def pick_worlds(sector: Sector, count: int, seed: int = 0,
                kinds: tuple | None = None) -> list:
    """Choose `count` worlds from a sector for a voyage — spread across
    subsectors so a sequence tours the sector rather than one corner of
    it."""
    pool = list(sector.worlds)
    if kinds:
        pool = [w for w in pool if w.body_type in kinds]
    if not pool:
        raise ValueError(f"no worlds of kinds {kinds} in {sector.name}")

    rng = random.Random(seed)
    by_sub: dict = {}
    for w in pool:
        by_sub.setdefault(w.subsector, []).append(w)
    for group in by_sub.values():
        rng.shuffle(group)

    # Round-robin across subsectors until we have enough.
    order = sorted(by_sub)
    rng.shuffle(order)
    out = []
    while len(out) < count and any(by_sub[k] for k in order):
        for k in order:
            if by_sub[k]:
                out.append(by_sub[k].pop())
                if len(out) >= count:
                    break
    return out
