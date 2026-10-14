"""World sources: the sister projects, plus a self-contained fallback.

`infinite_director.py` used to reach the Traveller generator through a
module-level `sys.path.append('/home/pmc/worldmaker')`, which meant the
module could not even be imported anywhere else. Providers replace that:
each one knows how to find its project (an explicit path, an environment
variable, or an installed package), and every one of them yields the
same neutral `World` objects.

    COSMIC_WORLDMAKER_PATH   -> traveller_world_generator (system/world gen)
    COSMIC_WEOROLD_PATH      -> weorold (terrestrial surface pipeline)
    COSMIC_ERITH_PATH        -> erith  (terrestrial surface pipeline)

`ProceduralProvider` needs none of them: it generates Traveller-legal
worlds from a seed alone, so a voyage manifest always renders, on any
machine, with or without the sister repos checked out. `get_provider()`
picks the best available source and silently falls back.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import random
import sys
from pathlib import Path
from typing import Iterator, Protocol

from .model import SECTOR_HEIGHT, SECTOR_WIDTH, Star, World
from .uwp import UWP, ehex_to_int

# Environment variables naming each sister project's checkout.
ENV_WORLDMAKER = "COSMIC_WORLDMAKER_PATH"
ENV_WEOROLD = "COSMIC_WEOROLD_PATH"
ENV_ERITH = "COSMIC_ERITH_PATH"


class ProviderUnavailable(RuntimeError):
    """The provider's project could not be located or imported."""


@contextlib.contextmanager
def _sys_path(extra: str | None) -> Iterator[None]:
    """Temporarily prepend a directory to sys.path. Unlike the old
    module-level append, this leaves the interpreter as it found it."""
    if not extra:
        yield
        return
    p = str(Path(extra).expanduser().resolve())
    sys.path.insert(0, p)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(p)


def _resolve_path(explicit: str | None, env_var: str) -> str | None:
    """An explicit path wins; else the environment variable; else None
    (meaning: hope the package is already importable)."""
    if explicit:
        return explicit
    return os.environ.get(env_var) or None


class WorldProvider(Protocol):
    """Anything that can produce worlds for the render pipeline."""

    name: str

    def available(self) -> bool:
        """True if this provider can actually run here."""

    def generate(self, seed: int, count: int = 1) -> list[World]:
        """Produce `count` worlds deterministically from `seed`."""


# ============================================================
# Procedural fallback — no external dependencies
# ============================================================

# Rough syllable stock for names. Deterministic given the seed; this is
# scaffolding, not a linguistics project.
_SYL_A = ("kar", "vel", "tor", "mir", "zan", "cal", "dre", "sol", "quin",
          "bel", "nar", "syr", "thal", "ver", "ori", "lex", "mor", "ish")
_SYL_B = ("dan", "ith", "mos", "ara", "eth", "ux", "ion", "ura", "esk",
          "andr", "oth", "iel", "ask", "umn", "ede", "orr")
_SYL_C = ("", "", "a", "is", "on", " Prime", " Secundus", " IV", " VII")


def _make_name(rng: random.Random) -> str:
    n = rng.choice(_SYL_A) + rng.choice(_SYL_B) + rng.choice(_SYL_C)
    return n[0].upper() + n[1:]


def _roll_2d6(rng: random.Random) -> int:
    return rng.randint(1, 6) + rng.randint(1, 6)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def roll_uwp(rng: random.Random) -> UWP:
    """Generate a Traveller-legal UWP with the classic 2d6 tables and
    their dependency chain (atmosphere depends on size, hydrographics on
    both, tech level on everything)."""
    size = _clamp(_roll_2d6(rng) - 2, 0, 10)

    if size == 0:
        atmosphere = 0
    else:
        atmosphere = _clamp(_roll_2d6(rng) - 7 + size, 0, 15)

    if size <= 1:
        hydro = 0
    else:
        mod = 0
        if atmosphere in (0, 1, 10, 11, 12):
            mod -= 4
        hydro = _clamp(_roll_2d6(rng) - 7 + size + mod, 0, 10)

    population = _clamp(_roll_2d6(rng) - 2, 0, 10)
    government = (_clamp(_roll_2d6(rng) - 7 + population, 0, 15)
                  if population else 0)
    law = (_clamp(_roll_2d6(rng) - 7 + government, 0, 9)
           if population else 0)

    # Starport quality: the classic A-X table.
    sp_roll = _roll_2d6(rng)
    starport = ("A" if sp_roll >= 11 else
                "B" if sp_roll >= 9 else
                "C" if sp_roll >= 7 else
                "D" if sp_roll >= 5 else
                "E" if sp_roll >= 3 else "X")

    tl_mod = {"A": 6, "B": 4, "C": 2, "D": 0, "E": 0, "X": -4}[starport]
    if size <= 1:
        tl_mod += 2
    if atmosphere <= 3 or atmosphere >= 10:
        tl_mod += 1
    if population >= 9:
        tl_mod += 2
    tech = _clamp(rng.randint(1, 6) + tl_mod, 0, 20) if population else 0

    return UWP(starport=starport, size=size, atmosphere=atmosphere,
               hydrographics=hydro, population=population,
               government=government, law_level=law, tech_level=tech)


def _trade_codes(uwp: UWP, temp_k: float) -> tuple:
    codes = []
    if uwp.atmosphere == 0:
        codes.append("Va")
    if uwp.hydrographics == 10:
        codes.append("Wa")
    if uwp.hydrographics == 0 and uwp.atmosphere >= 2:
        codes.append("De")
    if uwp.size == 0:
        codes.append("As")
    if uwp.population == 0:
        codes.append("Ba")
    if 4 <= uwp.atmosphere <= 9 and 4 <= uwp.hydrographics <= 8 \
            and 5 <= uwp.population <= 7:
        codes.append("Ag")
    if uwp.atmosphere in (0, 1, 2, 4, 7, 9) and uwp.population >= 9:
        codes.append("In")
    if temp_k < 240.0:
        codes.append("Fr")
    if temp_k > 330.0:
        codes.append("Ho")
    return tuple(codes)


def _mean_temp(star: Star, orbit_au: float, uwp: UWP,
               rng: random.Random) -> float:
    """Equilibrium temperature with a crude greenhouse term.

    T_eq = 278 K * L^(1/4) / sqrt(a), then scaled for albedo and the
    atmosphere's optical thickness."""
    lum = max(star.luminosity_sol, 1e-4)
    t_eq = 278.0 * (lum ** 0.25) / max(orbit_au, 0.02) ** 0.5
    # Greenhouse: thicker atmospheres trap more.
    greenhouse = 1.0 + 0.10 * uwp.pressure_atm
    if uwp.atmosphere in (11, 12):        # corrosive / insidious: runaway
        greenhouse = 1.0 + 0.55 * uwp.pressure_atm
    # Albedo: ice and cloud reflect, bare rock absorbs.
    albedo_factor = 1.0 - 0.12 * uwp.water_fraction
    return max(3.0, t_eq * greenhouse * albedo_factor * rng.uniform(0.94, 1.06))


class ProceduralProvider:
    """Seeded Traveller world generation with no external dependencies.

    This is the provider that always works — the pipeline's guarantee
    that a voyage manifest renders on a bare checkout.
    """

    name = "procedural"

    def __init__(self, sector: str = "Foreven"):
        self.sector = sector

    def available(self) -> bool:
        return True

    def generate(self, seed: int, count: int = 1) -> list[World]:
        out = []
        for i in range(count):
            # Each world gets its own substream, so world k is the same
            # whether you asked for 1 world or 500.
            rng = random.Random((seed * 1_000_003 + i) & 0x7FFFFFFF)
            out.append(self._one(rng, seed, i))
        return out

    @staticmethod
    def _giant_uwp(uwp: UWP, body_type: str, rng: random.Random) -> UWP:
        """Rewrite a rolled UWP so it describes a giant rather than a
        surface world: oversized, exotic atmosphere, no hydrosphere.
        Population and infrastructure survive — Traveller giants do host
        gas-mining stations."""
        size = {"mini_neptune": rng.randint(11, 13)}.get(
            body_type, rng.randint(14, 20))
        return UWP(
            starport=uwp.starport,
            size=min(size, 33),
            atmosphere=rng.choice((10, 11, 12, 15)),
            hydrographics=0,
            population=uwp.population if rng.random() < 0.35 else 0,
            government=uwp.government,
            law_level=uwp.law_level,
            tech_level=uwp.tech_level,
        )

    def _one(self, rng: random.Random, seed: int, index: int) -> World:
        uwp = roll_uwp(rng)
        spectral = rng.choices(
            ("O", "B", "A", "F", "G", "K", "M"),
            weights=(1, 2, 6, 12, 20, 25, 34), k=1)[0]
        star = Star(spectral_class=spectral, subclass=rng.randint(0, 9),
                    luminosity_class="V" if rng.random() > 0.08 else "III")
        # Orbit drawn near the star's habitable zone more often than not.
        hz = max(0.1, star.luminosity_sol ** 0.5)
        orbit = hz * rng.choice((0.35, 0.6, 0.9, 1.0, 1.1, 1.4, 2.2, 4.0))
        temp = _mean_temp(star, orbit, uwp, rng)

        body_type = "terrestrial"
        if uwp.size == 0:
            body_type = "asteroid"
        elif rng.random() < 0.18:
            body_type = rng.choices(
                ("gas_giant", "ringed_giant", "hot_jupiter",
                 "mini_neptune", "brown_dwarf"),
                weights=(30, 22, 18, 20, 10), k=1)[0]
            # A giant's profile has to agree with its body: no surface to
            # be 40% ocean, and the atmosphere is the planet.
            uwp = self._giant_uwp(uwp, body_type, rng)
            if body_type == "hot_jupiter":
                orbit = rng.uniform(0.02, 0.09)
            temp = _mean_temp(star, orbit, uwp, rng)

        return World(
            name=_make_name(rng),
            uwp=uwp,
            body_type=body_type,
            mean_temp_k=temp,
            star=star,
            orbit_au=orbit,
            hex_x=rng.randint(1, SECTOR_WIDTH),
            hex_y=rng.randint(1, SECTOR_HEIGHT),
            sector=self.sector,
            trade_codes=_trade_codes(uwp, temp),
            seed=(seed * 1_000_003 + index) & 0x7FFFFFFF,
            provider=self.name,
        )


# ============================================================
# worldmaker (Traveller system generation)
# ============================================================

class WorldmakerProvider:
    """Wraps the `traveller_world_generator` module from the worldmaker
    project. Falls back to nothing — callers use `available()` or
    `get_provider()` rather than catching import errors."""

    name = "worldmaker"

    def __init__(self, path: str | None = None, sector: str = "Foreven"):
        self.path = _resolve_path(path, ENV_WORLDMAKER)
        self.sector = sector

    def _import(self):
        with _sys_path(self.path):
            return importlib.import_module("traveller_world_generator")

    def available(self) -> bool:
        try:
            self._import()
            return True
        except Exception:
            return False

    def generate(self, seed: int, count: int = 1) -> list[World]:
        try:
            wm = self._import()
        except Exception as exc:
            raise ProviderUnavailable(
                f"worldmaker not importable (set {ENV_WORLDMAKER}): {exc}"
            ) from exc

        rng = random.Random(seed)
        out: list[World] = []
        while len(out) < count:
            # worldmaker seeds off the global RNG; seed it per system so
            # the sequence is reproducible.
            random.seed((seed * 7919 + len(out)) & 0x7FFFFFFF)
            system = wm.generate_full_system()
            for raw in getattr(system, "all_worlds", None) or []:
                out.append(self._convert(raw, rng, seed, len(out)))
                if len(out) >= count:
                    break
        return out[:count]

    def _convert(self, raw, rng: random.Random, seed: int, index: int) -> World:
        """Map a worldmaker world onto the neutral model.

        worldmaker's objects carry hex-digit code strings and a
        temperature; anything missing falls back to a rolled value, so a
        schema change upstream degrades instead of crashing."""
        fallback = roll_uwp(rng)

        def code(attr: str, default: int) -> int:
            val = getattr(raw, attr, None)
            if val is None or val == "":
                return default
            try:
                return ehex_to_int(str(val)[0])
            except Exception:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default

        uwp = UWP(
            starport=str(getattr(raw, "starport", None)
                         or fallback.starport)[0].upper(),
            size=code("size_code", fallback.size),
            atmosphere=code("atmosphere_code", fallback.atmosphere),
            hydrographics=code("hydrographics_code", fallback.hydrographics),
            population=code("population_code", fallback.population),
            government=code("government_code", fallback.government),
            law_level=code("law_level_code", fallback.law_level),
            tech_level=code("tech_level_code", fallback.tech_level),
        )

        temp = getattr(raw, "mean_temperature", 0.0) or 0.0
        body = str(getattr(raw, "body_type", "Terrestrial") or "").lower()
        body_type = {
            "terrestrial": "terrestrial",
            "gas giant": "gas_giant",
            "gasgiant": "gas_giant",
            "asteroid": "asteroid",
            "asteroid belt": "asteroid",
        }.get(body, "terrestrial")

        star = Star(
            spectral_class=str(getattr(raw, "spectral_class", "G") or "G")[0].upper(),
            subclass=int(getattr(raw, "spectral_subclass", 2) or 2) % 10,
        )
        orbit = float(getattr(raw, "orbit_au", 0.0) or 0.0) or 1.0
        if temp <= 0.0:
            temp = _mean_temp(star, orbit, uwp, rng)

        return World(
            name=str(getattr(raw, "name", "") or _make_name(rng)),
            uwp=uwp,
            body_type=body_type,
            mean_temp_k=float(temp),
            star=star,
            orbit_au=orbit,
            hex_x=int(getattr(raw, "hex_x", 0) or rng.randint(1, SECTOR_WIDTH)),
            hex_y=int(getattr(raw, "hex_y", 0) or rng.randint(1, SECTOR_HEIGHT)),
            sector=str(getattr(raw, "sector", None) or self.sector),
            trade_codes=tuple(getattr(raw, "trade_codes", ()) or ()),
            seed=(seed * 1_000_003 + index) & 0x7FFFFFFF,
            provider=self.name,
        )


# ============================================================
# weorold / Erith (terrestrial surface pipelines)
# ============================================================

class SurfacePipelineProvider:
    """Common wrapper for the terrestrial surface projects (weorold and
    Erith). Both take a temperature plus atmosphere/hydrographics codes
    and produce a planetary surface; this exposes that as an optional
    heightmap source for the terrain renderer.

    Surface generation is expensive and produces files, so it is never
    invoked during manifest composition — only when a shot is actually
    rendered and `heightmap_from` is set."""

    def __init__(self, module_name: str, env_var: str, name: str,
                 path: str | None = None):
        self.module_name = module_name
        self.env_var = env_var
        self.name = name
        self.path = _resolve_path(path, env_var)

    def _import(self):
        with _sys_path(self.path):
            return importlib.import_module(self.module_name)

    def available(self) -> bool:
        try:
            self._import()
            return True
        except Exception:
            return False

    def generate_surface(self, world: World, output_name: str,
                         test_mode: bool = False):
        """Run the project's pipeline for `world`. Returns whatever the
        pipeline returns (typically an output path)."""
        try:
            mod = self._import()
        except Exception as exc:
            raise ProviderUnavailable(
                f"{self.name} not importable (set {self.env_var}): {exc}"
            ) from exc

        run = getattr(mod, "run_full_pipeline", None)
        if run is None:
            raise ProviderUnavailable(
                f"{self.name} has no run_full_pipeline entry point")

        cwd = os.getcwd()
        try:
            if self.path:
                os.chdir(Path(self.path).expanduser().resolve())
            return run(
                output_name=output_name,
                mean_temp_k=world.mean_temp_k,
                hydro_code=world.uwp.hydrographics,
                atm_code=world.uwp.atmosphere,
                test_mode=test_mode,
            )
        finally:
            os.chdir(cwd)


def WeoroldProvider(path: str | None = None) -> SurfacePipelineProvider:
    return SurfacePipelineProvider("wp12_full_pipeline", ENV_WEOROLD,
                                   "weorold", path)


def ErithProvider(path: str | None = None) -> SurfacePipelineProvider:
    return SurfacePipelineProvider("erith_pipeline", ENV_ERITH,
                                   "erith", path)


# ============================================================
# Selection
# ============================================================

def get_provider(prefer: str | None = None, sector: str = "Foreven"):
    """Return the best available world provider.

    `prefer` names one explicitly ("worldmaker" / "procedural"); when it
    is unavailable, or unset, this falls back to the procedural
    generator so composition never fails for want of a sister repo.
    """
    if prefer == "procedural":
        return ProceduralProvider(sector=sector)
    if prefer in (None, "auto", "worldmaker"):
        wm = WorldmakerProvider(sector=sector)
        if wm.available():
            return wm
        if prefer == "worldmaker":
            raise ProviderUnavailable(
                f"worldmaker requested but not importable; set {ENV_WORLDMAKER}")
    return ProceduralProvider(sector=sector)


def provider_status() -> dict:
    """What is reachable from here — for CLI diagnostics."""
    return {
        "procedural": True,
        "worldmaker": WorldmakerProvider().available(),
        "weorold": WeoroldProvider().available(),
        "erith": ErithProvider().available(),
    }
