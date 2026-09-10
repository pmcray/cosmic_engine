# Worlds — Traveller planets as Cosmic Engine shots

The `worlds/` package is the bridge between the sister projects —
**worldmaker** (Traveller system generation), **weorold** and **Erith**
(terrestrial surface pipelines) — and the manifest-driven render
pipeline. It also ships a self-contained procedural generator, so
nothing in the pipeline depends on those repositories being present.

```python
from worlds import generate_sector, shot_for_world

foreven = generate_sector(seed=1977)      # ~400 worlds
world = foreven.by_hex("1721")
print(world.summary())
shot = shot_for_world(world)              # renderable Shot
```

## Why this exists

`infinite_director.py` used to reach worldmaker through a module-level
`sys.path.append('/home/pmc/worldmaker')` and `import
traveller_world_generator`. That made the module unimportable anywhere
else, and tied the whole director to one machine's directory layout.
Providers replace it: each knows how to find its project, none of them
is required, and all of them yield the same neutral `World` objects.

## Pointing at local checkouts

| Project | Environment variable | Used for |
|---------|---------------------|----------|
| worldmaker | `COSMIC_WORLDMAKER_PATH` | system and world generation |
| weorold | `COSMIC_WEOROLD_PATH` | terrestrial surface sequences |
| Erith | `COSMIC_ERITH_PATH` | terrestrial surface sequences |

```bash
export COSMIC_WORLDMAKER_PATH=~/src/worldmaker
python -m worlds.cli status
```

Nothing breaks when they are unset — `get_provider()` falls back to the
built-in procedural generator, and `generate_planetary_segment` falls
back to the engine's own terrain renderer. Asking for a provider by
name (`get_provider("worldmaker")`) raises rather than silently
substituting, so a deliberate choice is never quietly ignored.

Imports are scoped: a provider puts its checkout on `sys.path` only for
the duration of the import and removes it afterwards.

## The Foreven sector

A Traveller sector is a 32x40 hex grid, about a third of whose hexes
hold a system — roughly 400 worlds. Every world's identity is keyed on
`(sector seed, hex)` alone, so **Foreven 1721 is the same planet on
every machine, forever**, and stays the same whether you generate the
sector at full density or half:

```bash
python -m worlds.cli sector --seed 1977 --limit 20
python -m worlds.cli sector --seed 1977 --out foreven.sec   # .sec-style data
python -m worlds.cli world --hex 1721
```

Worlds are generated with the classic 2d6 tables and their dependency
chain — atmosphere depends on size, hydrographics on both, tech level on
everything — so every UWP is Traveller-legal. Gas giants get giant
profiles rather than terrestrial ones (oversized, exotic atmosphere, no
hydrosphere).

## From UWP to pixels

`worlds/adapter.py` turns Traveller's abstract codes into physical
rendering knobs:

| Code | Becomes |
|------|---------|
| hydrographics | water level (and whether there is a sea at all) |
| size → surface gravity | terrain relief — low gravity holds up taller mountains |
| atmosphere → pressure | haze strength, and erosion (thick air means fewer sharp octaves) |
| mean temperature | snow line; whether water is liquid, frozen, or boiled off |
| primary's spectral class | sunlight colour and palette choice |
| body type | which renderer runs |

Renderer selection:

```
terrestrial / asteroid                     -> terrain
gas_giant                                  -> volumetric_gas_giant
ringed_giant                               -> saturn_class
hot_jupiter / mini_neptune / brown_dwarf   -> exoplanet_atmosphere
```

Every adapted shot carries a `world` provenance block in its params
(name, UWP, sector, hex, subsector, star, provider), so a rendered frame
can always be traced back to the world that produced it — and the whole
thing is plain JSON, so a manifest built here renders on a machine that
has never heard of worldmaker.

## Touring the sector

```bash
# A voyage visiting twelve worlds, spread across subsectors
python -m worlds.cli manifest --count 12 --out voyages/foreven_tour.json

# ... then render it
python -c "
from director.graph import render_graph
from scene.manifest import load_manifest
import render.adapters
render_graph(load_manifest('voyages/foreven_tour.json'), 'tour.mp4')"
```

The voyage composer also places worlds automatically:
`t_traveller_world` (any surface), `t_traveller_habitable` (breathable
air and liquid water) and `t_traveller_giant` sit in the approach and
alien-worlds phases. The sector is generated once per voyage seed and
cached, so a voyage tours one coherent neighbourhood.

## Adding a provider

Implement `available()` and `generate(seed, count) -> list[World]`, and
return `World` objects. Nothing else in the engine needs to change — the
adapter and every renderer only ever see the neutral model. The
worldmaker wrapper is the worked example: it reads whatever attributes
the upstream objects happen to expose and falls back to a rolled value
for anything missing, so a schema change upstream degrades instead of
crashing.
