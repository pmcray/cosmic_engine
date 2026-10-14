"""Command line for the worlds package.

    # What sister projects can this machine reach?
    python -m worlds.cli status

    # Generate the Foreven sector and list it
    python -m worlds.cli sector --seed 1977 --limit 20

    # Look up one world and show the shot it would render as
    python -m worlds.cli world --hex 1721

    # Emit a voyage manifest touring N worlds of the sector
    python -m worlds.cli manifest --count 12 --out voyages/foreven.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sector(args):
    from .providers import get_provider
    from .sector import generate_sector
    return generate_sector(seed=args.seed, name=args.sector,
                           provider=get_provider(args.provider,
                                                 sector=args.sector))


def cmd_status(args) -> int:
    from .providers import (ENV_ERITH, ENV_WEOROLD, ENV_WORLDMAKER,
                            provider_status)
    status = provider_status()
    env = {"worldmaker": ENV_WORLDMAKER, "weorold": ENV_WEOROLD,
           "erith": ENV_ERITH}
    for name, ok in status.items():
        mark = "available" if ok else "not found"
        hint = f"   (set {env[name]})" if not ok and name in env else ""
        print(f"  {name:12} {mark}{hint}")
    return 0


def cmd_sector(args) -> int:
    sector = _sector(args)
    print(f"{sector.name} (seed {sector.seed}): {len(sector)} worlds — "
          f"{len(sector.terrestrials())} terrestrial, "
          f"{len(sector.giants())} giant, "
          f"{len(sector.habitable())} habitable")
    rows = sector.to_sec_lines()
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        print(row)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(sector.to_sec_lines()) + "\n")
        print(f"\nwrote {args.out}")
    return 0


def cmd_world(args) -> int:
    from .adapter import shot_for_world
    sector = _sector(args)
    world = sector.by_hex(args.hex)
    print(world.summary())
    print(f"  {world.uwp.describe()}")
    print(f"  subsector {world.subsector}, orbit {world.orbit_au:.2f} AU, "
          f"trade codes: {' '.join(world.trade_codes) or '—'}")
    shot = shot_for_world(world)
    print(f"\nrenders as '{shot.renderer}' with palette "
          f"'{shot.palette.name}':")
    params = {k: v for k, v in shot.params.items() if k != "world"}
    print(json.dumps(params, indent=2, default=float))
    return 0


def cmd_manifest(args) -> int:
    from scene.manifest import ShotGraph, Transition, dump_manifest
    from .adapter import shot_for_world
    from .sector import pick_worlds

    sector = _sector(args)
    worlds = pick_worlds(sector, args.count, seed=args.seed)
    w, h = (int(x) for x in args.resolution.split("x"))

    shots = [shot_for_world(world, duration_frames=args.shot_frames,
                            fps=args.fps, resolution=(w, h))
             for world in worlds]
    transitions = [
        Transition(from_shot=a.id, to_shot=b.id, kind="crossfade",
                   duration_frames=16,
                   params={"histogram_strength": 0.55, "smoothstep": True})
        for a, b in zip(shots, shots[1:])
    ]
    graph = ShotGraph(shots=shots, transitions=transitions,
                      title=f"{sector.name} tour (seed {args.seed})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump_manifest(graph, args.out)
    total = (sum(s.duration_frames for s in shots)
             - sum(t.duration_frames for t in transitions)) / args.fps
    print(f"{len(shots)} worlds, {total:.1f}s -> {args.out}")
    for s in shots:
        wd = s.params["world"]
        print(f"  {wd['hex']}  {wd['name']:20} {wd['uwp']:12} {s.renderer}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="worlds")
    parser.add_argument("--seed", type=int, default=1977)
    parser.add_argument("--sector", default="Foreven")
    parser.add_argument("--provider", default=None,
                        help="worldmaker | procedural (default: auto)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="which sister projects are reachable")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("sector", help="generate and list a sector")
    p.add_argument("--limit", type=int, default=25,
                   help="rows to print (0 = all)")
    p.add_argument("--out", help="write full .sec-style data here")
    p.set_defaults(func=cmd_sector)

    p = sub.add_parser("world", help="describe one world and its shot")
    p.add_argument("--hex", required=True, help="e.g. 1721")
    p.set_defaults(func=cmd_world)

    p = sub.add_parser("manifest", help="emit a voyage touring the sector")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--shot-frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--resolution", default="1920x1080")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_manifest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
