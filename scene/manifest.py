"""Shot manifest schema for the Cosmic Engine director graph.

A manifest is a JSON document describing a sequence of Shots and the
Transitions that join them. Each Shot names a renderer (registered via
`render.core.register_renderer`), the parameters that renderer accepts,
a camera path, a palette reference, and timing. The shot graph is the
single source of truth for what gets rendered — the director executes it,
the renderers consume it, the continuity engine reads palette/motion
metadata from it.

The schema is intentionally narrow: it carries only what is reproducible
(seeds, params, palette names, frame counts). Anything model-specific
lives inside the renderer plugin and is recorded as opaque `params`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


class ManifestError(ValueError):
    pass


@dataclass
class Camera:
    position: tuple[float, float, float] = (0.0, 0.0, -5.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 50.0
    # When path_to is set, the camera linearly interpolates from
    # (position, target) at frame 0 to (path_to.position, path_to.target)
    # at the last frame. Splines come later.
    path_to: "Camera | None" = None

    def at(self, t: float) -> "Camera":
        """Interpolate at t in [0, 1]. Returns a flat (non-pathed) Camera."""
        if self.path_to is None:
            return Camera(self.position, self.target, self.up, self.fov_deg, None)
        t = max(0.0, min(1.0, t))
        return Camera(
            position=_lerp3(self.position, self.path_to.position, t),
            target=_lerp3(self.target, self.path_to.target, t),
            up=_lerp3(self.up, self.path_to.up, t),
            fov_deg=self.fov_deg + (self.path_to.fov_deg - self.fov_deg) * t,
            path_to=None,
        )


@dataclass
class PaletteRef:
    name: str = "neutral"
    intensity: float = 1.0


@dataclass
class Shot:
    id: str
    renderer: str
    params: dict[str, Any] = field(default_factory=dict)
    camera: Camera = field(default_factory=Camera)
    palette: PaletteRef = field(default_factory=PaletteRef)
    duration_frames: int = 60
    fps: int = 24
    resolution: tuple[int, int] = (1920, 1080)
    seed: int = 0
    # A free-form hint the continuity engine uses to carry motion across
    # cuts ("zoom_in", "pan_left", "spin_cw", "still", ...). Optional.
    motion_hint: str | None = None


@dataclass
class Transition:
    from_shot: str
    to_shot: str
    kind: str = "crossfade"  # crossfade | slitscan | match_cut | hard_cut
    duration_frames: int = 18
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShotGraph:
    shots: list[Shot] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    title: str = "untitled"
    schema_version: int = SCHEMA_VERSION

    def shot_by_id(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise ManifestError(f"unknown shot id: {shot_id}")

    def validate(self) -> None:
        if not self.shots:
            raise ManifestError("graph has no shots")
        ids = [s.id for s in self.shots]
        if len(set(ids)) != len(ids):
            raise ManifestError(f"duplicate shot ids: {ids}")
        for s in self.shots:
            if s.duration_frames <= 0:
                raise ManifestError(f"shot {s.id} has non-positive duration")
            if s.fps <= 0:
                raise ManifestError(f"shot {s.id} has non-positive fps")
            w, h = s.resolution
            if w <= 0 or h <= 0:
                raise ManifestError(f"shot {s.id} has bad resolution {s.resolution}")
        for t in self.transitions:
            self.shot_by_id(t.from_shot)
            self.shot_by_id(t.to_shot)
            if t.duration_frames < 0:
                raise ManifestError(
                    f"transition {t.from_shot}->{t.to_shot} negative duration"
                )

    def iter_pairs(self) -> Iterable[tuple[Shot, Transition | None, Shot | None]]:
        """Yield (shot, transition_to_next_or_None, next_shot_or_None)."""
        by_pair = {(t.from_shot, t.to_shot): t for t in self.transitions}
        for i, s in enumerate(self.shots):
            nxt = self.shots[i + 1] if i + 1 < len(self.shots) else None
            trans = by_pair.get((s.id, nxt.id)) if nxt else None
            yield s, trans, nxt


def _lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _to_dict(obj):
    if is_dataclass(obj):
        d = asdict(obj)
        return d
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    return obj


def dump_manifest(graph: ShotGraph, path: str | Path | None = None) -> str:
    graph.validate()
    payload = _to_dict(graph)
    text = json.dumps(payload, indent=2)
    if path is not None:
        Path(path).write_text(text)
    return text


def load_manifest(path: str | Path) -> ShotGraph:
    data = json.loads(Path(path).read_text())
    return _graph_from_dict(data)


def _graph_from_dict(data: dict) -> ShotGraph:
    if "schema_version" in data and data["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported schema version {data['schema_version']} (expected {SCHEMA_VERSION})"
        )
    shots = [_shot_from_dict(s) for s in data.get("shots", [])]
    transitions = [Transition(**t) for t in data.get("transitions", [])]
    g = ShotGraph(
        shots=shots,
        transitions=transitions,
        title=data.get("title", "untitled"),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )
    g.validate()
    return g


def _shot_from_dict(d: dict) -> Shot:
    cam = _camera_from_dict(d.get("camera", {}))
    pal_d = d.get("palette", {})
    pal = PaletteRef(name=pal_d.get("name", "neutral"), intensity=pal_d.get("intensity", 1.0))
    res = tuple(d.get("resolution", (1920, 1080)))
    return Shot(
        id=d["id"],
        renderer=d["renderer"],
        params=dict(d.get("params", {})),
        camera=cam,
        palette=pal,
        duration_frames=int(d.get("duration_frames", 60)),
        fps=int(d.get("fps", 24)),
        resolution=res,
        seed=int(d.get("seed", 0)),
        motion_hint=d.get("motion_hint"),
    )


def _camera_from_dict(d: dict) -> Camera:
    if not d:
        return Camera()
    path_to = _camera_from_dict(d["path_to"]) if d.get("path_to") else None
    return Camera(
        position=tuple(d.get("position", (0.0, 0.0, -5.0))),
        target=tuple(d.get("target", (0.0, 0.0, 0.0))),
        up=tuple(d.get("up", (0.0, 1.0, 0.0))),
        fov_deg=float(d.get("fov_deg", 50.0)),
        path_to=path_to,
    )
