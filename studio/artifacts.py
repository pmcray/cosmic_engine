"""Artifact store for ShotGraph renders.

Each render produces a self-contained directory:

    renders/<timestamp>_<manifest_hash>/
        manifest.json          # canonical ShotGraph snapshot
        provenance.json        # git SHA, python, library versions
        timing.json            # per-shot wall-clock
        video.mp4              # finished tonemapped master
        metrics.json           # aggregate scores (filled by Critic)
        shots/
            <shot_id>/
                keyframes/
                    f0000.npy      # scene-linear float32, exact
                    f0000.png      # tonemapped sRGB, for humans + VLMs
                    ...
                metrics.json   # per-shot scores (filled by Critic)

The store is append-only at the directory level — new renders never
mutate old ones. Critics write metric sidecars in place; downstream
producers diff manifest snapshots between sibling artifacts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from scene.manifest import Shot, ShotGraph, dump_manifest

from .provenance import capture_provenance


DEFAULT_KEYFRAMES_PER_SHOT = 6


def _hash_manifest(graph: ShotGraph) -> str:
    text = dump_manifest(graph)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _pick_keyframe_indices(total_frames: int, k: int) -> set[int]:
    if total_frames <= 0:
        return set()
    if total_frames <= k:
        return set(range(total_frames))
    return {int(round(i * (total_frames - 1) / (k - 1))) for i in range(k)}


@dataclass
class ShotArtifact:
    shot: Shot
    root: Path
    keyframes_per_shot: int = DEFAULT_KEYFRAMES_PER_SHOT
    _picked: set[int] = field(default_factory=set)
    _seen: int = 0
    _saved: int = 0
    _t_start: float = field(default_factory=time.time)
    _t_end: float | None = None

    def __post_init__(self) -> None:
        (self.root / "keyframes").mkdir(parents=True, exist_ok=True)
        self._picked = _pick_keyframe_indices(
            self.shot.duration_frames, self.keyframes_per_shot
        )

    def on_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        self._seen += 1
        if frame_idx in self._picked:
            self._write_keyframe(frame_idx, frame)

    def _write_keyframe(self, idx: int, frame: np.ndarray) -> None:
        kdir = self.root / "keyframes"
        stem = f"f{idx:04d}"
        np.save(kdir / f"{stem}.npy", frame.astype(np.float32))
        try:
            from render.io import to_uint8_srgb
            try:
                from PIL import Image
                Image.fromarray(to_uint8_srgb(frame), mode="RGB").save(kdir / f"{stem}.png")
            except Exception:
                try:
                    import imageio.v3 as iio
                    iio.imwrite(kdir / f"{stem}.png", to_uint8_srgb(frame))
                except Exception:
                    pass
        except Exception:
            pass
        self._saved += 1

    def finalize(self) -> None:
        self._t_end = time.time()

    def summary(self) -> dict:
        return {
            "id": self.shot.id,
            "renderer": self.shot.renderer,
            "duration_frames": self.shot.duration_frames,
            "fps": self.shot.fps,
            "resolution": list(self.shot.resolution),
            "palette": self.shot.palette.name,
            "frames_seen": self._seen,
            "keyframes_saved": self._saved,
            "render_seconds": round(
                (self._t_end or time.time()) - self._t_start, 3
            ),
        }


@dataclass
class RenderArtifact:
    path: Path
    graph: ShotGraph
    keyframes_per_shot: int = DEFAULT_KEYFRAMES_PER_SHOT
    _shots: dict[str, ShotArtifact] = field(default_factory=dict)
    _t_start: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "shots").mkdir(exist_ok=True)
        (self.path / "manifest.json").write_text(dump_manifest(self.graph))
        (self.path / "provenance.json").write_text(
            json.dumps(capture_provenance(), indent=2)
        )

    @property
    def video_path(self) -> Path:
        return self.path / "video.mp4"

    def begin_shot(self, shot: Shot) -> ShotArtifact:
        sdir = self.path / "shots" / shot.id
        sa = ShotArtifact(
            shot=shot,
            root=sdir,
            keyframes_per_shot=self.keyframes_per_shot,
        )
        self._shots[shot.id] = sa
        return sa

    def get_shot(self, shot_id: str) -> ShotArtifact:
        return self._shots[shot_id]

    def finalize(self, video_path: Path | str | None = None) -> Path:
        if video_path is not None and Path(video_path) != self.video_path:
            shutil.copyfile(video_path, self.video_path)
        for sa in self._shots.values():
            if sa._t_end is None:
                sa.finalize()
        timing = {
            "total_seconds": round(time.time() - self._t_start, 3),
            "shots": {sid: sa.summary() for sid, sa in self._shots.items()},
        }
        (self.path / "timing.json").write_text(json.dumps(timing, indent=2))
        return self.path


class ArtifactStore:
    def __init__(self, root: str | Path = "renders"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def begin_render(
        self,
        graph: ShotGraph,
        label: str | None = None,
        keyframes_per_shot: int = DEFAULT_KEYFRAMES_PER_SHOT,
    ) -> RenderArtifact:
        ts = time.strftime("%Y%m%d_%H%M%S")
        sha = _hash_manifest(graph)
        slug = f"{ts}_{sha}"
        if label:
            slug = f"{slug}_{label}"
        out = self.root / slug
        if out.exists():
            out = self.root / f"{slug}_{int(time.time() * 1000) % 100000}"
        return RenderArtifact(
            path=out, graph=graph, keyframes_per_shot=keyframes_per_shot
        )

    def list_renders(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir() if p.is_dir())

    def latest(self) -> Path | None:
        renders = self.list_renders()
        return renders[-1] if renders else None

    def load_metrics(self, artifact_dir: Path | str) -> dict | None:
        p = Path(artifact_dir) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None


def load_keyframes(shot_dir: Path | str) -> Sequence[np.ndarray]:
    """Load scene-linear keyframes from a shot artifact directory."""
    kdir = Path(shot_dir) / "keyframes"
    if not kdir.exists():
        return []
    npys = sorted(kdir.glob("f*.npy"))
    return [np.load(p) for p in npys]
