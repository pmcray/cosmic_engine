"""Studio Session: orchestrates Producer + Critic + ArtifactStore.

A Session takes a base ShotGraph and explores its parameter space by
delegating mutations to a Producer, rendering each candidate through
the GraphRunner into the ArtifactStore, scoring with the Critic, and
recording an Attempt per iteration. The history is in-memory and
optionally persisted to a JSON sidecar.

Proxy mode downscales resolution and frame count on every produced
graph so exploration runs in seconds, not minutes. The user does a
single full-quality render at the end against the winning manifest.

Caching: before rendering, the store is scanned for an artifact whose
directory name embeds the same manifest hash. If found, the render is
skipped and the existing artifact is re-scored (cheap), letting the
Producer's RNG repeat exploration without re-paying GPU cost.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from director.graph import GraphRunner
from scene.manifest import ShotGraph

from .artifacts import ArtifactStore, _hash_manifest
from .metrics import MetricsCritic
from .producer import Producer


@dataclass
class Attempt:
    iteration: int
    artifact_path: Path
    score: float
    mutation: dict
    summary: dict
    cached: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifact_path"] = str(self.artifact_path)
        return d


class Session:
    def __init__(
        self,
        base_graph: ShotGraph,
        store: ArtifactStore,
        producer: Producer,
        critic: MetricsCritic | None = None,
        proxy: dict | None = None,
        write_video: bool = True,
    ):
        self.base = base_graph
        self.store = store
        self.producer = producer
        self.critic = critic or MetricsCritic()
        self.proxy = dict(proxy) if proxy else {}
        self.write_video = write_video
        self.history: list[Attempt] = []

    # ---- public API ---------------------------------------------------

    def explore(
        self,
        budget: int,
        label_prefix: str = "exp",
        on_attempt: Callable[[Attempt], None] | None = None,
    ) -> list[Attempt]:
        if not self.history:
            self._evaluate(self.base, {"param": "_base", "from": None, "to": None, "policy": "base"},
                           iteration=0, label=f"{label_prefix}_base", on_attempt=on_attempt)
        start = len(self.history)
        for i in range(start, start + budget):
            g, mutation = self.producer.propose(self.history)
            self._evaluate(g, mutation, iteration=i, label=f"{label_prefix}_{i:03d}",
                           on_attempt=on_attempt)
        return self.history

    def best(self) -> Attempt | None:
        return max(self.history, key=lambda a: a.score) if self.history else None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "base_manifest_hash": _hash_manifest(self.base),
            "proxy": self.proxy,
            "attempts": [a.to_dict() for a in self.history],
            "best": self.best().to_dict() if self.history else None,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def render_best_at_full_quality(
        self,
        artifact_label: str = "master",
    ) -> Path:
        b = self.best()
        if b is None:
            raise RuntimeError("no attempts yet; call explore() first")
        graph = self._load_attempt_graph(b)
        # un-apply proxy by copying over the corresponding fields from base
        graph = self._restore_full_quality(graph)
        runner = GraphRunner(
            graph,
            output_path="_unused.mp4",
            artifact_store=self.store,
            artifact_label=artifact_label,
        )
        return runner.run(write_video=True)

    # ---- internals ----------------------------------------------------

    def _evaluate(
        self,
        graph: ShotGraph,
        mutation: dict,
        iteration: int,
        label: str,
        on_attempt,
    ) -> Attempt:
        g_render = self._apply_proxy(graph)
        artifact_path, cached = self._render_or_reuse(g_render, label)
        result = self.critic.evaluate_artifact(artifact_path)
        attempt = Attempt(
            iteration=iteration,
            artifact_path=artifact_path,
            score=float(result.aggregate.get("composite_score", 0.0)),
            mutation=mutation,
            summary=result.aggregate,
            cached=cached,
        )
        self.history.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
        return attempt

    def _apply_proxy(self, graph: ShotGraph) -> ShotGraph:
        if not self.proxy:
            return graph
        g = copy.deepcopy(graph)
        res = self.proxy.get("resolution")
        dur_scale = self.proxy.get("duration_scale")
        dur_min = int(self.proxy.get("duration_min", 16))
        for s in g.shots:
            if res is not None:
                s.resolution = tuple(res)
            if dur_scale is not None:
                s.duration_frames = max(dur_min, int(s.duration_frames * dur_scale))
        if dur_scale is not None:
            for t in g.transitions:
                t.duration_frames = max(2, int(t.duration_frames * dur_scale))
        return g

    def _restore_full_quality(self, graph: ShotGraph) -> ShotGraph:
        g = copy.deepcopy(graph)
        base_by_id = {s.id: s for s in self.base.shots}
        base_trans = {(t.from_shot, t.to_shot): t for t in self.base.transitions}
        for s in g.shots:
            b = base_by_id.get(s.id)
            if b is None:
                continue
            s.resolution = tuple(b.resolution)
            s.duration_frames = b.duration_frames
        for t in g.transitions:
            b = base_trans.get((t.from_shot, t.to_shot))
            if b is not None:
                t.duration_frames = b.duration_frames
        return g

    def _render_or_reuse(self, graph: ShotGraph, label: str) -> tuple[Path, bool]:
        manifest_hash = _hash_manifest(graph)
        for existing in self.store.list_renders():
            if manifest_hash in existing.name:
                return existing, True
        runner = GraphRunner(
            graph,
            output_path="_unused.mp4",
            artifact_store=self.store,
            artifact_label=label,
        )
        path = runner.run(write_video=self.write_video)
        return path, False

    def _load_attempt_graph(self, attempt: Attempt) -> ShotGraph:
        from scene.manifest import load_manifest
        return load_manifest(attempt.artifact_path / "manifest.json")
