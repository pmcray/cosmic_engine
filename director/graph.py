"""Execute a ShotGraph: instantiate renderers, run continuity, write video.

The graph runner is the thing the Director calls. It does not know about
specific renderers; it only knows the registry. This is what replaces the
old `ffmpeg -c copy` concat in `InfiniteDirector`.

Pipeline per shot:
    1. Look up the renderer class by name.
    2. Iterate frames from the renderer.
    3. Buffer the trailing `transition.duration_frames` for the upcoming bridge.
    4. Once the next shot is up, run the TransitionEngine on the buffered
       tail of A and the head of B, and emit the bridge frames instead
       of the overlapping originals.
    5. Stream into VideoWriter.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np

from scene.manifest import ShotGraph, Shot, Transition
from scene.palette import get_palette
from render.core import get_renderer
from render.io import VideoWriter
from .continuity import TransitionEngine, TransitionPlan


def _make_plan(t: Transition | None) -> TransitionPlan:
    if t is None:
        return TransitionPlan(kind="hard_cut", duration_frames=0)
    return TransitionPlan(
        kind=t.kind,
        duration_frames=t.duration_frames,
        histogram_strength=t.params.get("histogram_strength", 0.7) if t.params else 0.7,
        smoothstep=t.params.get("smoothstep", True) if t.params else True,
        params=t.params or {},
    )


class GraphRunner:
    def __init__(
        self,
        graph: ShotGraph,
        output_path: str | Path,
        artifact_store=None,
        artifact_label: str | None = None,
    ):
        self.graph = graph
        self.output_path = Path(output_path)
        self.artifact_store = artifact_store
        self._artifact = None
        self._artifact_label = artifact_label

    def _shot_frames(self, shot: Shot) -> Iterator[np.ndarray]:
        cls = get_renderer(shot.renderer)
        renderer = cls(shot)
        shot_artifact = (
            self._artifact.begin_shot(shot) if self._artifact is not None else None
        )
        try:
            for idx, frame in enumerate(renderer.iter_frames()):
                if shot_artifact is not None:
                    shot_artifact.on_frame(idx, frame)
                yield frame
        finally:
            if shot_artifact is not None:
                shot_artifact.finalize()
            renderer.close()

    def run(self, write_video: bool = True) -> Path:
        self.graph.validate()
        fps = self.graph.shots[0].fps
        palette = get_palette(self.graph.shots[0].palette.name)
        if self.artifact_store is not None:
            self._artifact = self.artifact_store.begin_render(
                self.graph, label=self._artifact_label
            )
            self.output_path = self._artifact.video_path
        if write_video:
            with VideoWriter(self.output_path, fps=fps, exposure_stops=palette.exposure) as vw:
                self._stream(vw)
        else:
            self._stream(None)
        if self._artifact is not None:
            self._artifact.finalize(
                video_path=self.output_path if write_video else None
            )
            return self._artifact.path
        return self.output_path

    def _stream(self, vw: VideoWriter | None) -> None:
        def emit(frame):
            if vw is not None:
                vw.append(frame)

        pairs = list(self.graph.iter_pairs())
        prev_tail: list[np.ndarray] = []
        prev_plan: TransitionPlan | None = None
        for i, (shot, trans, nxt) in enumerate(pairs):
            plan = _make_plan(trans)
            tail_n = plan.duration_frames
            head_n = prev_plan.duration_frames if prev_plan else 0

            frames_iter = self._shot_frames(shot)
            head_buf: list[np.ndarray] = []
            for _ in range(head_n):
                try:
                    head_buf.append(next(frames_iter))
                except StopIteration:
                    break

            if prev_plan is not None and prev_tail and head_buf:
                engine = TransitionEngine(prev_plan)
                for bridge_frame in engine.bridge(prev_tail, head_buf):
                    emit(bridge_frame)

            tail_buf: deque[np.ndarray] = deque(maxlen=tail_n) if tail_n else deque()
            for f in frames_iter:
                if tail_n:
                    if len(tail_buf) == tail_buf.maxlen and tail_buf[0] is not None:
                        emit(tail_buf[0])
                    tail_buf.append(f)
                else:
                    emit(f)

            if tail_n == 0:
                prev_tail = []
            else:
                prev_tail = list(tail_buf)
            prev_plan = plan

        if prev_tail:
            for f in prev_tail:
                emit(f)


def render_graph(
    graph: ShotGraph,
    output_path: str | Path,
    artifact_store=None,
    artifact_label: str | None = None,
    write_video: bool = True,
) -> Path:
    return GraphRunner(
        graph,
        output_path,
        artifact_store=artifact_store,
        artifact_label=artifact_label,
    ).run(write_video=write_video)
