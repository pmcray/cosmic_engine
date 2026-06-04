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
    def __init__(self, graph: ShotGraph, output_path: str | Path):
        self.graph = graph
        self.output_path = Path(output_path)

    def _shot_frames(self, shot: Shot) -> Iterator[np.ndarray]:
        cls = get_renderer(shot.renderer)
        renderer = cls(shot)
        try:
            for frame in renderer.iter_frames():
                yield frame
        finally:
            renderer.close()

    def run(self) -> Path:
        self.graph.validate()
        fps = self.graph.shots[0].fps
        palette = get_palette(self.graph.shots[0].palette.name)
        with VideoWriter(self.output_path, fps=fps, exposure_stops=palette.exposure) as vw:
            self._stream(vw)
        return self.output_path

    def _stream(self, vw: VideoWriter) -> None:
        pairs = list(self.graph.iter_pairs())
        # Pre-buffer head frames of each shot's neighbor so transitions
        # can blend without re-rendering. We run shot generators on the
        # fly with a sliding tail buffer.
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

            # Emit bridge from previous tail into this shot's head.
            if prev_plan is not None and prev_tail and head_buf:
                engine = TransitionEngine(prev_plan)
                for bridge_frame in engine.bridge(prev_tail, head_buf):
                    vw.append(bridge_frame)
                # The bridge replaces the overlapped frames on both sides:
                # we've already consumed `head_n` frames of this shot, and
                # we did not write the tail of the previous shot.

            # Stream the body of this shot with a rolling tail buffer.
            tail_buf: deque[np.ndarray] = deque(maxlen=tail_n) if tail_n else deque()
            for f in frames_iter:
                if tail_n:
                    if len(tail_buf) == tail_buf.maxlen and tail_buf[0] is not None:
                        vw.append(tail_buf[0])
                    tail_buf.append(f)
                else:
                    vw.append(f)

            if tail_n == 0:
                # nothing buffered; nothing to hand off
                prev_tail = []
            else:
                prev_tail = list(tail_buf)
            prev_plan = plan

        # Last shot has no successor: flush the tail buffer.
        if prev_tail:
            for f in prev_tail:
                vw.append(f)


def render_graph(graph: ShotGraph, output_path: str | Path) -> Path:
    return GraphRunner(graph, output_path).run()
