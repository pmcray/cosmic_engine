"""Renderer abstraction shared by every shot in the graph.

A Renderer takes a Shot (parameters + camera + palette + seed) and
yields scene-linear RGB float frames in HWC order, shape
(height, width, 3), dtype float32, values nominally in [0, +inf) before
tone-mapping. The director consumes the iterator; the continuity engine
post-processes the frames; `render.io` writes them to disk.

Renderers register themselves via `@register_renderer("name")`. The
shot's `renderer` field is the registration name. Renderers that wrap
existing GPU code (Taichi kernels, diffusion pipelines) should do their
heavy init lazily in `__init__` and free GPU memory in `close()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator

import numpy as np

from scene.manifest import Shot, Camera


_REGISTRY: dict[str, Callable[[], "Renderer"]] = {}


def register_renderer(name: str) -> Callable[[type], type]:
    def wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"renderer '{name}' already registered")
        _REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return wrap


def get_renderer(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown renderer '{name}'. Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_renderers() -> list[str]:
    return sorted(_REGISTRY.keys())


class Renderer(ABC):
    """Stateless from the director's point of view, but may hold GPU buffers."""

    def __init__(self, shot: Shot):
        self.shot = shot
        self.width, self.height = shot.resolution

    @abstractmethod
    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        """Render one scene-linear RGB float32 frame, shape (H, W, 3)."""

    def iter_frames(self) -> Iterator[np.ndarray]:
        n = self.shot.duration_frames
        for i in range(n):
            t = i / max(1, n - 1)
            cam = self.shot.camera.at(t)
            yield self.render_frame(i, t, cam)

    def close(self) -> None:
        pass


class ConstantRenderer(Renderer):
    """Trivial renderer used for tests / placeholders. Fills a solid color."""

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        c = self.shot.params.get("color", (0.05, 0.05, 0.08))
        img = np.empty((self.height, self.width, 3), dtype=np.float32)
        img[..., 0] = c[0]
        img[..., 1] = c[1]
        img[..., 2] = c[2]
        return img


register_renderer("constant")(ConstantRenderer)
