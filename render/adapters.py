"""Adapters that present existing renderers to the new graph interface.

Existing engines (Taichi-based exotic physics, gas-giant camera, ZPHC
slit-scan tunnel) keep their original APIs. The adapters here wrap each
one into the `render.core.Renderer` contract so the graph executor can
drive them uniformly.

All taichi imports are deferred to instance construction so that the
graph schema itself remains importable on a CPU-only host.
"""
from __future__ import annotations

import numpy as np

from scene.manifest import Shot, Camera
from .core import Renderer, register_renderer


def _ensure_taichi():
    import taichi as ti
    if not getattr(_ensure_taichi, "_initialized", False):
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
        _ensure_taichi._initialized = True
    return ti


@register_renderer("exotic_physics")
class ExoticPhysicsRenderer(Renderer):
    """Wraps `encounters.exotic_physics.ExoticPhysicsEncounter`.

    Expected params:
        encounter_type: "blackhole" | "hypernova"
        samples: int (default 4)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from encounters.exotic_physics import ExoticPhysicsEncounter
        side = max(self.width, self.height)
        self._enc = ExoticPhysicsEncounter(
            encounter_type=shot.params.get("encounter_type", "blackhole"),
            res=side,
            samples=int(shot.params.get("samples", 4)),
        )

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        self._enc.frame = frame_idx
        self._enc.step()
        img = self._enc.pixels.to_numpy().astype(np.float32)
        return _resize_to(img, self.height, self.width)


@register_renderer("slitscan_tunnel")
class SlitScanTunnelRenderer(Renderer):
    """Wraps `infinite_director.ZPHCTransitionRenderer` as a standalone shot."""

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from infinite_director import ZPHCTransitionRenderer
        side = max(self.width, self.height)
        self._r = ZPHCTransitionRenderer(res=side)

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        time_arg = float(shot_param(self.shot, "time_scale", 0.05)) * frame_idx
        self._r.render_frame(time_arg, t)
        img = self._r.pixels.to_numpy().astype(np.float32)
        return _resize_to(img, self.height, self.width)


@register_renderer("kerr_black_hole")
class KerrBlackHoleRenderer(Renderer):
    """Wraps `render.kerr.KerrRenderer`.

    Expected params:
        spin: float in [0, 1) — dimensionless a/M (default 0.7)
        inclination_deg: viewing angle from disk normal (default 85)
        disk_outer: outer disk radius in units of M (default 12)
        steps: integrator steps (default 220)
        step_size: affine step in units of M (default 0.12)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        import math
        _ensure_taichi()
        from render.kerr import KerrRenderer
        side = max(self.width, self.height)
        self._r = KerrRenderer(
            res=side,
            spin=float(shot.params.get("spin", 0.7)),
            inclination=math.radians(float(shot.params.get("inclination_deg", 85.0))),
            disk_outer=float(shot.params.get("disk_outer", 12.0)),
            steps=int(shot.params.get("steps", 220)),
            step_size=float(shot.params.get("step_size", 0.12)),
        )

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        self._r.render_frame(float(frame_idx) * 0.05)
        img = self._r.pixels.to_numpy().astype(np.float32)
        return _resize_to(img, self.height, self.width)


@register_renderer("gas_giant")
class GasGiantRenderer(Renderer):
    """Wraps `render.camera.SphereCamera.render_gas_giant` with a fluid sim.

    Expected params:
        prewarm_frames: int (default 60) — initial fluid sim ticks
        rings: 0 | 1
        light_dir: (x,y,z) (default (0.707, 0.0, 0.707))
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.camera import SphereCamera
        from physics.fluid_solver import FluidEngine
        side = max(self.width, self.height)
        self._fluid = FluidEngine(res=512)
        self._cam = SphereCamera(
            fluid_res=512,
            render_res=side,
            has_rings=int(shot.params.get("rings", 0)),
            samples=int(shot.params.get("samples", 4)),
        )
        for f in range(int(shot.params.get("prewarm_frames", 60))):
            self._fluid.step(f)
        self._sim_frame = 0

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        self._fluid.step(self._sim_frame)
        self._sim_frame += 1
        lx, ly, lz = self.shot.params.get("light_dir", (0.707, 0.0, 0.707))
        self._cam.cam_pan = float(self.shot.params.get("pan", 0.0)) + t * float(
            self.shot.params.get("pan_rate", 0.0)
        )
        self._cam.cam_tilt = float(self.shot.params.get("tilt", -0.3))
        self._cam.render_gas_giant(self._fluid.dye, lx, ly, lz, float(frame_idx) * 0.016)
        img = self._cam.final_output.to_numpy().astype(np.float32)
        return _resize_to(img, self.height, self.width)


# ---- helpers ----

def shot_param(shot: Shot, key: str, default):
    return shot.params.get(key, default)


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Cheap nearest-neighbor resize. Replace with a sinc filter once a
    proper resampler is in. Renderer outputs are typically square; the
    graph asks for the shot's resolution."""
    ih, iw = img.shape[:2]
    if ih == h and iw == w:
        return img
    ys = (np.linspace(0, ih - 1, h)).astype(np.int32)
    xs = (np.linspace(0, iw - 1, w)).astype(np.int32)
    return img[ys[:, None], xs[None, :]]
