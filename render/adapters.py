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
        from render.zphc import ZPHCTransitionRenderer
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


@register_renderer("volumetric_gas_giant")
class VolumetricGasGiantRenderer(Renderer):
    """Volumetric Jupiter / gas-giant renderer with atmospheric scattering,
    multi-layer cloud composite, realistic banding, and parameterized
    storm features.

    Expected params:
        atm_thickness   : float, shell radius above planet (default 0.025)
        march_steps     : int, atmosphere ray-march steps (default 24)
        samples         : int, AA samples per pixel (default 2)
        tex_height      : int, baked texture height (default 512)
        tex_width       : int, baked texture width (default 1024)
        sun_dir         : (x, y, z), sun direction (default (0.707, 0.0, 0.707))
        grs_lon         : float, longitude of Great Red Spot (default 100.0)
        seed            : int, texture bake seed (default 0)
        push_in_factor  : float, how much closer the camera ends up at t=1
                          (default 1.0 = stay; 1.4 = +40% closer)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.volumetric_gas_giant import VolumetricGasGiantEngine
        self._engine = VolumetricGasGiantEngine(
            width=self.width,
            height=self.height,
            tex_height=int(shot.params.get("tex_height", 512)),
            tex_width=int(shot.params.get("tex_width", 1024)),
            atm_thickness=float(shot.params.get("atm_thickness", 0.025)),
            march_steps=int(shot.params.get("march_steps", 24)),
            samples=int(shot.params.get("samples", 2)),
            sun_dir=tuple(shot.params.get("sun_dir", (-0.55, 0.18, -0.81))),
            grs_lon=float(shot.params.get("grs_lon", 100.0)),
            seed=int(shot.params.get("seed", shot.seed)),
        )
        self._push_in = float(shot.params.get("push_in_factor", 1.0))

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        pos = np.asarray(camera.position, dtype=np.float32)
        if self._push_in != 1.0:
            scale = 1.0 / (1.0 + (self._push_in - 1.0) * t)
            pos = pos * scale
        self._engine.set_camera(
            position=tuple(pos.tolist()),
            target=tuple(camera.target),
            up_world=tuple(camera.up),
            fov_deg=float(camera.fov_deg),
        )
        self._engine.render(float(frame_idx) * 0.05)
        return self._engine.pixels.to_numpy().astype(np.float32)


@register_renderer("spiral_galaxy")
class SpiralGalaxyRenderer(Renderer):
    """Volumetric spiral-galaxy renderer (M51 / Whirlpool target).

    Pure procedural: no baked texture, so detail resolves at every zoom
    level via the footprint-aware noise toolkit in `render/noise.py`.

    Expected params (all optional, sensible defaults):
        R_disk, h_disk           : disk radial / vertical scales
        R_bulge, bulge_axis_ratio: bulge half-light radius and oblateness
        R_halo, h_halo           : extended halo ellipsoid
        n_arms                   : 2 = grand-design (M51), 4 = multi-arm
        pitch_deg                : log-spiral pitch (M51 ~ 12)
        arm_strength, arm_phase, arm_width
        dust_strength, dust_freq
        hii_threshold, hii_freq
        companion_pos            : (x, y, z) offset in galaxy radii
        companion_mass_ratio     : 0.0 = isolated, 0.5 = M51-like
        bridge_strength          : tidal bridge density
        inclination_deg          : 0 = face-on
        doppler_strength, doppler_sign
        march_steps, samples
        seed
        push_in_factor           : 1.0 = no zoom; >1 zooms in over the shot
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.spiral_galaxy import SpiralGalaxyEngine
        p = shot.params
        self._engine = SpiralGalaxyEngine(
            width=self.width,
            height=self.height,
            R_disk=float(p.get("R_disk", 1.0)),
            h_disk=float(p.get("h_disk", 0.05)),
            R_bulge=float(p.get("R_bulge", 0.18)),
            bulge_axis_ratio=float(p.get("bulge_axis_ratio", 0.85)),
            R_halo=float(p.get("R_halo", 1.6)),
            h_halo=float(p.get("h_halo", 0.45)),
            n_arms=int(p.get("n_arms", 2)),
            pitch_deg=float(p.get("pitch_deg", 12.0)),
            arm_strength=float(p.get("arm_strength", 0.85)),
            arm_phase=float(p.get("arm_phase", 0.0)),
            arm_width=float(p.get("arm_width", 0.45)),
            dust_strength=float(p.get("dust_strength", 2.2)),
            dust_freq=float(p.get("dust_freq", 10.0)),
            hii_threshold=float(p.get("hii_threshold", 0.22)),
            hii_freq=float(p.get("hii_freq", 14.0)),
            companion_pos=tuple(p.get("companion_pos", (1.05, 0.05, 0.35))),
            companion_mass_ratio=float(p.get("companion_mass_ratio", 0.5)),
            bridge_strength=float(p.get("bridge_strength", 0.6)),
            inclination_deg=float(p.get("inclination_deg", 20.0)),
            doppler_strength=float(p.get("doppler_strength", 0.06)),
            doppler_sign=float(p.get("doppler_sign", 1.0)),
            march_steps=int(p.get("march_steps", 72)),
            samples=int(p.get("samples", 2)),
            seed=int(p.get("seed", shot.seed)),
        )
        self._push_in = float(p.get("push_in_factor", 1.0))

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        pos = np.asarray(camera.position, dtype=np.float32)
        if self._push_in != 1.0:
            scale = 1.0 / (1.0 + (self._push_in - 1.0) * t)
            pos = pos * scale
        self._engine.set_camera(
            position=tuple(pos.tolist()),
            target=tuple(camera.target),
            up_world=tuple(camera.up),
            fov_deg=float(camera.fov_deg),
        )
        self._engine.render(float(frame_idx) * 0.05)
        return self._engine.pixels.to_numpy().astype(np.float32)


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
        # SphereCamera renders the Jovian shading conditionally on
        # `geo_engine.observer_attention`. The default of 0.0 produces a
        # green Pi-Lattice debug pattern instead of the textured planet
        # because the cast_ray composite is
        # `w_color * (1 - attention) + color * attention`. Force the
        # camera into the "observed" state so the cloud shader actually
        # runs end-to-end.
        self._cam.geo_engine.update_observer_attention(1.0)

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
