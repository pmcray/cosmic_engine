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

    The Kerr kernel builds its own camera from an observer distance and
    an orbital azimuth rather than a free camera, so the shot's Camera
    is read for its distance from the origin (giving a dolly when it has
    a `path_to`) and the rest of the move comes from the orbit and
    inclination sweeps below.

    Expected params:
        spin: float in [0, 1) — dimensionless a/M (default 0.7)
        inclination_deg: viewing angle from disk normal (default 85)
        inclination_deg_total: how far the inclination sweeps across the
            shot (default 0 — hold)
        disk_outer: outer disk radius in units of M (default 12)
        steps: integrator steps (default 220)
        step_size: affine step in units of M (default 0.12)
        orbit_deg: azimuth travelled across the shot (default 25)
        cam_dist: observer radius in units of M when the shot's camera
            sits at the origin (default 30)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        import math
        _ensure_taichi()
        from render.kerr import KerrRenderer
        p = shot.params
        side = max(self.width, self.height)
        self._incl0 = math.radians(float(p.get("inclination_deg", 85.0)))
        self._incl_sweep = math.radians(float(p.get("inclination_deg_total", 0.0)))
        self._orbit = math.radians(float(p.get("orbit_deg", 25.0)))
        self._default_dist = float(p.get("cam_dist", 30.0))
        self._r = KerrRenderer(
            res=side,
            spin=float(p.get("spin", 0.7)),
            inclination=self._incl0,
            disk_outer=float(p.get("disk_outer", 12.0)),
            steps=int(p.get("steps", 220)),
            step_size=float(p.get("step_size", 0.12)),
        )

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        dist = float(np.linalg.norm(np.asarray(camera.position, dtype=np.float64)))
        self._r.set_camera(
            distance=dist if dist > 1e-6 else self._default_dist,
            azimuth=t * self._orbit,
            inclination=self._incl0 + t * self._incl_sweep,
        )
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


@register_renderer("terrain")
class TerrainRenderer(Renderer):
    """Heightmap terrain via ridged fBM with sun, shadow, haze, snow line,
    optional water plane. Procedural detail at every scale.

    Expected params: terrain_amplitude, terrain_freq, octaves, snow_line,
    water_level, water_enable, sun_dir, haze_strength, max_dist,
    march_steps, samples, push_in_factor, seed.
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.terrain import TerrainEngine
        p = shot.params
        self._engine = TerrainEngine(
            width=self.width,
            height=self.height,
            terrain_amplitude=float(p.get("terrain_amplitude", 1.4)),
            terrain_freq=float(p.get("terrain_freq", 0.04)),
            octaves=int(p.get("octaves", 8)),
            snow_line=float(p.get("snow_line", 0.85)),
            water_level=float(p.get("water_level", -0.15)),
            water_enable=int(p.get("water_enable", 1)),
            sun_dir=tuple(p.get("sun_dir", (0.45, 0.55, -0.7))),
            haze_strength=float(p.get("haze_strength", 0.6)),
            max_dist=float(p.get("max_dist", 80.0)),
            march_steps=int(p.get("march_steps", 140)),
            samples=int(p.get("samples", 1)),
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


@register_renderer("ca_creatures")
class CACreaturesRenderer(Renderer):
    """Lenia-style continuous cellular-automata creatures.

    Expected params: grid_size; per-species A_R, A_kernel_mu, A_kernel_sigma,
    A_growth_mu, A_growth_sigma; same for B. Also dt, substeps_per_frame,
    warmup_steps, samples, seed.
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.ca_creatures import CACreaturesEngine
        p = shot.params
        self._engine = CACreaturesEngine(
            width=self.width,
            height=self.height,
            grid_size=int(p.get("grid_size", 256)),
            A_R=int(p.get("A_R", 13)),
            A_kernel_mu=float(p.get("A_kernel_mu", 0.5)),
            A_kernel_sigma=float(p.get("A_kernel_sigma", 0.15)),
            A_growth_mu=float(p.get("A_growth_mu", 0.15)),
            A_growth_sigma=float(p.get("A_growth_sigma", 0.015)),
            B_R=int(p.get("B_R", 24)),
            B_kernel_mu=float(p.get("B_kernel_mu", 0.55)),
            B_kernel_sigma=float(p.get("B_kernel_sigma", 0.12)),
            B_growth_mu=float(p.get("B_growth_mu", 0.12)),
            B_growth_sigma=float(p.get("B_growth_sigma", 0.020)),
            dt=float(p.get("dt", 0.1)),
            substeps_per_frame=int(p.get("substeps_per_frame", 2)),
            samples=int(p.get("samples", 1)),
            seed=int(p.get("seed", shot.seed)),
            warmup_steps=int(p.get("warmup_steps", 80)),
        )

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        # CA uses 2D pan/zoom from camera.position[0:2] (pan_x, pan_y)
        # and camera.position[2] negated as a zoom proxy.
        cp = camera.position
        zoom = max(0.1, 1.0 + cp[2] * 0.5)
        self._engine.set_camera(pan_x=float(cp[0]), pan_y=float(cp[1]), zoom=zoom)
        # Frame-addressable: Lenia is stateful, so the engine is asked
        # for a specific frame rather than told to take another step —
        # otherwise a re-render or a resumed chunk gets whatever state
        # the sim happened to reach.
        self._engine.render_at(frame_idx, float(frame_idx) * 0.05)
        return self._engine.pixels.to_numpy().astype(np.float32)


@register_renderer("exoplanet_atmosphere")
class ExoplanetAtmosphereRenderer(Renderer):
    """Exoplanet variants: hot_jupiter / mini_neptune / brown_dwarf."""

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.exoplanet_atmosphere import ExoplanetAtmosphereEngine
        p = shot.params
        self._engine = ExoplanetAtmosphereEngine(
            width=self.width,
            height=self.height,
            topology=str(p.get("topology", "hot_jupiter")),
            atm_thickness=float(p.get("atm_thickness", 0.07)),
            march_steps=int(p.get("march_steps", 30)),
            samples=int(p.get("samples", 2)),
            sun_dir=tuple(p.get("sun_dir", (-0.55, 0.18, -0.81))),
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


@register_renderer("cosmic_web")
class CosmicWebRenderer(Renderer):
    """Cosmic-web point-cloud fly-through. Renders the DESI tracer-class
    distribution (or a procedural fallback when no catalog is downloaded)
    splatted into a 3D emission + density grid, then volumetric ray-march
    with footprint-AA fine detail.

    Expected params:
        catalog_path     : str | null, .npz with positions/tracer/magnitudes
                           (if null, falls back to the procedural synth)
        box_size_mpc     : box half-extent (default 3000)
        grid_dim         : splat-grid resolution (default 128)
        n_synth_particles, n_synth_clusters, synth_seed
        march_steps      : ray-march sample count (default 128)
        samples          : AA samples (default 2)
        emission_gain    : multiplier on volumetric emission (default 4)
        extinction_strength
        background_gain
        push_in_factor   : cinematic zoom (default 1.0)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.cosmic_web import CosmicWebEngine
        from render.cosmic_pipeline import load_desi_npz
        p = shot.params
        positions = tracer = magnitudes = None
        catalog_path = p.get("catalog_path", None)
        if catalog_path:
            positions, tracer, magnitudes, _ = load_desi_npz(catalog_path)
        self._engine = CosmicWebEngine(
            width=self.width,
            height=self.height,
            positions=positions,
            tracer=tracer,
            magnitudes=magnitudes,
            box_size_mpc=float(p.get("box_size_mpc", 3000.0)),
            grid_dim=int(p.get("grid_dim", 128)),
            n_synth_particles=int(p.get("n_synth_particles", 120_000)),
            n_synth_clusters=int(p.get("n_synth_clusters", 400)),
            synth_seed=int(p.get("synth_seed", shot.seed)),
            march_steps=int(p.get("march_steps", 128)),
            samples=int(p.get("samples", 2)),
            emission_gain=float(p.get("emission_gain", 4.0)),
            extinction_strength=float(p.get("extinction_strength", 0.15)),
            background_gain=float(p.get("background_gain", 1.0)),
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


@register_renderer("saturn_class")
class SaturnClassRenderer(Renderer):
    """Saturn-class ringed gas-giant renderer: paler banded atmosphere,
    hexagonal north-polar vortex, full ring system (C, B, A + Cassini
    Division + Encke Gap + F ringlet), and bi-directional ring/planet
    shadow casting.

    Expected params:
        atm_thickness, march_steps, samples, sun_dir,
        hex_enable, hex_lat, hex_amp,
        rings_enable, ring_brightness, ring_shadows_enable,
        seed, push_in_factor
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.saturn_class import SaturnClassEngine
        p = shot.params
        self._engine = SaturnClassEngine(
            width=self.width,
            height=self.height,
            tex_height=int(p.get("tex_height", 512)),
            tex_width=int(p.get("tex_width", 1024)),
            atm_thickness=float(p.get("atm_thickness", 0.05)),
            march_steps=int(p.get("march_steps", 28)),
            samples=int(p.get("samples", 2)),
            sun_dir=tuple(p.get("sun_dir", (-0.55, 0.18, -0.81))),
            hex_enable=int(p.get("hex_enable", 1)),
            hex_lat=float(p.get("hex_lat", 78.0)),
            hex_amp=float(p.get("hex_amp", 0.45)),
            rings_enable=int(p.get("rings_enable", 1)),
            ring_brightness=float(p.get("ring_brightness", 1.0)),
            ring_shadows_enable=int(p.get("ring_shadows_enable", 1)),
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


@register_renderer("stellar_surface")
class StellarSurfaceRenderer(Renderer):
    """Sun-like / red-dwarf / brown-dwarf / O-star surface with
    limb darkening, two-scale Worley granulation, parametric sunspots,
    Halpha prominences, and a corona shell.

    Expected params:
        T_eff           : effective blackbody temperature (default 5800)
        limb_u          : linear limb-darkening coefficient (default 0.6)
        granule_scale   : Worley frequency for granules (default 22)
        supergranule_scale : Worley frequency for supergranules (4.5)
        corona_thickness : float (default 0.04)
        sunspots        : list of [lat, lon, radius_deg, darkness]
        prominences     : list of [lat, lon, height, length_deg]
        samples, seed, push_in_factor
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.stellar_surface import StellarSurfaceEngine
        p = shot.params
        self._engine = StellarSurfaceEngine(
            width=self.width,
            height=self.height,
            T_eff=float(p.get("T_eff", 5800.0)),
            limb_u=float(p.get("limb_u", 0.6)),
            granule_scale=float(p.get("granule_scale", 22.0)),
            supergranule_scale=float(p.get("supergranule_scale", 4.5)),
            corona_thickness=float(p.get("corona_thickness", 0.04)),
            sunspots=tuple(tuple(s) for s in p.get("sunspots", ())),
            prominences=tuple(tuple(s) for s in p.get("prominences", ())),
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


@register_renderer("diffuse_nebula")
class DiffuseNebulaRenderer(Renderer):
    """Volumetric diffuse-nebula renderer with six topology classes.

    Expected params:
        topology         : "pillars" | "crab" | "helix" | "veil"
                            | "orion" | "pleiades"
        R_bound          : bounding-sphere radius (default 1.5)
        intensity        : emission multiplier (default 1.0)
        dust_strength    : extinction multiplier (default 1.0)
        detail_strength  : procedural-detail amplitude (default 1.0)
        erosion_height, erosion_decay     : pillars-specific
        R_shell, shell_thickness          : crab/veil-specific
        R_torus, r_torus                  : helix-specific
        march_steps      : ray-march sample count (default 80)
        samples          : AA samples per pixel (default 2)
        seed             : int (default shot.seed)
        push_in_factor   : cinematic zoom (default 1.0)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.diffuse_nebula import DiffuseNebulaEngine
        p = shot.params
        self._engine = DiffuseNebulaEngine(
            width=self.width,
            height=self.height,
            topology=str(p.get("topology", "pillars")),
            R_bound=float(p.get("R_bound", 1.5)),
            intensity=float(p.get("intensity", 1.0)),
            dust_strength=float(p.get("dust_strength", 1.0)),
            detail_strength=float(p.get("detail_strength", 1.0)),
            erosion_height=float(p.get("erosion_height", 0.2)),
            erosion_decay=float(p.get("erosion_decay", 2.5)),
            R_shell=float(p.get("R_shell", 1.0)),
            shell_thickness=float(p.get("shell_thickness", 0.18)),
            R_torus=float(p.get("R_torus", 0.7)),
            r_torus=float(p.get("r_torus", 0.18)),
            march_steps=int(p.get("march_steps", 80)),
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


# ---- Odyssey Director acts as graph shots ----
#
# The three acts of odyssey_director.py, wrapped so the voyage composer
# can splice them into any sequence. The choreography curves live in
# render.odyssey_acts (pure Python, CPU-testable); these classes bind
# them to the Taichi engines. SphereCamera / StargateCorridor fields are
# (x, y) bottom-up, so frames are reoriented to the graph's (H, W, 3)
# top-down contract.


def _xy_to_rows(img: np.ndarray) -> np.ndarray:
    """(W, H, 3) x,y bottom-up Taichi field -> (H, W, 3) top-down frame."""
    return np.ascontiguousarray(np.flipud(img.transpose(1, 0, 2))).astype(np.float32)


@register_renderer("odyssey_jovian_approach")
class OdysseyJovianApproachRenderer(Renderer):
    """Act I of the Odyssey Director: photoreal Jovian fluid dynamics,
    a slow accelerating dolly toward the cloud deck, the sun sinking into
    a backlit crescent, optional white-out flash at the end.

    Expected params (all optional):
        fluid_res        : fluid grid resolution (default 256)
        samples          : AA samples per pixel (default 2)
        planet_type      : FluidEngine profile (default "jupiter")
        prewarm_steps    : fluid steps before frame 0 (default 100)
        vorticity_strength, band_freq, wind_mult, shear_mult,
        has_spot, has_pearls, meridional_damping
                         : fluid tuning (defaults = Odyssey Act I values)
        flash            : 0/1 — keep the terminal white-out (default 1)
        plus every keyword of render.odyssey_acts.jovian_approach_pose
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from physics.fluid_solver import FluidEngine
        from render.camera import SphereCamera
        from render import odyssey_acts

        self._acts = odyssey_acts
        p = shot.params
        fluid_res = int(p.get("fluid_res", 256))
        self._fluid = FluidEngine(res=fluid_res, dt=0.004,
                                  planet_type=p.get("planet_type", "jupiter"),
                                  seed=int(p.get("seed", shot.seed)))
        self._fluid.vorticity_strength = float(p.get("vorticity_strength", 2.0))
        self._fluid.band_freq = float(p.get("band_freq", 12.0))
        self._fluid.wind_mult = float(p.get("wind_mult", 1.5))
        self._fluid.shear_mult = float(p.get("shear_mult", 0.5))
        self._fluid.has_spot = int(p.get("has_spot", 1))
        self._fluid.has_pearls = int(p.get("has_pearls", 1))
        self._fluid.meridional_damping = float(p.get("meridional_damping", 0.9))

        self._cam = SphereCamera(fluid_res=fluid_res,
                                 render_res=max(self.width, self.height),
                                 has_rings=0,
                                 samples=int(p.get("samples", 2)),
                                 render_w=self.width, render_h=self.height)
        # Photoreal clouds, not the pi-lattice debug view.
        self._cam.geo_engine.update_observer_attention(1.0)

        self._pose_kwargs = {
            k: float(p[k]) for k in (
                "dist_start", "dist_fall", "dist_gamma", "pan_total",
                "tilt_start", "tilt_rise", "roll_onset", "roll_max",
                "sun_azimuth_offset", "sun_azimuth_sweep",
                "sun_ly_start", "sun_ly_fall", "flash_onset", "flash_gain",
            ) if k in p
        }
        if not int(p.get("flash", 1)):
            self._pose_kwargs["flash_gain"] = 0.0

        self._fluid_frame = 0
        for _ in range(int(p.get("prewarm_steps", 100))):
            self._fluid.step(self._fluid_frame)
            self._fluid_frame += 1
        self._prewarm = self._fluid_frame

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        # Keep the stateful sim in lockstep with the frame index so chunked
        # or resumed renders see the same atmosphere as a continuous run.
        while self._fluid_frame < self._prewarm + frame_idx:
            self._fluid.step(self._fluid_frame)
            self._fluid_frame += 1

        pose = self._acts.jovian_approach_pose(t, **self._pose_kwargs)
        cam = self._cam
        cam.cam_dist = pose["cam_dist"]
        cam.cam_pan = pose["cam_pan"]
        cam.cam_tilt = pose["cam_tilt"]
        cam.cam_roll = pose["cam_roll"]
        cam.exposure = pose["exposure"]
        lx, ly, lz = pose["sun_dir"]

        time_arg = float(frame_idx) / max(1, self.shot.fps)
        cam.render_gas_giant(self._fluid.dye, lx, ly, lz, time_arg)
        return _xy_to_rows(cam.get_image_data())


@register_renderer("stargate_corridor")
class StargateCorridorRenderer(Renderer):
    """Act II of the Odyssey Director: the widescreen slit-scan light
    corridor with colour epochs and the horizontal-to-vertical tumble.

    Expected params (all optional):
        samples : AA samples per pixel (default 2)
        plus every keyword of render.odyssey_acts.corridor_pose
        (tumble_start, tumble_width, tumble_angle, entry_flash,
        flash_decay, base_exposure)
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.stargate_corridor import StargateCorridor
        from render import odyssey_acts

        self._acts = odyssey_acts
        p = shot.params
        self._corridor = StargateCorridor(render_w=self.width,
                                          render_h=self.height,
                                          samples=int(p.get("samples", 2)))
        self._pose_kwargs = {
            k: float(p[k]) for k in (
                "tumble_start", "tumble_width", "tumble_angle",
                "entry_flash", "flash_decay", "base_exposure",
            ) if k in p
        }

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        pose = self._acts.corridor_pose(t, **self._pose_kwargs)
        time_arg = float(frame_idx) / max(1, self.shot.fps)
        self._corridor.render_frame(time_arg, pose["progress"],
                                    pose["roll"], pose["exposure"])
        img = np.clip(self._corridor.pixels.to_numpy(), 0.0, 1.0)
        return _xy_to_rows(img)


@register_renderer("odyssey_infinite")
class OdysseyInfiniteRenderer(Renderer):
    """Act III of the Odyssey Director: the gravitationally lensed black
    hole, a slow fall toward the event horizon with the mass ramping and
    a final fade to black.

    Expected params (all optional):
        samples  : AA samples per pixel (default 2)
        fluid_res: SphereCamera dye-texture resolution — unused by the
                   black-hole path, kept small (default 128)
        bh_steps : geodesic march steps (default 320)
        bh_dt    : geodesic step size (default 0.08)
        fade     : 0/1 — keep the terminal fade to black (default 1)
        plus every keyword of render.odyssey_acts.infinite_pose
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        _ensure_taichi()
        from render.camera import SphereCamera
        from render import odyssey_acts

        self._acts = odyssey_acts
        p = shot.params
        self._cam = SphereCamera(fluid_res=int(p.get("fluid_res", 128)),
                                 render_res=max(self.width, self.height),
                                 has_rings=0,
                                 samples=int(p.get("samples", 2)),
                                 render_w=self.width, render_h=self.height)
        # March budget is baked into the kernel at first compile.
        self._cam.bh_steps = int(p.get("bh_steps", 320))
        self._cam.bh_dt = float(p.get("bh_dt", 0.08))
        self._pose_kwargs = {
            k: float(p[k]) for k in (
                "dist_start", "dist_fall", "dist_gamma", "tilt_start",
                "tilt_rise", "pan_total", "mass_start", "mass_ramp",
                "fade_onset",
            ) if k in p
        }
        if not int(p.get("fade", 1)):
            self._pose_kwargs["fade_onset"] = 2.0  # never reached in [0, 1]

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        pose = self._acts.infinite_pose(t, **self._pose_kwargs)
        cam = self._cam
        cam.cam_dist = pose["cam_dist"]
        cam.cam_tilt = pose["cam_tilt"]
        cam.cam_pan = pose["cam_pan"]
        cam.cam_roll = pose["cam_roll"]
        cam.exposure = pose["exposure"]
        cam.geo_engine.update_black_hole_mass(pose["bh_mass"])

        time_arg = float(frame_idx) / max(1, self.shot.fps)
        cam.render_black_hole(time_arg)
        return _xy_to_rows(cam.get_image_data())


@register_renderer("fractal_dive")
class FractalDiveRenderer(Renderer):
    """Mandelbrot / Multibrot deep dive via perturbation theory
    (render.fractal_dive). Exponential zoom from scale_start to
    scale_end over the shot; fully deterministic (fixed supersampling
    grid, no RNG). Runs on the Taichi f64 engine when Taichi is
    available, else on the vectorised numpy engine.

    Expected params (all optional):
        center_re, center_im : dive centre as decimal strings — supply
                               enough digits for the target depth
        power                : Multibrot exponent d >= 2 (default 2)
        scale_start          : viewport half-height at t=0 (default 1.8)
        scale_end            : ... at t=1 (default 1e-12)
        rotation_deg_start   : frame rotation at t=0 (default 0)
        rotation_deg_total   : additional rotation across the shot
                               (default 120 — the slow corkscrew)
        samples              : supersampling grid side (default 2 = 4x)
        escape_radius        : bailout radius (default 64)
        iter_base, iter_per_decade : iteration budget curve
        color_cycles, color_phase  : palette cycling controls — how many
                               times the palette repeats across the
                               frame's escape-count range, and where it
                               starts
        interior_color       : RGB for non-escaping points (default black)
        engine               : "auto" | "taichi" | "numpy"
    """

    def __init__(self, shot: Shot):
        super().__init__(shot)
        from render import fractal_dive as fd
        from scene.palette import get_palette

        p = shot.params
        self._fd = fd
        self._power = int(p.get("power", 2))
        self._scale_start = float(p.get("scale_start", 1.8))
        self._scale_end = float(p.get("scale_end", 1e-12))
        self._rot0 = float(p.get("rotation_deg_start", 0.0)) * np.pi / 180.0
        self._rot_total = float(p.get("rotation_deg_total", 120.0)) * np.pi / 180.0
        self._samples = max(1, int(p.get("samples", 2)))
        self._escape_radius = float(p.get("escape_radius", 64.0))
        self._iter_base = int(p.get("iter_base", 600))
        self._iter_per_decade = int(p.get("iter_per_decade", 800))
        self._color_cycles = float(p.get("color_cycles", 4.0))
        self._color_phase = float(p.get("color_phase", 0.0))
        self._interior = tuple(p.get("interior_color", (0.0, 0.0, 0.0)))
        self._anchors = get_palette(shot.palette.name).anchors

        deepest = min(self._scale_start, self._scale_end)
        budget = fd.iter_budget(deepest, self._iter_base, self._iter_per_decade)
        self._orbit = fd.ReferenceOrbit(
            str(p.get("center_re", fd.DEFAULT_CENTER_RE)),
            str(p.get("center_im", fd.DEFAULT_CENTER_IM)),
            power=self._power,
            max_iter=budget,
            precision=fd.required_precision(deepest),
        )

        engine = p.get("engine", "auto")
        self._engine = None
        if engine in ("auto", "taichi") and fd._HAS_TAICHI:
            _ensure_taichi()
            self._engine = fd.FractalDiveEngine(self.width, self.height,
                                                self._orbit)
        elif engine == "taichi":
            raise RuntimeError("fractal_dive: taichi engine requested "
                               "but taichi is not installed")

    def render_frame(self, frame_idx: int, t: float, camera: Camera) -> np.ndarray:
        fd = self._fd
        scale = fd.dive_scale(t, self._scale_start, self._scale_end)
        rotation = self._rot0 + t * self._rot_total
        max_iter = fd.iter_budget(scale, self._iter_base,
                                  self._iter_per_decade)

        acc = np.zeros((self.height, self.width, 3), dtype=np.float64)
        offsets = fd.supersample_offsets(self._samples)
        for (dx, dy) in offsets:
            if self._engine is not None:
                mu = self._engine.compute_mu(scale, rotation, max_iter,
                                             self._escape_radius, dx, dy)
            else:
                dc = fd.pixel_deltas(self.width, self.height, scale,
                                     rotation, dx, dy)
                mu = fd.perturbation_grid(self._orbit, dc, max_iter,
                                          self._escape_radius)
            acc += fd.colorize(mu, self._anchors, self._color_cycles,
                               self._color_phase, self._interior)
        return (acc / len(offsets)).astype(np.float32)


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
