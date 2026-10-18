"""Deterministic Stratified Flow Renderer (`render.stratified_flow`).

Implements two coupled multi-phase fluid / granular physical simulations:

1. `mode='kelvin_helmholtz'` (Default — The Rocking Two-Liquid Wave Machine):
   * 2D two-layer immiscible Boussinesq fluid driven by a slow harmonic
     rocking gravity vector:
         g(t) = g0 * (sin(theta(t)), -cos(theta(t))),
         theta(t) = theta_max * sin(omega * t + phi_0)
     plus baroclinic vorticity generation (grad(rho) x g(t)) along the
     interface and desingularized Birkhoff-Rott vortex-sheet roll-up.
   * Forms long internal gravity swells that steepen and roll up into
     crisp, multi-turn Kelvin-Helmholtz billow spirals without diffusing
     into numerical mush. Rendered with refractive fluid shading,
     specular interface highlights, and shot palette mapping.

2. `mode='sand_strata'` (The Invertible Falling-Sand Picture):
   * Multi-species granular-fluid sedimentation after frame inversion:
     4 distinct grain classes (heavy dark basalt, dense crimson garnet,
     mid-density amber quartz, fine pale silica) varying in density,
     terminal settling velocity, angle of repose, and palette color.
   * Grains fall through a viscous fluid column obstructed by a rising,
     coalescing line of buoyant air bubbles. As plumes breach gaps between
     bubbles, they accumulate onto a bottom heightfield pile that undergoes
     deterministic angle-of-repose avalanche relaxation (Self-Organized
     Criticality) to build sharply stratified, cross-bedded geological
     layers over time.

3. Determinism, Frame-Addressability & MetaPost Vector Export:
   * All micro-perturbations and bubble/grain seeding are keyed on
     `render/detrng.py` (`py_rand01`, `py_hash_u32`) and `--seed`.
   * Supports deterministic state stepping from frame 0 to frame `frame_idx`
     with cached state for sequential rendering.
   * `export_metapost(state, out_path, scale_bp=240.0)` emits a valid,
     self-contained MetaPost figure (`prologues := 3; beginfig(1); ... endfig;`)
     of the overturning Kelvin-Helmholtz interface contour or the stacked
     sedimentary horizon polylines.
"""
from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from render.detrng import py_hash_u32, py_rand01
from render.fractal_dive import (
    _HAS_TAICHI,
    arch_supports_f64,
    ensure_taichi_f64,
    get_taichi_arch,
    supersample_offsets,
)

try:
    import taichi as ti
except Exception:  # pragma: no cover
    ti = None


VALID_MODES = ("kelvin_helmholtz", "sand_strata")

# Salt constants for deterministic RNG streams via render.detrng
S_KH_MODE = 0x13579BDF
S_KH_PHASE = 0x2468ACE1
S_SAND_BUBBLE = 0x31415927
S_SAND_PLUME = 0x27182818
S_SAND_GRAIN = 0x16180339


# ============================================================
# Palette Helpers
# ============================================================

def resolve_palette_anchors(palette) -> np.ndarray:
    """Resolve a palette name, `Palette` instance, or anchor sequence to an (M, 3) float64 array."""
    if isinstance(palette, str):
        from scene.palette import get_palette
        anchors = get_palette(palette).anchors
    elif hasattr(palette, "anchors"):
        anchors = palette.anchors
    else:
        anchors = palette
    arr = np.asarray(anchors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        raise ValueError("palette must provide at least 2 RGB anchors of shape (M, 3)")
    return np.clip(arr, 0.0, 1.0)


def sample_palette(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of scalar `values` in [0, 1] through `anchors` (M, 3)."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    m = anchors.shape[0]
    scaled = v * (m - 1)
    idx = np.minimum(np.floor(scaled).astype(np.int32), m - 2)
    frac = (scaled - idx)[..., None]
    return anchors[idx] + (anchors[idx + 1] - anchors[idx]) * frac


def _det_noise_1d(n: int, seed: int, salt: int) -> np.ndarray:
    """Deterministic 1D array of uniform [0, 1) floats from `render.detrng.py_rand01`."""
    return np.array(
        [py_rand01(i, int(seed), int(salt), i * 17 + 3) for i in range(int(n))],
        dtype=np.float64,
    )


def _det_hash_2d(ix: np.ndarray, iy: np.ndarray, seed: int, salt: int = S_SAND_GRAIN) -> np.ndarray:
    """Vectorized 32-bit PCG hash matching `render.detrng._py_pcg` for 2D pixel grain."""
    m32 = np.uint64(0xFFFFFFFF)
    c1 = np.uint64(747796405)
    c2 = np.uint64(2891336453)
    c3 = np.uint64(277803737)

    def pcg(x: np.ndarray) -> np.ndarray:
        state = (x * c1 + c2) & m32
        shift = (state >> np.uint64(28)) + np.uint64(4)
        word = (((state >> shift) ^ state) * c3) & m32
        return ((word >> np.uint64(22)) ^ word) & m32

    a = ix.astype(np.int64) & 0xFFFFFFFF
    b = iy.astype(np.int64) & 0xFFFFFFFF
    c = np.uint64(int(seed) & 0xFFFFFFFF)
    d = np.uint64(int(salt) & 0xFFFFFFFF)
    h = pcg(a.astype(np.uint64))
    h = pcg(h ^ b.astype(np.uint64))
    h = pcg(h ^ c)
    h = pcg(h ^ d)
    return h.astype(np.float64) / 4294967296.0


# ============================================================
# Simulation State Data Structures
# ============================================================

@dataclass
class GrainSpecies:
    """Physical properties of a granular sediment class in `sand_strata` mode."""
    name: str
    density: float            # g/cm^3
    settling_velocity: float  # terminal fall speed in normalized units/step
    repose_deg: float         # angle of repose in degrees
    palette_coord: float      # coordinate in [0, 1] along active shot palette


DEFAULT_GRAIN_SPECIES: tuple[GrainSpecies, ...] = (
    GrainSpecies("basalt", density=3.15, settling_velocity=0.022, repose_deg=39.0, palette_coord=0.14),
    GrainSpecies("garnet", density=2.80, settling_velocity=0.017, repose_deg=34.5, palette_coord=0.42),
    GrainSpecies("amber_quartz", density=2.45, settling_velocity=0.013, repose_deg=30.0, palette_coord=0.72),
    GrainSpecies("pale_silica", density=2.10, settling_velocity=0.009, repose_deg=25.5, palette_coord=0.94),
)


@dataclass
class StratifiedFlowState:
    """Complete deterministic state of the stratified flow simulation at a given frame."""
    mode: str
    frame_idx: int
    time: float
    seed: int

    # Mode 1: kelvin_helmholtz state
    kh_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kh_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kh_gamma: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kh_S: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kh_centers_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kh_centers_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kh_circ: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    gravity_angle: float = 0.0
    gravity_vec: tuple[float, float] = (0.0, -1.0)

    # Mode 2: sand_strata state
    grid_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    horizons: list[np.ndarray] = field(default_factory=list)         # stacked heightfields h_0(x) <= h_1(x) <= ...
    horizon_species: list[int] = field(default_factory=list)         # grain species index per stratum
    reservoir_top: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    reservoir_bot: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    bubbles: list[tuple[float, float, float]] = field(default_factory=list)  # (bx, by, br)
    plumes: list[dict] = field(default_factory=list)                 # active falling plumes


# ============================================================
# Mode 1: Kelvin-Helmholtz Rocking Two-Liquid Wave Machine
# ============================================================

def rocking_gravity(t_sec: float,
                    g0: float = 1.0,
                    theta_max_deg: float = 14.0,
                    omega: float = 1.35,
                    phi0: float = 0.25) -> tuple[float, tuple[float, float]]:
    """Compute the rocking tilt angle `theta(t)` and gravity vector `g(t)`:
        theta(t) = theta_max * sin(omega * t + phi0)
        g(t) = g0 * (sin(theta(t)), -cos(theta(t)))
    """
    theta_max = math.radians(float(theta_max_deg))
    theta = theta_max * math.sin(float(omega) * float(t_sec) + float(phi0))
    gx = float(g0) * math.sin(theta)
    gy = -float(g0) * math.cos(theta)
    return theta, (gx, gy)


def _init_kelvin_helmholtz_state(seed: int = 42,
                                 n_points: int = 480,
                                 n_billows: int = 3,
                                 atwood: float = 0.18,
                                 shear_velocity: float = 0.52) -> StratifiedFlowState:
    """Initialize the two-layer immiscible Boussinesq interface and baroclinic vortex sheet."""
    s_param = np.linspace(0.0, 1.0, int(n_points), endpoint=False, dtype=np.float64)
    # Deterministic phase & micro-perturbation keyed on render.detrng
    phase_main = 2.0 * math.pi * py_rand01(seed, 1, S_KH_PHASE, 0)
    phase_sub = 2.0 * math.pi * py_rand01(seed, 2, S_KH_PHASE, 1)
    jitter = (_det_noise_1d(n_points, seed, S_KH_MODE) - 0.5) * 0.0025

    k_main = 2.0 * math.pi * float(n_billows)
    # Long internal swell + initial Kelvin-Helmholtz eigenmode perturbation
    y0 = (
        0.50
        + 0.022 * np.sin(2.0 * math.pi * s_param + phase_sub)
        + 0.016 * np.sin(k_main * s_param + phase_main)
        + jitter
    )
    x0 = s_param + 0.008 * np.cos(k_main * s_param + phase_main)

    # Initial vortex sheet circulation density gamma(s) from shear + baroclinic torque
    gamma0 = float(shear_velocity) * (1.0 + 0.35 * np.cos(k_main * s_param + phase_main))

    # Primary billow vortex cores along the wave crests
    crest_s = np.array(
        [(float(m) + 0.5 - phase_main / (2.0 * math.pi)) / float(n_billows) for m in range(int(n_billows))],
        dtype=np.float64,
    ) % 1.0
    crest_y = 0.50 + 0.022 * np.sin(2.0 * math.pi * crest_s + phase_sub)
    circ0 = np.full(int(n_billows), float(shear_velocity) / float(n_billows), dtype=np.float64)

    theta0, gvec0 = rocking_gravity(0.0)
    return StratifiedFlowState(
        mode="kelvin_helmholtz",
        frame_idx=0,
        time=0.0,
        seed=int(seed),
        kh_x=x0,
        kh_y=y0,
        kh_gamma=gamma0,
        kh_S=s_param.copy(),
        kh_centers_x=crest_s,
        kh_centers_y=crest_y,
        kh_circ=circ0,
        gravity_angle=theta0,
        gravity_vec=gvec0,
    )


def _step_kelvin_helmholtz(state: StratifiedFlowState,
                           dt: float = 0.025,
                           atwood: float = 0.18,
                           viscosity: float = 0.004,
                           shear_velocity: float = 0.52,
                           theta_max_deg: float = 14.0,
                           rock_omega: float = 1.35) -> None:
    """Advance the Boussinesq two-liquid interface by one deterministic timestep `dt`.

    Combines:
      1. Rocking gravity vector g(t) = g0 * (sin(theta(t)), -cos(theta(t)))
         driving a large-scale internal gravity sloshing swell.
      2. Baroclinic vorticity generation d(gamma)/dt = 2 * A_T * (g(t) . ds) - nu * gamma
         along the interface.
      3. Desingularized periodic Birkhoff-Rott / Stuart cat's-eye vortex roll-up
         so the interface steepens and winds into crisp multi-turn Kelvin-Helmholtz
         billow spirals without numerical diffusion.
    """
    t_next = state.time + float(dt)
    theta, (gx, gy) = rocking_gravity(t_next, theta_max_deg=theta_max_deg, omega=rock_omega)
    state.gravity_angle = theta
    state.gravity_vec = (gx, gy)

    x = state.kh_x
    y = state.kh_y
    gamma = state.kh_gamma

    # Tangent vector along the interface for baroclinic torque grad(rho) x g(t)
    dx = np.roll(x, -1) - np.roll(x, 1)
    dx = np.where(dx < -0.5, dx + 1.0, np.where(dx > 0.5, dx - 1.0, dx))
    dy = np.roll(y, -1) - np.roll(y, 1)
    ds_norm = np.maximum(1e-6, np.hypot(dx, dy))
    tx = dx / ds_norm
    ty = dy / ds_norm

    # Baroclinic vorticity generation: (grad(rho) x g)_z ~ -2 * A_T * (g_x * tx + g_y * ty)
    baroclinic = -2.0 * float(atwood) * (gx * tx + gy * ty)
    gamma = (1.0 - float(viscosity) * dt) * gamma + dt * baroclinic

    # Drift billow vortex centers with rocking gravity swell
    n_billows = len(state.kh_centers_x)
    swell_drift = 0.08 * gx * dt
    state.kh_centers_x = (state.kh_centers_x + swell_drift) % 1.0
    state.kh_centers_y = 0.50 + 0.045 * math.sin(theta) * np.sin(2.0 * math.pi * state.kh_centers_x + 0.4)
    # Baroclinic amplification of billow circulation
    state.kh_circ = state.kh_circ + dt * float(atwood) * (0.32 * abs(gy) + 0.45 * abs(gx)) * (
        1.0 / max(1, n_billows)
    ) * np.exp(-0.35 * state.time)

    # Evaluate the velocity field at every Lagrangian interface marker (x, y)
    # from the periodic vortex array + background Boussinesq shear and rocking swell
    u = np.zeros_like(x)
    v = np.zeros_like(y)

    # Rocking internal gravity sloshing swell
    u += 0.22 * math.sin(theta) * np.tanh((y - 0.50) * 8.0)
    v += 0.06 * gx * np.cos(2.0 * math.pi * x)

    # Desingularized periodic vortex-sheet + billow core velocity (Krasny / Stuart kernel)
    delta_core = 0.052
    for m in range(n_billows):
        cx = state.kh_centers_x[m]
        cy = state.kh_centers_y[m]
        circ = state.kh_circ[m] * (1.0 + 0.85 * float(shear_velocity))
        # Periodic horizontal offset in [-0.5, 0.5)
        rx = (x - cx + 0.5) % 1.0 - 0.5
        ry = y - cy
        r2 = rx * rx + ry * ry
        # Smooth localized swirl kernel that winds markers into multi-turn spirals
        # while decaying cleanly away from the interface
        r_env = np.exp(-r2 / (2.0 * (0.145 / max(1.0, 0.55 * n_billows)) ** 2))
        omega_loc = 1.65 * circ / (r2 + delta_core * delta_core) * r_env
        # Elliptical shear strain that stretches the braids between billows
        u += -omega_loc * ry + 0.28 * float(shear_velocity) * np.tanh(ry * 9.0) * (1.0 - 0.55 * r_env)
        v += omega_loc * rx * 0.72

    # Secondary harmonic billow modulation from gamma(s)
    u += 0.025 * (gamma - np.mean(gamma)) * np.sin(2.0 * math.pi * (y - 0.5))

    # RK2 midpoint advection step for crisp spiral trajectories
    x_mid = x + 0.5 * dt * u
    y_mid = np.clip(y + 0.5 * dt * v, 0.06, 0.94)

    u_mid = 0.22 * math.sin(theta) * np.tanh((y_mid - 0.50) * 8.0)
    v_mid = 0.06 * gx * np.cos(2.0 * math.pi * x_mid)
    for m in range(n_billows):
        cx = state.kh_centers_x[m]
        cy = state.kh_centers_y[m]
        circ = state.kh_circ[m] * (1.0 + 0.85 * float(shear_velocity))
        rx = (x_mid - cx + 0.5) % 1.0 - 0.5
        ry = y_mid - cy
        r2 = rx * rx + ry * ry
        r_env = np.exp(-r2 / (2.0 * (0.145 / max(1.0, 0.55 * n_billows)) ** 2))
        omega_loc = 1.65 * circ / (r2 + delta_core * delta_core) * r_env
        u_mid += -omega_loc * ry + 0.28 * float(shear_velocity) * np.tanh(ry * 9.0) * (1.0 - 0.55 * r_env)
        v_mid += omega_loc * rx * 0.72

    state.kh_x = x + dt * u_mid
    state.kh_y = np.clip(y + dt * v_mid, 0.06, 0.94)
    state.kh_gamma = gamma
    state.time = t_next


# ============================================================
# Mode 2: Sand Strata — The Invertible Falling-Sand Picture
# ============================================================

def _relax_angle_of_repose(top_h: np.ndarray,
                           base_h: np.ndarray,
                           dx: float,
                           repose_deg: float,
                           iterations: int = 24) -> np.ndarray:
    """Deterministic Self-Organized Criticality (SOC) / angle-of-repose avalanche
    relaxation on a 1D heightfield `top_h(x) >= base_h(x)`.

    Whenever the local surface slope |dh/dx| exceeds tan(repose_deg), excess
    sand topples symmetrically downslope to adjacent columns while respecting
    the underlying cemented stratum `base_h(x)`.
    """
    h = top_h.copy()
    max_dh = math.tan(math.radians(float(repose_deg))) * float(dx)
    for _ in range(int(iterations)):
        # Left-to-right and right-to-left symmetric flux computation
        diff_r = h[:-1] - h[1:]
        excess_r = np.where(
            np.abs(diff_r) > max_dh,
            0.35 * (np.abs(diff_r) - max_dh) * np.sign(diff_r),
            0.0,
        )
        # Clamp flux so neither donor column drops below `base_h`
        avail_left = np.maximum(0.0, h[:-1] - base_h[:-1])
        avail_right = np.maximum(0.0, h[1:] - base_h[1:])
        flux = np.where(
            excess_r > 0.0,
            np.minimum(excess_r, 0.45 * avail_left),
            -np.minimum(-excess_r, 0.45 * avail_right),
        )
        if float(np.max(np.abs(flux))) < 1e-7:
            break
        h[:-1] -= flux
        h[1:] += flux
        h = np.maximum(h, base_h)
    return h


def _init_sand_strata_state(seed: int = 42,
                            n_cols: int = 320,
                            n_bubbles: int = 18) -> StratifiedFlowState:
    """Initialize the inverted falling-sand picture with upper sand reservoir,
    buoyant bubble barrier, and initial basal substrate."""
    gx = np.linspace(0.0, 1.0, int(n_cols), dtype=np.float64)
    # Basal dark floor heightfield h_0(x)
    h0 = 0.035 + 0.012 * np.sin(2.0 * math.pi * gx * 2.0 + py_rand01(seed, 0, S_SAND_GRAIN, 0) * 6.28)
    horizons = [h0.copy()]
    horizon_species = [0]

    # Pre-seed an initial thin basal basalt layer so even frame 0 has a valid floor
    h1 = h0 + 0.018 * (0.6 + 0.4 * np.cos(2.0 * math.pi * gx * 3.0 + 1.1))
    h1 = _relax_angle_of_repose(h1, h0, 1.0 / (n_cols - 1), DEFAULT_GRAIN_SPECIES[0].repose_deg)
    horizons.append(h1)
    horizon_species.append(0)

    # Upper sand reservoir supported by the bubble line after inversion
    res_bot = 0.75 + 0.012 * np.sin(2.0 * math.pi * gx * 4.0 + 0.5)
    res_top = 0.94 - 0.010 * np.cos(2.0 * math.pi * gx * 2.0)

    # Deterministic line of buoyant air bubbles obstructing the sand reservoir
    bubbles: list[tuple[float, float, float]] = []
    for i in range(int(n_bubbles)):
        bx = (i + 0.5) / float(n_bubbles) + (py_rand01(i, seed, S_SAND_BUBBLE, 1) - 0.5) * (0.45 / n_bubbles)
        bx = float(np.clip(bx, 0.03, 0.97))
        by = 0.735 + (py_rand01(i, seed, S_SAND_BUBBLE, 2) - 0.5) * 0.025
        br = 0.018 + 0.014 * py_rand01(i, seed, S_SAND_BUBBLE, 3)
        bubbles.append((bx, by, br))
    bubbles.sort(key=lambda b: b[0])

    return StratifiedFlowState(
        mode="sand_strata",
        frame_idx=0,
        time=0.0,
        seed=int(seed),
        grid_x=gx,
        horizons=horizons,
        horizon_species=horizon_species,
        reservoir_top=res_top,
        reservoir_bot=res_bot,
        bubbles=bubbles,
        plumes=[],
    )


def _step_sand_strata(state: StratifiedFlowState,
                      dt: float = 0.04) -> None:
    """Advance the falling-sand sedimentation & bubble-barrier system by one step.

    1. Bubbles slowly rise, oscillate, and coalesce/shift gaps.
    2. Gaps between adjacent bubbles emit multi-species sand plumes whose
       dominant species shifts over time (heavy basalt first, then garnet,
       amber quartz, and fine pale silica) due to differential terminal
       settling velocities.
    3. Sand plumes deposit onto the bottom heightfield pile and relax via
       angle-of-repose SOC avalanching into distinct cross-bedded strata.
    """
    state.time += float(dt)
    gx = state.grid_x
    n_cols = len(gx)
    dx = 1.0 / max(1, n_cols - 1)

    # 1. Update buoyant bubbles (slow rise + lateral oscillation)
    new_bubbles: list[tuple[float, float, float]] = []
    for idx, (bx, by, br) in enumerate(state.bubbles):
        phase = 2.0 * math.pi * py_rand01(idx, state.seed, S_SAND_BUBBLE, 10)
        bx_new = bx + 0.0025 * math.sin(1.8 * state.time + phase)
        by_new = min(0.84, by + 0.0008 * float(dt) * (1.0 + 15.0 * br))
        br_new = br * (1.0 + 0.015 * math.cos(2.4 * state.time + idx))
        new_bubbles.append((float(np.clip(bx_new, 0.03, 0.97)), float(by_new), float(br_new)))
    new_bubbles.sort(key=lambda b: b[0])
    state.bubbles = new_bubbles

    # 2. Identify breach gaps between bubbles where plumes stream downward
    gaps: list[tuple[float, float, float]] = []  # (gap_x, gap_y, gap_width)
    for i in range(len(new_bubbles) - 1):
        b1 = new_bubbles[i]
        b2 = new_bubbles[i + 1]
        clear_dist = (b2[0] - b1[0]) - (b1[2] + b2[2])
        if clear_dist > 0.004:
            gx_mid = 0.5 * (b1[0] + b1[2] + b2[0] - b2[2])
            gy_mid = 0.5 * (b1[1] + b2[1])
            gaps.append((gx_mid, gy_mid, clear_dist))

    # Ensure at least 5 active gaps across the frame so rich dunes form everywhere
    if len(gaps) < 5:
        for k in range(6):
            gx_fallback = 0.10 + 0.16 * k + 0.03 * math.sin(state.time + k)
            gaps.append((gx_fallback, 0.74, 0.025))

    # 3. Determine active sedimentary stratum based on simulation progress
    # Each stratum corresponds to a distinct sedimentation wave & grain class
    n_species = len(DEFAULT_GRAIN_SPECIES)
    target_stratum = min(14, 1 + int(state.time / 0.16))
    while len(state.horizons) - 1 < target_stratum:
        prev_h = state.horizons[-1]
        sp_idx = (len(state.horizons) - 1) % n_species
        state.horizons.append(prev_h.copy())
        state.horizon_species.append(sp_idx)

    active_sp_idx = state.horizon_species[-1]
    species = DEFAULT_GRAIN_SPECIES[active_sp_idx]
    top_h = state.horizons[-1]
    base_h = state.horizons[-2]

    # 4. Deposit sand from active breach gaps & record plume geometry for rendering/MetaPost
    active_plumes: list[dict] = []
    for g_idx, (gap_x, gap_y, gap_w) in enumerate(gaps):
        # Lateral wander of plume in fluid wake
        wander = 0.022 * math.sin(2.5 * state.time + g_idx * 1.7 + state.seed * 0.1)
        landing_x = float(np.clip(gap_x + wander, 0.04, 0.96))
        flux_amp = species.settling_velocity * float(dt) * (0.85 + 12.0 * gap_w)
        sigma = 0.028 + 0.45 * gap_w

        # Deposit Gaussian packet onto top heightfield
        packet = flux_amp * np.exp(-0.5 * ((gx - landing_x) / sigma) ** 2)
        top_h = top_h + packet

        # Deplete upper sand reservoir above the gap
        deplete = 0.55 * flux_amp * np.exp(-0.5 * ((gx - gap_x) / (1.3 * sigma)) ** 2)
        state.reservoir_top = np.maximum(state.reservoir_bot + 0.008, state.reservoir_top - deplete)

        # Record plume trajectory from (gap_x, gap_y) down to pile surface at landing_x
        col_idx = int(np.clip(round(landing_x * (n_cols - 1)), 0, n_cols - 1))
        pile_y = float(top_h[col_idx])
        active_plumes.append({
            "x_top": gap_x,
            "y_top": gap_y,
            "x_bot": landing_x,
            "y_bot": pile_y,
            "width": max(0.008, min(0.032, gap_w * 0.85)),
            "species": active_sp_idx,
        })

    # 5. Deterministic angle-of-repose SOC avalanche relaxation on the active horizon
    top_h = _relax_angle_of_repose(
        top_h, base_h, dx=dx, repose_deg=species.repose_deg, iterations=28
    )
    state.horizons[-1] = np.clip(top_h, base_h, 0.68)
    state.plumes = active_plumes


# ============================================================
# Frame-Addressable Simulation Runner
# ============================================================

class StratifiedFlowSimulator:
    """Deterministic, frame-addressable simulator for `kelvin_helmholtz` and `sand_strata`.

    Caches the most recent state so sequential frame evaluation (0, 1, 2, ...) steps
    forward incrementally in O(1) substeps per frame, while out-of-order or repeated
    frame requests deterministically reset or reuse the exact state for `frame_idx`.
    """

    def __init__(self,
                 mode: str = "kelvin_helmholtz",
                 seed: int = 42,
                 warmup_steps: int = 18,
                 substeps_per_frame: int = 2,
                 dt: float = 0.028,
                 atwood: float = 0.18,
                 viscosity: float = 0.004,
                 shear_velocity: float = 0.52,
                 n_billows: int = 3,
                 theta_max_deg: float = 14.0,
                 rock_omega: float = 1.35):
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
        self.mode = mode
        self.seed = int(seed)
        self.warmup_steps = max(0, int(warmup_steps))
        self.substeps_per_frame = max(1, int(substeps_per_frame))
        self.dt = float(dt)
        self.atwood = float(atwood)
        self.viscosity = float(viscosity)
        self.shear_velocity = float(shear_velocity)
        self.n_billows = max(1, int(n_billows))
        self.theta_max_deg = float(theta_max_deg)
        self.rock_omega = float(rock_omega)

        self._state: StratifiedFlowState | None = None
        self._reset_and_warmup()

    def _reset_and_warmup(self) -> None:
        if self.mode == "kelvin_helmholtz":
            st = _init_kelvin_helmholtz_state(
                seed=self.seed,
                n_billows=self.n_billows,
                atwood=self.atwood,
                shear_velocity=self.shear_velocity,
            )
            for _ in range(self.warmup_steps):
                _step_kelvin_helmholtz(
                    st,
                    dt=self.dt,
                    atwood=self.atwood,
                    viscosity=self.viscosity,
                    shear_velocity=self.shear_velocity,
                    theta_max_deg=self.theta_max_deg,
                    rock_omega=self.rock_omega,
                )
        else:
            st = _init_sand_strata_state(seed=self.seed)
            for _ in range(self.warmup_steps):
                _step_sand_strata(st, dt=self.dt)
        st.frame_idx = 0
        self._state = st

    def state_at(self, frame_idx: int) -> StratifiedFlowState:
        """Return a snapshot of the simulation state at exact `frame_idx >= 0`."""
        target = max(0, int(frame_idx))
        if self._state is None or self._state.frame_idx > target:
            self._reset_and_warmup()
        assert self._state is not None
        while self._state.frame_idx < target:
            for _ in range(self.substeps_per_frame):
                if self.mode == "kelvin_helmholtz":
                    _step_kelvin_helmholtz(
                        self._state,
                        dt=self.dt,
                        atwood=self.atwood,
                        viscosity=self.viscosity,
                        shear_velocity=self.shear_velocity,
                        theta_max_deg=self.theta_max_deg,
                        rock_omega=self.rock_omega,
                    )
                else:
                    _step_sand_strata(self._state, dt=self.dt)
            self._state.frame_idx += 1
        return copy.deepcopy(self._state)


# ============================================================
# High-Resolution Physical Rasterizers
# ============================================================

def render_kelvin_helmholtz(state: StratifiedFlowState,
                            width: int,
                            height: int,
                            palette="trumbull_2001",
                            sub_dx: float = 0.0,
                            sub_dy: float = 0.0) -> np.ndarray:
    """Render the two-layer Boussinesq Kelvin-Helmholtz fluid with multi-turn billow
    spirals, refractive internal caustic shading, and specular meniscus highlights.
    """
    anchors = resolve_palette_anchors(palette)
    xs = (np.arange(width, dtype=np.float64) + 0.5 + float(sub_dx)) / float(width)
    ys = 1.0 - (np.arange(height, dtype=np.float64) + 0.5 + float(sub_dy)) / float(height)
    gx, gy = np.meshgrid(xs, ys)

    # Apply subtle visual rocking tilt matching the wave machine's harmonic tilt
    tilt = 0.35 * state.gravity_angle
    ct, st = math.cos(tilt), math.sin(tilt)
    rx_frame = (gx - 0.5) * ct - (gy - 0.5) * st + 0.5
    ry_frame = (gx - 0.5) * st + (gy - 0.5) * ct + 0.5

    # Reconstruct the Eulerian vortex-wound Lagrangian coordinate (bx, by) at every pixel
    # via exact incompressible (area-preserving) vortex-core swirl + Boussinesq shear.
    # Because exact rotation around each billow core preserves streamline topology
    # without explicit-Euler radial overshoot, phi(x, y) = by - h0(bx) produces
    # razor-sharp, multi-turn Kelvin-Helmholtz billow spirals with zero numerical mush.
    n_billows = len(state.kh_centers_x)
    swell_shift = 0.08 * math.sin(state.gravity_angle)
    bx = rx_frame - swell_shift * np.tanh((ry_frame - 0.50) * 6.0)
    by = ry_frame - 0.032 * math.sin(state.gravity_angle) * np.sin(2.0 * math.pi * rx_frame)

    delta_core = 0.048
    sigma_b = 0.24 / float(max(1, n_billows))
    for m in range(n_billows):
        cx = state.kh_centers_x[m]
        cy = state.kh_centers_y[m]
        # Smooth periodic coordinate relative to billow m
        dx_m = (bx - cx + 0.5) % 1.0 - 0.5
        dy_m = by - cy
        # Elliptical aspect of the shear billow cat's-eye
        r2 = (dx_m * 0.90) ** 2 + (dy_m * 1.15) ** 2
        # Smooth compactly-vanishing window so periodic wrap at |dx_m|=0.5 is C^infinity zero
        window = np.clip(1.0 - (dx_m / 0.45) ** 2, 0.0, 1.0) ** 3
        env = np.exp(-r2 / (2.0 * sigma_b * sigma_b)) * window
        # Multi-turn swirl angle grows with integrated baroclinic circulation (2-3 crisp turns)
        swirl_angle = (2.4 * float(state.kh_circ[m]) * float(n_billows) + 1.65 * state.time) * (
            (delta_core * delta_core + 0.018) / (r2 + delta_core * delta_core)
        ) * env
        cs = np.cos(swirl_angle)
        sn = np.sin(swirl_angle)
        rot_dx = dx_m * cs + (dy_m * 1.12) * sn
        rot_dy = (-dx_m / 1.12) * sn + dy_m * cs
        bx = bx + (rot_dx - dx_m)
        by = by + (rot_dy - dy_m)

    # Initial interface profile h0(bx)
    phase_main = 2.0 * math.pi * py_rand01(state.seed, 1, S_KH_PHASE, 0)
    phase_sub = 2.0 * math.pi * py_rand01(state.seed, 2, S_KH_PHASE, 1)
    h0_bx = (
        0.50
        + 0.022 * np.sin(2.0 * math.pi * bx + phase_sub)
        + 0.018 * np.sin(2.0 * math.pi * n_billows * bx + phase_main)
    )
    phi = by - h0_bx

    # Smooth two-liquid Boussinesq density field rho in [0, 1] (0 = upper light layer, 1 = lower dense layer)
    sharpness = 65.0
    rho = 0.5 * (1.0 - np.tanh(sharpness * phi))

    # Interface gradient for refractive meniscus highlight
    dy_grad, dx_grad = np.gradient(phi)
    interface_band = np.exp(-((phi * 52.0) ** 2))

    # Internal refractive swirl & vorticity glow inside the billow cores
    vorticity_glow = np.zeros_like(phi)
    for m in range(n_billows):
        cx = state.kh_centers_x[m]
        cy = state.kh_centers_y[m]
        dx_m = (rx_frame - cx + 0.5) % 1.0 - 0.5
        dy_m = ry_frame - cy
        r2 = dx_m * dx_m + dy_m * dy_m
        vorticity_glow += np.exp(-r2 / (2.0 * 0.085 ** 2))

    # Map upper light fluid, lower dense fluid, and baroclinic spiral filaments through shot palette
    caustic = 0.5 + 0.5 * np.sin(2.0 * math.pi * (2.0 * bx + 1.5 * by))
    palette_u = (
        (1.0 - rho) * (0.08 + 0.22 * (1.0 - np.clip(ry_frame, 0.0, 1.0)) + 0.05 * caustic)
        + rho * (0.52 + 0.26 * np.clip(1.0 - ry_frame, 0.0, 1.0) + 0.14 * vorticity_glow * caustic)
    )
    palette_u = np.clip(palette_u + 0.22 * interface_band, 0.0, 1.0)
    rgb = sample_palette(palette_u, anchors)

    # Specular Fresnel highlight along the overturning Kelvin-Helmholtz meniscus
    spec = interface_band * np.clip(0.55 + 0.45 * (dy_grad / np.maximum(1e-6, np.hypot(dx_grad, dy_grad))), 0.0, 1.0)
    highlight_col = 0.35 * anchors[-1] + 0.65 * np.array([1.0, 0.98, 0.92])
    rgb = rgb * (1.0 - 0.55 * spec[..., None]) + highlight_col[None, None, :] * (0.65 * spec[..., None])

    # Subtle internal depth absorption toward top and bottom tank walls
    wall_vignette = np.clip(4.0 * gy * (1.0 - gy), 0.35, 1.0) ** 0.35
    rgb = rgb * wall_vignette[..., None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def render_sand_strata(state: StratifiedFlowState,
                       width: int,
                       height: int,
                       palette="trumbull_2001",
                       sub_dx: float = 0.0,
                       sub_dy: float = 0.0) -> np.ndarray:
    """Render the invertible falling-sand picture with multi-species cross-bedded
    sedimentary horizons, active falling plumes, and buoyant air bubble barrier.
    """
    anchors = resolve_palette_anchors(palette)
    xs = (np.arange(width, dtype=np.float64) + 0.5 + float(sub_dx)) / float(width)
    ys = 1.0 - (np.arange(height, dtype=np.float64) + 0.5 + float(sub_dy)) / float(height)

    # 1. Deep backlit fluid column background
    bg_coord = np.clip(0.04 + 0.18 * (ys[:, None] ** 1.4), 0.0, 0.35)
    img = np.broadcast_to(sample_palette(bg_coord, anchors), (height, width, 3)).copy()

    # Deterministic pixel-scale granular micro-texture from render.detrng
    px_grid, py_grid = np.meshgrid(np.arange(width), np.arange(height))
    grain_noise = _det_hash_2d(px_grid, py_grid, state.seed, S_SAND_GRAIN)

    # Interpolate 1D heightfields onto pixel columns `xs`
    gx = state.grid_x
    interp_horizons = [np.interp(xs, gx, h) for h in state.horizons]
    res_bot_x = np.interp(xs, gx, state.reservoir_bot)
    res_top_x = np.interp(xs, gx, state.reservoir_top)

    # 2. Render active falling granular plumes between bubble gaps and the bottom dune pile
    for p_idx, plume in enumerate(state.plumes):
        x_top = plume["x_top"]
        y_top = plume["y_top"]
        x_bot = plume["x_bot"]
        y_bot = plume["y_bot"]
        pw = plume["width"]
        sp_idx = plume["species"]
        sp = DEFAULT_GRAIN_SPECIES[sp_idx]
        sp_col = sample_palette(np.array([sp.palette_coord]), anchors)[0]

        # Vertical mask between y_bot and y_top
        y_col = ys[:, None]
        in_y = (y_col >= y_bot - 0.01) & (y_col <= y_top + 0.01)
        if not np.any(in_y):
            continue
        frac_y = np.clip((y_top - y_col) / max(1e-4, y_top - y_bot), 0.0, 1.0)
        # Plume centerline curves gently from gap_x to landing_x with harmonic ripple
        x_center = (1.0 - frac_y) * x_top + frac_y * x_bot + 0.004 * np.sin(18.0 * frac_y - 4.0 * state.time + p_idx)
        # Plume spreads slightly as it falls through viscous fluid
        w_y = pw * (0.65 + 0.55 * frac_y)
        dist_x = np.abs(xs[None, :] - x_center)
        plume_density = np.where(
            in_y,
            np.exp(-0.5 * (dist_x / w_y) ** 2) * (0.45 + 0.55 * grain_noise),
            0.0,
        )
        alpha = np.clip(plume_density * 0.85, 0.0, 0.92)[..., None]
        img = img * (1.0 - alpha) + sp_col[None, None, :] * (0.85 + 0.30 * grain_noise[..., None]) * alpha

    # 3. Render stacked, cross-bedded bottom sedimentary strata
    px_per_y = float(height)
    for k in range(1, len(interp_horizons)):
        h_low = interp_horizons[k - 1][None, :]
        h_high = interp_horizons[k][None, :]
        thickness = h_high - h_low
        if float(np.max(thickness)) < 1e-4:
            continue
        sp_idx = state.horizon_species[k]
        sp = DEFAULT_GRAIN_SPECIES[sp_idx]
        base_coord = sp.palette_coord
        # Internal sub-lamination & granular texture within each stratum
        rel_h = np.clip((ys[:, None] - h_low) / np.maximum(1e-4, thickness), 0.0, 1.0)
        lamina_wave = 0.04 * np.sin(2.0 * math.pi * (3.0 * rel_h + 1.5 * xs[None, :]))
        grain_mod = (grain_noise - 0.5) * 0.08
        stratum_rgb = sample_palette(np.clip(base_coord + lamina_wave + grain_mod, 0.02, 0.98), anchors)

        # Darkened horizon contact line at the top of each stratum for geological clarity
        dist_top_px = (h_high - ys[:, None]) * px_per_y
        dist_bot_px = (ys[:, None] - h_low) * px_per_y
        cov = np.clip(0.5 + np.minimum(dist_top_px, dist_bot_px), 0.0, 1.0)
        horizon_line = np.exp(-0.5 * (dist_top_px / 1.15) ** 2)
        stratum_rgb = stratum_rgb * (1.0 - 0.42 * horizon_line[..., None])

        mask = cov[..., None]
        img = img * (1.0 - mask) + stratum_rgb * mask

    # 4. Render upper inverted sand reservoir above the bubble line
    res_thick = np.maximum(1e-4, res_top_x - res_bot_x)[None, :]
    in_res_px = np.minimum((res_top_x[None, :] - ys[:, None]) * px_per_y, (ys[:, None] - res_bot_x[None, :]) * px_per_y)
    res_cov = np.clip(0.5 + in_res_px, 0.0, 1.0)
    if np.any(res_cov > 0.0):
        res_rel = np.clip((ys[:, None] - res_bot_x[None, :]) / res_thick, 0.0, 1.0)
        # Mixed multi-species reservoir banding
        res_coord = np.clip(0.25 + 0.62 * res_rel + (grain_noise - 0.5) * 0.12, 0.05, 0.95)
        res_rgb = sample_palette(res_coord, anchors)
        img = img * (1.0 - res_cov[..., None]) + res_rgb * res_cov[..., None]

    # 5. Render buoyant air bubbles with Fresnel rim lighting & specular highlights
    aspect = float(width) / float(height)
    rim_col = 0.25 * anchors[-1] + 0.75 * np.array([0.92, 0.97, 1.0])
    for bx, by, br in state.bubbles:
        dx_b = (xs[None, :] - bx) * aspect
        dy_b = ys[:, None] - by
        r_b = np.hypot(dx_b, dy_b)
        dist_px = (br - r_b) * px_per_y
        b_cov = np.clip(0.5 + dist_px, 0.0, 1.0)
        if not np.any(b_cov > 0.0):
            continue
        r_norm = np.clip(r_b / max(1e-5, br), 0.0, 1.0)
        # Fresnel rim + upper-left specular glint
        fresnel = (r_norm ** 2.4) * 0.85
        glint = np.exp(-(((dx_b / br + 0.35) ** 2 + (dy_b / br - 0.38) ** 2) / 0.08))
        bubble_rgb = img * (0.35 + 0.25 * (1.0 - r_norm[..., None])) + rim_col[None, None, :] * (
            fresnel[..., None] + 0.9 * glint[..., None]
        )
        img = img * (1.0 - b_cov[..., None]) + np.clip(bubble_rgb, 0.0, 1.0) * b_cov[..., None]

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def render_stratified_frame(state: StratifiedFlowState,
                            width: int,
                            height: int,
                            palette="trumbull_2001",
                            sub_dx: float = 0.0,
                            sub_dy: float = 0.0) -> np.ndarray:
    """Dispatch `state` to the appropriate physical mode renderer."""
    if state.mode == "kelvin_helmholtz":
        return render_kelvin_helmholtz(state, width, height, palette=palette, sub_dx=sub_dx, sub_dy=sub_dy)
    elif state.mode == "sand_strata":
        return render_sand_strata(state, width, height, palette=palette, sub_dx=sub_dx, sub_dy=sub_dy)
    raise ValueError(f"unknown mode {state.mode!r}")


if _HAS_TAICHI:
    @ti.data_oriented
    class StratifiedFlowEngine:
        """Taichi-compatible wrapper for StratifiedFlowSimulator with CPU/NumPy safety."""

        def __init__(self, width: int, height: int, simulator: StratifiedFlowSimulator):
            self.width = int(width)
            self.height = int(height)
            self.simulator = simulator
            self._use_fallback = not ensure_taichi_f64()

        def render(self,
                   state: StratifiedFlowState,
                   palette="trumbull_2001",
                   sub_dx: float = 0.0,
                   sub_dy: float = 0.0) -> np.ndarray:
            return render_stratified_frame(
                state, self.width, self.height, palette=palette, sub_dx=sub_dx, sub_dy=sub_dy
            )


# ============================================================
# Native MetaPost Contour & Horizon Exporter (--export-metapost)
# ============================================================

def export_metapost(state: StratifiedFlowState,
                    out_path: str | Path,
                    scale_bp: float = 240.0,
                    palette="trumbull_2001") -> str:
    """Export a self-contained, valid MetaPost figure (`beginfig(1); ... endfig;`)
    of the overturning Kelvin-Helmholtz interface contour or the stacked sedimentary
    horizon polylines in `sand_strata` mode.
    """
    anchors = resolve_palette_anchors(palette)
    w_bp = float(scale_bp) * (16.0 / 9.0)
    h_bp = float(scale_bp)

    bg_r, bg_g, bg_b = [float(c) for c in anchors[0] * 0.35]
    hi_r, hi_g, hi_b = [float(c) for c in anchors[-1]]

    lines: list[str] = [
        "% Generated by cosmic_engine render.stratified_flow",
        f"% mode: {state.mode}, frame: {state.frame_idx}, time: {state.time:.3f}",
        "prologues := 3;",
        'outputtemplate := "%j-%c.mps";',
        "beginfig(1);",
        f"  numeric W, H; W := {w_bp:.3f}bp; H := {h_bp:.3f}bp;",
        "  path frame_box, p;",
        "  frame_box := (0, 0)--(W, 0)--(W, H)--(0, H)--cycle;",
        f"  fill frame_box withcolor ({bg_r:.4f}, {bg_g:.4f}, {bg_b:.4f});",
    ]

    if state.mode == "kelvin_helmholtz":
        # Fill lower dense layer bounded by the overturning Kelvin-Helmholtz contour
        xs = state.kh_x
        ys = state.kh_y
        # Subsample to ~160 points so MetaPost spline compilation stays fast and exact
        step = max(1, len(xs) // 160)
        pts_x = np.clip(xs[::step], 0.0, 1.0) * w_bp
        pts_y = np.clip(ys[::step], 0.02, 0.98) * h_bp

        dense_col = sample_palette(np.array([0.62]), anchors)[0]
        dr, dg, db = [float(c) for c in dense_col]

        contour_pts = "..".join(f"({px:.2f}bp,{py:.2f}bp)" for px, py in zip(pts_x, pts_y))
        lines.append(
            f"  p := (0, 0)--({pts_x[0]:.2f}bp,{pts_y[0]:.2f}bp)..{contour_pts}..(W,{pts_y[-1]:.2f}bp)--(W, 0)--cycle;"
        )
        lines.append(f"  fill p withcolor ({dr:.4f}, {dg:.4f}, {db:.4f});")

        # Draw the overturning Kelvin-Helmholtz interface contour
        lines.append("  pickup pencircle scaled 1.4bp;")
        lines.append(f"  p := {contour_pts};")
        lines.append(f"  draw p withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});")

        # Draw billow core streamlines
        lines.append("  pickup pencircle scaled 0.7bp;")
        for m in range(len(state.kh_centers_x)):
            cx = float(state.kh_centers_x[m]) * w_bp
            cy = float(state.kh_centers_y[m]) * h_bp
            for r_frac in (0.03, 0.055):
                rx = r_frac * w_bp
                ry = r_frac * h_bp * 0.65
                lines.append(
                    f"  draw fullcircle xscaled {2*rx:.2f}bp yscaled {2*ry:.2f}bp shifted ({cx:.2f}bp,{cy:.2f}bp) "
                    f"withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});"
                )
    else:
        # sand_strata mode: export stacked sedimentary horizon polylines, plumes, and bubble barrier
        gx = state.grid_x
        step = max(1, len(gx) // 100)
        xs_sub = gx[::step] * w_bp

        # Active falling plumes
        lines.append("  pickup pencircle scaled 1.8bp;")
        for plume in state.plumes:
            sp = DEFAULT_GRAIN_SPECIES[plume["species"]]
            pr, pg, pb = [float(c) for c in sample_palette(np.array([sp.palette_coord]), anchors)[0]]
            x1, y1 = plume["x_top"] * w_bp, plume["y_top"] * h_bp
            x2, y2 = plume["x_bot"] * w_bp, plume["y_bot"] * h_bp
            lines.append(
                f"  draw ({x1:.2f}bp,{y1:.2f}bp)--({x2:.2f}bp,{y2:.2f}bp) withcolor ({pr:.4f}, {pg:.4f}, {pb:.4f});"
            )

        # Stacked sedimentary horizons (filled polygons between h_{k-1}(x) and h_k(x))
        lines.append("  pickup pencircle scaled 0.65bp;")
        for k in range(1, len(state.horizons)):
            h_low = state.horizons[k - 1][::step] * h_bp
            h_high = state.horizons[k][::step] * h_bp
            if float(np.max(h_high - h_low)) < 0.15:
                continue
            sp = DEFAULT_GRAIN_SPECIES[state.horizon_species[k]]
            sr, sg, sb = [float(c) for c in sample_palette(np.array([sp.palette_coord]), anchors)[0]]
            top_poly = "--".join(f"({x:.2f}bp,{y:.2f}bp)" for x, y in zip(xs_sub, h_high))
            bot_poly = "--".join(f"({x:.2f}bp,{y:.2f}bp)" for x, y in zip(reversed(xs_sub), reversed(h_low)))
            lines.append(f"  p := {top_poly}--{bot_poly}--cycle;")
            lines.append(f"  fill p withcolor ({sr:.4f}, {sg:.4f}, {sb:.4f});")
            lines.append(f"  draw ({top_poly}) withcolor ({bg_r:.4f}, {bg_g:.4f}, {bg_b:.4f});")

        # Buoyant air bubble barrier
        lines.append("  pickup pencircle scaled 0.9bp;")
        for bx, by, br in state.bubbles:
            cx, cy = bx * w_bp, by * h_bp
            diam = 2.0 * br * h_bp
            lines.append(
                f"  draw fullcircle scaled {diam:.2f}bp shifted ({cx:.2f}bp,{cy:.2f}bp) "
                f"withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});"
            )

    lines.append("  clip currentpicture to frame_box;")
    lines.append("  pickup pencircle scaled 1.0bp;")
    lines.append(f"  draw frame_box withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});")
    lines.append("endfig;")
    lines.append("end;")

    content = "\n".join(lines) + "\n"
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(content, encoding="utf-8")
    return content


# ============================================================
# CLI
# ============================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stratified_flow")
    parser.add_argument("--mode", default="kelvin_helmholtz", choices=VALID_MODES)
    parser.add_argument("--frame", type=int, default=48,
                        help="target simulation frame index to render")
    parser.add_argument("--warmup-steps", type=int, default=18)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.028)
    parser.add_argument("--atwood", type=float, default=0.18)
    parser.add_argument("--viscosity", type=float, default=0.004)
    parser.add_argument("--shear-velocity", type=float, default=0.52)
    parser.add_argument("--n-billows", type=int, default=3)
    parser.add_argument("--palette", default="trumbull_2001")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-bp", type=float, default=240.0)
    parser.add_argument("--export-metapost", default=None,
                        help="optional path to write a native MetaPost (.mp) figure")
    parser.add_argument("--out", default=None,
                        help="output image (.png or .npy)")
    args = parser.parse_args(argv)

    sim = StratifiedFlowSimulator(
        mode=args.mode,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        substeps_per_frame=args.substeps,
        dt=args.dt,
        atwood=args.atwood,
        viscosity=args.viscosity,
        shear_velocity=args.shear_velocity,
        n_billows=args.n_billows,
    )
    state = sim.state_at(args.frame)

    if args.export_metapost:
        export_metapost(state, args.export_metapost, scale_bp=args.scale_bp, palette=args.palette)
        print(f"exported MetaPost contour/horizon figure to {args.export_metapost} (mode={args.mode}, frame={args.frame})")

    if args.out:
        offsets = supersample_offsets(max(1, int(args.samples)))
        acc = np.zeros((args.height, args.width, 3), dtype=np.float64)
        for dx, dy in offsets:
            acc += render_stratified_frame(
                state, args.width, args.height, palette=args.palette, sub_dx=dx, sub_dy=dy
            )
        rgb = (acc / len(offsets)).astype(np.float32)
        if args.out.endswith(".npy"):
            np.save(args.out, rgb)
        else:
            img8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            try:
                import imageio.v3 as iio
                iio.imwrite(args.out, img8)
            except ImportError:
                import cv2
                cv2.imwrite(args.out, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.out} ({args.width}x{args.height}, mode={args.mode}, frame={args.frame})")

    if not args.out and not args.export_metapost:
        parser.error("at least one of --out or --export-metapost is required")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
