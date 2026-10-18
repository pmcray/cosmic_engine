"""Sealed Glass Ecosphere & Ashby Ultrastable Homeostat (`render.ecosphere`).

Models a hermetically sealed spherical glass ecosystem (inspired by the NASA
Ames / Joe Hanson closed EcoSphere and W. Ross Ashby's *Design for a Brain*
four-unit Homeostat) driven by an external circadian light vector `L(t)`:

1. Coupled 4-Variable Biosphere ODE & 4-Unit Ashby Homeostat:
   * Tracks Nutrient substrate `N(t)`, filamentous Algae / phytoplankton `A(t)`,
     grazing *Halocaridina rubra* red shrimp density `S(t)`, and dissolved
     Oxygen/Carbon balance `O2(t) in [0, 1]`.
   * Simultaneously integrates a coupled 4-unit Ashby Homeostat feedback state
     `x(t) in [-1, 1]^4` governed by a 4x4 uniselector synaptic coupling matrix `M`
     and driven by ecological disequilibrium from the target homeostatic setpoints
     `(N*, A*, S*, O2*)`.

2. Ashby Ultrastable Step-Function:
   * Monitors the physiological essential variable `eta(t) = |O2(t) - O2*|`.
   * When `eta(t)` breaches its critical threshold `eta_crit` (from incipient
     algal bloom or anoxic collapse), triggers a deterministic Ashby uniselector
     step-reconfiguration (via `render/detrng.py`) that switches `M` to a new
     stabilizing feedback configuration and perturbs the spatial Gray-Scott
     reaction-diffusion parameters `(F, k)`, guiding the biosphere back into an
     ultrastable homeostatic equilibrium.

3. 2D Spatial Reaction-Diffusion (Gray-Scott / Turing Morphogenesis), L-System
   Gorgonian Frond & Swimming *Halocaridina* Red Shrimp Agents:
   * Integrates a deterministic 2D Gray-Scott reaction-diffusion lattice `(U, V)`
     inside the circular orb cross-section to generate organic algal filaments
     and Turing spots modulated by light and `A(t)`.
   * Generates a procedural branching Gorgonian sea-fan L-system skeleton rooted
     on the basal basalt/aragonite gravel bed.
   * Advances deterministic swimming *Halocaridina rubra* red shrimp trajectories
     that graze along the gorgonian branches and algal filaments.

4. Refractive Glass Orb Shader & Native MetaPost Exporter (`--export-metapost`):
   * Renders the 3D/2D sealed glass sphere with Fresnel rim reflections, Snell's
     law internal refraction, solar caustic focus at the gravel substrate, meniscus
     waterline, L-system gorgonian frond silhouette, spatial Turing/Gray-Scott
     algal filaments, swimming red shrimp trajectories, and an optional diegetic
     cybernetic telemetry overlay / 4-unit Ashby Homeostat phase-portrait trace.
   * `export_metapost(state, out_path, scale_bp=200.0)` and CLI `--export-metapost`
     emit a self-contained MetaPost vector diagram (`prologues := 3;`) of the
     spherical boundary, L-system gorgonian branch, reaction-diffusion contour
     polygons, and 4-unit Ashby Homeostat phase-space trajectory that compiles
     cleanly with `mpost`.
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
from render.stratified_flow import (
    _det_hash_2d,
    resolve_palette_anchors,
    sample_palette,
)

try:
    import taichi as ti
except Exception:  # pragma: no cover
    ti = None


# Deterministic stream salts for render.detrng
S_ECO_UNISELECTOR = 0x1A2B3C4D
S_ECO_RD_INIT = 0x2B3C4D5E
S_ECO_LSYS = 0x3C4D5E6F
S_ECO_SHRIMP = 0x4D5E6F71
S_ECO_GRAVEL = 0x5E6F7182
S_ECO_BUBBLE = 0x6F718293


# ============================================================
# Circadian Light Vector & Ecological Setpoints
# ============================================================

def circadian_light_vector(t_sec: float,
                           period_sec: float = 2.0,
                           phi0: float = 0.0) -> tuple[float, tuple[float, float, float]]:
    """Compute the external circadian solar light intensity `I(t) in [0.06, 1.0]`
    and unit 3D direction vector `(lx, ly, lz)` at simulation time `t_sec`.
    """
    omega = 2.0 * math.pi / max(1e-6, float(period_sec))
    phase = omega * float(t_sec) + float(phi0)
    # Smooth diurnal insolation cycle with twilight floor
    raw_sin = math.sin(phase)
    intensity = 0.06 + 0.94 * (0.5 * (1.0 + raw_sin)) ** 1.35

    # Sun sweeps across the upper hemisphere of the sealed glass orb
    az = 0.72 * math.cos(phase)
    el = 0.32 + 0.52 * (0.5 * (1.0 + raw_sin))
    lx = math.sin(az) * math.cos(el)
    ly = math.sin(el)
    lz = -math.cos(az) * math.cos(el)
    norm = max(1e-12, math.sqrt(lx * lx + ly * ly + lz * lz))
    return float(intensity), (lx / norm, ly / norm, lz / norm)


def sample_uniselector_matrix(seed: int, step_index: int) -> tuple[np.ndarray, float, float]:
    """Deterministically sample a 4x4 Ashby Homeostat uniselector coupling matrix `M`
    and Gray-Scott reaction-diffusion parameters `(F, k)` via `render/detrng.py`.

    Step 0 is intentionally poised with weak regulatory damping so an external
    perturbation or circadian bloom can push `eta(t) = |O2(t) - O2*|` past `eta_crit`,
    whereas subsequent uniselector steps (`step_index >= 1`) engage strong negative-feedback
    ultrastable diagonal damping and balanced antisymmetric cross-couplings.
    """
    m = np.zeros((4, 4), dtype=np.float64)
    s_idx = int(step_index)
    sd = int(seed)

    # 25 discrete uniselector positions per synaptic weight (Ashby 1952)
    for i in range(4):
        for j in range(4):
            u = py_rand01(sd, s_idx, i * 4 + j, S_ECO_UNISELECTOR)
            q_level = (int(u * 25.0) - 12) / 12.0  # quantized in {-1.0, ..., +1.0}
            if i == j:
                if s_idx == 0:
                    m[i, j] = -0.18 - 0.10 * abs(q_level)
                else:
                    # Strong negative self-feedback ensures Hurwitz stability
                    m[i, j] = -1.25 - 0.55 * abs(q_level)
            else:
                scale = 0.42 if s_idx == 0 else 0.24
                m[i, j] = scale * q_level

    if s_idx >= 1:
        # Skew-symmetric bias on off-diagonal terms promotes damped spiral convergence
        off = 0.5 * (m - m.T)
        diag = np.diag(np.diag(m))
        m = diag + 0.65 * off

    # Gray-Scott coral/mitosis/filament regime parameters (F, k)
    u_f = py_rand01(sd, s_idx, 101, S_ECO_UNISELECTOR)
    u_k = py_rand01(sd, s_idx, 202, S_ECO_UNISELECTOR)
    if s_idx == 0:
        F = 0.034 + 0.004 * u_f
        k = 0.059 + 0.002 * u_k
    else:
        # Coral / filamentous labyrinthine Turing pattern regime
        F = 0.037 + 0.006 * u_f
        k = 0.060 + 0.003 * (u_k - 0.5)
    return m, float(F), float(k)


# ============================================================
# Procedural L-System Gorgonian Frond & Gravel Substrate
# ============================================================

def build_gorgonian_lsystem(seed: int = 42,
                            iterations: int = 4) -> list[tuple[float, float, float, float, int]]:
    """Generate a 2D branching Gorgonian sea-fan skeleton inside the unit disk (`r < 0.88`).

    Returns a list of line segments `(x0, y0, x1, y1, depth)` in normalized orb
    coordinates `[-1, 1]^2` where `(0, -0.70)` is the gravel bed crown.
    """
    axiom = "F"
    rules = {
        "F": "F[+F]F[-F][F]",
    }
    sentence = axiom
    for _ in range(max(1, min(5, int(iterations)))):
        out = []
        for idx, ch in enumerate(sentence):
            if ch == "F":
                # Deterministic rule variation keyed on detrng
                r = py_rand01(int(seed), idx, len(out), S_ECO_LSYS)
                if r < 0.38:
                    out.append("F[+F][-F]")
                elif r < 0.76:
                    out.append("F[+F]F[-F]")
                else:
                    out.append("F[-F]F[+F]")
            else:
                out.append(ch)
        sentence = "".join(out)

    segments: list[tuple[float, float, float, float, int]] = []
    x, y = 0.0, -0.68
    angle = math.pi * 0.5  # pointing straight up
    step_len = 0.34 / (1.68 ** max(0, iterations - 1))
    stack: list[tuple[float, float, float, int, float]] = []
    depth = 0

    for idx, ch in enumerate(sentence):
        jitter = (py_rand01(int(seed), idx, depth, S_ECO_LSYS) - 0.5) * 0.09
        if ch == "F":
            cur_len = step_len * (0.86 ** min(depth, 5))
            nx = x + cur_len * math.cos(angle + jitter * 0.3)
            ny = y + cur_len * math.sin(angle + jitter * 0.3)
            # Keep inside the lower/central region of the glass sphere
            r_norm = math.hypot(nx, ny)
            if r_norm < 0.78 and ny < 0.52:
                segments.append((float(x), float(y), float(nx), float(ny), int(depth)))
                x, y = nx, ny
        elif ch == "+":
            angle += math.radians(24.0) + jitter
        elif ch == "-":
            angle -= math.radians(24.0) + jitter
        elif ch == "[":
            stack.append((x, y, angle, depth, step_len))
            depth += 1
        elif ch == "]" and stack:
            x, y, angle, depth, step_len = stack.pop()

    return segments


# ============================================================
# Ecosphere State & Simulator
# ============================================================

@dataclass
class EcosphereState:
    """Complete snapshot of the Sealed Glass Ecosphere & Ashby Homeostat at `frame_idx`."""

    frame_idx: int
    time: float
    seed: int

    # 4-variable biosphere state: Nutrient N, Algae A, Shrimp S, Oxygen O2
    N: float
    A: float
    S: float
    O2: float
    O2_star: float
    eta: float
    eta_crit: float

    # Ashby Homeostat state
    ashby_units: np.ndarray              # shape (4,) in [-1, 1]
    coupling_matrix: np.ndarray          # shape (4, 4)
    uniselector_steps: int               # cumulative step-reconfigurations
    last_step_time: float                # time of most recent uniselector step
    step_events: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    # History arrays for phase-portrait & telemetry trace
    history_t: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    history_vars: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    history_ashby: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    history_eta: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    # Circadian light vector
    light_intensity: float = 0.8
    light_dir: tuple[float, float, float] = (0.35, 0.75, -0.55)

    # 2D Gray-Scott reaction-diffusion fields inside the orb
    F: float = 0.037
    k: float = 0.060
    rd_U: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.float64))
    rd_V: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.float64))

    # L-system gorgonian segments & swimming red shrimp trajectories
    gorgonian_segments: list[tuple[float, float, float, float, int]] = field(default_factory=list)
    shrimp_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    shrimp_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    shrimp_trails: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 2), dtype=np.float64))


class EcosphereSimulator:
    """Deterministic, frame-addressable simulator for the Sealed Glass Ecosphere
    and W. Ross Ashby's Ultrastable Homeostat.
    """

    def __init__(self,
                 seed: int = 42,
                 rd_size: int = 72,
                 n_shrimp: int = 9,
                 trail_len: int = 18,
                 warmup_steps: int = 16,
                 substeps_per_frame: int = 4,
                 dt: float = 0.042,
                 circadian_period: float = 1.8,
                 O2_star: float = 0.62,
                 eta_crit: float = 0.14,
                 perturbation_frame: int = 16,
                 perturbation_strength: float = 0.21,
                 telemetry_overlay: bool = True):
        self.seed = int(seed)
        self.rd_size = max(32, int(rd_size))
        self.n_shrimp = max(2, int(n_shrimp))
        self.trail_len = max(4, int(trail_len))
        self.warmup_steps = max(0, int(warmup_steps))
        self.substeps_per_frame = max(1, int(substeps_per_frame))
        self.dt = float(dt)
        self.circadian_period = float(circadian_period)
        self.O2_star = float(O2_star)
        self.eta_crit = float(eta_crit)
        self.perturbation_frame = int(perturbation_frame)
        self.perturbation_strength = float(perturbation_strength)
        self.telemetry_overlay = bool(telemetry_overlay)

        self._gorgonian = build_gorgonian_lsystem(seed=self.seed, iterations=4)
        self._cache: dict[int, EcosphereState] = {}
        self._init_base_state()

    def _init_base_state(self) -> None:
        n = self.rd_size
        y_lin = np.linspace(-1.0, 1.0, n)
        x_lin = np.linspace(-1.0, 1.0, n)
        xx, yy = np.meshgrid(x_lin, y_lin)
        rr = np.hypot(xx, yy)
        self._disk_mask = (rr <= 0.92).astype(np.float64)

        U = np.ones((n, n), dtype=np.float64)
        V = np.zeros((n, n), dtype=np.float64)

        # Seed Turing / Gray-Scott algal colonies along the gorgonian branches and central water column
        n_seeds = 14
        for m in range(n_seeds):
            if m < len(self._gorgonian) and m % 2 == 0:
                seg = self._gorgonian[(m * 7) % len(self._gorgonian)]
                sx = 0.5 * (seg[0] + seg[2])
                sy = 0.5 * (seg[1] + seg[3])
            else:
                ang = 2.0 * math.pi * py_rand01(self.seed, m, 1, S_ECO_RD_INIT)
                rad = 0.62 * math.sqrt(py_rand01(self.seed, m, 2, S_ECO_RD_INIT))
                sx = rad * math.cos(ang)
                sy = rad * math.sin(ang) * 0.8 + 0.05
            dist2 = (xx - sx) ** 2 + (yy - sy) ** 2
            patch = np.exp(-dist2 / (2.0 * 0.075 ** 2))
            U -= 0.48 * patch
            V += 0.26 * patch

        # Deterministic micro-noise
        ix, iy = np.meshgrid(np.arange(n), np.arange(n))
        noise = _det_hash_2d(ix, iy, seed=self.seed, salt=S_ECO_RD_INIT)
        U = np.clip(U + 0.02 * (noise - 0.5), 0.0, 1.0)
        V = np.clip(V + 0.01 * (noise - 0.5), 0.0, 1.0) * self._disk_mask

        M0, F0, k0 = sample_uniselector_matrix(self.seed, 0)

        # Initialize Halocaridina rubra shrimp agents
        shrimp_pos = np.zeros((self.n_shrimp, 2), dtype=np.float64)
        shrimp_vel = np.zeros((self.n_shrimp, 2), dtype=np.float64)
        for s_i in range(self.n_shrimp):
            ang = 2.0 * math.pi * py_rand01(self.seed, s_i, 10, S_ECO_SHRIMP)
            rad = 0.18 + 0.50 * py_rand01(self.seed, s_i, 11, S_ECO_SHRIMP)
            shrimp_pos[s_i, 0] = rad * math.cos(ang)
            shrimp_pos[s_i, 1] = -0.15 + 0.55 * math.sin(ang)
            v_ang = 2.0 * math.pi * py_rand01(self.seed, s_i, 12, S_ECO_SHRIMP)
            shrimp_vel[s_i, 0] = 0.25 * math.cos(v_ang)
            shrimp_vel[s_i, 1] = 0.20 * math.sin(v_ang)

        trails = np.repeat(shrimp_pos[:, None, :], self.trail_len, axis=1)

        # Ecological variables near equilibrium before circadian cycle & bloom perturbation
        N0 = 0.54
        A0 = 0.42
        S0 = 0.38
        O2_0 = self.O2_star - 0.03
        ashby0 = np.array([0.04, -0.05, 0.03, -0.03], dtype=np.float64)

        light_I, light_dir = circadian_light_vector(0.0, period_sec=self.circadian_period)

        st = EcosphereState(
            frame_idx=-1,
            time=0.0,
            seed=self.seed,
            N=N0,
            A=A0,
            S=S0,
            O2=O2_0,
            O2_star=self.O2_star,
            eta=abs(O2_0 - self.O2_star),
            eta_crit=self.eta_crit,
            ashby_units=ashby0,
            coupling_matrix=M0,
            uniselector_steps=0,
            last_step_time=-10.0,
            step_events=[],
            history_t=np.array([0.0], dtype=np.float64),
            history_vars=np.array([[N0, A0, S0, O2_0]], dtype=np.float64),
            history_ashby=ashby0[None, :].copy(),
            history_eta=np.array([abs(O2_0 - self.O2_star)], dtype=np.float64),
            light_intensity=light_I,
            light_dir=light_dir,
            F=F0,
            k=k0,
            rd_U=U,
            rd_V=V,
            gorgonian_segments=self._gorgonian,
            shrimp_positions=shrimp_pos,
            shrimp_velocities=shrimp_vel,
            shrimp_trails=trails,
        )

        # Warm up reaction-diffusion lattice so frame 0 already displays rich Turing filaments
        for w in range(self.warmup_steps):
            self._step_rd(st, dt_rd=1.0, light_I=0.75)

        st.frame_idx = 0
        self._cache[0] = copy.deepcopy(st)

    def _step_rd(self, st: EcosphereState, dt_rd: float, light_I: float) -> None:
        """Advance 2D Gray-Scott reaction-diffusion inside the circular orb boundary."""
        U = st.rd_U
        V = st.rd_V
        # 5-point discrete Laplacian with zero-flux Neumann boundary via roll
        lap_U = (
            np.roll(U, 1, axis=0)
            + np.roll(U, -1, axis=0)
            + np.roll(U, 1, axis=1)
            + np.roll(U, -1, axis=1)
            - 4.0 * U
        )
        lap_V = (
            np.roll(V, 1, axis=0)
            + np.roll(V, -1, axis=0)
            + np.roll(V, 1, axis=1)
            + np.roll(V, -1, axis=1)
            - 4.0 * V
        )
        Du = 0.16
        Dv = 0.08
        # Modulate feed rate softly by circadian light and algal biomass
        F_eff = st.F * (0.88 + 0.22 * light_I)
        k_eff = st.k * (0.96 + 0.08 * st.S)

        uvv = U * V * V
        U_new = U + dt_rd * (Du * lap_U - uvv + F_eff * (1.0 - U))
        V_new = V + dt_rd * (Dv * lap_V + uvv - (F_eff + k_eff) * V)

        # Confine active reaction-diffusion to the spherical interior
        st.rd_U = np.clip(U_new * self._disk_mask + (1.0 - self._disk_mask), 0.0, 1.0)
        st.rd_V = np.clip(V_new * self._disk_mask, 0.0, 1.0)

    def _advance_one_frame(self, prev: EcosphereState, target_frame: int) -> EcosphereState:
        st = copy.deepcopy(prev)
        st.frame_idx = int(target_frame)

        # Target ecological homeostatic setpoints
        N_star, A_star, S_star, O2_star = 0.52, 0.44, 0.38, self.O2_star

        for sub in range(self.substeps_per_frame):
            t_cur = st.time + self.dt
            st.time = t_cur
            light_I, light_dir = circadian_light_vector(t_cur, period_sec=self.circadian_period)
            st.light_intensity = light_I
            st.light_dir = light_dir

            # Inject deterministic algal bloom / anoxic thermal shock at perturbation_frame
            bloom_impulse = 0.0
            o2_shock = 0.0
            if target_frame == self.perturbation_frame and sub == 0 and st.uniselector_steps == 0:
                bloom_impulse = self.perturbation_strength * 0.85
                o2_shock = -self.perturbation_strength

            # 1. Coupled 4-variable ecological ODE + Ashby cybernetic control input
            # Deviations from homeostatic equilibrium
            err = np.array([
                st.N - N_star,
                st.A - A_star,
                st.S - S_star,
                st.O2 - O2_star,
            ], dtype=np.float64)

            # Ashby 4-unit Homeostat dynamics: dx/dt = M x - gain * err
            # Once the uniselector has stepped (uniselector_steps >= 1), the reconfigured
            # matrix M actively damps both the homeostat units and the ecological error.
            d_ashby = st.coupling_matrix @ st.ashby_units - 1.35 * err
            st.ashby_units = np.tanh(st.ashby_units + self.dt * d_ashby)

            reg_gain = 0.15 if st.uniselector_steps == 0 else 2.65
            u_ctrl = -reg_gain * err + 0.35 * st.ashby_units

            # Ecological rates:
            # Photosynthesis: light + nutrients -> algae + O2
            photo = 1.15 * light_I * st.N * st.A / (0.30 + st.N)
            # Grazing: shrimp consume algae
            graze = 0.92 * st.A * st.S / (0.35 + st.A)
            # Microbial remineralization of detritus -> nutrients
            remin = 0.48 * (0.28 * st.A + 0.55 * st.S) * (0.35 + st.O2)
            # Respiration consumes O2
            resp = 0.54 * st.S + 0.36 * st.A * (1.15 - 0.55 * light_I)

            dN = remin - 0.85 * photo + 0.25 * (N_star - st.N) + 0.35 * u_ctrl[0]
            dA = photo - graze - 0.22 * st.A + bloom_impulse / self.dt + 0.35 * u_ctrl[1]
            dS = 0.62 * graze * (0.4 + st.O2) - 0.26 * st.S + 0.30 * u_ctrl[2]
            dO2 = 1.18 * photo - resp + 0.12 * (light_I - 0.48) + o2_shock / self.dt + 0.85 * u_ctrl[3]

            st.N = float(np.clip(st.N + self.dt * dN, 0.05, 0.98))
            st.A = float(np.clip(st.A + self.dt * dA, 0.05, 0.98))
            st.S = float(np.clip(st.S + self.dt * dS, 0.05, 0.98))
            st.O2 = float(np.clip(st.O2 + self.dt * dO2, 0.02, 0.98))

            # Essential variable eta(t) = |O2(t) - O2*|
            st.eta = abs(st.O2 - st.O2_star)

            # 2. Ashby Ultrastable Step-Function check
            cooldown_elapsed = (t_cur - st.last_step_time) >= (3.5 * self.dt)
            if st.eta > st.eta_crit and cooldown_elapsed:
                st.uniselector_steps += 1
                st.last_step_time = t_cur
                new_M, new_F, new_k = sample_uniselector_matrix(self.seed, st.uniselector_steps)
                st.coupling_matrix = new_M
                st.F = new_F
                st.k = new_k
                st.step_events.append((target_frame, float(t_cur), float(st.eta), new_F, new_k))

                # Uniselector step impulse perturbs the reaction-diffusion field & re-oxygenates
                ix, iy = np.meshgrid(np.arange(self.rd_size), np.arange(self.rd_size))
                step_wave = _det_hash_2d(
                    ix, iy, seed=self.seed + st.uniselector_steps * 31, salt=S_ECO_UNISELECTOR
                )
                st.rd_V = np.clip(
                    st.rd_V * 0.88 + 0.08 * step_wave * self._disk_mask, 0.0, 1.0
                )
                # Direct homeostatic valve correction toward O2*
                st.O2 = float(st.O2 + 0.42 * (st.O2_star - st.O2))
                st.eta = abs(st.O2 - st.O2_star)

            # 3. Advance 2D Gray-Scott reaction-diffusion
            self._step_rd(st, dt_rd=0.85, light_I=light_I)

        # 4. Advance swimming Halocaridina rubra shrimp agents
        dt_frame = self.dt * self.substeps_per_frame
        pos = st.shrimp_positions.copy()
        vel = st.shrimp_velocities.copy()
        for i in range(self.n_shrimp):
            # Target foraging attractor along gorgonian tips and circadian phototaxis
            seg = self._gorgonian[(i * 11 + target_frame // 8) % len(self._gorgonian)]
            tx = 0.5 * (seg[0] + seg[2]) + 0.14 * st.light_dir[0] * st.light_intensity
            ty = 0.5 * (seg[1] + seg[3]) + 0.10 * math.sin(0.9 * st.time + i * 1.3)
            # Deterministic darting impulse keyed on detrng
            jx = (py_rand01(self.seed, target_frame, i, S_ECO_SHRIMP) - 0.5) * 0.45
            jy = (py_rand01(self.seed, target_frame, i + 100, S_ECO_SHRIMP) - 0.5) * 0.35

            ax = 1.6 * (tx - pos[i, 0]) + jx - 1.2 * vel[i, 0]
            ay = 1.6 * (ty - pos[i, 1]) + jy - 1.2 * vel[i, 1]
            vel[i, 0] += dt_frame * ax
            vel[i, 1] += dt_frame * ay
            spd = math.hypot(vel[i, 0], vel[i, 1])
            max_spd = 0.42 * (0.7 + 0.5 * st.O2)
            if spd > max_spd:
                vel[i] *= max_spd / spd

            pos[i] += dt_frame * vel[i]
            # Soft spherical glass wall & gravel floor reflection
            r_p = math.hypot(pos[i, 0], pos[i, 1])
            if r_p > 0.76:
                nx_w, ny_w = pos[i, 0] / r_p, pos[i, 1] / r_p
                pos[i, 0] = 0.75 * nx_w
                pos[i, 1] = 0.75 * ny_w
                dot = vel[i, 0] * nx_w + vel[i, 1] * ny_w
                if dot > 0:
                    vel[i, 0] -= 1.8 * dot * nx_w
                    vel[i, 1] -= 1.8 * dot * ny_w
            if pos[i, 1] < -0.58:
                pos[i, 1] = -0.58
                vel[i, 1] = abs(vel[i, 1]) * 0.7
            elif pos[i, 1] > 0.62:
                pos[i, 1] = 0.62
                vel[i, 1] = -abs(vel[i, 1]) * 0.7

        st.shrimp_positions = pos
        st.shrimp_velocities = vel
        st.shrimp_trails = np.concatenate([st.shrimp_trails[:, 1:, :], pos[:, None, :]], axis=1)

        # Append history for phase-portrait & telemetry
        st.history_t = np.append(st.history_t, st.time)
        st.history_vars = np.vstack([st.history_vars, [[st.N, st.A, st.S, st.O2]]])
        st.history_ashby = np.vstack([st.history_ashby, st.ashby_units[None, :]])
        st.history_eta = np.append(st.history_eta, st.eta)

        return st

    def state_at(self, frame_idx: int) -> EcosphereState:
        """Return the exact deterministic `EcosphereState` at `frame_idx >= 0`."""
        target = max(0, int(frame_idx))
        if target in self._cache:
            return copy.deepcopy(self._cache[target])

        # Find highest cached frame <= target
        cached_frames = [k for k in self._cache if k <= target]
        curr_idx = max(cached_frames)
        curr_state = self._cache[curr_idx]

        while curr_idx < target:
            curr_idx += 1
            curr_state = self._advance_one_frame(curr_state, curr_idx)
            # Cache checkpoints and recent frames for fast sequential playback
            if curr_idx % 8 == 0 or curr_idx == target:
                self._cache[curr_idx] = copy.deepcopy(curr_state)

        return copy.deepcopy(curr_state)


# ============================================================
# Refractive Glass Orb Renderer
# ============================================================

def _bilinear_sample_2d(field_2d: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear interpolation of `field_2d` (H_g, W_g) over `u, v in [-1, 1]`."""
    hg, wg = field_2d.shape
    fx = np.clip((u + 1.0) * 0.5 * (wg - 1), 0.0, wg - 1.0001)
    fy = np.clip((v + 1.0) * 0.5 * (hg - 1), 0.0, hg - 1.0001)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, wg - 1)
    y1 = np.minimum(y0 + 1, hg - 1)
    wx = fx - x0
    wy = fy - y0
    val = (
        (1.0 - wx) * (1.0 - wy) * field_2d[y0, x0]
        + wx * (1.0 - wy) * field_2d[y0, x1]
        + (1.0 - wx) * wy * field_2d[y1, x0]
        + wx * wy * field_2d[y1, x1]
    )
    return val


def _draw_segment_sdf(xx: np.ndarray,
                      yy: np.ndarray,
                      x0: float,
                      y0: float,
                      x1: float,
                      y1: float) -> np.ndarray:
    """Compute squared distance from grid `(xx, yy)` to line segment `(x0, y0)-(x1, y1)`."""
    dx = x1 - x0
    dy = y1 - y0
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return (xx - x0) ** 2 + (yy - y0) ** 2
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / l2, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    return (xx - px) ** 2 + (yy - py) ** 2


def render_ecosphere_frame(state: EcosphereState,
                           width: int,
                           height: int,
                           palette="trumbull_2001",
                           orb_scale: float = 0.84,
                           telemetry_overlay: bool = True,
                           sub_dx: float = 0.0,
                           sub_dy: float = 0.0) -> np.ndarray:
    """Render the Sealed Glass Ecosphere & Ashby Homeostat to an `(H, W, 3)` float32 RGB image."""
    w = int(width)
    h = int(height)
    anchors = resolve_palette_anchors(palette)

    # Normalized screen coordinates with y pointing up and aspect correction
    aspect = w / max(1.0, float(height))
    py = (np.arange(h, dtype=np.float64) + 0.5 + sub_dy) / h
    px = (np.arange(w, dtype=np.float64) + 0.5 + sub_dx) / w
    sy = (1.0 - 2.0 * py)[:, None] / max(0.25, float(orb_scale))
    sx = ((2.0 * px - 1.0) * aspect)[None, :] / max(0.25, float(orb_scale))

    xx = np.broadcast_to(sx, (h, w))
    yy = np.broadcast_to(sy, (h, w))
    r2 = xx * xx + yy * yy
    r = np.sqrt(r2)
    inside = r <= 1.0

    # 1. Background laboratory / cosmic void with circadian ambient glow
    lx, ly, lz = state.light_dir
    light_I = state.light_intensity
    bg_grad = np.clip(0.12 + 0.25 * (0.5 * (yy + 1.0)) + 0.18 * light_I * np.exp(-0.55 * r2), 0.0, 1.0)
    bg_col = sample_palette(bg_grad * 0.32, anchors) * (0.28 + 0.35 * light_I)
    # Subtle glass pedestal silhouette below the sphere (r_x < 0.42, y < -0.94)
    ped_mask = (yy < -0.92) & ( np.abs(xx) < 0.36 + 0.14 * (-yy - 0.92) )
    ped_shade = np.exp(-6.0 * (xx ** 2)) * (0.18 + 0.15 * light_I)
    bg_col = np.where(ped_mask[..., None], sample_palette(ped_shade, anchors) * 0.45, bg_col)

    # 2. 3D Glass Sphere Normal, Snell Refraction & Fresnel Rim
    nz = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, r2)))
    nx = np.where(inside, xx, 0.0)
    ny = np.where(inside, yy, 0.0)

    # Schlick Fresnel approximation (n_air = 1.0, n_glass = 1.51)
    R0 = ((1.51 - 1.0) / (1.51 + 1.0)) ** 2
    cos_theta = np.clip(nz, 0.0, 1.0)
    fresnel = R0 + (1.0 - R0) * ((1.0 - cos_theta) ** 5)

    # Refracted interior coordinates (borosilicate shell + brackish water column)
    refract_scale = 1.0 / (1.0 + 0.32 * nz)
    ux = xx * refract_scale
    uy = yy * refract_scale

    # 3. Brackish Water Column & Dissolved Oxygen / Circadian Lighting
    water_depth = np.clip(0.55 - 0.45 * uy, 0.0, 1.0)
    o2_tint = np.clip(0.32 + 0.45 * state.O2 - 0.18 * water_depth, 0.05, 0.95)
    water_col = sample_palette(o2_tint, anchors) * (0.35 + 0.55 * light_I)
    # Add aqua/teal photosynthetic scattering proportional to algal & oxygen balance
    aqua_vec = np.array([0.08, 0.46, 0.52], dtype=np.float64)
    water_col = 0.62 * water_col + 0.38 * aqua_vec * (0.4 + 0.6 * state.O2) * (0.4 + 0.6 * light_I)

    # Upper air-pocket meniscus at uy ~ +0.68
    meniscus_y = 0.68 + 0.012 * np.sin(5.0 * ux + 1.5 * state.time)
    in_air_cap = uy > meniscus_y
    meniscus_line = np.exp(-((uy - meniscus_y) / 0.014) ** 2)
    water_col = np.where(
        in_air_cap[..., None],
        water_col * 0.55 + 0.22 * light_I,
        water_col,
    )
    water_col += meniscus_line[..., None] * np.array([0.65, 0.82, 0.92]) * (0.4 + 0.5 * light_I)

    # 4. Basal Basalt / Aragonite Gravel Substrate & Internal Solar Caustic Focus
    gravel_horizon = -0.64 + 0.035 * np.cos(3.2 * ux) + 0.015 * np.sin(9.5 * ux + 0.8)
    in_gravel = uy < gravel_horizon
    ix_g = np.floor((ux + 1.0) * 64.0).astype(np.int32)
    iy_g = np.floor((uy + 1.0) * 64.0).astype(np.int32)
    pebble_hash = _det_hash_2d(ix_g, iy_g, seed=state.seed, salt=S_ECO_GRAVEL)
    gravel_col = sample_palette(0.15 + 0.45 * pebble_hash, anchors) * (0.32 + 0.48 * light_I)

    # Internal lens caustic focused on the opposite lower hemisphere from light_dir
    caustic_cx = -0.38 * lx
    caustic_cy = -0.52 - 0.15 * ly
    caustic_r2 = ((ux - caustic_cx) / 0.34) ** 2 + ((uy - caustic_cy) / 0.16) ** 2
    caustic_ring = np.exp(-((caustic_r2 - 0.45) / 0.28) ** 2) * np.exp(-0.7 * caustic_r2)
    caustic_glow = caustic_ring * light_I * 0.75
    gravel_col += caustic_glow[..., None] * np.array([0.95, 0.88, 0.62])

    water_col = np.where(in_gravel[..., None], gravel_col, water_col)
    water_col += (1.0 - in_gravel.astype(np.float64))[..., None] * caustic_glow[..., None] * 0.28

    # 5. Spatial 2D Gray-Scott Reaction-Diffusion Algal Filaments
    rd_v = _bilinear_sample_2d(state.rd_V, ux, uy)
    algal_mask = np.clip((rd_v - 0.08) / 0.28, 0.0, 1.0) * (1.0 - in_gravel.astype(np.float64)) * (1.0 - in_air_cap.astype(np.float64))
    algal_col = (
        0.55 * sample_palette(np.clip(0.45 + 0.4 * rd_v, 0.0, 1.0), anchors)
        + 0.45 * np.array([0.18, 0.82, 0.44])
    ) * (0.45 + 0.55 * light_I)
    water_col = water_col * (1.0 - 0.68 * algal_mask[..., None]) + algal_col * (0.68 * algal_mask[..., None])

    # 6. Branching Gorgonian L-System Sea-Fan Silhouette
    gorg_sdf = np.full((h, w), 1e6, dtype=np.float64)
    for x0, y0, x1, y1, depth in state.gorgonian_segments:
        d2 = _draw_segment_sdf(ux, uy, x0, y0, x1, y1)
        thickness = (0.015 * (0.78 ** min(depth, 4))) ** 2
        gorg_sdf = np.minimum(gorg_sdf, d2 / max(1e-8, thickness))
    gorg_alpha = np.exp(-gorg_sdf * 1.35) * (1.0 - in_gravel.astype(np.float64))
    gorg_col = np.array([0.24, 0.14, 0.10], dtype=np.float64) * (0.45 + 0.55 * light_I)
    water_col = water_col * (1.0 - 0.88 * gorg_alpha[..., None]) + gorg_col * (0.88 * gorg_alpha[..., None])

    # 7. Swimming Halocaridina rubra Red Shrimp & Wake Trails
    shrimp_layer = np.zeros((h, w), dtype=np.float64)
    trail_layer = np.zeros((h, w), dtype=np.float64)
    for s_i in range(len(state.shrimp_positions)):
        sx_p, sy_p = state.shrimp_positions[s_i]
        vx_p, vy_p = state.shrimp_velocities[s_i]
        ang = math.atan2(vy_p, vx_p)
        ca, sa = math.cos(ang), math.sin(ang)
        dx_s = ux - sx_p
        dy_s = uy - sy_p
        # Oriented elliptical body
        u_par = dx_s * ca + dy_s * sa
        u_perp = -dx_s * sa + dy_s * ca
        body_d2 = (u_par / 0.032) ** 2 + (u_perp / 0.013) ** 2
        shrimp_layer = np.maximum(shrimp_layer, np.exp(-body_d2 * 1.8))

        # Swimming trajectory wake
        pts = state.shrimp_trails[s_i]
        for t_idx in range(0, len(pts) - 1, 2):
            p0 = pts[t_idx]
            p1 = pts[t_idx + 1]
            td2 = _draw_segment_sdf(ux, uy, float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
            fade = (t_idx + 1) / len(pts)
            trail_layer = np.maximum(trail_layer, fade * np.exp(-td2 / (0.006 ** 2)))

    shrimp_crimson = np.array([0.96, 0.22, 0.18], dtype=np.float64) * (0.55 + 0.45 * light_I)
    trail_glow = sample_palette(np.full((h, w), 0.78), anchors) * 0.65
    water_col = water_col * (1.0 - 0.42 * trail_layer[..., None]) + trail_glow * (0.42 * trail_layer[..., None])
    water_col = water_col * (1.0 - 0.92 * shrimp_layer[..., None]) + shrimp_crimson * (0.92 * shrimp_layer[..., None])

    # 8. Rising Oxygen Micro-Bubbles (proportional to photosynthetic O2 production)
    n_bubbles = 12
    bubble_layer = np.zeros((h, w), dtype=np.float64)
    for b_i in range(n_bubbles):
        bx0 = (py_rand01(state.seed, b_i, 1, S_ECO_BUBBLE) - 0.5) * 1.1
        by_phase = py_rand01(state.seed, b_i, 2, S_ECO_BUBBLE)
        by = -0.55 + ((by_phase + 0.22 * state.time * (0.5 + 0.5 * state.O2)) % 1.20)
        bx = bx0 + 0.025 * math.sin(4.0 * state.time + b_i)
        if math.hypot(bx, by) < 0.76 and by < 0.67:
            br = 0.008 + 0.005 * py_rand01(state.seed, b_i, 3, S_ECO_BUBBLE)
            bd2 = (ux - bx) ** 2 + (uy - by) ** 2
            bubble_layer = np.maximum(bubble_layer, np.exp(-(( np.sqrt(bd2) - br ) / 0.004) ** 2))
    water_col += bubble_layer[..., None] * np.array([0.85, 0.95, 1.0]) * (0.4 + 0.6 * light_I)

    # 9. Specular Highlights, Diffuse Shading & Fresnel Glass Envelope
    ndotl = np.maximum(0.0, nx * lx + ny * ly + nz * lz)
    # Half-vector between light_dir and viewer (0, 0, 1)
    hx, hy, hz = lx, ly, lz + 1.0
    h_norm = max(1e-12, math.sqrt(hx * hx + hy * hy + hz * hz))
    hx, hy, hz = hx / h_norm, hy / h_norm, hz / h_norm
    ndoth = np.maximum(0.0, nx * hx + ny * hy + nz * hz)
    spec_primary = (ndoth ** 64.0) * (0.65 + 0.65 * light_I)
    # Secondary internal reflection arc on opposite rim
    ndoth2 = np.maximum(0.0, -nx * hx - ny * hy + nz * hz)
    spec_secondary = (ndoth2 ** 28.0) * 0.28 * light_I

    rim_col = sample_palette(np.clip(0.65 + 0.3 * ndotl, 0.0, 1.0), anchors)
    orb_col = (
        water_col * (0.68 + 0.42 * ndotl[..., None]) * (1.0 - 0.55 * fresnel[..., None])
        + rim_col * (0.75 * fresnel[..., None])
        + (spec_primary + spec_secondary)[..., None] * np.array([1.0, 0.98, 0.94])
    )

    # Ashby Uniselector Step-Reconfiguration Flash / Ring Pulse
    time_since_step = state.time - state.last_step_time
    if state.uniselector_steps > 0 and 0.0 <= time_since_step < 0.45:
        pulse_r = time_since_step / 0.45
        pulse_ring = np.exp(-((r - pulse_r) / 0.045) ** 2) * (1.0 - pulse_r)
        orb_col += pulse_ring[..., None] * np.array([1.0, 0.78, 0.25]) * 0.75

    # Anti-aliased spherical boundary blend
    px_size = 2.0 / max(1.0, float(min(w, h)) * orb_scale)
    edge_alpha = np.clip((1.0 - r) / px_size + 0.5, 0.0, 1.0)
    rgb = bg_col * (1.0 - edge_alpha[..., None]) + orb_col * edge_alpha[..., None]

    # 10. Optional Diegetic Cybernetic Telemetry Overlay (Ashby Homeostat Phase Portrait)
    if telemetry_overlay and aspect >= 1.25 and len(state.history_ashby) >= 2:
        # Place diegetic phase-portrait ring & 4-unit trajectory in the right margin
        hud_cx = 0.74 * aspect / max(0.25, float(orb_scale))
        hud_cy = 0.0
        hud_r = 0.34
        dx_h = xx - hud_cx
        dy_h = yy - hud_cy
        r_h = np.hypot(dx_h, dy_h)

        # Outer HUD reticle ring and critical threshold circle
        reticle = np.exp(-((r_h - hud_r) / (px_size * 1.1)) ** 2) * 0.42
        crit_r = hud_r * min(0.9, state.eta_crit / 0.22)
        crit_ring = np.exp(-((r_h - crit_r) / (px_size * 0.9)) ** 2) * 0.32
        rgb += reticle[..., None] * np.array([0.35, 0.80, 0.92])
        rgb += crit_ring[..., None] * np.array([0.95, 0.42, 0.28])

        # Draw recent Ashby Homeostat (x0, x1) phase-space trajectory
        hist = state.history_ashby[-28:]
        for k_i in range(len(hist) - 1):
            p0x = hud_cx + float(hist[k_i, 0]) * hud_r * 0.82
            p0y = hud_cy + float(hist[k_i, 1]) * hud_r * 0.82
            p1x = hud_cx + float(hist[k_i + 1, 0]) * hud_r * 0.82
            p1y = hud_cy + float(hist[k_i + 1, 1]) * hud_r * 0.82
            sd2 = _draw_segment_sdf(xx, yy, p0x, p0y, p1x, p1y)
            w_fade = (k_i + 1) / len(hist)
            trace_mask = w_fade * np.exp(-sd2 / ((px_size * 1.35) ** 2))
            rgb += trace_mask[..., None] * np.array([0.25, 0.95, 0.72]) * 0.55

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


if _HAS_TAICHI:
    @ti.data_oriented
    class EcosphereEngine:
        """Taichi-compatible wrapper for `EcosphereSimulator` with CPU/NumPy parity."""

        def __init__(self, width: int, height: int, simulator: EcosphereSimulator):
            self.width = int(width)
            self.height = int(height)
            self.simulator = simulator
            self._use_fallback = not ensure_taichi_f64()

        def render(self,
                   state: EcosphereState,
                   palette="trumbull_2001",
                   orb_scale: float = 0.84,
                   telemetry_overlay: bool = True,
                   sub_dx: float = 0.0,
                   sub_dy: float = 0.0) -> np.ndarray:
            return render_ecosphere_frame(
                state,
                self.width,
                self.height,
                palette=palette,
                orb_scale=orb_scale,
                telemetry_overlay=telemetry_overlay,
                sub_dx=sub_dx,
                sub_dy=sub_dy,
            )


# ============================================================
# Native MetaPost Exporter (--export-metapost)
# ============================================================

def _extract_rd_contour_polygons(rd_V: np.ndarray,
                                 threshold: float = 0.16,
                                 max_blobs: int = 16) -> list[np.ndarray]:
    """Extract smooth closed contour polygons in normalized orb coordinates `[-1, 1]^2`
    from local maxima of the 2D Gray-Scott reaction-diffusion field `rd_V`.
    """
    n = rd_V.shape[0]
    y_lin = np.linspace(-1.0, 1.0, n)
    x_lin = np.linspace(-1.0, 1.0, n)
    work = rd_V.copy()
    polygons: list[np.ndarray] = []

    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    for _ in range(max_blobs):
        flat_idx = int(np.argmax(work))
        iy, ix = divmod(flat_idx, n)
        peak_val = float(work[iy, ix])
        if peak_val < threshold:
            break
        cx = float(x_lin[ix])
        cy = float(y_lin[iy])
        if math.hypot(cx, cy) > 0.78:
            work[iy, ix] = 0.0
            continue

        # Trace radial contour around the Turing spot / filament node
        pts = []
        for ang in angles:
            r_found = 0.035
            for r_probe in np.linspace(0.02, 0.14, 8):
                px = cx + r_probe * math.cos(ang)
                py = cy + r_probe * math.sin(ang)
                gx = int(np.clip((px + 1.0) * 0.5 * (n - 1), 0, n - 1))
                gy = int(np.clip((py + 1.0) * 0.5 * (n - 1), 0, n - 1))
                if rd_V[gy, gx] >= threshold * 0.85:
                    r_found = float(r_probe)
                else:
                    break
            pts.append([cx + r_found * math.cos(ang), cy + r_found * math.sin(ang)])
        polygons.append(np.array(pts, dtype=np.float64))

        # Suppress neighborhood to find next distinct contour
        y0, y1 = max(0, iy - 5), min(n, iy + 6)
        x0, x1 = max(0, ix - 5), min(n, ix + 6)
        work[y0:y1, x0:x1] = 0.0

    return polygons


def export_metapost(state: EcosphereState,
                    out_path: str | Path,
                    scale_bp: float = 200.0,
                    palette="trumbull_2001") -> str:
    """Export a self-contained, valid MetaPost diagram (`prologues := 3; beginfig(1); ... endfig;`)
    containing:
      1. The hermetically sealed spherical glass boundary, Fresnel rim, and meniscus waterline.
      2. The L-system Gorgonian sea-fan branching skeleton rooted on the gravel substrate.
      3. The 2D Gray-Scott reaction-diffusion contour polygons (`rd_V`).
      4. Swimming *Halocaridina* red shrimp trajectories.
      5. The 4-unit W. Ross Ashby Ultrastable Homeostat phase-space trajectory and uniselector step markers.
    """
    anchors = resolve_palette_anchors(palette)
    R = float(scale_bp)
    w_bp = 3.55 * R
    h_bp = 2.25 * R
    orb_cx = -0.62 * R
    orb_cy = 0.0
    hud_cx = 0.92 * R
    hud_cy = 0.0
    hud_r = 0.62 * R

    bg_r, bg_g, bg_b = [float(c) for c in anchors[0] * 0.30]
    hi_r, hi_g, hi_b = [float(c) for c in anchors[-1]]
    algal_r, algal_g, algal_b = [float(c) for c in sample_palette(np.array([0.55]), anchors)[0]]
    rim_r, rim_g, rim_b = [float(c) for c in sample_palette(np.array([0.75]), anchors)[0]]

    lines: list[str] = [
        "% Generated by cosmic_engine render.ecosphere (Sealed Glass Ecosphere & Ashby Homeostat)",
        f"% frame: {state.frame_idx}, time: {state.time:.3f}, uniselector_steps: {state.uniselector_steps}, eta: {state.eta:.4f}",
        "prologues := 3;",
        'outputtemplate := "%j-%c.mps";',
        "beginfig(1);",
        f"  numeric R, W, H; R := {R:.3f}bp; W := {w_bp:.3f}bp; H := {h_bp:.3f}bp;",
        "  path plate_box, orb_circle, p;",
        "  plate_box := (-1.75*R, -1.12*R)--(1.80*R, -1.12*R)--(1.80*R, 1.12*R)--(-1.75*R, 1.12*R)--cycle;",
        f"  fill plate_box withcolor ({bg_r:.4f}, {bg_g:.4f}, {bg_b:.4f});",
        "",
        "  % 1. Sealed Spherical Glass Boundary & Water Column",
        f"  orb_circle := fullcircle scaled (1.84*R) shifted ({orb_cx:.2f}bp, {orb_cy:.2f}bp);",
        f"  fill orb_circle withcolor ({bg_r * 1.4:.4f}, {bg_g * 1.6 + 0.06:.4f}, {bg_b * 1.8 + 0.10:.4f});",
    ]

    # Water meniscus chord inside the orb
    men_y = orb_cy + 0.68 * (0.92 * R)
    men_dx = math.sqrt(max(0.0, (0.92 * R) ** 2 - (0.68 * 0.92 * R) ** 2))
    lines.append("  pickup pencircle scaled 0.9bp;")
    lines.append(
        f"  draw ({orb_cx - men_dx:.2f}bp, {men_y:.2f}bp)--({orb_cx + men_dx:.2f}bp, {men_y:.2f}bp) "
        f"withcolor ({rim_r:.4f}, {rim_g:.4f}, {rim_b:.4f});"
    )

    # 2. Reaction-Diffusion Contour Polygons (Gray-Scott Turing morphogenesis)
    lines.append("  % 2. Reaction-Diffusion Algal Contour Polygons")
    rd_polys = _extract_rd_contour_polygons(state.rd_V, threshold=0.14, max_blobs=14)
    lines.append("  pickup pencircle scaled 0.65bp;")
    for poly in rd_polys:
        pts_mp = "..".join(
            f"({orb_cx + float(pt[0]) * 0.90 * R:.2f}bp, {orb_cy + float(pt[1]) * 0.90 * R:.2f}bp)"
            for pt in poly
        )
        lines.append(f"  p := {pts_mp}..cycle;")
        lines.append(f"  fill p withcolor ({0.15 + 0.45 * algal_r:.4f}, {0.45 + 0.45 * algal_g:.4f}, {0.25 + 0.35 * algal_b:.4f});")
        lines.append(f"  draw p withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});")

    # 3. L-System Gorgonian Branching Sea-Fan
    lines.append("  % 3. L-System Gorgonian Sea-Fan Skeleton")
    for x0, y0, x1, y1, depth in state.gorgonian_segments:
        pen_w = max(0.45, 1.55 * (0.76 ** min(depth, 4)))
        gx0 = orb_cx + x0 * 0.90 * R
        gy0 = orb_cy + y0 * 0.90 * R
        gx1 = orb_cx + x1 * 0.90 * R
        gy1 = orb_cy + y1 * 0.90 * R
        lines.append(
            f"  draw ({gx0:.2f}bp,{gy0:.2f}bp)--({gx1:.2f}bp,{gy1:.2f}bp) "
            f"withpen pencircle scaled {pen_w:.2f}bp withcolor (0.5800, 0.3600, 0.2400);"
        )

    # 4. Swimming Halocaridina rubra Red Shrimp Trajectories
    lines.append("  % 4. Halocaridina Red Shrimp Trajectories")
    lines.append("  pickup pencircle scaled 0.85bp;")
    for s_i in range(len(state.shrimp_positions)):
        tr = state.shrimp_trails[s_i][::2]
        tr_str = "--".join(
            f"({orb_cx + float(pt[0]) * 0.90 * R:.2f}bp,{orb_cy + float(pt[1]) * 0.90 * R:.2f}bp)"
            for pt in tr
        )
        lines.append(f"  draw {tr_str} withcolor (0.9500, 0.2600, 0.2000);")
        sx_p = orb_cx + float(state.shrimp_positions[s_i, 0]) * 0.90 * R
        sy_p = orb_cy + float(state.shrimp_positions[s_i, 1]) * 0.90 * R
        lines.append(
            f"  fill fullcircle scaled 3.6bp shifted ({sx_p:.2f}bp,{sy_p:.2f}bp) "
            "withcolor (0.9800, 0.2200, 0.1800);"
        )

    # Draw outer spherical boundary rim
    lines.append("  pickup pencircle scaled 1.6bp;")
    lines.append(f"  draw orb_circle withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});")

    # 5. 4-Unit Ashby Ultrastable Homeostat Phase-Space Trajectory
    lines.append("  % 5. 4-Unit Ashby Homeostat Phase-Space Trajectory")
    lines.append("  path homeostat_ring, crit_ring;")
    lines.append(f"  homeostat_ring := fullcircle scaled ({2.0 * hud_r:.2f}bp) shifted ({hud_cx:.2f}bp, {hud_cy:.2f}bp);")
    crit_rad = hud_r * min(0.88, state.eta_crit / 0.22)
    lines.append(f"  crit_ring := fullcircle scaled ({2.0 * crit_rad:.2f}bp) shifted ({hud_cx:.2f}bp, {hud_cy:.2f}bp);")
    lines.append("  pickup pencircle scaled 1.0bp;")
    lines.append(f"  draw homeostat_ring withcolor ({rim_r:.4f}, {rim_g:.4f}, {rim_b:.4f});")
    lines.append("  pickup pencircle scaled 0.75bp;")
    lines.append("  draw crit_ring dashed evenly withcolor (0.9500, 0.4200, 0.2800);")

    # Phase space axes
    lines.append(
        f"  draw ({hud_cx - hud_r:.2f}bp, {hud_cy:.2f}bp)--({hud_cx + hud_r:.2f}bp, {hud_cy:.2f}bp) "
        "withcolor (0.3500, 0.4500, 0.5500);"
    )
    lines.append(
        f"  draw ({hud_cx:.2f}bp, {hud_cy - hud_r:.2f}bp)--({hud_cx:.2f}bp, {hud_cy + hud_r:.2f}bp) "
        "withcolor (0.3500, 0.4500, 0.5500);"
    )

    if len(state.history_ashby) >= 2:
        hist = state.history_ashby
        step = max(1, len(hist) // 80)
        h_sub = hist[::step]
        # Primary pair (unit 0 vs unit 1) and secondary pair (unit 2 vs unit 3)
        pts_12 = "--".join(
            f"({hud_cx + float(u[0]) * hud_r * 0.85:.2f}bp,{hud_cy + float(u[1]) * hud_r * 0.85:.2f}bp)"
            for u in h_sub
        )
        pts_34 = "--".join(
            f"({hud_cx + float(u[2]) * hud_r * 0.85:.2f}bp,{hud_cy + float(u[3]) * hud_r * 0.85:.2f}bp)"
            for u in h_sub
        )
        lines.append("  pickup pencircle scaled 1.15bp;")
        lines.append(f"  draw {pts_12} withcolor (0.2500, 0.9500, 0.7200);")
        lines.append("  pickup pencircle scaled 0.85bp;")
        lines.append(f"  draw {pts_34} withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});")

    lines.append("  clip currentpicture to plate_box;")
    lines.append("  pickup pencircle scaled 1.1bp;")
    lines.append(f"  draw plate_box withcolor ({hi_r:.4f}, {hi_g:.4f}, {hi_b:.4f});")
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

def _write_image_or_npy(out_path: str, rgb: np.ndarray) -> None:
    if out_path.endswith(".npy"):
        np.save(out_path, rgb)
    else:
        img8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            import imageio.v3 as iio
            iio.imwrite(out_path, img8)
        except ImportError:
            import cv2
            cv2.imwrite(out_path, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ecosphere")
    parser.add_argument("--frame", type=int, default=48,
                        help="target simulation frame index to render")
    parser.add_argument("--rd-size", type=int, default=72)
    parser.add_argument("--n-shrimp", type=int, default=9)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.042)
    parser.add_argument("--circadian-period", type=float, default=1.8)
    parser.add_argument("--o2-star", type=float, default=0.62)
    parser.add_argument("--eta-crit", type=float, default=0.14)
    parser.add_argument("--perturbation-frame", type=int, default=16)
    parser.add_argument("--perturbation-strength", type=float, default=0.21)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--palette", default="trumbull_2001")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-bp", type=float, default=200.0)
    parser.add_argument("--export-metapost", default=None,
                        help="optional path to write a native MetaPost (.mp) figure")
    parser.add_argument("--out", default=None,
                        help="output image (.png or .npy)")
    args = parser.parse_args(argv)

    if not args.out and not args.export_metapost:
        parser.error("at least one of --out or --export-metapost is required")

    sim = EcosphereSimulator(
        seed=args.seed,
        rd_size=args.rd_size,
        n_shrimp=args.n_shrimp,
        warmup_steps=args.warmup_steps,
        substeps_per_frame=args.substeps,
        dt=args.dt,
        circadian_period=args.circadian_period,
        O2_star=args.o2_star,
        eta_crit=args.eta_crit,
        perturbation_frame=args.perturbation_frame,
        perturbation_strength=args.perturbation_strength,
        telemetry_overlay=not args.no_telemetry,
    )
    state = sim.state_at(args.frame)

    if args.export_metapost:
        export_metapost(state, args.export_metapost, scale_bp=args.scale_bp, palette=args.palette)
        print(
            f"exported MetaPost ecosphere figure to {args.export_metapost} "
            f"(frame={args.frame}, uniselector_steps={state.uniselector_steps}, eta={state.eta:.4f})"
        )

    if args.out:
        offsets = supersample_offsets(max(1, int(args.samples)))
        acc = np.zeros((args.height, args.width, 3), dtype=np.float64)
        for dx, dy in offsets:
            acc += render_ecosphere_frame(
                state,
                args.width,
                args.height,
                palette=args.palette,
                telemetry_overlay=not args.no_telemetry,
                sub_dx=dx,
                sub_dy=dy,
            )
        rgb = (acc / len(offsets)).astype(np.float32)
        _write_image_or_npy(args.out, rgb)
        print(
            f"wrote {args.out} ({args.width}x{args.height}, frame={args.frame}, "
            f"uniselector_steps={state.uniselector_steps}, O2={state.O2:.3f}, eta={state.eta:.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
