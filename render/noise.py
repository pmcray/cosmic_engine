"""Procedural detail synthesis toolkit for Cosmic Engine renderers.

The system-wide invariant: every renderer that produces high-frequency
texture should source it from these primitives rather than rolling its
own. The footprint-aware fBM variants gate each octave by the
screen-space pixel footprint so detail synthesized via this toolkit
NEVER aliases at any zoom level. A nebula or a spiral galaxy looks
amazing at Hubble resolution AND at a Webb-style close-up because the
detail is procedurally infinite, not painted at a fixed pixel grid.

Two layers:

  np_*  : CPU numpy helpers, used by bake pipelines that emit
          low-frequency structure textures (band tables, storm
          placements, density maps).

  ti_*  : Taichi @ti.func helpers, used inside renderer kernels for
          high-frequency detail that resolves under zoom.

Primitives provided:

  Numpy (CPU):
    np_fbm_2d            -- bilinear-upsampled fractional Brownian motion

  Taichi (GPU @ti.func):
    hash3                -- cheap 3D hash
    value_noise_3d       -- trilinear value noise on unit cell grid
    fbm_3d               -- plain multi-octave fBM
    fbm_3d_footprint     -- Nyquist-gated fBM (use whenever output is
                            shown on screen and the camera might zoom)
    ridged_fbm_3d        -- |1 - 2n| transform; sharp ridges for
                            lightning, mountain crests, filaments
    domain_warp_fbm      -- two-pass; low-freq warp + footprint-gated
                            fBM; produces eddy/swirl patterns
    curl_noise_3d        -- divergence-free 3D vector field; advect
                            particles / displace samples for fluid-look
    worley_3d            -- distance to nearest jittered seed; crater
                            fields, packed-cell patterns, dust grains

The principle is the only one that matters: pick a base frequency,
let the footprint kill the octaves that would alias. Detail emerges
under zoom; never falls apart.
"""

import math

import numpy as np

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover - GPU-only dep
    ti = None
    _HAS_TAICHI = False


# ============================================================
# Numpy primitives (CPU-side bake pipelines)
# ============================================================

def np_fbm_2d(shape, octaves=5, persistence=0.55, seed=0):
    """Bilinear-upsampled fBM. Returns float32 zero-mean unit-stddev."""
    H, W = shape
    rng = np.random.default_rng(seed)
    val = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    ny, nx = 4, 8
    for _ in range(octaves):
        n = rng.standard_normal((ny, nx)).astype(np.float32)
        yi = np.linspace(0, ny - 1, H).astype(np.float32)
        xi = np.linspace(0, nx - 1, W).astype(np.float32)
        y0 = yi.astype(np.int32)
        x0 = xi.astype(np.int32)
        y1 = np.minimum(y0 + 1, ny - 1)
        x1 = np.minimum(x0 + 1, nx - 1)
        fy = (yi - y0).astype(np.float32)[:, None]
        fx = (xi - x0).astype(np.float32)[None, :]
        n00 = n[y0[:, None], x0[None, :]]
        n01 = n[y0[:, None], x1[None, :]]
        n10 = n[y1[:, None], x0[None, :]]
        n11 = n[y1[:, None], x1[None, :]]
        lerp = (n00 * (1 - fx) + n01 * fx) * (1 - fy) + (n10 * (1 - fx) + n11 * fx) * fy
        val += lerp * amp
        amp *= persistence
        ny *= 2
        nx *= 2
    val -= val.mean()
    s = val.std()
    if s > 0:
        val /= s
    return val.astype(np.float32)


# ============================================================
# Taichi primitives (GPU @ti.func, callable from any @ti.kernel)
# ============================================================

if _HAS_TAICHI:

    @ti.func
    def hash3(p):
        h = ti.sin(p[0] * 127.1 + p[1] * 311.7 + p[2] * 74.7) * 43758.5453
        return h - ti.floor(h)

    @ti.func
    def value_noise_3d(p):
        pi_v = ti.Vector([ti.floor(p[0]), ti.floor(p[1]), ti.floor(p[2])])
        pf = p - pi_v
        u = pf * pf * (3.0 - 2.0 * pf)
        n000 = hash3(pi_v)
        n100 = hash3(pi_v + ti.Vector([1.0, 0.0, 0.0]))
        n010 = hash3(pi_v + ti.Vector([0.0, 1.0, 0.0]))
        n110 = hash3(pi_v + ti.Vector([1.0, 1.0, 0.0]))
        n001 = hash3(pi_v + ti.Vector([0.0, 0.0, 1.0]))
        n101 = hash3(pi_v + ti.Vector([1.0, 0.0, 1.0]))
        n011 = hash3(pi_v + ti.Vector([0.0, 1.0, 1.0]))
        n111 = hash3(pi_v + ti.Vector([1.0, 1.0, 1.0]))
        nx00 = n000 * (1.0 - u[0]) + n100 * u[0]
        nx10 = n010 * (1.0 - u[0]) + n110 * u[0]
        nx01 = n001 * (1.0 - u[0]) + n101 * u[0]
        nx11 = n011 * (1.0 - u[0]) + n111 * u[0]
        nxy0 = nx00 * (1.0 - u[1]) + nx10 * u[1]
        nxy1 = nx01 * (1.0 - u[1]) + nx11 * u[1]
        return nxy0 * (1.0 - u[2]) + nxy1 * u[2]

    @ti.func
    def fbm_3d(p, octaves):
        """Plain multi-octave fBM (no footprint AA). Use when the
        result is not pixel-bound -- volumetric density fields,
        secondary modulations, etc."""
        val = 0.0
        amp = 1.0
        freq = 1.0
        for _ in range(octaves):
            val += (value_noise_3d(p * freq) - 0.5) * amp
            amp *= 0.55
            freq *= 2.1
        return val

    @ti.func
    def fbm_3d_footprint(p, footprint, base_freq, octaves):
        """Nyquist-gated fBM. Octaves whose period falls below 2*footprint
        are smoothly faded out. Use whenever the result will appear in
        a screen-space pixel that the camera might zoom on."""
        val = 0.0
        amp = 1.0
        freq = base_freq
        for _ in range(octaves):
            period = 1.0 / freq
            x = ti.max(0.0, ti.min(1.0, footprint / period - 0.4))
            weight = 1.0 - x * x * (3.0 - 2.0 * x)
            val += (value_noise_3d(p * freq) - 0.5) * amp * weight
            amp *= 0.55
            freq *= 2.1
        return val

    @ti.func
    def ridged_fbm_3d(p, footprint, base_freq, octaves):
        """Ridged multi-fractal: |1 - 2n| produces sharp ridge crests.
        Good for lightning, mountain ridges, dust filaments aligned to
        flow, dendritic structures."""
        val = 0.0
        amp = 1.0
        freq = base_freq
        for _ in range(octaves):
            period = 1.0 / freq
            x = ti.max(0.0, ti.min(1.0, footprint / period - 0.4))
            weight = 1.0 - x * x * (3.0 - 2.0 * x)
            n = value_noise_3d(p * freq)
            ridge = 1.0 - ti.abs(1.0 - 2.0 * n)
            val += ridge * ridge * amp * weight
            amp *= 0.5
            freq *= 2.1
        return val

    @ti.func
    def domain_warp_fbm(p, footprint, base_freq, octaves, warp_freq, warp_amount):
        """Two-pass: a low-frequency noise field computes a 3D warp
        vector that perturbs the sample position before fBM
        evaluation. Produces eddy / swirl patterns instead of
        isotropic noise -- the right look for clouds, dust lanes,
        gas-giant turbulence, nebula filaments."""
        warp_p = p * warp_freq
        wx = value_noise_3d(warp_p) - 0.5
        wy = value_noise_3d(warp_p + ti.Vector([7.3, 0.0, 0.0])) - 0.5
        wz = value_noise_3d(warp_p + ti.Vector([0.0, 13.7, 0.0])) - 0.5
        warp_vec = ti.Vector([wx, wy, wz]) * warp_amount
        return fbm_3d_footprint(p + warp_vec, footprint, base_freq, octaves)

    @ti.func
    def worley_3d(p):
        """Distance to nearest seed point in jittered unit cells.
        Returns a value in roughly [0, 1.5]; lower means closer to a
        seed. Crater fields, packed cellular patterns, dust grain
        distributions, neuron-cell layouts."""
        pi_v = ti.Vector([ti.floor(p[0]), ti.floor(p[1]), ti.floor(p[2])])
        min_dist = 2.0
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                for dz in ti.static(range(-1, 2)):
                    cell = pi_v + ti.Vector([float(dx), float(dy), float(dz)])
                    h1 = hash3(cell)
                    h2 = hash3(cell + ti.Vector([3.7, 0.0, 0.0]))
                    h3 = hash3(cell + ti.Vector([0.0, 7.3, 0.0]))
                    seed = cell + ti.Vector([h1, h2, h3])
                    d = (p - seed).norm()
                    min_dist = ti.min(min_dist, d)
        return min_dist

    @ti.func
    def curl_noise_3d(p):
        """Divergence-free 3D vector field. Derived from finite
        differences of three independent scalar potential fields. Use
        to advect samples or distort fields so they flow with the
        coherence of incompressible motion -- nebula filaments,
        smoke plumes, gas-giant zonal jets, ocean wave systems."""
        eps = 0.01
        e1 = ti.Vector([eps, 0.0, 0.0])
        e2 = ti.Vector([0.0, eps, 0.0])
        e3 = ti.Vector([0.0, 0.0, eps])
        o1 = ti.Vector([0.0, 0.0, 0.0])
        o2 = ti.Vector([23.4, 31.7, 11.3])
        o3 = ti.Vector([41.1, 7.3, 19.5])
        inv2e = 1.0 / (2.0 * eps)
        psi1_dy = (value_noise_3d(p + o1 + e2) - value_noise_3d(p + o1 - e2)) * inv2e
        psi1_dz = (value_noise_3d(p + o1 + e3) - value_noise_3d(p + o1 - e3)) * inv2e
        psi2_dx = (value_noise_3d(p + o2 + e1) - value_noise_3d(p + o2 - e1)) * inv2e
        psi2_dz = (value_noise_3d(p + o2 + e3) - value_noise_3d(p + o2 - e3)) * inv2e
        psi3_dx = (value_noise_3d(p + o3 + e1) - value_noise_3d(p + o3 - e1)) * inv2e
        psi3_dy = (value_noise_3d(p + o3 + e2) - value_noise_3d(p + o3 - e2)) * inv2e
        return ti.Vector([
            psi3_dy - psi2_dz,
            psi1_dz - psi3_dx,
            psi2_dx - psi1_dy,
        ])

else:
    # CPU-only stubs so the symbol always exists. Calling raises
    # immediately rather than failing at GPU kernel compile time.
    def hash3(*a, **k):
        raise RuntimeError("hash3 requires Taichi")

    def value_noise_3d(*a, **k):
        raise RuntimeError("value_noise_3d requires Taichi")

    def fbm_3d(*a, **k):
        raise RuntimeError("fbm_3d requires Taichi")

    def fbm_3d_footprint(*a, **k):
        raise RuntimeError("fbm_3d_footprint requires Taichi")

    def ridged_fbm_3d(*a, **k):
        raise RuntimeError("ridged_fbm_3d requires Taichi")

    def domain_warp_fbm(*a, **k):
        raise RuntimeError("domain_warp_fbm requires Taichi")

    def worley_3d(*a, **k):
        raise RuntimeError("worley_3d requires Taichi")

    def curl_noise_3d(*a, **k):
        raise RuntimeError("curl_noise_3d requires Taichi")


# ============================================================
# Recommended octave / frequency defaults
# ============================================================
#
# Renderers should pick a `base_freq` that maps to "one full period
# around the object" -- e.g. base_freq=40 means 40 cycles around the
# circumference. The first few octaves carry the low-freq look; the
# footprint gate handles the rest.
#
# Per-domain suggestions:
#   gas giant / cloud detail   : base_freq=40,  octaves=6
#   galaxy dust lanes          : base_freq=10,  octaves=7
#   nebula filaments           : base_freq=8,   octaves=7
#   mountain ridges (ridged)   : base_freq=20,  octaves=8
#   planetary surface texture  : base_freq=80,  octaves=6
#   crater fields (worley)     : freq=12 (single pass)
#   advection warps (curl)     : freq=6  (single pass)
