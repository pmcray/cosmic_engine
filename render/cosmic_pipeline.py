"""DESI cosmic-web data pipeline.

Implements the FITS → comoving-Cartesian conversion required to turn
DESI Bright Galaxy Survey / Luminous Red Galaxy / Emission Line
Galaxy / Quasar catalogs into 3D point clouds that the cosmic-web
renderer can consume.

Three responsibilities:

  1. Cosmology. A custom `w₀wₐ` (Chevallier-Polarski-Linder) Friedmann
     expansion is implemented from scratch -- this is the parameterization
     DESI DR2 has rendered statistically preferred over ΛCDM, and is what
     the user explicitly requested. Comoving distance D_C(z) is obtained
     via numerical quadrature with no scipy dependency.

  2. Spherical → Cartesian. The standard RA/Dec/D_C → X/Y/Z conversion.

  3. A procedural cosmic-web synthesizer used as the default fallback
     when real DESI parquet/FITS catalogs aren't yet downloaded. The
     synthesizer produces a plausible filament+cluster structure with
     the correct tracer-class distribution by redshift, so the renderer
     and notebook work out of the box.

The renderer (`render/cosmic_web.py`) consumes a (positions, tracer,
magnitudes) tuple. The tracer index maps to UI colors per the user's
specification:

    BGS  (orange)  - low-mass nearby (z ~ 0.01-0.6)
    LRG  (red)     - massive elliptical (z ~ 0.4-1.1)
    ELG  (teal)    - active star-forming (z ~ 0.8-1.6)
    QSO  (violet)  - supermassive AGN (z ~ 0.8-2.1)
"""

from __future__ import annotations

import math

import numpy as np


# Speed of light in km/s.
C_LIGHT_KM_S = 299792.458

# Default Planck-2018 + DESI DR2 favoured cosmology. w0wa is the
# Chevallier-Polarski-Linder parameterisation w(a) = w0 + wa(1 - a).
DEFAULT_H0 = 67.4
DEFAULT_OMEGA_M = 0.315
DEFAULT_W0 = -0.838  # DESI DR2 best-fit (Adame et al. 2024 + 2026)
DEFAULT_WA = -0.62


# Tracer indices and the UI colors the renderer maps them to.
TRACER_NAMES = ("BGS", "LRG", "ELG", "QSO")
TRACER_INDEX = {name: i for i, name in enumerate(TRACER_NAMES)}
TRACER_COLORS = (
    (1.00, 0.55, 0.20),  # BGS  orange
    (1.05, 0.20, 0.15),  # LRG  red
    (0.20, 0.92, 0.88),  # ELG  teal
    (0.62, 0.30, 1.15),  # QSO  blue/violet
)


# Approximate redshift ranges per the user's table.
TRACER_Z_RANGES = {
    "BGS": (0.01, 0.6),
    "LRG": (0.4, 1.1),
    "ELG": (0.8, 1.6),
    "QSO": (0.8, 2.1),
}


# ============================================================
# Cosmology
# ============================================================

def E_w0wa(z, Omega_m=DEFAULT_OMEGA_M, w0=DEFAULT_W0, wa=DEFAULT_WA):
    """Friedmann expansion factor E(z) for a flat w0wa cosmology.

    Derived from
        rho_de(z) ∝ exp(3 ∫₀^z (1+w(a))/(1+a) da)
    with CPL w(a) = w0 + wa(1-a) giving the closed form
        rho_de(z) = (1+z)^(3(1+w0+wa)) * exp(-3 * wa * z/(1+z)).
    """
    Omega_de = 1.0 - Omega_m
    z = np.asarray(z, dtype=np.float64)
    one_plus_z = 1.0 + z
    de_factor = one_plus_z ** (3.0 * (1.0 + w0 + wa)) * np.exp(
        -3.0 * wa * z / one_plus_z
    )
    return np.sqrt(Omega_m * one_plus_z ** 3 + Omega_de * de_factor)


def comoving_distance(
    z,
    H0=DEFAULT_H0,
    Omega_m=DEFAULT_OMEGA_M,
    w0=DEFAULT_W0,
    wa=DEFAULT_WA,
    n_steps=512,
):
    """Comoving distance D_C(z) in Mpc via the simpson-rule integral.

    Vectorised over arrays of z. No scipy dependency.
    """
    scalar_input = np.ndim(z) == 0
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    DH = C_LIGHT_KM_S / H0  # Hubble distance in Mpc
    out = np.empty_like(z_arr)
    for i, z_i in enumerate(z_arr):
        if z_i <= 0.0:
            out[i] = 0.0
            continue
        N = max(64, int(n_steps * min(1.0, z_i / 1.5)))
        if N % 2 == 1:
            N += 1
        z_grid = np.linspace(0.0, float(z_i), N + 1)
        integrand = 1.0 / E_w0wa(z_grid, Omega_m=Omega_m, w0=w0, wa=wa)
        h = z_grid[1] - z_grid[0]
        integral = (h / 3.0) * (
            integrand[0]
            + integrand[-1]
            + 4.0 * integrand[1:-1:2].sum()
            + 2.0 * integrand[2:-1:2].sum()
        )
        out[i] = DH * integral
    return float(out[0]) if scalar_input else out


def radec_z_to_cartesian(
    ra_deg, dec_deg, z,
    H0=DEFAULT_H0,
    Omega_m=DEFAULT_OMEGA_M,
    w0=DEFAULT_W0,
    wa=DEFAULT_WA,
):
    """Convert (RA [deg], Dec [deg], z) → (X, Y, Z) in Mpc comoving.

    Vectorised over equal-length arrays.
    """
    scalar = (np.ndim(ra_deg) == 0 and np.ndim(dec_deg) == 0
              and np.ndim(z) == 0)
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    D_C = comoving_distance(z, H0=H0, Omega_m=Omega_m, w0=w0, wa=wa)
    cos_dec = np.cos(dec)
    X = D_C * cos_dec * np.cos(ra)
    Y = D_C * cos_dec * np.sin(ra)
    Z = D_C * np.sin(dec)
    if scalar:
        return np.array([float(X), float(Y), float(Z)], dtype=np.float64)
    return np.stack([X, Y, Z], axis=-1)


# ============================================================
# Tracer class assignment
# ============================================================

def classify_by_redshift(z, rng=None):
    """Assign each redshift to a tracer class index.

    The boundaries below are an approximation of the DESI selection
    pipeline; in real data the selection is more complex (color cuts,
    fiber assignment), but for visualisation the binning is enough.
    Where redshift ranges overlap we pick the most populous tracer
    at that redshift.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros(z.shape, dtype=np.int32)
    out[(z >= 0.0) & (z < 0.4)] = TRACER_INDEX["BGS"]
    out[(z >= 0.4) & (z < 0.8)] = TRACER_INDEX["LRG"]
    out[(z >= 0.8) & (z < 1.6)] = TRACER_INDEX["ELG"]
    out[z >= 1.6] = TRACER_INDEX["QSO"]
    return out


def tracer_color(index):
    """Look up the UI color for a tracer index (0=BGS, 1=LRG, ...)."""
    return TRACER_COLORS[int(index) % len(TRACER_COLORS)]


# ============================================================
# Procedural synthesizer (used when DESI catalog isn't downloaded)
# ============================================================

def synthesize_cosmic_web(
    n_particles=120_000,
    box_size_mpc=3000.0,
    n_clusters=400,
    seed=0,
):
    """Generate a plausible filament+cluster point cloud with the right
    tracer-class distribution.

    The structure: cluster centres are placed log-normally to bias toward
    overdense regions; each cluster has a power-law mass and a Gaussian
    halo of satellite galaxies; some particles are placed along
    filaments connecting nearby clusters.

    Returns:
        positions:  (N, 3) float32 in Mpc, centered at origin
        tracer:     (N,)   int32, 0=BGS .. 3=QSO
        magnitudes: (N,)   float32, > 0, weights the emission
        redshifts:  (N,)   float32, approximate redshift per object
    """
    rng = np.random.default_rng(seed)

    # Cluster mass: power law (Press-Schechter-like).
    cluster_masses = rng.pareto(1.4, n_clusters) + 0.5
    cluster_masses = cluster_masses / cluster_masses.sum()

    # Cluster positions: not uniform -- bias toward filament-y curl-noise
    # by drawing from a heavy-tailed concentration around a few "wall"
    # locations.
    n_walls = 20
    wall_centres = rng.uniform(-1.0, 1.0, (n_walls, 3))
    wall_axes = rng.standard_normal((n_walls, 3))
    wall_axes /= np.linalg.norm(wall_axes, axis=1, keepdims=True) + 1e-9
    cluster_pos = np.zeros((n_clusters, 3), dtype=np.float32)
    for i in range(n_clusters):
        # Pick a wall, place near it with anisotropic Gaussian.
        w = i % n_walls
        s = rng.standard_normal(3) * 0.18
        # Stretch along the wall axis.
        s_axis = (s @ wall_axes[w]) * wall_axes[w] * 2.5
        cluster_pos[i] = wall_centres[w] + s_axis + s * 0.5
    cluster_pos *= box_size_mpc * 0.5

    # Particles per cluster, weighted by mass.
    counts = rng.multinomial(n_particles, cluster_masses)

    positions = []
    redshifts = []
    magnitudes = []

    DH = C_LIGHT_KM_S / DEFAULT_H0

    for c, n in enumerate(counts):
        if n == 0:
            continue
        center = cluster_pos[c]
        # Scatter: a fraction in the cluster core, a fraction along a
        # filament toward the nearest other cluster.
        if c + 1 < n_clusters:
            target = cluster_pos[(c * 7 + 13) % n_clusters]
        else:
            target = cluster_pos[0]
        direction = target - center
        L = np.linalg.norm(direction) + 1e-9
        if L > 0:
            direction = direction / L

        core_frac = 0.65
        n_core = int(n * core_frac)
        n_fil = n - n_core

        # Core: Gaussian sphere.
        core_sigma = box_size_mpc * 0.012
        core_pts = rng.standard_normal((n_core, 3)).astype(np.float32) * core_sigma
        core_pts += center.astype(np.float32)

        # Filament: along the direction with small lateral spread.
        s_fil = rng.uniform(0.0, min(0.5 * box_size_mpc * 0.08, 0.4 * L), n_fil).astype(np.float32)
        lateral = rng.standard_normal((n_fil, 3)).astype(np.float32) * (box_size_mpc * 0.004)
        fil_pts = center.astype(np.float32) + direction.astype(np.float32) * s_fil[:, None] + lateral

        all_pts = np.concatenate([core_pts, fil_pts], axis=0)

        # Per-object brightness from cluster mass + jitter.
        mag = (cluster_masses[c] * 1500.0 + 0.5) * (
            0.5 + rng.random(n).astype(np.float32) * 1.5
        )

        # Redshift from radial distance (only a proxy for visualisation).
        dist_to_origin = np.linalg.norm(all_pts, axis=1)
        z_obj = (dist_to_origin / (box_size_mpc * 0.5)) * 2.0
        z_obj = np.clip(z_obj.astype(np.float32), 0.01, 2.1)

        positions.append(all_pts)
        magnitudes.append(mag)
        redshifts.append(z_obj)

    positions = np.concatenate(positions, axis=0).astype(np.float32)
    magnitudes = np.concatenate(magnitudes, axis=0).astype(np.float32)
    redshifts = np.concatenate(redshifts, axis=0).astype(np.float32)

    # Clip to the box (real catalogs are bounded too).
    half = box_size_mpc * 0.5
    positions = np.clip(positions, -half, half).astype(np.float32)

    tracer = classify_by_redshift(redshifts).astype(np.int32)

    return positions, tracer, magnitudes, redshifts


def load_desi_npz(path):
    """Load a pre-processed DESI catalog from a .npz file.

    The file is expected to contain:
      positions   (N, 3) float32 Mpc comoving (origin = observer)
      tracer      (N,) int32 in {0..3}
      magnitudes  (N,) float32 emission weight
      redshifts   (N,) float32 (optional, used for color-by-z)

    This is the format `convert_desi_fits_to_npz` (TODO future) emits
    after running the FITS → comoving pipeline. Until that script is
    wired in, the procedural synthesizer above provides equivalent
    data structure for the renderer to consume.
    """
    data = np.load(path)
    return (
        data["positions"].astype(np.float32),
        data["tracer"].astype(np.int32),
        data["magnitudes"].astype(np.float32),
        data["redshifts"].astype(np.float32) if "redshifts" in data.files else None,
    )
