"""Volumetric Jupiter / gas-giant renderer (Juno-quality target).

A fork of the existing `gas_giant` adapter. The new pipeline keeps the
old one untouched so the Studio loop can score them side-by-side.

What it does differently:

  1. Atmospheric ray-march on a thin Rayleigh-Mie shell around the
     planet. Limb darkening, twilight terminator, and forward-scatter
     glow all fall out of the same integral.
  2. Three-layer cloud composite (NH3 ice, NH4SH, water clouds) baked
     into the surface color via Beer-Lambert. Rare windows where all
     three layers are thin reveal the deep blue layer ("5-micron
     hotspots").
  3. Realistic latitudinal banding from the Jovian zonal-wind table
     (SPR -> SSTZ -> ... -> NPR) with per-band drift speeds derived
     from observed jet velocities.
  4. Storm features parameterized from real Jovian locations:
       - Great Red Spot at lat=-22.5
       - White ovals along the South Temperate Belt
       - Brown barges along the North Equatorial Belt
       - Polar cyclone arrays (1 central + 8 surrounding at lat=+/-83),
         matching the Juno polar discovery
  5. Kelvin-Helmholtz roll perturbations at zone/belt boundaries.

The bake (numpy) happens once at construction. The render kernel
(Taichi) consumes the baked textures and re-samples them per frame
with per-band longitude offsets so the bands and storms drift
independently. No fluid simulation is needed — band-rate shearing of
the static texture gives the apparent motion.

Camera, sun direction, and feature longitudes are all driven from the
Renderer adapter so the Producer can mutate them through manifests.
"""

import math

import numpy as np

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover - GPU-only dep
    ti = None
    _HAS_TAICHI = False


# --- Static configuration tables -----------------------------------------

# (lat_min, lat_max, name, base_color RGB, density_mult, wind m/s)
JOVIAN_BANDS = (
    (-90, -75, "SPR",  (0.45, 0.35, 0.28), 0.92,  +25.0),
    (-75, -55, "SSTZ", (0.85, 0.78, 0.62), 0.82,  -25.0),
    (-55, -42, "STZ",  (0.88, 0.80, 0.65), 0.88,  -20.0),
    (-42, -28, "STB",  (0.45, 0.32, 0.22), 0.95,  +50.0),
    (-28, -20, "STrZ", (0.90, 0.82, 0.68), 0.86,  +40.0),
    (-20,  -7, "SEB",  (0.62, 0.38, 0.27), 0.96,  -40.0),
    ( -7,   7, "EZ",   (0.92, 0.86, 0.70), 0.83, +100.0),
    (  7,  20, "NEB",  (0.58, 0.36, 0.25), 0.96,  -20.0),
    ( 20,  28, "NTrZ", (0.90, 0.82, 0.68), 0.86,  +40.0),
    ( 28,  42, "NTB",  (0.48, 0.34, 0.24), 0.93, +130.0),
    ( 42,  55, "NTZ",  (0.86, 0.78, 0.62), 0.85,  +50.0),
    ( 55,  75, "NNTZ", (0.84, 0.76, 0.60), 0.80,  -20.0),
    ( 75,  90, "NPR",  (0.42, 0.32, 0.26), 0.90,  +30.0),
)

LAYER_COLORS = (
    (0.93, 0.89, 0.78),  # NH3 ice (top)
    (0.66, 0.42, 0.24),  # NH4SH (middle)
    (0.16, 0.22, 0.42),  # H2O ice (deep)
)


# --- Numpy baking pipeline -----------------------------------------------

def _fbm_2d(shape, octaves=5, persistence=0.55, seed=0):
    """Cheap bilinear-upsampled fractional Brownian motion."""
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


def _ellipse_mask(lat_grid, lon_grid, c_lat, c_lon, r_lat, r_lon, intensity=1.0, falloff=2.0):
    dlon = ((lon_grid - c_lon + 180.0) % 360.0) - 180.0
    dlat = lat_grid - c_lat
    e = (dlat / r_lat) ** 2 + (dlon / r_lon) ** 2
    return (np.exp(-e * falloff) * intensity).astype(np.float32)


def _apply_overlay(color_map, alpha_map, mask, color):
    a = mask
    keep = 1.0 - a
    color_map[..., 0] = color_map[..., 0] * keep + color[0] * a
    color_map[..., 1] = color_map[..., 1] * keep + color[1] * a
    color_map[..., 2] = color_map[..., 2] * keep + color[2] * a
    np.maximum(alpha_map, mask, out=alpha_map)


def _smooth_2d(arr, n_passes=2):
    """Light separable box blur for anti-aliasing band boundaries.

    Operates in-place semantically. Longitude wraps via np.roll; the
    latitude axis uses edge-replicate by clipping the rolled indices.
    A 3-tap kernel iterated n_passes times approximates a Gaussian.
    """
    for _ in range(n_passes):
        # Latitude axis: edge-replicate by clamping the shift.
        up = np.concatenate([arr[:1], arr[:-1]], axis=0)
        dn = np.concatenate([arr[1:], arr[-1:]], axis=0)
        arr = (up + arr + dn) / 3.0
        # Longitude axis: wraps naturally.
        arr = (np.roll(arr, 1, axis=1) + arr + np.roll(arr, -1, axis=1)) / 3.0
    return arr.astype(np.float32)


def _saturate(color_arr, amount=1.3):
    """Multiply chromatic distance from luminance to deepen colors."""
    luma = color_arr.mean(axis=-1, keepdims=True)
    return (luma + (color_arr - luma) * amount).astype(np.float32)


def bake_textures(
    height=512,
    width=1024,
    seed=0,
    grs_lon=-60.0,
    white_oval_lons=(-150.0, -80.0, 30.0, 110.0, 180.0),
    brown_barge_lons=(-100.0, 0.0, 100.0),
):
    """Bake the planet-surface color and per-row drift omega.

    Returns:
        surface_color: float32 (H, W, 3) — composite of three cloud
            decks, lit by ambient Lambert at 1.0 sun.
        layer_density: float32 (H, W, 3) — per-layer optical depth.
        band_omega: float32 (H,) — radians per simulated second of
            longitude drift for each row; resolves to band rotation in
            the kernel.
    """
    lat = np.linspace(-90.0, 90.0, height, dtype=np.float32)
    lon = np.linspace(-180.0, 180.0, width, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    base_color = np.zeros((height, width, 3), dtype=np.float32)
    base_density = np.zeros((height, width), dtype=np.float32)
    band_omega_per_row = np.zeros((height,), dtype=np.float32)

    R_eq_m = 4.4e8 / (2.0 * math.pi)
    drift_aesthetic = 8.0e4

    # ---- 1. Wavy band boundaries -------------------------------------
    # Real Jovian bands are not straight — they undulate from K-H rolls.
    # Displace the latitude at which we look up the band assignment by
    # a multi-scale fBM + a long-wavelength sinusoid.
    lat_wave_fbm = _fbm_2d((height, width), octaves=4, persistence=0.55,
                           seed=seed + 11) * 2.2
    lat_wave_sin = (
        np.sin(lon2d * (math.pi / 180.0) * 2.0) * 1.0
        + np.sin(lon2d * (math.pi / 180.0) * 5.0) * 0.45
    ).astype(np.float32)
    effective_lat = lat2d + lat_wave_fbm + lat_wave_sin

    # Per-pixel band assignment (wavy). Per-row omega still uses real lat.
    for (lo, hi, _name, color, dens, wind) in JOVIAN_BANDS:
        mask = (effective_lat >= lo) & (effective_lat < hi)
        for k in range(3):
            base_color[mask, k] = color[k]
        base_density[mask] = dens
        row_mask = (lat >= lo) & (lat <= hi)
        band_omega_per_row[row_mask] = (wind / R_eq_m) * drift_aesthetic

    band_omega_per_row = np.convolve(
        band_omega_per_row, np.ones(5) / 5.0, mode="same"
    ).astype(np.float32)

    # ---- 2. Chromatic band texture (turbulent eddies in COLOR) -------
    # Two-octave fBM modulates each channel slightly differently so the
    # noise reads as varying pigment, not just brightness.
    chrom_a = _fbm_2d((height, width), octaves=6, persistence=0.55, seed=seed + 21)
    chrom_b = _fbm_2d((height, width), octaves=5, persistence=0.45, seed=seed + 22)
    base_color[..., 0] += chrom_a * 0.07 + chrom_b * 0.04
    base_color[..., 1] += chrom_a * 0.05 + chrom_b * 0.025
    base_color[..., 2] += chrom_a * 0.025 + chrom_b * 0.015

    # Belts kick up an extra noise-driven darkening (turbulent eddies).
    is_belt = base_density > 0.93
    belt_eddy = np.abs(_fbm_2d((height, width), octaves=5, persistence=0.5,
                               seed=seed + 23)) * 0.10
    base_color[is_belt, 0] -= belt_eddy[is_belt]
    base_color[is_belt, 1] -= belt_eddy[is_belt] * 0.8
    base_color[is_belt, 2] -= belt_eddy[is_belt] * 0.6

    # ---- 3. Multi-scale fBM in band density --------------------------
    n_big = _fbm_2d((height, width), octaves=6, persistence=0.55, seed=seed)
    n_small = _fbm_2d((height, width), octaves=5, persistence=0.45, seed=seed + 1)
    band_density = base_density + n_big * 0.18 + n_small * 0.08
    band_density[is_belt] += np.abs(n_small[is_belt]) * 0.10

    # ---- 4. Kelvin-Helmholtz rolls visible in color + density --------
    for lat_b, wavelen in (
        (-20.0, 18.0), (-7.0, 24.0), (7.0, 24.0), (20.0, 18.0), (28.0, 15.0),
    ):
        env = np.exp(-((lat2d - lat_b) / 1.8) ** 2).astype(np.float32)
        roll = np.sin(
            lon2d * (360.0 / wavelen) * math.pi / 180.0
            + 0.4 * np.sin(lon2d * 0.05)
        ).astype(np.float32)
        band_density += env * roll * 0.07
        # K-H rolls also locally darken/lighten color so they're visible
        # in the final composite, not just hidden in cloud density.
        base_color[..., 0] -= env * roll * 0.06
        base_color[..., 1] -= env * roll * 0.045
        base_color[..., 2] -= env * roll * 0.03

    # ---- 4.5 Anti-alias the wavy band assignment + boost saturation --
    base_color = _smooth_2d(np.clip(base_color, 0.0, 1.5), n_passes=2)
    base_color = _saturate(base_color, amount=1.28)

    # ---- 4.6 Polar darkening (brownish-blue at the poles) ------------
    polar = np.clip((np.abs(lat2d) - 60.0) / 30.0, 0.0, 1.0)
    polar_tint = np.array([0.38, 0.40, 0.46], dtype=np.float32)
    base_color = (
        base_color * (1.0 - polar[..., None] * 0.45)
        + polar_tint[None, None, :] * polar[..., None] * 0.45
    ).astype(np.float32)

    # ---- 6. Three-layer cloud composite -------------------------------
    # Layer 1 (NH3 ice) density follows storm_color "whitishness".
    # Layer 2 (NH4SH) is the background brown beneath the top layer.
    # Layer 3 (H2O) only shows where both layers above are thin.
    rng = np.random.default_rng(seed + 7)

    # Top layer is thicker in zones (whiter base) and thinner in belts.
    layer_top_d = np.clip(0.4 + 0.7 * (base_density - 0.85) * 5.0
                          + 0.4 * _fbm_2d((height, width), octaves=5,
                                          persistence=0.5, seed=seed + 2), 0.0, 2.5)
    # Middle layer fills in everywhere with belt-scale variation.
    layer_mid_d = np.clip(0.6 + 0.4 * _fbm_2d((height, width), octaves=5,
                                              persistence=0.5, seed=seed + 3), 0.0, 2.5)
    # Deep layer is sparse — only present where both layers above thin (hotspots).
    sparse_mask = (rng.random((height, width)) > 0.985).astype(np.float32)
    layer_deep_d = sparse_mask * (0.6 + 0.4 * rng.random((height, width)).astype(np.float32))

    # Back-to-front composite, Beer-Lambert.
    a3 = 1.0 - np.exp(-layer_deep_d * 2.5)
    surface = (np.array(LAYER_COLORS[2], dtype=np.float32)[None, None, :]
               * a3[..., None])
    a2 = 1.0 - np.exp(-layer_mid_d * 1.6)
    surface = (np.array(LAYER_COLORS[1], dtype=np.float32)[None, None, :]
               * a2[..., None] + surface * (1.0 - a2[..., None]))
    a1 = 1.0 - np.exp(-layer_top_d * 1.2)
    surface = (np.array(LAYER_COLORS[0], dtype=np.float32)[None, None, :]
               * a1[..., None] + surface * (1.0 - a1[..., None]))

    blended = surface * 0.40 + base_color * 0.60

    # ---- 7. Storms applied AFTER the layer blend (full strength) -----
    storm_alpha = np.zeros((height, width), dtype=np.float32)

    # 7a. Great Red Spot: wider, lower-falloff so the halo extends, more
    # saturated reds across the three concentric ellipses, plus a
    # trailing wake on the east side.
    grs_outer = _ellipse_mask(lat2d, lon2d, -22.5, grs_lon, 11.5, 26.0,
                              intensity=0.90, falloff=1.6)
    grs_body = _ellipse_mask(lat2d, lon2d, -22.5, grs_lon, 7.0, 16.0,
                             intensity=0.96, falloff=2.0)
    grs_core = _ellipse_mask(lat2d, lon2d, -22.5, grs_lon, 3.2, 8.0,
                             intensity=0.90, falloff=2.6)
    _apply_overlay(blended, storm_alpha, grs_outer * 0.85,
                   color=(0.95, 0.44, 0.20))
    _apply_overlay(blended, storm_alpha, grs_body,
                   color=(0.88, 0.24, 0.10))
    _apply_overlay(blended, storm_alpha, grs_core,
                   color=(0.72, 0.13, 0.06))

    wake_lon = grs_lon + 26.0
    wake_mask = _ellipse_mask(lat2d, lon2d, -22.5, wake_lon, 4.5, 22.0,
                              intensity=0.45, falloff=1.4)
    wake_roll = np.clip(
        np.sin((lon2d - grs_lon) * 0.5 * math.pi / 180.0) * 0.6, 0.0, 1.0
    )
    _apply_overlay(blended, storm_alpha, wake_mask * wake_roll,
                   color=(0.85, 0.50, 0.30))

    # 7b. White ovals along the STZ.
    for lon_c in white_oval_lons:
        m = _ellipse_mask(lat2d, lon2d, -33.0, lon_c, 2.6, 5.4,
                          intensity=0.78, falloff=2.3)
        _apply_overlay(blended, storm_alpha, m, color=(0.96, 0.94, 0.88))

    # 7c. Brown barges along the NEB.
    for lon_c in brown_barge_lons:
        m = _ellipse_mask(lat2d, lon2d, 18.0, lon_c, 2.0, 13.0,
                          intensity=0.62, falloff=1.9)
        _apply_overlay(blended, storm_alpha, m, color=(0.26, 0.15, 0.08))

    # 7d. Polar cyclones: central + 8 surrounding at lat = +/-83.
    for sign in (+1, -1):
        m_c = _ellipse_mask(lat2d, lon2d, sign * 89.0, 0.0, 4.0, 30.0,
                            intensity=0.55, falloff=2.0)
        _apply_overlay(blended, storm_alpha, m_c, color=(0.78, 0.58, 0.40))
        for k in range(8):
            lc = -180.0 + 45.0 * k
            m_k = _ellipse_mask(lat2d, lon2d, sign * 83.0, lc, 3.0, 20.0,
                                intensity=0.55, falloff=2.0)
            _apply_overlay(blended, storm_alpha, m_k,
                           color=(0.82, 0.62, 0.42))

    layer_density = np.stack([layer_top_d, layer_mid_d, layer_deep_d], axis=-1).astype(np.float32)
    return blended.astype(np.float32), layer_density, band_omega_per_row


# --- Taichi renderer -----------------------------------------------------

if not _HAS_TAICHI:

    class VolumetricGasGiantEngine:  # pragma: no cover - GPU-only path
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "VolumetricGasGiantEngine requires Taichi. Install taichi and "
                "ensure a GPU runtime is available."
            )

else:

  @ti.data_oriented
  class VolumetricGasGiantEngine:
    """Volumetric Jovian renderer.

    Renders directly into a (height, width) buffer (no separate
    resize step), so the planet stays circular at any output aspect.
    Planet sits at the origin with `planet_radius = 1`; the atmosphere
    shell extends to `1 + atm_thickness`.
    """

    def __init__(
        self,
        width=1024,
        height=1024,
        tex_height=512,
        tex_width=1024,
        atm_thickness=0.06,
        march_steps=30,
        samples=2,
        sun_dir=(-0.55, 0.18, -0.81),
        grs_lon=-60.0,
        seed=0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.aspect = float(self.width) / float(self.height)
        self.tex_h = int(tex_height)
        self.tex_w = int(tex_width)
        self.atm_thickness = float(atm_thickness)
        self.march_steps = int(march_steps)
        self.samples = int(samples)

        # Output buffer is (H, W) so the renderer can produce 16:9
        # (or any rectangular) frames natively.
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

        # Baked textures.
        surface_color, layer_density, band_omega = bake_textures(
            height=self.tex_h, width=self.tex_w, seed=seed, grs_lon=grs_lon,
        )
        self.surface_color = ti.Vector.field(3, dtype=ti.f32, shape=(self.tex_h, self.tex_w))
        self.layer_density = ti.Vector.field(3, dtype=ti.f32, shape=(self.tex_h, self.tex_w))
        self.band_omega = ti.field(dtype=ti.f32, shape=(self.tex_h,))
        self.surface_color.from_numpy(surface_color)
        self.layer_density.from_numpy(layer_density)
        self.band_omega.from_numpy(band_omega)

        # Camera + sun parameters in a flat 16-float buffer.
        # [0..2]  camera position
        # [3..5]  camera forward
        # [6..8]  camera right
        # [9..11] camera up
        # [12]    fov_scale = tan(vertical_fov/2)
        # [13..15] sun direction (normalized)
        self.cam = ti.field(dtype=ti.f32, shape=(16,))
        self._set_default_camera()
        self.set_sun(sun_dir)

    # ---- Python-side helpers ------------------------------------------

    def _set_default_camera(self):
        self.set_camera(
            position=(0.0, 0.0, -3.2),
            target=(0.0, 0.0, 0.0),
            up_world=(0.0, 1.0, 0.0),
            fov_deg=35.0,
        )

    def set_camera(self, position, target, up_world, fov_deg):
        position = np.asarray(position, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up_world = np.asarray(up_world, dtype=np.float32)
        fwd = target - position
        fwd /= np.linalg.norm(fwd) + 1e-8
        right = np.cross(fwd, up_world)
        nr = np.linalg.norm(right)
        if nr < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= nr
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up) + 1e-8
        fov_scale = math.tan(math.radians(float(fov_deg)) * 0.5)
        arr = np.zeros(16, dtype=np.float32)
        arr[0:3] = position
        arr[3:6] = fwd
        arr[6:9] = right
        arr[9:12] = up
        arr[12] = fov_scale
        arr[13:16] = self.cam.to_numpy()[13:16] if hasattr(self.cam, "to_numpy") else 0.0
        self.cam.from_numpy(arr)

    def set_sun(self, direction):
        d = np.asarray(direction, dtype=np.float32)
        d /= np.linalg.norm(d) + 1e-8
        arr = self.cam.to_numpy()
        arr[13:16] = d
        self.cam.from_numpy(arr)

    def render(self, t: float) -> np.ndarray:
        self.render_kernel(float(t))
        return self.pixels.to_numpy()

    # ---- Taichi kernels ------------------------------------------------

    @ti.func
    def _sample_surface(self, lat_deg, lon_deg, t):
        # Per-row drift in longitude: lon_eff = lon_sample - omega * t
        row = ti.cast(
            ti.min(self.tex_h - 1, ti.max(0, int((lat_deg + 90.0) / 180.0 * self.tex_h))),
            ti.i32,
        )
        omega = self.band_omega[row]
        lon_eff = lon_deg - omega * t * 57.2957795  # rad -> deg
        # Wrap into [-180, 180].
        lon_eff = lon_eff - 360.0 * ti.floor((lon_eff + 180.0) / 360.0)
        col = ti.cast(
            ti.min(self.tex_w - 1, ti.max(0, int((lon_eff + 180.0) / 360.0 * self.tex_w))),
            ti.i32,
        )
        return self.surface_color[row, col]

    # ---- Procedural detail synthesis -----------------------------------
    # The baked surface_color carries low-frequency structure (bands,
    # named storms, K-H rolls). High-frequency cloud filigree is
    # synthesized procedurally in world space here, so as the camera
    # zooms in, additional fBM octaves resolve rather than the texels
    # of a fixed-resolution texture becoming visible. Footprint-aware
    # octave weighting fades any octave whose period is smaller than
    # the screen-space pixel footprint, keeping the output alias-free
    # at every zoom level.

    @ti.func
    def _hash3(self, p):
        h = ti.sin(p[0] * 127.1 + p[1] * 311.7 + p[2] * 74.7) * 43758.5453
        return h - ti.floor(h)

    @ti.func
    def _value_noise_3d(self, p):
        pi = ti.Vector([ti.floor(p[0]), ti.floor(p[1]), ti.floor(p[2])])
        pf = p - pi
        u = pf * pf * (3.0 - 2.0 * pf)

        n000 = self._hash3(pi)
        n100 = self._hash3(pi + ti.Vector([1.0, 0.0, 0.0]))
        n010 = self._hash3(pi + ti.Vector([0.0, 1.0, 0.0]))
        n110 = self._hash3(pi + ti.Vector([1.0, 1.0, 0.0]))
        n001 = self._hash3(pi + ti.Vector([0.0, 0.0, 1.0]))
        n101 = self._hash3(pi + ti.Vector([1.0, 0.0, 1.0]))
        n011 = self._hash3(pi + ti.Vector([0.0, 1.0, 1.0]))
        n111 = self._hash3(pi + ti.Vector([1.0, 1.0, 1.0]))

        nx00 = n000 * (1.0 - u[0]) + n100 * u[0]
        nx10 = n010 * (1.0 - u[0]) + n110 * u[0]
        nx01 = n001 * (1.0 - u[0]) + n101 * u[0]
        nx11 = n011 * (1.0 - u[0]) + n111 * u[0]

        nxy0 = nx00 * (1.0 - u[1]) + nx10 * u[1]
        nxy1 = nx01 * (1.0 - u[1]) + nx11 * u[1]

        return nxy0 * (1.0 - u[2]) + nxy1 * u[2]

    @ti.func
    def _detail_fbm(self, p, footprint):
        # 6 octaves starting at 40 cycles around the planet. Each octave's
        # contribution is gated by a smoothstep: octaves whose period is
        # below ~2 * footprint (Nyquist limit) are faded out so the
        # detail never aliases regardless of zoom.
        val = 0.0
        amp = 1.0
        freq = 40.0
        for _ in range(6):
            period = 1.0 / freq
            x = ti.max(0.0, ti.min(1.0, footprint / period - 0.4))
            weight = 1.0 - x * x * (3.0 - 2.0 * x)
            val += (self._value_noise_3d(p * freq) - 0.5) * amp * weight
            amp *= 0.55
            freq *= 2.1
        return val

    @ti.func
    def _surface_detail(self, p_surf, footprint):
        # Low-freq domain warp creates eddy-like swirls in the high-freq
        # detail. Two passes: warp the sample point, then fBM at the
        # warped position.
        warp_p = p_surf * 12.0
        wx = self._value_noise_3d(warp_p) - 0.5
        wy = self._value_noise_3d(warp_p + ti.Vector([7.3, 0.0, 0.0])) - 0.5
        wz = self._value_noise_3d(warp_p + ti.Vector([0.0, 13.7, 0.0])) - 0.5
        warp_vec = ti.Vector([wx, wy, wz]) * 0.35
        return self._detail_fbm(p_surf + warp_vec, footprint)

    @ti.kernel
    def render_kernel(self, t: ti.f32):
        cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
        cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
        cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
        cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
        fov_scale = self.cam[12]
        sun_dir = ti.Vector([self.cam[13], self.cam[14], self.cam[15]])

        R_p = 1.0
        R_atm = 1.0 + self.atm_thickness
        N = self.march_steps

        rayleigh_color = ti.Vector([0.32, 0.58, 1.05])
        mie_color = ti.Vector([1.05, 0.97, 0.85])
        sun_color = ti.Vector([1.0, 0.97, 0.92])
        # Rim-glow tint: cool blue Rayleigh haze that paints onto the
        # surface where the line of sight grazes the lit limb.
        rim_haze = ti.Vector([0.50, 0.66, 1.05])

        for i, j in self.pixels:
            accum = ti.Vector([0.0, 0.0, 0.0])

            for s in range(self.samples):
                # Stratified jitter.
                jx = (ti.random() - 0.5) / self.samples
                jy = (ti.random() - 0.5) / self.samples
                # u in [-1, 1] across width, v in [-1, 1] across height.
                # Multiply u by aspect so a unit angular step right matches
                # a unit angular step up -- keeps the planet circular at any
                # output aspect (fov_scale is half-vertical-fov).
                u = ((float(j) + 0.5 + jx) / self.width) * 2.0 - 1.0
                v = ((float(i) + 0.5 + jy) / self.height) * 2.0 - 1.0
                v = -v

                rd = (
                    cam_fwd
                    + cam_right * (u * fov_scale * self.aspect)
                    + cam_up * (v * fov_scale)
                )
                rd = rd / (rd.norm() + 1e-8)

                # Sphere intersections (with planet center at origin).
                b = rd.dot(cam_pos)
                co = cam_pos.dot(cam_pos) - R_atm * R_atm
                cp = cam_pos.dot(cam_pos) - R_p * R_p
                d_atm = b * b - co
                d_planet = b * b - cp

                color = self._background(rd)

                if d_atm > 0.0:
                    t_in = -b - ti.sqrt(d_atm)
                    t_out = -b + ti.sqrt(d_atm)
                    if t_out > 0.0:
                        t_start = ti.max(0.0, t_in)
                        t_end = t_out
                        hit_planet = 0
                        if d_planet > 0.0:
                            tp = -b - ti.sqrt(d_planet)
                            if tp > t_start:
                                t_end = tp
                                hit_planet = 1

                        scattered = ti.Vector([0.0, 0.0, 0.0])
                        transmittance = 1.0
                        cos_view = -rd.dot(sun_dir)
                        rayleigh_phase = 0.0596831 * (1.0 + cos_view * cos_view)
                        g = 0.76
                        mie_phase = (1.0 - g * g) / (
                            4.0 * 3.14159 * ti.pow(1.0 + g * g - 2.0 * g * cos_view, 1.5)
                        )

                        dt = (t_end - t_start) / N
                        for k in range(N):
                            p = cam_pos + rd * (t_start + dt * (float(k) + 0.5))
                            r = p.norm()
                            alt = (r - R_p) / (R_atm - R_p)
                            alt = ti.max(0.0, ti.min(1.0, alt))
                            # Steeper density profile keeps the haze concentrated
                            # near the surface even though the shell is thicker --
                            # so the planet stays sharp and the halo softens with
                            # altitude.
                            density = ti.exp(-alt * 5.0)

                            n_p = p / (r + 1e-6)
                            cos_sun = sun_dir.dot(n_p)
                            sun_term = ti.max(0.0, (cos_sun + 0.08) / 1.08)

                            in_scatter = sun_term * (
                                rayleigh_color * rayleigh_phase * 2.1
                                + mie_color * mie_phase * 0.55
                            ) * density

                            scattered += in_scatter * transmittance * dt
                            sigma_t = density * 1.0
                            transmittance *= ti.exp(-sigma_t * dt)
                            if transmittance < 0.01:
                                break

                        # Surface contribution (cloud composite seen through atmosphere).
                        if hit_planet == 1:
                            p_surf = cam_pos + rd * t_end
                            r_s = p_surf.norm() + 1e-6
                            n_surf = p_surf / r_s
                            # lat in [-90, 90], lon in [-180, 180].
                            lat_deg = ti.asin(ti.max(-1.0, ti.min(1.0, n_surf[1]))) * 57.2957795
                            lon_deg = ti.atan2(n_surf[2], n_surf[0]) * 57.2957795
                            base_col = self._sample_surface(lat_deg, lon_deg, t)

                            # Procedural detail: world-space fBM that resolves
                            # finer scales as the camera approaches. Footprint
                            # = pixel angular size on the surface, divided by
                            # the foreshortening of the local normal so the
                            # equivalent surface area per pixel is captured.
                            view_n_planet = ti.max(0.05, -rd.dot(n_surf))
                            pixel_size = (2.0 * fov_scale * t_end) / float(self.height)
                            footprint = pixel_size / view_n_planet
                            detail = self._surface_detail(p_surf, footprint)
                            # Brightness modulation reads as 3D cloud relief;
                            # additive warm tint adds chromatic eddy variation.
                            warm = ti.Vector([0.10, 0.06, 0.03])
                            base_col = base_col * (1.0 + detail * 0.45) + detail * warm
                            base_col = ti.Vector([
                                ti.max(0.0, base_col[0]),
                                ti.max(0.0, base_col[1]),
                                ti.max(0.0, base_col[2]),
                            ])

                            cos_n_sun = n_surf.dot(sun_dir)
                            # Soft terminator + ambient floor for the night side.
                            soft = ti.pow(ti.max(0.0, cos_n_sun + 0.08) / 1.08, 0.7)
                            # Forward-scattered glow at the day/night line (Mie multi-scatter proxy).
                            fwd_glow = 0.35 * ti.pow(
                                ti.max(0.0, cos_view + 0.5), 2.5
                            ) * ti.max(0.0, cos_n_sun)
                            ambient = 0.015
                            surface_lit = sun_color * (soft + fwd_glow) + ti.Vector(
                                [ambient, ambient, ambient * 1.3]
                            )
                            scattered += base_col * surface_lit * transmittance

                            # Atmospheric rim glow: a fresnel-like term that
                            # paints a cool blue Rayleigh halo onto the lit
                            # limb. Grazing rays (n . -rd small) get the most
                            # contribution; the sun-side floor keeps the
                            # night limb dark.
                            view_n = ti.max(0.0, -rd.dot(n_surf))
                            rim_fres = ti.pow(1.0 - view_n, 3.0)
                            rim_sun = ti.max(0.0, (cos_n_sun + 0.20) / 1.20)
                            rim_strength = rim_fres * rim_sun * 0.55
                            scattered += rim_haze * rim_strength * transmittance

                        color = scattered + color * transmittance

                accum += color

            self.pixels[i, j] = accum / float(self.samples)

    @ti.func
    def _background(self, rd):
        # Simple starry background — high-frequency hash that nets a sparse star field.
        col = ti.Vector([0.005, 0.006, 0.013])
        h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
        h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
        s = h * h2
        if s > 0.9985:
            col += ti.Vector([1.0, 1.0, 0.95])
        return col
