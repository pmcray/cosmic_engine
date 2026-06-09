"""Saturn-class ringed gas-giant volumetric renderer.

Extends the volumetric gas-giant pattern with:

  - Saturn-tuned band table (paler creams + golds, single fast EZ jet)
  - A parametric ring system (C ring, B ring, Cassini Division, A ring,
    Encke gap, F ringlet) rendered as a thin annular density in the
    planet's equatorial plane
  - Bi-directional shadow casting: ring shadow projected onto the
    planet, planet shadow projected onto the rings (both via simple
    ray-sphere / ray-plane tests)
  - Hexagonal north-polar vortex (six-fold Rossby-wave pattern, the
    Cassini discovery)

Same architectural invariants as the other volumetric renderers:
no baked planet texture beyond the 1D band table, all detail
synthesized via render/noise.py footprint-gated primitives, no
pixelation under zoom.

The "Saturn-class" naming is deliberate: the renderer is parameterized
so it can also produce Uranus-tilted, Neptune-blue, or exoplanet
ringed-giant variants by swapping band color and ring radii.
"""

import math

import numpy as np

from render.noise import (
    domain_warp_fbm,
    fbm_3d_footprint,
    ridged_fbm_3d,
    value_noise_3d,
)

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover
    ti = None
    _HAS_TAICHI = False


# Saturn-class band table: (lat_min, lat_max, name, base_color, density, wind m/s).
# Wind speeds from Cassini observations. Saturn's equatorial jet is enormous
# (~470 m/s eastward), several times Jupiter's.
SATURN_BANDS = (
    (-90, -75, "SPR",  (0.55, 0.52, 0.45), 0.85, +25.0),
    (-75, -55, "STZ",  (0.85, 0.80, 0.65), 0.78, -20.0),
    (-55, -42, "SBZ",  (0.78, 0.72, 0.58), 0.82, +40.0),
    (-42, -28, "SSB",  (0.62, 0.55, 0.42), 0.88, -30.0),
    (-28,  -7, "STrZ", (0.88, 0.82, 0.65), 0.80, +20.0),
    ( -7,   7, "EZ",   (0.92, 0.86, 0.68), 0.80, +280.0),
    (  7,  28, "NTrZ", (0.86, 0.78, 0.62), 0.80, +60.0),
    ( 28,  42, "NSB",  (0.62, 0.55, 0.42), 0.88, -20.0),
    ( 42,  55, "NBZ",  (0.78, 0.72, 0.58), 0.82, +30.0),
    ( 55,  75, "NTZ",  (0.85, 0.80, 0.65), 0.78, +20.0),
    ( 75,  90, "NPR",  (0.55, 0.52, 0.45), 0.85, +30.0),
)


# Ring system (in units of planet radius). The standard Saturn rings.
RING_SEGMENTS = (
    # (r_inner, r_outer, base_density, base_color_rgb, name)
    (1.24, 1.53, 0.30, (0.55, 0.48, 0.40), "C"),
    (1.53, 1.95, 0.95, (0.92, 0.88, 0.78), "B"),
    (1.95, 2.03, 0.02, (0.30, 0.28, 0.25), "Cassini_Division"),
    (2.03, 2.21, 0.70, (0.85, 0.80, 0.72), "A_inner"),
    (2.21, 2.214, 0.05, (0.25, 0.24, 0.22), "Encke_Gap"),
    (2.214, 2.27, 0.65, (0.82, 0.78, 0.70), "A_outer"),
    (2.32, 2.33, 0.40, (0.95, 0.92, 0.85), "F"),
)


# ============================================================
# Numpy CPU references (testable without GPU)
# ============================================================

def _np_saturn_band_color(lat_deg):
    """Look up the Saturn band color at a given latitude."""
    for lo, hi, _name, color, _dens, _wind in SATURN_BANDS:
        if lo <= lat_deg <= hi:
            return color
    return SATURN_BANDS[-1][3]


def _np_ring_density(r):
    """Look up ring density at radial distance r (in planet radii).
    Returns (density, (r, g, b)) of the ring segment containing r,
    or (0.0, (0,0,0)) if outside all rings."""
    for r_in, r_out, dens, color, _name in RING_SEGMENTS:
        if r_in <= r < r_out:
            return float(dens), color
    return 0.0, (0.0, 0.0, 0.0)


def _np_hex_mask(lat_deg, lon_deg, hex_lat=78.0, hex_amp=0.45):
    """Six-fold symmetric mask for the north-polar hexagon.
    Returns positive when inside the hex envelope near hex_lat."""
    if lat_deg < hex_lat - 12.0 or lat_deg > hex_lat + 12.0:
        return 0.0
    lat_env = math.exp(-((lat_deg - hex_lat) / 6.0) ** 2)
    # 6-fold cos with phase offset for the corner orientation.
    hex_petal = math.cos(6.0 * math.radians(lon_deg))
    return lat_env * max(0.0, hex_amp + 0.5 * hex_petal)


# ============================================================
# Taichi engine
# ============================================================

if not _HAS_TAICHI:

    class SaturnClassEngine:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError(
                "SaturnClassEngine requires Taichi and a GPU runtime."
            )

else:

    @ti.data_oriented
    class SaturnClassEngine:
        """Saturn-class ringed gas-giant renderer.

        Planet sits at origin with radius 1. Rings lie in the planet's
        equatorial plane (XZ plane; planet's rotation axis = +Y).
        Camera at (..., ..., -R) looking toward origin matches the
        established convention.
        """

        def __init__(
            self,
            width=1280,
            height=720,
            tex_height=512,
            tex_width=1024,
            # atmosphere
            atm_thickness=0.05,
            march_steps=28,
            samples=2,
            sun_dir=(-0.55, 0.18, -0.81),
            # hexagonal polar vortex (north pole)
            hex_enable=1,
            hex_lat=78.0,
            hex_amp=0.45,
            # rings
            rings_enable=1,
            ring_samples=24,  # samples per ray through ring plane
            ring_brightness=1.0,
            # ring/planet shadows
            ring_shadows_enable=1,
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
            self.ring_samples = int(ring_samples)

            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            # ---- bake the band texture (numpy) ----
            surface_color, band_omega = self._bake_atmosphere(
                self.tex_h, self.tex_w, seed,
            )
            self.surface_color = ti.Vector.field(3, dtype=ti.f32, shape=(self.tex_h, self.tex_w))
            self.band_omega = ti.field(dtype=ti.f32, shape=(self.tex_h,))
            self.surface_color.from_numpy(surface_color)
            self.band_omega.from_numpy(band_omega)

            # ---- ring lookup table ----
            # Pack ring segments into a flat (N, 5) field: [r_in, r_out, dens, R, G, B].
            n_segs = len(RING_SEGMENTS)
            self.n_ring_segs = n_segs
            self.ring_table = ti.field(dtype=ti.f32, shape=(n_segs, 6))
            rt_np = np.zeros((n_segs, 6), dtype=np.float32)
            for i, (r_in, r_out, dens, color, _name) in enumerate(RING_SEGMENTS):
                rt_np[i, 0] = r_in
                rt_np[i, 1] = r_out
                rt_np[i, 2] = dens
                rt_np[i, 3] = color[0]
                rt_np[i, 4] = color[1]
                rt_np[i, 5] = color[2]
            self.ring_table.from_numpy(rt_np)

            # ---- params ----
            self.gp = ti.field(dtype=ti.f32, shape=(16,))
            gp_arr = np.zeros(16, dtype=np.float32)
            gp_arr[0] = float(hex_enable)
            gp_arr[1] = hex_lat
            gp_arr[2] = hex_amp
            gp_arr[3] = float(rings_enable)
            gp_arr[4] = ring_brightness
            gp_arr[5] = float(ring_shadows_enable)
            gp_arr[6] = self.atm_thickness
            self.gp.from_numpy(gp_arr)

            # camera + sun (same 16-slot layout as the other engines)
            self.cam = ti.field(dtype=ti.f32, shape=(16,))
            self._set_default_camera()
            self.set_sun(sun_dir)

        # ---- bake helpers (numpy) -----------------------------------

        def _bake_atmosphere(self, H, W, seed):
            """Bake (H, W, 3) base color + (H,) per-row drift omega.
            Saturn-style: paler than Jupiter, wavy boundaries, no GRS."""
            lat = np.linspace(-90.0, 90.0, H, dtype=np.float32)
            lon = np.linspace(-180.0, 180.0, W, dtype=np.float32)
            lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

            R_eq_m = 1.16e8 / (2.0 * math.pi)  # Saturn equatorial circumference
            drift_aesthetic = 6.0e4

            from render.noise import np_fbm_2d as _np_fbm
            lat_wave = _np_fbm((H, W), octaves=4, persistence=0.55, seed=seed + 1) * 1.8
            lat_wave += np.sin(lon2d * math.pi / 180.0 * 2.0) * 0.9
            effective_lat = lat2d + lat_wave

            base_color = np.zeros((H, W, 3), dtype=np.float32)
            band_omega = np.zeros((H,), dtype=np.float32)
            for (lo, hi, _name, color, _dens, wind) in SATURN_BANDS:
                mask = (effective_lat >= lo) & (effective_lat < hi)
                for k in range(3):
                    base_color[mask, k] = color[k]
                row_mask = (lat >= lo) & (lat <= hi)
                band_omega[row_mask] = (wind / R_eq_m) * drift_aesthetic
            band_omega = np.convolve(band_omega, np.ones(5) / 5.0, mode="same").astype(np.float32)

            # Chromatic turbulence (less than Jupiter; Saturn is hazier).
            chrom = _np_fbm((H, W), octaves=5, persistence=0.5, seed=seed + 3)
            base_color[..., 0] += chrom * 0.045
            base_color[..., 1] += chrom * 0.035
            base_color[..., 2] += chrom * 0.020

            # Light separable smoothing for clean band boundaries.
            for _ in range(2):
                base_color = (
                    np.concatenate([base_color[:1], base_color[:-1]], axis=0)
                    + base_color
                    + np.concatenate([base_color[1:], base_color[-1:]], axis=0)
                ) / 3.0
                base_color = (
                    np.roll(base_color, 1, axis=1)
                    + base_color
                    + np.roll(base_color, -1, axis=1)
                ) / 3.0
            base_color = np.clip(base_color, 0.0, 1.5).astype(np.float32)
            return base_color, band_omega

        # ---- python-side ---------------------------------------------

        def _set_default_camera(self):
            self.set_camera(
                position=(0.0, 0.7, -4.5),
                target=(0.0, -0.2, 0.0),
                up_world=(0.0, 1.0, 0.0),
                fov_deg=30.0,
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

        # ---- kernel helpers ----------------------------------------

        @ti.func
        def _sample_surface(self, lat_deg, lon_deg, t):
            row = ti.cast(
                ti.min(self.tex_h - 1,
                       ti.max(0, int((lat_deg + 90.0) / 180.0 * self.tex_h))),
                ti.i32,
            )
            omega = self.band_omega[row]
            lon_eff = lon_deg - omega * t * 57.2957795
            lon_eff = lon_eff - 360.0 * ti.floor((lon_eff + 180.0) / 360.0)
            col = ti.cast(
                ti.min(self.tex_w - 1,
                       ti.max(0, int((lon_eff + 180.0) / 360.0 * self.tex_w))),
                ti.i32,
            )
            return self.surface_color[row, col]

        @ti.func
        def _sample_ring(self, r):
            """Look up ring density and color at radial distance r."""
            dens = 0.0
            col = ti.Vector([0.0, 0.0, 0.0])
            for k in range(self.n_ring_segs):
                r_in = self.ring_table[k, 0]
                r_out = self.ring_table[k, 1]
                if r >= r_in and r < r_out:
                    dens = self.ring_table[k, 2]
                    col = ti.Vector([
                        self.ring_table[k, 3],
                        self.ring_table[k, 4],
                        self.ring_table[k, 5],
                    ])
            return dens, col

        @ti.func
        def _hex_perturbation(self, lat_deg, lon_deg, hex_lat, hex_amp):
            """Darken/brighten near the north pole following a six-fold
            Rossby-wave cosine."""
            lat_env = ti.exp(-((lat_deg - hex_lat) / 6.0) ** 2)
            hex_petal = ti.cos(6.0 * lon_deg * 0.01745329)
            return lat_env * (hex_amp + 0.45 * hex_petal)

        @ti.func
        def _background(self, rd):
            col = ti.Vector([0.004, 0.005, 0.012])
            h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
            h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
            s = h * h2
            if s > 0.9985:
                col += ti.Vector([1.0, 1.0, 0.95])
            return col

        # ---- main kernel -------------------------------------------

        @ti.kernel
        def render_kernel(self, t: ti.f32):
            cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
            cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
            cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
            cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
            fov_scale = self.cam[12]
            sun_dir = ti.Vector([self.cam[13], self.cam[14], self.cam[15]])

            hex_enable = self.gp[0]
            hex_lat = self.gp[1]
            hex_amp = self.gp[2]
            rings_enable = self.gp[3]
            ring_brightness = self.gp[4]
            ring_shadows_enable = self.gp[5]
            atm_thickness = self.gp[6]

            R_p = 1.0
            R_atm = 1.0 + atm_thickness

            warm = ti.Vector([1.0, 0.95, 0.85])
            rayleigh_color = ti.Vector([0.32, 0.58, 1.05])
            mie_color = ti.Vector([1.05, 0.97, 0.85])
            sun_color = ti.Vector([1.0, 0.97, 0.92])

            pixel_size = 2.0 * fov_scale / float(self.height)

            for i, j in self.pixels:
                accum = ti.Vector([0.0, 0.0, 0.0])

                for s in range(self.samples):
                    jx = (ti.random() - 0.5) / self.samples
                    jy = (ti.random() - 0.5) / self.samples
                    u = ((float(j) + 0.5 + jx) / self.width) * 2.0 - 1.0
                    v = ((float(i) + 0.5 + jy) / self.height) * 2.0 - 1.0
                    v = -v
                    rd = (
                        cam_fwd
                        + cam_right * (u * fov_scale * self.aspect)
                        + cam_up * (v * fov_scale)
                    )
                    rd = rd / (rd.norm() + 1e-8)

                    color = self._background(rd)

                    # Intersect atmosphere + planet.
                    b_p = rd.dot(cam_pos)
                    co_atm = cam_pos.dot(cam_pos) - R_atm * R_atm
                    cp_p = cam_pos.dot(cam_pos) - R_p * R_p
                    d_atm = b_p * b_p - co_atm
                    d_planet = b_p * b_p - cp_p

                    # Track planet front-face hit, including the t value.
                    hit_planet = 0
                    t_planet = 1e10
                    if d_planet > 0.0:
                        tp = -b_p - ti.sqrt(d_planet)
                        if tp > 0.0:
                            hit_planet = 1
                            t_planet = tp

                    # Ring plane intersection (XZ plane: y = 0).
                    hit_ring = 0
                    t_ring = 1e10
                    ring_col = ti.Vector([0.0, 0.0, 0.0])
                    ring_alpha = 0.0
                    if rings_enable > 0.5 and ti.abs(rd[1]) > 1e-5:
                        t_r = -cam_pos[1] / rd[1]
                        if t_r > 0.0:
                            p_ring = cam_pos + rd * t_r
                            r_ring = ti.sqrt(p_ring[0] ** 2 + p_ring[2] ** 2)
                            dens, color_seg = self._sample_ring(r_ring)
                            if dens > 0.001:
                                hit_ring = 1
                                t_ring = t_r
                                ring_col = color_seg
                                ring_alpha = dens
                                # Subtle radial noise so the rings aren't flat.
                                wob = (value_noise_3d(
                                    ti.Vector([r_ring * 35.0,
                                               ti.atan2(p_ring[2], p_ring[0]) * 9.0,
                                               0.0])
                                ) - 0.5) * 0.20
                                ring_alpha = ti.max(0.0, ring_alpha + wob)

                    # Atmosphere ray-march.
                    atm_color = ti.Vector([0.0, 0.0, 0.0])
                    transmittance = 1.0
                    if d_atm > 0.0:
                        t_in = -b_p - ti.sqrt(d_atm)
                        t_out = -b_p + ti.sqrt(d_atm)
                        if t_out > 0.0:
                            t_start = ti.max(0.0, t_in)
                            t_end = t_out
                            if hit_planet == 1 and t_planet < t_end:
                                t_end = t_planet
                            N = self.march_steps
                            dt = (t_end - t_start) / float(N)
                            cos_view = -rd.dot(sun_dir)
                            rayleigh_phase = 0.0596831 * (1.0 + cos_view * cos_view)
                            g = 0.76
                            mie_phase = (1.0 - g * g) / (
                                4.0 * 3.14159
                                * ti.pow(1.0 + g * g - 2.0 * g * cos_view, 1.5)
                            )
                            for k in range(N):
                                p = cam_pos + rd * (t_start + dt * (float(k) + 0.5))
                                r = p.norm()
                                alt = ti.max(0.0, ti.min(1.0,
                                    (r - R_p) / (R_atm - R_p)))
                                density = ti.exp(-alt * 5.0)
                                n_p = p / (r + 1e-6)
                                cos_sun = sun_dir.dot(n_p)
                                sun_term = ti.max(0.0, (cos_sun + 0.08) / 1.08)
                                in_scat = sun_term * (
                                    rayleigh_color * rayleigh_phase * 2.1
                                    + mie_color * mie_phase * 0.55
                                ) * density
                                atm_color += in_scat * transmittance * dt
                                sigma_t = density * 1.0
                                transmittance *= ti.exp(-sigma_t * dt)
                                if transmittance < 0.01:
                                    break

                    # Planet surface contribution.
                    planet_color = ti.Vector([0.0, 0.0, 0.0])
                    if hit_planet == 1:
                        p_surf = cam_pos + rd * t_planet
                        r_s = p_surf.norm() + 1e-6
                        n_surf = p_surf / r_s
                        lat_deg = ti.asin(ti.max(-1.0, ti.min(1.0, n_surf[1]))) * 57.2957795
                        lon_deg = ti.atan2(n_surf[2], n_surf[0]) * 57.2957795
                        base_col = self._sample_surface(lat_deg, lon_deg, t)

                        # Procedural detail (footprint-AA).
                        footprint = pixel_size * t_planet
                        detail = domain_warp_fbm(p_surf, footprint, 40.0, 6, 12.0, 0.35)
                        warm_tint = ti.Vector([0.08, 0.05, 0.02])
                        base_col = base_col * (1.0 + detail * 0.35) + detail * warm_tint
                        base_col = ti.Vector([
                            ti.max(0.0, base_col[0]),
                            ti.max(0.0, base_col[1]),
                            ti.max(0.0, base_col[2]),
                        ])

                        # Hexagonal vortex (north pole only).
                        if hex_enable > 0.5:
                            hex_d = self._hex_perturbation(lat_deg, lon_deg,
                                                           hex_lat, hex_amp)
                            base_col = base_col + ti.Vector([0.04, 0.02, -0.04]) * hex_d

                        cos_n_sun = n_surf.dot(sun_dir)
                        soft = ti.pow(ti.max(0.0, cos_n_sun + 0.08) / 1.08, 0.7)
                        ambient = 0.015
                        surface_lit = sun_color * soft + ti.Vector(
                            [ambient, ambient, ambient * 1.3]
                        )

                        # Ring shadow on planet: cast a ray from the surface
                        # toward the sun, check if it crosses the ring plane.
                        ring_shadow = 1.0
                        if ring_shadows_enable > 0.5 and ti.abs(sun_dir[1]) > 1e-3:
                            t_s = -p_surf[1] / sun_dir[1]
                            if t_s > 0.0:
                                p_cross = p_surf + sun_dir * t_s
                                r_cross = ti.sqrt(p_cross[0] ** 2 + p_cross[2] ** 2)
                                shadow_dens, _ = self._sample_ring(r_cross)
                                # 1 - alpha * darkening; partial transparency.
                                ring_shadow = 1.0 - shadow_dens * 0.85

                        planet_color = base_col * surface_lit * ring_shadow * transmittance

                    # Composite: atmosphere + planet, then ring on top with
                    # planet-shadow modulation.
                    color_no_ring = atm_color + planet_color + color * transmittance

                    if hit_ring == 1:
                        # Planet shadow on rings.
                        ring_lit = 1.0
                        if ring_shadows_enable > 0.5:
                            p_ring = cam_pos + rd * t_ring
                            to_sun = sun_dir
                            b_s = to_sun.dot(p_ring)
                            c_s = p_ring.dot(p_ring) - R_p * R_p
                            disc_s = b_s * b_s - c_s
                            if disc_s > 0.0:
                                t_hit = -b_s - ti.sqrt(disc_s)
                                if t_hit > 0.0:
                                    ring_lit = 0.18  # deep umbra

                        # Ring lighting from sun direction.
                        light_intensity = ti.max(0.15, ti.abs(sun_dir[1])) * ring_lit
                        ring_emission = ring_col * light_intensity * ring_brightness

                        # Whether the ring is in front of or behind the planet.
                        # If the ring intersection is closer than the planet
                        # hit, the ring occludes the planet.
                        if hit_planet == 0 or t_ring < t_planet:
                            color = ring_emission * ring_alpha + color_no_ring * (1.0 - ring_alpha)
                        else:
                            color = color_no_ring  # planet in front of ring
                    else:
                        color = color_no_ring

                    accum += color

                self.pixels[i, j] = accum / float(self.samples)
