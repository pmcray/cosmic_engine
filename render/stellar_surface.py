"""Stellar surface volumetric renderer.

Renders a sun-like (or brown-dwarf-cool, or red-supergiant-cool, or
hot-blue) star surface with:

  - Limb-darkened photosphere via the standard linear limb-darkening
    law I(mu) = I_0 * (1 - u + u * mu)
  - Granulation cells (Rayleigh-Benard convection pattern) at two
    scales -- granules and supergranules -- via Worley noise from
    the render.noise toolkit
  - Sunspots placed parametrically at given lat/lon with dark umbra
    + brighter penumbra rings
  - Faculae / network brightening surrounding sunspots
  - Prominences: pink Halpha plasma arcs anchored at the limb,
    rendered as a thin glowing torus above the photosphere
  - A glowing corona ring at the silhouette (very low density,
    extends beyond the disc)
  - Effective-temperature color via a Planck blackbody approximation

Same architectural invariants as the other volumetric renderers:
no baked texture, all detail synthesized via render/noise.py
footprint-gated primitives, anti-pixelation on zoom.

The renderer is general-purpose: by varying T_eff and granule_scale
you can make it look like the Sun (5800K), a M-dwarf (3500K), a
brown dwarf (1500K with silicate clouds), or a hot O-star (40000K
blue-white).
"""

import math

import numpy as np

from render.noise import (
    domain_warp_fbm,
    fbm_3d_footprint,
    value_noise_3d,
    worley_3d,
)

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover
    ti = None
    _HAS_TAICHI = False


def _planck_color(T):
    """Approximate sRGB color for a blackbody at temperature T (K).
    Returns (r, g, b) in [0, 1.5+] range (HDR; ACES handles it)."""
    # Empirical fit to the Planck curve, scaled so 5800K ~ (1, 1, 1).
    T_n = float(T) / 5800.0
    if T < 6500:
        r = 1.0
        g = 0.39 + 0.45 * math.log(max(T, 1500) / 2500.0)
        b = 0.04 + 0.55 * math.log(max(T, 2000) / 2000.0)
    else:
        r = 1.0 / max(0.5, (T_n ** 0.4))
        g = 0.85 + 0.15 / max(0.5, T_n)
        b = 1.0 + 0.4 * math.log(T_n)
    return (max(0.0, min(2.0, r)),
            max(0.0, min(2.0, g)),
            max(0.0, min(2.0, b)))


# ============================================================
# Numpy CPU references (testable without GPU)
# ============================================================

def _np_limb_darkening(cos_view, u=0.6):
    """Linear limb-darkening law I(mu) = I_0 (1 - u + u * mu)."""
    mu = max(0.0, min(1.0, cos_view))
    return (1.0 - u + u * mu)


def _np_sunspot_mask(lat_deg, lon_deg, spot_lat, spot_lon, r_deg):
    """Smooth angular-distance Gaussian for a single sunspot."""
    dlat = lat_deg - spot_lat
    dlon = lon_deg - spot_lon
    # Account for cos(lat) longitude compression near poles.
    dlon *= math.cos(math.radians(spot_lat))
    d2 = dlat * dlat + dlon * dlon
    return math.exp(-d2 / (r_deg * r_deg))


# ============================================================
# Taichi engine
# ============================================================

if not _HAS_TAICHI:

    class StellarSurfaceEngine:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError(
                "StellarSurfaceEngine requires Taichi and a GPU runtime."
            )

else:

    @ti.data_oriented
    class StellarSurfaceEngine:

        def __init__(
            self,
            width=1280,
            height=720,
            # photosphere
            T_eff=5800.0,
            limb_u=0.6,
            granule_scale=22.0,
            supergranule_scale=4.5,
            # corona shell extent above the photosphere
            corona_thickness=0.04,
            # sunspots: each is (lat_deg, lon_deg, radius_deg, depth)
            sunspots=(),
            # prominences: each is (lat_deg, lon_deg, height_above_R, length_deg)
            prominences=(),
            # ray-march
            samples=2,
            seed=0,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.samples = int(samples)

            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            # Parameter buffer.
            self.gp = ti.field(dtype=ti.f32, shape=(16,))
            arr = np.zeros(16, dtype=np.float32)
            r, g, b = _planck_color(T_eff)
            arr[0] = r
            arr[1] = g
            arr[2] = b
            arr[3] = limb_u
            arr[4] = granule_scale
            arr[5] = supergranule_scale
            arr[6] = corona_thickness
            arr[7] = float(seed)
            self.gp.from_numpy(arr)

            # Sunspots: pad to fixed slots.
            self.n_spots_max = 8
            self.n_spots = int(min(len(sunspots), self.n_spots_max))
            self.spots = ti.field(dtype=ti.f32, shape=(self.n_spots_max, 4))
            s_arr = np.zeros((self.n_spots_max, 4), dtype=np.float32)
            for i, sp in enumerate(sunspots[: self.n_spots_max]):
                s_arr[i, 0] = float(sp[0])  # lat
                s_arr[i, 1] = float(sp[1])  # lon
                s_arr[i, 2] = float(sp[2])  # radius deg
                s_arr[i, 3] = float(sp[3])  # darkness 0..1
            self.spots.from_numpy(s_arr)

            # Prominences.
            self.n_prom_max = 6
            self.n_prom = int(min(len(prominences), self.n_prom_max))
            self.prom = ti.field(dtype=ti.f32, shape=(self.n_prom_max, 4))
            p_arr = np.zeros((self.n_prom_max, 4), dtype=np.float32)
            for i, pr in enumerate(prominences[: self.n_prom_max]):
                p_arr[i, 0] = float(pr[0])  # anchor lat
                p_arr[i, 1] = float(pr[1])  # anchor lon
                p_arr[i, 2] = float(pr[2])  # height above radius
                p_arr[i, 3] = float(pr[3])  # length deg
            self.prom.from_numpy(p_arr)

            # Camera.
            self.cam = ti.field(dtype=ti.f32, shape=(16,))
            self._set_default_camera()

        def _set_default_camera(self):
            self.set_camera(
                position=(0.0, 0.0, -3.0),
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
            self.cam.from_numpy(arr)

        def render(self, t: float) -> np.ndarray:
            self.render_kernel(float(t))
            return self.pixels.to_numpy()

        # ---- kernel helpers ----------------------------------------

        @ti.func
        def _background(self, rd):
            col = ti.Vector([0.004, 0.005, 0.012])
            h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
            h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
            if h * h2 > 0.9985:
                col += ti.Vector([1.0, 1.0, 0.95])
            return col

        @ti.func
        def _granulation(self, p, granule_scale, supergranule_scale):
            """Two-scale Worley pattern for granules + supergranules.
            Returns a per-pixel brightness modulation in roughly [0.7, 1.3]."""
            w_g = worley_3d(p * granule_scale)
            w_sg = worley_3d(p * supergranule_scale)
            # Distance from a seed = darker (cell boundary), so invert
            # and softclamp so the cell CENTERS read as bright granules.
            granule_b = ti.max(0.0, 1.0 - w_g * 2.5)
            super_b = ti.max(0.0, 1.0 - w_sg * 2.0)
            return 0.85 + granule_b * 0.45 + super_b * 0.15 - 0.20

        @ti.func
        def _sunspot_darkness(self, lat_deg, lon_deg):
            """Aggregate darkness from all sunspots. 1.0 = unaffected,
            0.0 = full umbra."""
            visibility = 1.0
            for k in range(self.n_spots_max):
                if k < self.n_spots:
                    s_lat = self.spots[k, 0]
                    s_lon = self.spots[k, 1]
                    r_deg = self.spots[k, 2]
                    dark = self.spots[k, 3]
                    dlat = lat_deg - s_lat
                    dlon = (lon_deg - s_lon) * ti.cos(s_lat * 0.01745329)
                    d2 = dlat * dlat + dlon * dlon
                    umbra = ti.exp(-d2 / (r_deg * r_deg))
                    penumbra = ti.exp(-d2 / (r_deg * 1.8) ** 2)
                    # Penumbra adds a brightening ring; umbra darkens center.
                    visibility *= 1.0 - dark * umbra
                    visibility += 0.06 * (penumbra - umbra)
            return ti.max(0.0, visibility)

        @ti.func
        def _prominence_emission(self, p):
            """Emit pink Halpha along arcs anchored at the limb."""
            R_p = 1.0
            em = ti.Vector([0.0, 0.0, 0.0])
            for k in range(self.n_prom_max):
                if k < self.n_prom:
                    anchor_lat = self.prom[k, 0]
                    anchor_lon = self.prom[k, 1]
                    height = self.prom[k, 2]
                    length = self.prom[k, 3]
                    # Anchor position on the sphere surface.
                    cla = ti.cos(anchor_lat * 0.01745329)
                    sla = ti.sin(anchor_lat * 0.01745329)
                    clo = ti.cos(anchor_lon * 0.01745329)
                    slo = ti.sin(anchor_lon * 0.01745329)
                    anchor = ti.Vector([cla * clo, sla, cla * slo])
                    # Distance from sample point to the loop arc top
                    # (approximated as point at anchor + height * normal).
                    loop_top = anchor * (1.0 + height)
                    d = (p - loop_top).norm()
                    # Bright Halpha glow within a small radius of the loop top.
                    g = ti.exp(-d * d * 25.0 / (length * length * 0.01 + 1e-3))
                    em += ti.Vector([1.20, 0.40, 0.55]) * g * 1.4
            return em

        # ---- main kernel -------------------------------------------

        @ti.kernel
        def render_kernel(self, t: ti.f32):
            cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
            cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
            cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
            cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
            fov_scale = self.cam[12]

            base_color = ti.Vector([self.gp[0], self.gp[1], self.gp[2]])
            limb_u = self.gp[3]
            granule_scale = self.gp[4]
            supergranule_scale = self.gp[5]
            corona_thickness = self.gp[6]

            R_p = 1.0
            R_corona = 1.0 + corona_thickness
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

                    # Photosphere intersection.
                    b = rd.dot(cam_pos)
                    cp = cam_pos.dot(cam_pos) - R_p * R_p
                    d_p = b * b - cp

                    hit_planet = 0
                    t_planet = 1e10
                    if d_p > 0.0:
                        tp = -b - ti.sqrt(d_p)
                        if tp > 0.0:
                            hit_planet = 1
                            t_planet = tp

                    # Corona / chromosphere shell intersection (always
                    # extends a bit beyond the disc).
                    cc = cam_pos.dot(cam_pos) - R_corona * R_corona
                    d_c = b * b - cc
                    corona_color = ti.Vector([0.0, 0.0, 0.0])
                    if d_c > 0.0:
                        t_c_in = -b - ti.sqrt(d_c)
                        t_c_out = -b + ti.sqrt(d_c)
                        if t_c_out > 0.0:
                            t_start = ti.max(0.0, t_c_in)
                            t_end = t_c_out
                            if hit_planet == 1 and t_planet < t_end:
                                t_end = t_planet
                            N = 8
                            dt = (t_end - t_start) / float(N)
                            for k in range(N):
                                p = cam_pos + rd * (t_start + dt * (float(k) + 0.5))
                                r = p.norm()
                                alt = ti.max(0.0,
                                    (r - R_p) / ti.max(1e-4, R_corona - R_p))
                                halo = ti.exp(-alt * 3.0) * 0.18
                                # Prominences.
                                prom = self._prominence_emission(p)
                                corona_color += (
                                    base_color * halo + prom
                                ) * dt * 4.0

                    if hit_planet == 1:
                        p_surf = cam_pos + rd * t_planet
                        r_s = p_surf.norm() + 1e-6
                        n_surf = p_surf / r_s
                        lat_deg = ti.asin(ti.max(-1.0, ti.min(1.0, n_surf[1]))) * 57.2957795
                        lon_deg = ti.atan2(n_surf[2], n_surf[0]) * 57.2957795

                        cos_view = -rd.dot(n_surf)
                        mu = ti.max(0.0, cos_view)
                        limb = 1.0 - limb_u + limb_u * mu

                        # Granulation pattern in 3D world space (so it
                        # follows the surface naturally).
                        gran = self._granulation(p_surf,
                                                 granule_scale,
                                                 supergranule_scale)

                        # Procedural fine-scale detail (footprint-AA).
                        footprint = pixel_size * t_planet
                        fine = fbm_3d_footprint(p_surf, footprint, 90.0, 5) * 0.12

                        # Sunspot darkness.
                        spot = self._sunspot_darkness(lat_deg, lon_deg)

                        photosphere = base_color * limb * (gran + fine) * spot
                        # Modest highlight to the limb-near-darkening transition.
                        photosphere = ti.Vector([
                            ti.max(0.0, photosphere[0]),
                            ti.max(0.0, photosphere[1]),
                            ti.max(0.0, photosphere[2]),
                        ])
                        color = photosphere + corona_color
                    else:
                        # Off-disc: just the corona shell + background.
                        color = corona_color + color

                    accum += color

                self.pixels[i, j] = accum / float(self.samples)
