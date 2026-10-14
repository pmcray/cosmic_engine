"""Diffuse nebula volumetric renderer.

Six topology classes share one volumetric emission kernel:

  pillars   - Pillars of Creation (M16): vertical columns eroded from
              above, surrounded by pink HII glow, dust-thick interiors
  crab      - Crab supernova remnant (M1): expanding ovoid shell with
              red filaments + interior bluish synchrotron + central
              pulsar point source
  helix     - Helix planetary nebula (NGC 7293): green-cyan torus
              (OIII) with axial bipolar lobes (red Halpha) and central
              white dwarf
  veil      - Veil SNR: thin shock-front shell with twin Doppler
              layers (approaching blue + receding red)
  orion     - Orion HII region (M42): turbulent pink-purple emission
              with embedded young-star cluster and dust silhouettes
  pleiades  - Pleiades reflection nebula: diffuse blue dust cloud
              scattering foreground starlight

Topology selector switches the per-step density + emission function;
the ray-march, per-channel transmittance, and background star field
are shared with the spiral galaxy renderer's design.

Architecturally identical to render/spiral_galaxy.py: no baked
texture, pure parametric + procedural via render/noise.py
footprint-gated primitives, per-channel transmittance so dust
extinction reddens the background naturally.
"""

import math

import numpy as np

from render.noise import (
    domain_warp_fbm,
    fbm_3d_footprint,
    ridged_fbm_3d,
    value_noise_3d,
    worley_3d,
)

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover - GPU-only dep
    ti = None
    _HAS_TAICHI = False
from render.detrng import rand_centered, tick, S_AA_X, S_AA_Y


# Topology identifiers used in the kernel dispatch.
TOPOLOGY_IDS = {
    "pillars": 0,
    "crab": 1,
    "helix": 2,
    "veil": 3,
    "orion": 4,
    "pleiades": 5,
}


def topology_id(name):
    """Look up a topology id from its string name. Raises on unknown."""
    if name not in TOPOLOGY_IDS:
        raise ValueError(
            f"unknown nebula topology '{name}'. "
            f"Known: {sorted(TOPOLOGY_IDS.keys())}"
        )
    return TOPOLOGY_IDS[name]


# ============================================================
# Numpy CPU references (testable without GPU)
# ============================================================

def _np_torus_density(p, R_major=0.7, r_minor=0.18):
    """Helix/Saturn-style torus: Gaussian-thickness ring at R_major."""
    p = np.asarray(p, dtype=np.float64)
    r_xz = math.sqrt(p[0] * p[0] + p[2] * p[2])
    d = math.sqrt((r_xz - R_major) ** 2 + p[1] * p[1])
    return float(math.exp(-(d * d) / (r_minor * r_minor)))


def _np_shell_density(p, R=1.0, thickness=0.1):
    """Spherical shell at radius R with Gaussian thickness."""
    p = np.asarray(p, dtype=np.float64)
    r = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
    d = abs(r - R)
    return float(math.exp(-(d * d) / (thickness * thickness)))


def _np_pillar_erosion(p, erosion_height=0.2, decay=2.5):
    """Pillar erosion: density drops as p[1] exceeds erosion_height."""
    p = np.asarray(p, dtype=np.float64)
    return float(max(0.0, 1.0 - max(0.0, p[1] - erosion_height) * decay))


# ============================================================
# Taichi engine
# ============================================================

if not _HAS_TAICHI:

    class DiffuseNebulaEngine:  # pragma: no cover - GPU-only path
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "DiffuseNebulaEngine requires Taichi. Install taichi and use "
                "a GPU runtime."
            )

else:

    @ti.data_oriented
    class DiffuseNebulaEngine:
        """Volumetric diffuse-nebula renderer.

        Coordinate convention: standard camera at (0, 0, -R_view) looking
        toward origin with up=(0, 1, 0); the nebula occupies roughly
        the unit ball at the origin (bounded by R_bound).
        """

        def __init__(
            self,
            width=1280,
            height=720,
            topology="pillars",
            R_bound=1.5,
            # generic feature controls
            intensity=1.0,
            dust_strength=1.0,
            detail_strength=1.0,
            # topology-specific knobs (Pillars)
            erosion_height=0.2,
            erosion_decay=2.5,
            # topology-specific knobs (Crab / Veil)
            R_shell=1.0,
            shell_thickness=0.18,
            # topology-specific knobs (Helix)
            R_torus=0.7,
            r_torus=0.18,
            # ray-march
            march_steps=80,
            samples=2,
            # misc
            seed=0,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.march_steps = int(march_steps)
            self.samples = int(samples)

            # Output buffer.
            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            # Param buffer (24 floats; topology_id is gp[0] cast to int).
            self.gp = ti.field(dtype=ti.f32, shape=(24,))
            arr = np.zeros(24, dtype=np.float32)
            arr[0] = float(topology_id(topology))
            arr[1] = R_bound
            arr[2] = intensity
            arr[3] = dust_strength
            arr[4] = detail_strength
            arr[5] = erosion_height
            arr[6] = erosion_decay
            arr[7] = R_shell
            arr[8] = shell_thickness
            arr[9] = R_torus
            arr[10] = r_torus
            arr[11] = float(seed)
            self.gp.from_numpy(arr)

            # Camera + view buffer (same 16-slot layout as the galaxy).
            self.cam = ti.field(dtype=ti.f32, shape=(16,))
            self._set_default_camera()

        # ---- Python-side helpers ------------------------------------

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

        # ---- Background ---------------------------------------------

        @ti.func
        def _background(self, rd):
            col = ti.Vector([0.004, 0.005, 0.012])
            h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
            h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
            s = h * h2
            if s > 0.9985:
                col += ti.Vector([1.0, 1.0, 0.95])
            return col

        # ---- Topology sample functions ------------------------------
        # Each returns a 6-vector: (em_r, em_g, em_b, sigma_r, sigma_g, sigma_b).

        @ti.func
        def _sample_pillars(self, p, footprint, intensity, dust_k, detail_k,
                            erosion_h, erosion_d):
            # Vertical pillars eroded from above (+y direction).
            # Columnar density: lateral ridged fBM, slow in y.
            col_p = ti.Vector([p[0] * 2.0, p[1] * 0.3, p[2] * 2.0])
            col = ridged_fbm_3d(col_p, footprint, 5.0, 7) * detail_k

            # Erosion mask: density drops as y exceeds erosion_h.
            erosion = ti.max(0.0, 1.0 - ti.max(0.0, p[1] - erosion_h) * erosion_d)
            body = col * erosion * 1.2

            # HII glow surrounds the pillar tops (UV-illuminated rims).
            rim = ti.exp(-(p[1] - 0.05) * (p[1] - 0.05) * 6.0) * (col + 0.4) * 0.7

            # A handful of embedded stars (young massive stars in M16).
            star_n = value_noise_3d(p * 18.0)
            star_burst = 0.0
            if star_n > 0.978:
                star_burst = (star_n - 0.978) * 45.0

            # Colors.
            em_hii = ti.Vector([1.30, 0.55, 0.40]) * rim * intensity
            em_glow = ti.Vector([0.85, 0.45, 0.55]) * body * 0.18 * intensity
            em_star = ti.Vector([1.0, 0.95, 0.85]) * star_burst * intensity * 0.6
            em = em_hii + em_glow + em_star

            # Per-channel extinction: dust reddens (blue absorbed most).
            sigma_base = body * 3.5 * dust_k
            sigma_r = sigma_base * 0.7
            sigma_g = sigma_base * 1.0
            sigma_b = sigma_base * 1.35
            return ti.Vector([em[0], em[1], em[2], sigma_r, sigma_g, sigma_b])

        @ti.func
        def _sample_crab(self, p, footprint, intensity, dust_k, detail_k,
                         R_sh, shell_thk):
            r = p.norm() + 1e-6
            # Shell mask: Gaussian thickness at R_sh.
            shell_d = ti.abs(r - R_sh)
            shell_mask = ti.exp(-(shell_d * shell_d) / (shell_thk * shell_thk))

            # Filaments: ridged fBM riding the shell surface.
            fil = ridged_fbm_3d(p * 3.0, footprint, 7.0, 7) * detail_k
            fil_dens = fil * shell_mask * 1.5

            # Interior synchrotron glow: bluish, falls off as r/R_sh.
            interior = ti.max(0.0, 1.0 - r / R_sh)
            syn = interior * interior * 0.6

            # Central pulsar: bright point source.
            pulsar = ti.exp(-r * 28.0) * 6.0

            # Colors.
            em_fil = ti.Vector([1.45, 0.55, 0.35]) * fil_dens * intensity
            em_syn = ti.Vector([0.30, 0.55, 1.10]) * syn * intensity * 0.85
            em_pulsar = ti.Vector([1.0, 0.95, 0.90]) * pulsar * intensity
            em = em_fil + em_syn + em_pulsar

            # Mild dust extinction, slightly chromatic.
            sigma_base = fil_dens * 0.45 * dust_k
            return ti.Vector([em[0], em[1], em[2],
                              sigma_base * 0.85, sigma_base, sigma_base * 1.15])

        @ti.func
        def _sample_helix(self, p, footprint, intensity, dust_k, detail_k,
                          R_maj, r_min):
            # Torus axis = y; ring in xz plane.
            r_xz = ti.sqrt(p[0] * p[0] + p[2] * p[2]) + 1e-6
            tor_d = ti.sqrt((r_xz - R_maj) * (r_xz - R_maj) + p[1] * p[1])
            tor_mask = ti.exp(-(tor_d * tor_d) / (r_min * r_min))

            # Knots: high-freq noise modulating the torus density.
            knot_n = value_noise_3d(p * 14.0)
            knot = ti.max(0.0, knot_n - 0.42) * 1.8 * detail_k
            tor_dens = tor_mask * (1.0 + knot)

            # Bipolar lobes along y.
            lobe_perp = ti.sqrt(p[0] * p[0] + p[2] * p[2])
            lobe = ti.exp(-lobe_perp * 4.5) * ti.exp(-(p[1] * p[1]) * 1.8)
            lobe_n = value_noise_3d(p * 4.0) + 0.4
            lobe_dens = lobe * lobe_n * 0.55

            # Central white dwarf.
            r = p.norm() + 1e-6
            wd = ti.exp(-r * 22.0) * 8.0

            # Colors: torus = teal OIII, lobes = warm Halpha, wd = white.
            em_tor = ti.Vector([0.30, 1.05, 0.85]) * tor_dens * intensity
            em_lobe = ti.Vector([1.10, 0.40, 0.55]) * lobe_dens * intensity
            em_wd = ti.Vector([1.10, 1.00, 1.05]) * wd * intensity
            em = em_tor + em_lobe + em_wd

            # Very mild extinction (planetary nebulae are optically thin).
            sigma_base = tor_dens * 0.18 * dust_k
            return ti.Vector([em[0], em[1], em[2],
                              sigma_base, sigma_base, sigma_base])

        @ti.func
        def _sample_veil(self, p, footprint, intensity, dust_k, detail_k,
                         R_sh, shell_thk):
            # Very thin shock-front shell.
            r = p.norm() + 1e-6
            shell_d = ti.abs(r - R_sh)
            shell_mask = ti.exp(-(shell_d * shell_d) / (shell_thk * shell_thk))

            # Network filaments along the shell.
            fil = ridged_fbm_3d(p * 4.5, footprint, 9.0, 6) * detail_k

            # Doppler split: front side blueshifted, back redshifted.
            # Use z relative to camera (rd-aligned) as a proxy -- here
            # we just use the world z so the front face cools and back warms.
            blue = ti.max(0.0, -p[2]) * 0.9
            red = ti.max(0.0, p[2]) * 0.9
            dens = shell_mask * fil * 1.4

            em_blue = ti.Vector([0.30, 0.70, 1.10]) * dens * (1.0 + blue) * intensity
            em_red = ti.Vector([1.10, 0.40, 0.30]) * dens * red * intensity
            em = em_blue + em_red

            sigma_base = dens * 0.25 * dust_k
            return ti.Vector([em[0], em[1], em[2],
                              sigma_base * 0.9, sigma_base, sigma_base * 1.1])

        @ti.func
        def _sample_orion(self, p, footprint, intensity, dust_k, detail_k,
                          R_bound):
            r = p.norm() + 1e-6
            envelope = ti.exp(-r * 1.2)

            # Turbulent emission via domain-warped fBM.
            turb = (domain_warp_fbm(p, footprint, 5.0, 6, 2.5, 0.30) + 0.5) * detail_k
            neb_dens = envelope * ti.max(0.0, turb) * 1.4

            # Dust silhouettes carved through.
            dust = ti.max(0.0, ridged_fbm_3d(p * 1.4, footprint, 4.0, 5)) * detail_k

            # Embedded stars (Trapezium-like cluster near origin).
            star_n = value_noise_3d(p * 22.0)
            star = 0.0
            if star_n > 0.975:
                star = (star_n - 0.975) * 35.0

            em_neb = ti.Vector([1.20, 0.55, 0.95]) * neb_dens * intensity
            em_star = ti.Vector([1.0, 0.95, 0.85]) * star * intensity * 0.55
            em = em_neb + em_star

            # Dust columns extinguish, blueward more.
            sigma_base = dust * envelope * 0.95 * dust_k
            return ti.Vector([em[0], em[1], em[2],
                              sigma_base * 0.70, sigma_base, sigma_base * 1.30])

        @ti.func
        def _sample_pleiades(self, p, footprint, intensity, dust_k, detail_k,
                             R_bound):
            r = p.norm() + 1e-6
            envelope = ti.exp(-r * 0.9)
            cloud_n = (value_noise_3d(p * 1.8) + 0.45) * detail_k
            cloud = envelope * ti.max(0.0, cloud_n) * 0.95

            # Brighter foreground stars.
            star_n = value_noise_3d(p * 12.0)
            star = 0.0
            if star_n > 0.983:
                star = (star_n - 0.983) * 55.0

            em_cloud = ti.Vector([0.40, 0.62, 1.10]) * cloud * intensity * 0.6
            em_star = ti.Vector([1.0, 0.95, 0.85]) * star * intensity
            em = em_cloud + em_star

            # Very low extinction (reflection nebulae are nearly transparent).
            sigma_base = cloud * 0.12 * dust_k
            return ti.Vector([em[0], em[1], em[2],
                              sigma_base * 0.85, sigma_base, sigma_base * 1.20])

        # ---- Ray-march kernel ---------------------------------------

        @ti.kernel
        def render_kernel(self, t: ti.f32):
            cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
            cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
            cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
            cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
            fov_scale = self.cam[12]

            topology_id = ti.cast(self.gp[0], ti.i32)
            R_bound = self.gp[1]
            intensity = self.gp[2]
            dust_k = self.gp[3]
            detail_k = self.gp[4]
            erosion_h = self.gp[5]
            erosion_d = self.gp[6]
            R_sh = self.gp[7]
            shell_thk = self.gp[8]
            R_maj = self.gp[9]
            r_min = self.gp[10]

            pixel_size = 2.0 * fov_scale / float(self.height)

            for i, j in self.pixels:
                accum = ti.Vector([0.0, 0.0, 0.0])

                for s in range(self.samples):
                    jx = rand_centered(i, j, s, tick(t) + S_AA_X) / self.samples
                    jy = rand_centered(i, j, s, tick(t) + S_AA_Y) / self.samples
                    u = ((float(j) + 0.5 + jx) / self.width) * 2.0 - 1.0
                    v = ((float(i) + 0.5 + jy) / self.height) * 2.0 - 1.0
                    v = -v

                    rd = (
                        cam_fwd
                        + cam_right * (u * fov_scale * self.aspect)
                        + cam_up * (v * fov_scale)
                    )
                    rd = rd / (rd.norm() + 1e-8)

                    # Bounding-sphere intersection.
                    b = rd.dot(cam_pos)
                    c_s = cam_pos.dot(cam_pos) - R_bound * R_bound
                    disc = b * b - c_s

                    color = self._background(rd)
                    if disc > 0.0:
                        t_in = -b - ti.sqrt(disc)
                        t_out = -b + ti.sqrt(disc)
                        if t_out > 0.0:
                            t_start = ti.max(0.0, t_in)
                            t_end = t_out

                            accum_em = ti.Vector([0.0, 0.0, 0.0])
                            trans = ti.Vector([1.0, 1.0, 1.0])

                            N = self.march_steps
                            dt = (t_end - t_start) / float(N)

                            for k in range(N):
                                t_k = t_start + dt * (float(k) + 0.5)
                                p = cam_pos + rd * t_k
                                footprint = pixel_size * t_k

                                # Topology dispatch. Returns 6-vec:
                                # [em_r, em_g, em_b, sigma_r, sigma_g, sigma_b].
                                result = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                                if topology_id == 0:
                                    result = self._sample_pillars(
                                        p, footprint, intensity, dust_k, detail_k,
                                        erosion_h, erosion_d,
                                    )
                                elif topology_id == 1:
                                    result = self._sample_crab(
                                        p, footprint, intensity, dust_k, detail_k,
                                        R_sh, shell_thk,
                                    )
                                elif topology_id == 2:
                                    result = self._sample_helix(
                                        p, footprint, intensity, dust_k, detail_k,
                                        R_maj, r_min,
                                    )
                                elif topology_id == 3:
                                    result = self._sample_veil(
                                        p, footprint, intensity, dust_k, detail_k,
                                        R_sh, shell_thk,
                                    )
                                elif topology_id == 4:
                                    result = self._sample_orion(
                                        p, footprint, intensity, dust_k, detail_k,
                                        R_bound,
                                    )
                                elif topology_id == 5:
                                    result = self._sample_pleiades(
                                        p, footprint, intensity, dust_k, detail_k,
                                        R_bound,
                                    )

                                em = ti.Vector([result[0], result[1], result[2]])
                                sigma_r = result[3]
                                sigma_g = result[4]
                                sigma_b = result[5]

                                accum_em += em * trans * dt
                                trans = ti.Vector([
                                    trans[0] * ti.exp(-sigma_r * dt),
                                    trans[1] * ti.exp(-sigma_g * dt),
                                    trans[2] * ti.exp(-sigma_b * dt),
                                ])

                                t_avg = (trans[0] + trans[1] + trans[2]) * (1.0 / 3.0)
                                if t_avg < 0.01:
                                    break

                            color = accum_em + ti.Vector([
                                color[0] * trans[0],
                                color[1] * trans[1],
                                color[2] * trans[2],
                            ])

                    accum += color

                self.pixels[i, j] = accum / float(self.samples)
