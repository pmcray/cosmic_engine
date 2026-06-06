"""Spiral galaxy volumetric renderer (M51 / Whirlpool target).

A face-on (and tiltable) spiral-galaxy renderer modeled to honour both
ends of the M51 reference spectrum: Lord Rosse's 1845 sketch
(grand-design two-arm structure visible at any zoom) and the JWST
mid-infrared image (intricate dust-lane filaments, HII regions along
arms, glowing nuclear region, the bridge to NGC 5195).

Architectural decisions:

  - No baked texture at all. The galaxy is a parametric structure
    (Sersic bulge, exponential disk, log-spiral arm modulation, halo
    ellipsoid) plus procedural detail evaluated per-step in the
    ray-march. As the camera zooms in, additional fBM octaves
    contribute via the footprint-gated primitives in `render.noise`.
    No fixed-resolution texture means no pixelation cap.

  - Emission-only volumetric integration. Galaxies emit; they are
    not lit by an external source. The integrand at each step is
    `emission * transmittance * dt`, and `transmittance` is per-RGB
    so dust extinction reddens the disk (blue absorbed more) the way
    Webb shows it.

  - Disk in the XY plane (normal = +Z) so the standard face-on camera
    at (0, 0, -R) looking toward +Z matches the gas-giant convention.

  - The M51 companion (NGC 5195) is parameterized; set
    `companion_mass_ratio = 0` to render an isolated spiral.

Public API mirrors `volumetric_gas_giant.VolumetricGasGiantEngine`:
construction, `set_camera`, `render(t)`. The adapter in
`render/adapters.py` is what the manifest references via the renderer
name `"spiral_galaxy"`.
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


# Sersic n=4 normalization coefficient. exp(-b_n * (x^(1/n) - 1)).
_SERSIC_B4 = 7.669


# ============================================================
# Numpy CPU references (testable without GPU)
# ============================================================

def _np_sersic(x, n_inv=0.25):
    x = np.maximum(x, 1e-4)
    return np.exp(-_SERSIC_B4 * (np.power(x, n_inv) - 1.0))


def _np_bulge(p, R_b, q):
    p = np.asarray(p, dtype=np.float32)
    rr = np.sqrt(p[0] * p[0] + p[1] * p[1] + (p[2] / q) ** 2)
    return float(_np_sersic(rr / R_b, 0.25))


def _np_disk(p, R_disk, h_disk):
    p = np.asarray(p, dtype=np.float32)
    r_cyl = np.sqrt(p[0] * p[0] + p[1] * p[1])
    return float(np.exp(-r_cyl / R_disk) * np.exp(-abs(p[2]) / h_disk))


def _np_arm_modulation(
    r_cyl, theta, n_arms=2, pitch_deg=12.0, phi=0.0,
    strength=0.85, width=0.45,
):
    pitch_rad = math.radians(pitch_deg)
    k_log = n_arms / math.tan(pitch_rad)
    psi = n_arms * theta - k_log * np.log(np.maximum(r_cyl, 1e-3)) + phi
    c = np.cos(psi)
    ridge_base = np.maximum(0.0, 0.5 + 0.5 * c)
    ridge = np.power(ridge_base, 1.0 / max(0.05, width))
    return 1.0 + strength * (2.0 * ridge - 1.0)


def _np_companion_bridge(p, comp_pos, mass_ratio, bridge_strength):
    if mass_ratio <= 0.0 and bridge_strength <= 0.0:
        return 0.0
    p = np.asarray(p, dtype=np.float64)
    comp_pos = np.asarray(comp_pos, dtype=np.float64)
    pc = p - comp_pos
    rr = float(np.sqrt(pc[0] * pc[0] + pc[1] * pc[1] + (pc[2] / 0.85) ** 2))
    d_c = float(_np_sersic(rr / (0.18 * max(1e-3, mass_ratio)), 0.25)) * mass_ratio
    L2 = float(comp_pos.dot(comp_pos)) + 1e-4
    s = max(0.0, min(1.0, float(p.dot(comp_pos)) / L2))
    perp = p - comp_pos * s
    d_perp2 = float(perp.dot(perp))
    bridge = bridge_strength / (1.0 + 25.0 * d_perp2) * math.exp(
        -4.0 * (s - 0.5) ** 2
    )
    return d_c + bridge


# ============================================================
# Taichi engine
# ============================================================

if not _HAS_TAICHI:

    class SpiralGalaxyEngine:  # pragma: no cover - GPU-only path
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "SpiralGalaxyEngine requires Taichi. Install taichi and use "
                "a GPU runtime."
            )

else:

    @ti.data_oriented
    class SpiralGalaxyEngine:
        """Volumetric spiral-galaxy renderer.

        Disk lies in XY plane (normal = +Z). Standard face-on camera at
        (0, 0, -R_view) looking toward origin with up=(0,1,0).
        """

        def __init__(
            self,
            width=1280,
            height=720,
            # geometry — units: galaxy radii
            R_disk=1.0,
            h_disk=0.05,
            R_bulge=0.18,
            bulge_axis_ratio=0.85,
            R_halo=1.6,
            h_halo=0.45,
            # spiral pattern
            n_arms=2,
            pitch_deg=12.0,
            arm_strength=0.85,
            arm_phase=0.0,
            arm_width=0.45,
            # dust + HII
            dust_strength=2.2,
            dust_freq=10.0,
            hii_threshold=0.22,
            hii_freq=14.0,
            # M51 companion
            companion_pos=(1.05, 0.05, 0.35),
            companion_mass_ratio=0.5,
            bridge_strength=0.6,
            # view / Doppler
            inclination_deg=20.0,
            doppler_strength=0.06,
            doppler_sign=1.0,
            # ray-march
            march_steps=72,
            samples=2,
            # misc
            seed=0,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.march_steps = int(march_steps)
            self.samples = int(samples)

            # Output buffer (H, W) so the renderer is aspect-native.
            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            # Galaxy parameters packed into a flat field so the kernel
            # can read them. Order is documented in `_param_layout`.
            self.gp = ti.field(dtype=ti.f32, shape=(48,))
            arr = np.zeros(48, dtype=np.float32)
            arr[0] = R_disk
            arr[1] = h_disk
            arr[2] = R_bulge
            arr[3] = bulge_axis_ratio
            arr[4] = R_halo
            arr[5] = h_halo
            arr[6] = float(n_arms)
            arr[7] = float(pitch_deg)
            arr[8] = arm_strength
            arr[9] = arm_phase
            arr[10] = arm_width
            arr[11] = dust_strength
            arr[12] = dust_freq
            arr[13] = hii_threshold
            arr[14] = hii_freq
            arr[15] = float(companion_pos[0])
            arr[16] = float(companion_pos[1])
            arr[17] = float(companion_pos[2])
            arr[18] = companion_mass_ratio
            arr[19] = bridge_strength
            arr[20] = math.radians(inclination_deg)
            arr[21] = doppler_strength
            arr[22] = doppler_sign
            # Pitch-derived constant k_log = n_arms / tan(pitch).
            pitch_rad = math.radians(pitch_deg)
            arr[23] = float(n_arms) / max(1e-4, math.tan(pitch_rad))
            # Bounding sphere radius for the ray-march.
            comp_r = math.sqrt(sum(c * c for c in companion_pos))
            arr[24] = max(R_halo + 0.2, comp_r + 0.5)
            self.gp.from_numpy(arr)

            # Camera + view buffer; layout matches volumetric_gas_giant
            # for consistency (pos, fwd, right, up, fov_scale, view_aux).
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
            # The remaining 3 slots carry a view-azimuth direction used
            # for the Doppler tint (angle of the camera projection in the
            # disk plane). We compute it from the camera forward.
            fwd_xy = np.array([fwd[0], fwd[1], 0.0], dtype=np.float32)
            n_fwd_xy = np.linalg.norm(fwd_xy)
            if n_fwd_xy > 1e-5:
                fwd_xy /= n_fwd_xy
            arr[13:16] = fwd_xy
            self.cam.from_numpy(arr)

        def render(self, t: float) -> np.ndarray:
            self.render_kernel(float(t))
            return self.pixels.to_numpy()

        # ---- Density-field helpers (kernel-side) --------------------

        @ti.func
        def _sersic(self, x, n_inv):
            return ti.exp(-7.669 * (ti.pow(ti.max(x, 1e-4), n_inv) - 1.0))

        @ti.func
        def _d_bulge(self, p, R_b, q):
            rr = ti.sqrt(p[0] * p[0] + p[1] * p[1] + (p[2] / q) * (p[2] / q))
            return self._sersic(rr / R_b, 0.25)

        @ti.func
        def _d_disk(self, p, R_disk, h_disk, arm_mod):
            r_cyl = ti.sqrt(p[0] * p[0] + p[1] * p[1])
            return ti.exp(-r_cyl / R_disk) * ti.exp(-ti.abs(p[2]) / h_disk) * arm_mod

        @ti.func
        def _d_halo(self, p, R_h, h_h):
            zr = p[2] / ti.max(1e-4, h_h)
            rr = ti.sqrt(p[0] * p[0] + p[1] * p[1] + zr * zr)
            return ti.exp(-rr / R_h) * 0.06

        @ti.func
        def _arm_modulation(self, r_cyl, theta, n_arms, k_log, phi,
                            arm_strength, arm_width):
            psi = n_arms * theta - k_log * ti.log(ti.max(r_cyl, 1e-3)) + phi
            c = ti.cos(psi)
            ridge_base = ti.max(0.0, 0.5 + 0.5 * c)
            ridge = ti.pow(ridge_base, 1.0 / ti.max(0.05, arm_width))
            return 1.0 + arm_strength * (2.0 * ridge - 1.0)

        @ti.func
        def _d_companion_bridge(self, p, comp_pos, mass_ratio, bridge_strength):
            pc = p - comp_pos
            zr = pc[2] / 0.85
            rr = ti.sqrt(pc[0] * pc[0] + pc[1] * pc[1] + zr * zr)
            R_c = 0.18 * ti.max(1e-3, mass_ratio)
            d_c = self._sersic(rr / R_c, 0.25) * mass_ratio

            L2 = comp_pos.dot(comp_pos) + 1e-4
            s = ti.max(0.0, ti.min(1.0, p.dot(comp_pos) / L2))
            perp = p - comp_pos * s
            d_perp2 = perp.dot(perp)
            bridge = bridge_strength / (1.0 + 25.0 * d_perp2) * ti.exp(
                -4.0 * (s - 0.5) * (s - 0.5)
            )
            return d_c + bridge

        @ti.func
        def _background(self, rd):
            col = ti.Vector([0.005, 0.006, 0.013])
            h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
            h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
            s = h * h2
            if s > 0.9985:
                col += ti.Vector([1.0, 1.0, 0.95])
            return col

        # ---- Ray-march kernel ---------------------------------------

        @ti.kernel
        def render_kernel(self, t: ti.f32):
            cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
            cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
            cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
            cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
            fov_scale = self.cam[12]
            view_az = ti.Vector([self.cam[13], self.cam[14], self.cam[15]])
            phi_view = ti.atan2(view_az[1], view_az[0])

            # Unpack galaxy params.
            R_disk = self.gp[0]
            h_disk = self.gp[1]
            R_bulge = self.gp[2]
            bulge_q = self.gp[3]
            R_halo = self.gp[4]
            h_halo = self.gp[5]
            n_arms = self.gp[6]
            arm_strength = self.gp[8]
            arm_phase = self.gp[9]
            arm_width = self.gp[10]
            dust_strength = self.gp[11]
            dust_freq = self.gp[12]
            hii_threshold = self.gp[13]
            hii_freq = self.gp[14]
            comp_pos = ti.Vector([self.gp[15], self.gp[16], self.gp[17]])
            comp_mass = self.gp[18]
            bridge_strength = self.gp[19]
            incl = self.gp[20]
            doppler_strength = self.gp[21]
            doppler_sign = self.gp[22]
            k_log = self.gp[23]
            R_bound = self.gp[24]

            # Emission palette.
            warm_yellow = ti.Vector([1.00, 0.86, 0.60])
            white_yellow = ti.Vector([0.95, 0.92, 0.82])
            blue_white = ti.Vector([0.75, 0.85, 1.05])
            pink = ti.Vector([1.10, 0.55, 0.55])
            dim_warm = ti.Vector([0.55, 0.42, 0.32])

            pixel_size = 2.0 * fov_scale / float(self.height)
            sin_incl = ti.sin(incl)

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

                    # Intersect bounding sphere.
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

                                r_cyl = ti.sqrt(p[0] * p[0] + p[1] * p[1]) + 1e-6
                                theta = ti.atan2(p[1], p[0])
                                z = p[2]

                                arm_mod = self._arm_modulation(
                                    r_cyl, theta, n_arms, k_log, arm_phase,
                                    arm_strength, arm_width,
                                )
                                # Procedural swirl perturbation on top of the
                                # analytic arm cosine (footprint-AA).
                                swirl = domain_warp_fbm(
                                    p, footprint, 6.0, 6, 3.0, 0.20
                                )
                                arm_mod_p = arm_mod * (1.0 + 0.35 * swirl)

                                d_bulge = self._d_bulge(p, R_bulge, bulge_q)
                                d_disk = self._d_disk(p, R_disk, h_disk, arm_mod_p)
                                d_halo = self._d_halo(p, R_halo, h_halo)

                                # Skip the more expensive samples outside the
                                # arm-active / mid-plane regions.
                                d_dust = 0.0
                                d_hii = 0.0
                                if ti.abs(z) < 3.0 * h_disk:
                                    d_dust = ridged_fbm_3d(
                                        p, footprint, dust_freq, 7
                                    ) * arm_mod_p * ti.exp(
                                        -(z * z) / (0.6 * h_disk * h_disk)
                                    )
                                if arm_mod_p > 1.0 and ti.abs(z) < 2.0 * h_disk:
                                    w = worley_3d(p * hii_freq)
                                    d_hii = ti.max(
                                        0.0, 1.0 - w / hii_threshold
                                    ) * (arm_mod_p - 1.0)

                                d_comp = 0.0
                                if comp_mass > 0.0 or bridge_strength > 0.0:
                                    d_comp = self._d_companion_bridge(
                                        p, comp_pos, comp_mass, bridge_strength
                                    )

                                arm_only = ti.max(0.0, arm_mod_p - 1.0)
                                em = (
                                    warm_yellow * d_bulge * 2.5
                                    + white_yellow * d_disk
                                    + blue_white * d_disk * arm_only * 1.2
                                    + pink * d_hii * 2.0
                                    + dim_warm * d_halo
                                    + warm_yellow * d_comp * 2.0
                                )

                                # Doppler tint -- proportional to inclination,
                                # depends on the local azimuth relative to the
                                # camera projection in the disk plane.
                                doppler = (
                                    doppler_strength
                                    * sin_incl
                                    * doppler_sign
                                    * ti.cos(theta - phi_view)
                                )
                                em = ti.Vector([
                                    em[0] * (1.0 + doppler),
                                    em[1],
                                    em[2] * (1.0 - doppler),
                                ])

                                # Per-channel extinction (dust reddens disk).
                                sigma_base = (d_disk + d_bulge) * 0.4
                                dust_ext = d_dust * dust_strength
                                sigma_r = sigma_base + dust_ext * 0.7
                                sigma_g = sigma_base + dust_ext * 1.0
                                sigma_b = sigma_base + dust_ext * 1.2

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
