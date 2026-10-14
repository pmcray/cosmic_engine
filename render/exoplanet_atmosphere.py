"""Exoplanet atmosphere variants.

Three topology classes share one volumetric atmosphere kernel:

  hot_jupiter   - tidally locked, magma hotspot on dayside, alkali-metal
                  haze (red/orange dominant), extreme equatorial jet
  mini_neptune  - methane haze (deep blue), almost no surface features
                  visible, thicker atmosphere, smaller core
  brown_dwarf   - silicate cloud bands, alternating dark/light at
                  multiple latitudes, often very deep red-brown

Same architectural invariants as volumetric_gas_giant: no baked
texture beyond the band-color lookup, all detail via render/noise.py
footprint-gated primitives. A single kernel branches on a topology id.
"""

import math

import numpy as np

from render.noise import (
    domain_warp_fbm,
    fbm_3d_footprint,
    value_noise_3d,
)

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover
    ti = None
    _HAS_TAICHI = False
from render.detrng import rand_centered, tick, S_AA_X, S_AA_Y


TOPOLOGY_IDS = {
    "hot_jupiter": 0,
    "mini_neptune": 1,
    "brown_dwarf": 2,
}


def topology_id(name: str) -> int:
    if name not in TOPOLOGY_IDS:
        raise ValueError(
            f"unknown exoplanet topology '{name}'. "
            f"Known: {sorted(TOPOLOGY_IDS.keys())}"
        )
    return TOPOLOGY_IDS[name]


if not _HAS_TAICHI:

    class ExoplanetAtmosphereEngine:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError(
                "ExoplanetAtmosphereEngine requires Taichi + GPU."
            )

else:

    @ti.data_oriented
    class ExoplanetAtmosphereEngine:
        def __init__(
            self,
            width=1280,
            height=720,
            topology="hot_jupiter",
            atm_thickness=0.07,
            march_steps=30,
            samples=2,
            sun_dir=(-0.55, 0.18, -0.81),
            seed=0,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.atm_thickness = float(atm_thickness)
            self.march_steps = int(march_steps)
            self.samples = int(samples)

            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            self.gp = ti.field(dtype=ti.f32, shape=(8,))
            arr = np.array([
                float(topology_id(topology)),
                float(atm_thickness),
                float(seed),
                0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            self.gp.from_numpy(arr)

            self.cam = ti.field(dtype=ti.f32, shape=(16,))
            self._set_default_camera()
            self.set_sun(sun_dir)

        def _set_default_camera(self):
            self.set_camera(
                position=(0.0, 0.2, -3.4),
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

        # ---- topology surface helpers ------------------------------

        @ti.func
        def _hot_jupiter_surface(self, n, sun_dir, footprint, p):
            lat = ti.asin(ti.max(-1.0, ti.min(1.0, n[1])))
            # Cosine angle to sub-stellar point: dayside hot, nightside dark.
            day = ti.max(0.0, n.dot(sun_dir))
            base = ti.Vector([0.85, 0.45, 0.30]) * (0.3 + 0.7 * day)
            band = ti.cos(lat * 14.0) * 0.15
            base += ti.Vector([0.2, 0.05, -0.10]) * band
            # Magma hotspot near sub-stellar point.
            hot = ti.pow(day, 12.0) * 1.2
            base += ti.Vector([1.0, 0.55, 0.2]) * hot
            # Procedural detail.
            d = domain_warp_fbm(p, footprint, 32.0, 5, 10.0, 0.40)
            base = base * (1.0 + d * 0.45)
            return base

        @ti.func
        def _mini_neptune_surface(self, n, sun_dir, footprint, p):
            lat = ti.asin(ti.max(-1.0, ti.min(1.0, n[1])))
            day = ti.max(0.0, n.dot(sun_dir))
            # Deep methane blue; very subtle banding.
            base = ti.Vector([0.18, 0.32, 0.65]) * (0.4 + 0.6 * day)
            band = ti.cos(lat * 6.0) * 0.06
            base += ti.Vector([0.0, 0.03, 0.08]) * band
            d = fbm_3d_footprint(p, footprint, 28.0, 5)
            base = base * (1.0 + d * 0.25)
            return base

        @ti.func
        def _brown_dwarf_surface(self, n, sun_dir, footprint, p):
            lat = ti.asin(ti.max(-1.0, ti.min(1.0, n[1])))
            day = ti.max(0.0, n.dot(sun_dir))
            # Deep red-brown body with alternating silicate cloud bands.
            band = ti.cos(lat * 22.0) * 0.5 + 0.5
            base_a = ti.Vector([0.55, 0.30, 0.18])
            base_b = ti.Vector([0.30, 0.16, 0.10])
            base = base_a * band + base_b * (1.0 - band)
            base *= 0.20 + 0.80 * day
            # Stormy turbulence.
            d = domain_warp_fbm(p, footprint, 48.0, 6, 14.0, 0.45)
            base = base * (1.0 + d * 0.55)
            return base

        @ti.func
        def _background(self, rd):
            col = ti.Vector([0.004, 0.005, 0.012])
            h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
            h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
            if h * h2 > 0.9985:
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

            topo = ti.cast(self.gp[0], ti.i32)
            atm = self.gp[1]

            R_p = 1.0
            R_atm = 1.0 + atm
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

                    color = self._background(rd)

                    b = rd.dot(cam_pos)
                    cp = cam_pos.dot(cam_pos) - R_p * R_p
                    d_p = b * b - cp
                    if d_p > 0.0:
                        tp = -b - ti.sqrt(d_p)
                        if tp > 0.0:
                            p_surf = cam_pos + rd * tp
                            n = p_surf / (p_surf.norm() + 1e-6)
                            footprint = pixel_size * tp
                            base = ti.Vector([0.0, 0.0, 0.0])
                            if topo == 0:
                                base = self._hot_jupiter_surface(n, sun_dir, footprint, p_surf)
                            elif topo == 1:
                                base = self._mini_neptune_surface(n, sun_dir, footprint, p_surf)
                            elif topo == 2:
                                base = self._brown_dwarf_surface(n, sun_dir, footprint, p_surf)
                            # Soft terminator.
                            soft = ti.max(0.0, n.dot(sun_dir) + 0.08) / 1.08
                            color = base * (0.05 + 0.95 * ti.pow(soft, 0.65))

                    accum += color

                self.pixels[i, j] = accum / float(self.samples)
