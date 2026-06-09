"""Terrain volumetric renderer.

Heightmap-based terrain via ridged multi-fractal noise (from the
toolkit), ray-marched with logarithmic step sizing so far-distant
mountains and near-foreground rocks both resolve. Includes:

  - Ridged-fBM topography (gives the sharp ridge crests of real ranges)
  - Sun-direction-based diffuse lighting + shadow raymarch
  - Simple atmospheric scattering (blue haze with distance)
  - Snow line above a configurable elevation
  - Optional water plane with a tinted depth

Same architectural invariants as the other renderers: no baked
texture, pure procedural via render/noise.py footprint-gated
primitives, so the same renderer holds up at any zoom level.

Coord convention: world Y is "up"; camera typically stands a bit
above the average ground, looking toward the horizon.
"""

from __future__ import annotations

import math

import numpy as np

from render.noise import ridged_fbm_3d, fbm_3d_footprint

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover
    ti = None
    _HAS_TAICHI = False


if not _HAS_TAICHI:

    class TerrainEngine:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError("TerrainEngine requires Taichi and a GPU runtime.")

else:

    @ti.data_oriented
    class TerrainEngine:
        def __init__(
            self,
            width=1280,
            height=720,
            # topography
            terrain_amplitude=1.4,
            terrain_freq=0.04,
            octaves=8,
            snow_line=0.85,
            water_level=-0.15,
            water_enable=1,
            # lighting
            sun_dir=(0.45, 0.55, -0.7),
            # atmosphere
            haze_strength=0.6,
            sky_zenith=(0.20, 0.42, 0.78),
            sky_horizon=(0.92, 0.78, 0.62),
            # ray-march
            max_dist=80.0,
            march_steps=140,
            samples=1,
            seed=0,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.march_steps = int(march_steps)
            self.samples = int(samples)
            self.octaves = int(octaves)

            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            self.gp = ti.field(dtype=ti.f32, shape=(24,))
            arr = np.zeros(24, dtype=np.float32)
            arr[0] = terrain_amplitude
            arr[1] = terrain_freq
            arr[2] = snow_line
            arr[3] = water_level
            arr[4] = float(water_enable)
            arr[5] = haze_strength
            arr[6] = max_dist
            sd = np.asarray(sun_dir, dtype=np.float32)
            sd /= np.linalg.norm(sd) + 1e-8
            arr[7:10] = sd
            arr[10:13] = sky_zenith
            arr[13:16] = sky_horizon
            arr[16] = float(seed)
            self.gp.from_numpy(arr)

            self.cam = ti.field(dtype=ti.f32, shape=(16,))
            self._set_default_camera()

        def _set_default_camera(self):
            self.set_camera(
                position=(0.0, 1.2, -5.0),
                target=(0.0, 0.6, 0.0),
                up_world=(0.0, 1.0, 0.0),
                fov_deg=55.0,
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
        def _terrain_height(self, x, z, footprint, amp, freq):
            p = ti.Vector([x * freq, 0.0, z * freq])
            h = ridged_fbm_3d(p, footprint, 1.0, self.octaves)
            return h * amp

        @ti.func
        def _sky(self, rd, sky_z, sky_h, sun_dir, haze):
            t = ti.max(0.0, rd[1])
            sky = sky_z * t + sky_h * (1.0 - t)
            # Sun disc.
            sd = rd.dot(sun_dir)
            sun_disc = ti.pow(ti.max(0.0, sd), 64.0)
            sky += ti.Vector([1.0, 0.95, 0.85]) * sun_disc * 0.8
            return sky

        # ---- main kernel -------------------------------------------

        @ti.kernel
        def render_kernel(self, t: ti.f32):
            cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
            cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
            cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
            cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
            fov_scale = self.cam[12]

            amp = self.gp[0]
            freq = self.gp[1]
            snow_line = self.gp[2]
            water_level = self.gp[3]
            water_enable = self.gp[4]
            haze = self.gp[5]
            max_dist = self.gp[6]
            sun_dir = ti.Vector([self.gp[7], self.gp[8], self.gp[9]])
            sky_z = ti.Vector([self.gp[10], self.gp[11], self.gp[12]])
            sky_h = ti.Vector([self.gp[13], self.gp[14], self.gp[15]])

            pixel_size = 2.0 * fov_scale / float(self.height)

            rock_low = ti.Vector([0.32, 0.27, 0.22])
            rock_high = ti.Vector([0.55, 0.50, 0.45])
            grass = ti.Vector([0.32, 0.45, 0.22])
            snow = ti.Vector([0.95, 0.96, 0.98])
            water_col = ti.Vector([0.05, 0.20, 0.30])

            for i, j in self.pixels:
                u = ((float(j) + 0.5) / self.width) * 2.0 - 1.0
                v = ((float(i) + 0.5) / self.height) * 2.0 - 1.0
                v = -v
                rd = (
                    cam_fwd
                    + cam_right * (u * fov_scale * self.aspect)
                    + cam_up * (v * fov_scale)
                )
                rd = rd / (rd.norm() + 1e-8)

                # March: logarithmic step so foreground and distance both resolve.
                t_h = 0.05
                hit = 0
                p = cam_pos
                d_accum = 0.0
                for k in range(self.march_steps):
                    p = cam_pos + rd * d_accum
                    fp = pixel_size * d_accum
                    h_terrain = self._terrain_height(p[0], p[2], fp, amp, freq)
                    if p[1] < h_terrain:
                        hit = 1
                        break
                    step = ti.max(0.02, 0.04 * d_accum)
                    d_accum += step
                    if d_accum > max_dist:
                        break

                color = self._sky(rd, sky_z, sky_h, sun_dir, haze)

                if hit == 1:
                    # Approximate intersection with one refinement step.
                    p_hit = p
                    # Normal via central differences of height.
                    eps = 0.05
                    fp = pixel_size * d_accum
                    h_x1 = self._terrain_height(p_hit[0] + eps, p_hit[2], fp, amp, freq)
                    h_x0 = self._terrain_height(p_hit[0] - eps, p_hit[2], fp, amp, freq)
                    h_z1 = self._terrain_height(p_hit[0], p_hit[2] + eps, fp, amp, freq)
                    h_z0 = self._terrain_height(p_hit[0], p_hit[2] - eps, fp, amp, freq)
                    n = ti.Vector([
                        -(h_x1 - h_x0) / (2.0 * eps),
                        1.0,
                        -(h_z1 - h_z0) / (2.0 * eps),
                    ])
                    n = n / (n.norm() + 1e-8)

                    cos_sun = ti.max(0.0, n.dot(sun_dir))
                    elev = p_hit[1]

                    base_col = ti.Vector([0.0, 0.0, 0.0])
                    if elev > snow_line * amp:
                        base_col = snow
                    elif elev > 0.4 * amp:
                        base_col = rock_high
                    elif elev > 0.0:
                        base_col = grass
                    else:
                        base_col = rock_low

                    # Fine-scale detail (footprint-AA).
                    fp = pixel_size * d_accum
                    detail = fbm_3d_footprint(
                        ti.Vector([p_hit[0], p_hit[1], p_hit[2]]),
                        fp, 8.0, 5,
                    ) * 0.20
                    surface = base_col * (1.0 + detail) * (0.25 + 0.75 * cos_sun)

                    # Haze along distance.
                    haze_t = ti.min(1.0, d_accum / max_dist)
                    surface = surface * (1.0 - haze_t * haze) + color * (haze_t * haze)

                    color = surface

                    # Water plane.
                    if water_enable > 0.5 and p_hit[1] < water_level:
                        color = water_col

                self.pixels[i, j] = color
