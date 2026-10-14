"""Cosmic-web point-cloud renderer.

The "fly through a universe where galaxies are grains of sand" view:
millions of galaxies sourced from DESI (or synthesized procedurally
when the catalog isn't yet downloaded) splatted into a 3D emission
+ density grid, then ray-marched through with the same kernel pattern
used by every other Cosmic Engine volumetric renderer.

Pipeline:

  1. Particles -> tracer-color-weighted emission grid (CPU, numpy).
     The grid resolution is small (default 128^3) so memory is bounded
     regardless of how many particles we splat.

  2. Trilinear-interpolated volumetric ray-march in the kernel. Each
     ray accumulates emission * transmittance * dt; extinction comes
     from the same grid's local density.

  3. Per-channel transmittance preserved (as in every other renderer)
     so the dust-equivalent here -- intergalactic absorption -- can
     reddens light from background galaxies.

  4. Anti-pixelation via the standard footprint-gated fBM: at every
     ray-march step we layer a small procedural emission term so
     close-ups reveal finer "grain-of-sand" detail than the splat grid
     alone could produce. As the camera zooms in, the toolkit's
     footprint gating opens up additional octaves.

The grid acceleration structure means rendering scales O(grid_voxels)
not O(particles). 1 million galaxies render no slower than 100 000.

Tracer-class palette comes from `data.cosmic_pipeline.TRACER_COLORS`:
BGS orange / LRG red / ELG teal / QSO violet, per the user's spec.
"""

import math

import numpy as np

from render.cosmic_pipeline import (
    TRACER_COLORS,
    synthesize_cosmic_web,
)
from render.noise import fbm_3d_footprint, value_noise_3d

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover
    ti = None
    _HAS_TAICHI = False
from render.detrng import rand_centered, tick, S_AA_X, S_AA_Y


def splat_particles_to_grid(
    positions, tracer, magnitudes,
    box_size_mpc=3000.0,
    grid_dim=128,
    splat_sigma_vox=1.2,
):
    """Splat (N, 3) particles into a (grid, grid, grid, 3) emission grid
    and a (grid, grid, grid) density grid using trilinear weights.

    Returns (emission, density) as float32 arrays. Center of the grid
    corresponds to world origin. The grid extends ±box_size_mpc/2 on
    each axis.

    The Gaussian splat is approximated by writing each particle into
    its 8 nearest voxels with trilinear weights, then applying one
    pass of a 3x3x3 box blur so the structure reads as smooth
    filaments at grid scale rather than per-voxel speckle.
    """
    G = int(grid_dim)
    voxel_size = box_size_mpc / G

    emission = np.zeros((G, G, G, 3), dtype=np.float32)
    density = np.zeros((G, G, G), dtype=np.float32)

    # Map world coords to voxel coords.
    half = box_size_mpc * 0.5
    f = (positions + half) / voxel_size  # in voxel space, real-valued
    f = np.clip(f, 0.0, G - 1.0001)
    i0 = f.astype(np.int32)
    fr = (f - i0).astype(np.float32)

    # Trilinear weights to 8 corners.
    w0 = 1.0 - fr
    # Per-particle color contribution.
    colors_arr = np.array(TRACER_COLORS, dtype=np.float32)
    per_particle_color = colors_arr[tracer]  # (N, 3)
    mass = magnitudes.astype(np.float32)
    em_contrib = per_particle_color * mass[:, None]  # (N, 3)

    for dz in (0, 1):
        wz = w0[:, 2] if dz == 0 else fr[:, 2]
        for dy in (0, 1):
            wy = w0[:, 1] if dy == 0 else fr[:, 1]
            for dx in (0, 1):
                wx = w0[:, 0] if dx == 0 else fr[:, 0]
                w = wx * wy * wz
                ix = i0[:, 0] + dx
                iy = i0[:, 1] + dy
                iz = i0[:, 2] + dz
                np.add.at(density, (ix, iy, iz), w * mass)
                np.add.at(emission[..., 0], (ix, iy, iz), w * em_contrib[:, 0])
                np.add.at(emission[..., 1], (ix, iy, iz), w * em_contrib[:, 1])
                np.add.at(emission[..., 2], (ix, iy, iz), w * em_contrib[:, 2])

    # Light separable 3-tap blur so visible filaments aren't blocky.
    def _blur_one_axis(arr, axis):
        a = np.roll(arr, 1, axis=axis)
        b = np.roll(arr, -1, axis=axis)
        return (a + arr + b) / 3.0

    for _ in range(2):
        density = _blur_one_axis(_blur_one_axis(_blur_one_axis(density, 0), 1), 2)
        for c in range(3):
            emission[..., c] = _blur_one_axis(
                _blur_one_axis(_blur_one_axis(emission[..., c], 0), 1), 2
            )

    # Normalize so the brightest voxel doesn't blow out.
    max_d = float(density.max())
    if max_d > 0:
        density /= max_d
        emission /= max_d

    return emission.astype(np.float32), density.astype(np.float32)


if not _HAS_TAICHI:

    class CosmicWebEngine:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError(
                "CosmicWebEngine requires Taichi and a GPU runtime."
            )

else:

    @ti.data_oriented
    class CosmicWebEngine:
        """Volumetric cosmic-web fly-through renderer.

        Coord convention: galaxies live in a box ±box_size_mpc/2 around
        the origin (the observer's home). The camera flies through this
        volume; the box itself is what the kernel intersects to bound
        the ray-march.
        """

        def __init__(
            self,
            width=1280,
            height=720,
            # data: either a precomputed catalog or procedurally synth
            positions=None,
            tracer=None,
            magnitudes=None,
            box_size_mpc=3000.0,
            grid_dim=128,
            # synthesizer fallback
            n_synth_particles=120_000,
            n_synth_clusters=400,
            synth_seed=0,
            # ray-march
            march_steps=128,
            samples=2,
            # presentation
            emission_gain=4.0,
            extinction_strength=0.15,
            background_gain=1.0,
            seed=0,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.march_steps = int(march_steps)
            self.samples = int(samples)
            self.box_size = float(box_size_mpc)
            self.grid_dim = int(grid_dim)

            # ---- Synthesize / use catalog ----
            if positions is None:
                positions, tracer, magnitudes, _zs = synthesize_cosmic_web(
                    n_particles=n_synth_particles,
                    box_size_mpc=box_size_mpc,
                    n_clusters=n_synth_clusters,
                    seed=synth_seed,
                )

            # Bake into emission + density grids.
            emission, density = splat_particles_to_grid(
                positions, tracer, magnitudes,
                box_size_mpc=box_size_mpc,
                grid_dim=grid_dim,
            )

            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))
            self.emission = ti.Vector.field(3, dtype=ti.f32,
                                            shape=(grid_dim, grid_dim, grid_dim))
            self.density = ti.field(dtype=ti.f32,
                                    shape=(grid_dim, grid_dim, grid_dim))
            self.emission.from_numpy(emission)
            self.density.from_numpy(density)

            # Parameters.
            self.gp = ti.field(dtype=ti.f32, shape=(16,))
            arr = np.zeros(16, dtype=np.float32)
            arr[0] = emission_gain
            arr[1] = extinction_strength
            arr[2] = background_gain
            arr[3] = box_size_mpc
            arr[4] = float(grid_dim)
            arr[5] = box_size_mpc * 0.5  # half-extent
            self.gp.from_numpy(arr)

            # Camera.
            self.cam = ti.field(dtype=ti.f32, shape=(16,))
            self._set_default_camera()

        def _set_default_camera(self):
            half = self.box_size * 0.5
            self.set_camera(
                position=(0.0, 0.0, -half * 1.2),
                target=(0.0, 0.0, 0.0),
                up_world=(0.0, 1.0, 0.0),
                fov_deg=45.0,
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
        def _sample_grid(self, p, half):
            """Trilinear sample (emission, density) from the 3D grid.
            Returns (em_r, em_g, em_b, density) as a Vector(4)."""
            G = ti.cast(self.gp[4], ti.i32)
            voxel = self.gp[3] / float(G)
            # Map world position to [0, G-1] voxel coords.
            fp = (p + ti.Vector([half, half, half])) / voxel
            # Out-of-bounds -> zero contribution.
            inb = (
                fp[0] >= 0.0 and fp[0] < float(G - 1)
                and fp[1] >= 0.0 and fp[1] < float(G - 1)
                and fp[2] >= 0.0 and fp[2] < float(G - 1)
            )
            r0 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            if inb:
                ix = ti.cast(ti.floor(fp[0]), ti.i32)
                iy = ti.cast(ti.floor(fp[1]), ti.i32)
                iz = ti.cast(ti.floor(fp[2]), ti.i32)
                fx = fp[0] - float(ix)
                fy = fp[1] - float(iy)
                fz = fp[2] - float(iz)

                e000 = self.emission[ix, iy, iz]; d000 = self.density[ix, iy, iz]
                e100 = self.emission[ix + 1, iy, iz]; d100 = self.density[ix + 1, iy, iz]
                e010 = self.emission[ix, iy + 1, iz]; d010 = self.density[ix, iy + 1, iz]
                e110 = self.emission[ix + 1, iy + 1, iz]; d110 = self.density[ix + 1, iy + 1, iz]
                e001 = self.emission[ix, iy, iz + 1]; d001 = self.density[ix, iy, iz + 1]
                e101 = self.emission[ix + 1, iy, iz + 1]; d101 = self.density[ix + 1, iy, iz + 1]
                e011 = self.emission[ix, iy + 1, iz + 1]; d011 = self.density[ix, iy + 1, iz + 1]
                e111 = self.emission[ix + 1, iy + 1, iz + 1]; d111 = self.density[ix + 1, iy + 1, iz + 1]

                ex00 = e000 * (1.0 - fx) + e100 * fx
                ex10 = e010 * (1.0 - fx) + e110 * fx
                ex01 = e001 * (1.0 - fx) + e101 * fx
                ex11 = e011 * (1.0 - fx) + e111 * fx
                exy0 = ex00 * (1.0 - fy) + ex10 * fy
                exy1 = ex01 * (1.0 - fy) + ex11 * fy
                em = exy0 * (1.0 - fz) + exy1 * fz

                dx00 = d000 * (1.0 - fx) + d100 * fx
                dx10 = d010 * (1.0 - fx) + d110 * fx
                dx01 = d001 * (1.0 - fx) + d101 * fx
                dx11 = d011 * (1.0 - fx) + d111 * fx
                dxy0 = dx00 * (1.0 - fy) + dx10 * fy
                dxy1 = dx01 * (1.0 - fy) + dx11 * fy
                den = dxy0 * (1.0 - fz) + dxy1 * fz
                r0 = ti.Vector([em[0], em[1], em[2], den])
            return r0

        @ti.func
        def _box_intersect(self, ro, rd, half):
            """Slab method for axis-aligned cube of half-extent `half`."""
            t_min = -1e10
            t_max = 1e10
            for k in ti.static(range(3)):
                if ti.abs(rd[k]) < 1e-8:
                    if ro[k] < -half or ro[k] > half:
                        t_min = 1e10
                        t_max = -1e10
                else:
                    inv = 1.0 / rd[k]
                    t1 = (-half - ro[k]) * inv
                    t2 = (half - ro[k]) * inv
                    if t1 > t2:
                        t1, t2 = t2, t1
                    t_min = ti.max(t_min, t1)
                    t_max = ti.min(t_max, t2)
            return t_min, t_max

        @ti.func
        def _background(self, rd):
            # Very subtle cosmic-background gradient + sparse foreground stars.
            col = ti.Vector([0.001, 0.0015, 0.005])
            h = ti.sin(rd[0] * 173.7) * ti.sin(rd[1] * 121.3) * ti.sin(rd[2] * 89.5)
            h2 = ti.sin(rd[0] * 311.1 + rd[1] * 271.7 + rd[2] * 199.3)
            if h * h2 > 0.997:
                col += ti.Vector([1.0, 1.0, 0.95]) * 0.5
            return col

        # ---- main kernel -------------------------------------------

        @ti.kernel
        def render_kernel(self, t: ti.f32):
            cam_pos = ti.Vector([self.cam[0], self.cam[1], self.cam[2]])
            cam_fwd = ti.Vector([self.cam[3], self.cam[4], self.cam[5]])
            cam_right = ti.Vector([self.cam[6], self.cam[7], self.cam[8]])
            cam_up = ti.Vector([self.cam[9], self.cam[10], self.cam[11]])
            fov_scale = self.cam[12]

            emission_gain = self.gp[0]
            extinction_strength = self.gp[1]
            background_gain = self.gp[2]
            half = self.gp[5]

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

                    color = self._background(rd) * background_gain

                    t_min, t_max = self._box_intersect(cam_pos, rd, half)
                    if t_max > t_min and t_max > 0.0:
                        t_start = ti.max(0.0, t_min)
                        t_end = t_max
                        N = self.march_steps
                        dt = (t_end - t_start) / float(N)

                        accum_em = ti.Vector([0.0, 0.0, 0.0])
                        trans = ti.Vector([1.0, 1.0, 1.0])

                        for k in range(N):
                            t_k = t_start + dt * (float(k) + 0.5)
                            p = cam_pos + rd * t_k

                            r4 = self._sample_grid(p, half)
                            em = ti.Vector([r4[0], r4[1], r4[2]]) * emission_gain
                            den = r4[3]

                            # Procedural sub-grid detail so close-ups don't
                            # reveal the splat-grid lattice. Footprint-AA.
                            footprint = pixel_size * t_k
                            fine = fbm_3d_footprint(
                                p * (1.0 / (half * 0.05)),
                                footprint / (half * 0.05),
                                3.0, 5,
                            )
                            em = em * (1.0 + 0.55 * fine)

                            accum_em += em * trans * dt
                            sigma = den * extinction_strength
                            # Slight chromatic extinction (red wins).
                            trans = ti.Vector([
                                trans[0] * ti.exp(-sigma * dt * 0.8),
                                trans[1] * ti.exp(-sigma * dt * 1.0),
                                trans[2] * ti.exp(-sigma * dt * 1.25),
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
