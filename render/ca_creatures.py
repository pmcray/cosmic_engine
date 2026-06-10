"""Continuous cellular-automata "creatures" renderer (Lenia-style).

A continuous-state 2D cellular automaton that evolves in time per
frame. Generates plausible "alien yet real" organic blobs that move,
split, and interact. The renderer is a thin Taichi wrapper around
the standard Lenia update rule:

    A_{t+1}(x) = clip(A_t(x) + dt * G(K * A_t(x)), 0, 1)

where K is a smoothed annular kernel and G is a Gaussian growth
function. The Lenia parameters (kernel radius R, mu, sigma, growth
mu_g, sigma_g) define which "species" emerges. Different parameter
seeds give different morphologies; the same parameters reproduce
the same creature.

Two channels for two species so they interact: a slower magenta
predator species, a faster cyan prey species, each with its own
kernel and growth. The renderer composites both channels into a
glowing 2D field over a dark background.

Camera convention is 2D: position (x, z) parameterizes pan; the y
component controls zoom (larger y = wider field of view).
"""

import math

import numpy as np

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover
    ti = None
    _HAS_TAICHI = False


def _lenia_kernel(grid_size: int, R: float, mu: float, sigma: float):
    """Generate a smooth annular Lenia kernel of given peak radius R."""
    coords = np.arange(grid_size, dtype=np.float64) - grid_size / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2) / R
    K = np.exp(-((rr - mu) ** 2) / (2.0 * sigma ** 2))
    K[rr > 1.0] = 0.0
    K /= K.sum() + 1e-12
    return K.astype(np.float32)


def _gaussian_growth(u, mu, sigma):
    return 2.0 * np.exp(-((u - mu) ** 2) / (2.0 * sigma ** 2)) - 1.0


def _seed_state(grid_size: int, seed: int = 0, n_seeds: int = 8, seed_radius: int = 12):
    rng = np.random.default_rng(seed)
    state = np.zeros((grid_size, grid_size), dtype=np.float32)
    for _ in range(n_seeds):
        cy = rng.integers(seed_radius, grid_size - seed_radius)
        cx = rng.integers(seed_radius, grid_size - seed_radius)
        for dy in range(-seed_radius, seed_radius + 1):
            for dx in range(-seed_radius, seed_radius + 1):
                if dx * dx + dy * dy <= seed_radius * seed_radius:
                    state[cy + dy, cx + dx] = rng.random()
    return state


if not _HAS_TAICHI:

    class CACreaturesEngine:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError("CACreaturesEngine requires Taichi and a GPU runtime.")

else:

    @ti.data_oriented
    class CACreaturesEngine:
        def __init__(
            self,
            width=1280,
            height=720,
            grid_size=256,
            # species A (cyan prey)
            A_R=13, A_kernel_mu=0.5, A_kernel_sigma=0.15,
            A_growth_mu=0.15, A_growth_sigma=0.015,
            # species B (magenta predator)
            B_R=24, B_kernel_mu=0.55, B_kernel_sigma=0.12,
            B_growth_mu=0.12, B_growth_sigma=0.020,
            dt=0.1,
            substeps_per_frame=2,
            samples=1,
            seed=0,
            warmup_steps=80,
        ):
            self.width = int(width)
            self.height = int(height)
            self.aspect = float(self.width) / float(self.height)
            self.grid_size = int(grid_size)
            self.substeps = int(substeps_per_frame)
            self.samples = int(samples)
            self.dt = float(dt)

            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))

            # Lenia state for both species.
            self.A = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size))
            self.B = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size))
            # Kernel sums (precomputed convolution buffers).
            self.A_tmp = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size))
            self.B_tmp = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size))

            # Kernels stored as Taichi fields.
            self.KA = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size))
            self.KB = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size))
            self.KA.from_numpy(_lenia_kernel(self.grid_size, A_R, A_kernel_mu, A_kernel_sigma))
            self.KB.from_numpy(_lenia_kernel(self.grid_size, B_R, B_kernel_mu, B_kernel_sigma))

            self.gp = ti.field(dtype=ti.f32, shape=(8,))
            arr = np.array([
                A_growth_mu, A_growth_sigma,
                B_growth_mu, B_growth_sigma,
                self.dt, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            self.gp.from_numpy(arr)

            # Seed state and warm up the simulation so creatures form.
            seedA = _seed_state(self.grid_size, seed=seed, n_seeds=10, seed_radius=A_R)
            seedB = _seed_state(self.grid_size, seed=seed + 7, n_seeds=4, seed_radius=B_R)
            self.A.from_numpy(seedA)
            self.B.from_numpy(seedB)
            for _ in range(int(warmup_steps)):
                self.step()

            self.cam = ti.field(dtype=ti.f32, shape=(8,))
            self._set_default_camera()

        def _set_default_camera(self):
            self.set_camera(pan_x=0.0, pan_y=0.0, zoom=1.0)

        def set_camera(self, pan_x=0.0, pan_y=0.0, zoom=1.0, **_):
            arr = np.zeros(8, dtype=np.float32)
            arr[0] = float(pan_x)
            arr[1] = float(pan_y)
            arr[2] = float(zoom)
            self.cam.from_numpy(arr)

        def render(self, t: float) -> np.ndarray:
            for _ in range(self.substeps):
                self.step()
            self.render_kernel()
            return self.pixels.to_numpy()

        # ---- Lenia step --------------------------------------------

        @ti.kernel
        def step(self):
            # Convolve A with KA (circular FFT-style via direct grid).
            # For tractable per-frame cost, sample sparsely with a
            # 9x9 stencil weighted by the central portion of the kernel.
            G = ti.static(9)
            half = G // 2
            for i, j in self.A_tmp:
                s_a = 0.0
                s_b = 0.0
                w_a = 0.0
                w_b = 0.0
                for dy in ti.static(range(G)):
                    for dx in ti.static(range(G)):
                        yk = (i + dy - half) % self.grid_size
                        xk = (j + dx - half) % self.grid_size
                        ka = self.KA[(self.grid_size // 2 + dy - half),
                                     (self.grid_size // 2 + dx - half)]
                        kb = self.KB[(self.grid_size // 2 + dy - half),
                                     (self.grid_size // 2 + dx - half)]
                        s_a += self.A[yk, xk] * ka
                        s_b += self.B[yk, xk] * kb
                        w_a += ka
                        w_b += kb
                if w_a > 1e-9:
                    s_a /= w_a
                if w_b > 1e-9:
                    s_b /= w_b
                self.A_tmp[i, j] = s_a
                self.B_tmp[i, j] = s_b

            mu_a = self.gp[0]
            sig_a = self.gp[1]
            mu_b = self.gp[2]
            sig_b = self.gp[3]
            dt = self.gp[4]

            for i, j in self.A:
                u_a = self.A_tmp[i, j]
                u_b = self.B_tmp[i, j]
                g_a = 2.0 * ti.exp(-((u_a - mu_a) ** 2) / (2.0 * sig_a * sig_a)) - 1.0
                g_b = 2.0 * ti.exp(-((u_b - mu_b) ** 2) / (2.0 * sig_b * sig_b)) - 1.0
                # Predator-prey coupling: B eats A; A grows where B is sparse.
                interact_a = -self.B[i, j] * 0.15
                interact_b = self.A[i, j] * 0.05
                self.A[i, j] = ti.max(0.0, ti.min(1.0,
                    self.A[i, j] + dt * (g_a + interact_a)))
                self.B[i, j] = ti.max(0.0, ti.min(1.0,
                    self.B[i, j] + dt * (g_b + interact_b)))

        @ti.kernel
        def render_kernel(self):
            pan_x = self.cam[0]
            pan_y = self.cam[1]
            zoom = self.cam[2]
            for i, j in self.pixels:
                u = ((float(j) + 0.5) / self.width) * 2.0 - 1.0
                v = ((float(i) + 0.5) / self.height) * 2.0 - 1.0
                v = -v
                # Map (u, v) into grid coords with zoom + pan.
                gx = (u * zoom + pan_x) * (self.grid_size * 0.5) + self.grid_size * 0.5
                gy = (v * zoom + pan_y) * (self.grid_size * 0.5) + self.grid_size * 0.5
                ix = ti.cast(ti.max(0.0, ti.min(float(self.grid_size - 1), gx)), ti.i32)
                iy = ti.cast(ti.max(0.0, ti.min(float(self.grid_size - 1), gy)), ti.i32)
                a = self.A[iy, ix]
                b = self.B[iy, ix]
                # Cyan for A, magenta for B; bright glow where both overlap.
                col = ti.Vector([
                    a * 0.20 + b * 1.10,
                    a * 1.05 + b * 0.30,
                    a * 1.10 + b * 0.85,
                ])
                self.pixels[i, j] = col
