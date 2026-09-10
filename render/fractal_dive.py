"""Mandelbrot / Multibrot deep-dive renderer (perturbation theory).

The missing scenario class for the long-form voyages: an exponential
dive into the Mandelbrot set (or its power-d Multibrot cousins), far
past the ~1e-13 scale where naive float64 iteration dissolves into
pixelated mush.

Method
------
Classic perturbation rendering (K.I. Martin's SuperFractalThing note,
2013), with the rebasing refinement (Zhuoran, fractalforums 2022) that
retires most glitch handling:

  1. One *reference orbit* is iterated at the dive's centre in
     arbitrary precision (stdlib `decimal`, precision auto-scaled to
     the zoom depth), then stored as a float64 shadow:
         Z_0 = 0,  Z_{n+1} = Z_n^d + C
  2. Every pixel iterates only its tiny offset δ from the reference,
     entirely in float64:
         δ_{n+1} = Σ_{k=1..d} binom(d,k) Z_n^{d-k} δ^k  +  δc
     (for d=2 the familiar 2·Z·δ + δ² + δc).
  3. Rebasing: whenever |Z_n + δ| < |δ| (the pixel orbit is nearer the
     origin than the reference is) or the reference runs out, set
     δ ← Z_n + δ and jump back to Z_0 = 0. This keeps δ small and
     kills the classic Pauldelbrot glitches without secondary
     references.

Escape values are converted to a smooth (fractional) iteration count
and coloured through the shot palette's anchors on the CPU.

Two engines share the exact same math:

  * `perturbation_grid`   — vectorised numpy; the reference
                            implementation, used by tests and CPU hosts.
  * `FractalDiveEngine`   — Taichi f64 kernel for production zooms.

Everything is deterministic: the same manifest renders the same frames
bit-for-bit (supersampling uses a fixed sub-pixel grid, no RNG at all).

CLI (single still):

    python -m render.fractal_dive --scale 1e-8 --out dive.npy
"""

import argparse
import math
from decimal import Decimal, getcontext, localcontext

import numpy as np

try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover - GPU-only dep
    ti = None
    _HAS_TAICHI = False


# A classic deep-dive location on the seahorse-valley spiral: structure
# at every depth, never escapes early.
DEFAULT_CENTER_RE = "-0.743643887037158704752191506114774"
DEFAULT_CENTER_IM = "0.131825904205311970493132056385139"


# ============================================================
# Reference orbit (arbitrary precision, CPU)
# ============================================================

def required_precision(scale: float) -> int:
    """Decimal digits needed to resolve pixel offsets at `scale`
    (viewport half-width in complex units), with guard digits."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return max(30, 20 + int(math.ceil(-math.log10(scale))))


class ReferenceOrbit:
    """High-precision orbit of the dive centre, stored as a float64
    shadow (`z`: complex128 array, Z_0 = 0 first).

    The orbit stops early if the centre escapes; perturbation handles
    the truncation by rebasing.
    """

    def __init__(self, center_re: str = DEFAULT_CENTER_RE,
                 center_im: str = DEFAULT_CENTER_IM,
                 power: int = 2,
                 max_iter: int = 2048,
                 precision: int = 60,
                 escape_radius: float = 1e10):
        if power < 2 or power != int(power):
            raise ValueError(f"power must be an integer >= 2, got {power}")
        self.power = int(power)
        self.center_re = str(center_re)
        self.center_im = str(center_im)
        self.max_iter = int(max_iter)
        self.precision = int(precision)

        esc2 = float(escape_radius) ** 2
        with localcontext() as ctx:
            ctx.prec = self.precision
            cr = Decimal(self.center_re)
            ci = Decimal(self.center_im)
            zr = Decimal(0)
            zi = Decimal(0)
            zs = [0.0 + 0.0j]
            for _ in range(self.max_iter):
                # z^d by repeated complex multiplication (d is small).
                pr, pi = zr, zi
                for _k in range(self.power - 1):
                    pr, pi = pr * zr - pi * zi, pr * zi + pi * zr
                zr, zi = pr + cr, pi + ci
                fr, fi = float(zr), float(zi)
                zs.append(complex(fr, fi))
                if fr * fr + fi * fi > esc2:
                    break
        self.z = np.array(zs, dtype=np.complex128)

    def __len__(self) -> int:
        return len(self.z)


# ============================================================
# Perturbation iteration (numpy reference engine)
# ============================================================

def _binoms(d: int) -> np.ndarray:
    return np.array([math.comb(d, k) for k in range(d + 1)], dtype=np.float64)


def perturbation_grid(orbit: ReferenceOrbit,
                      dc: np.ndarray,
                      max_iter: int,
                      escape_radius: float = 64.0) -> np.ndarray:
    """Iterate every δc in `dc` (complex128, any shape) against the
    reference orbit. Returns the smooth iteration count per point, or
    -1.0 where the point never escaped (interior)."""
    d = orbit.power
    binom = _binoms(d)
    Z = orbit.z
    n_ref = len(Z)
    esc2 = float(escape_radius) ** 2
    log_d = math.log(d)
    log_esc = math.log(escape_radius)

    shape = dc.shape
    dc = dc.ravel()
    delta = np.zeros_like(dc)
    m = np.zeros(dc.shape, dtype=np.int64)     # reference index per point
    mu = np.full(dc.shape, -1.0, dtype=np.float64)
    active = np.ones(dc.shape, dtype=bool)

    for n in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        Zm = Z[m[idx]]
        z = Zm + delta[idx]
        z2 = z.real * z.real + z.imag * z.imag

        escaped = z2 > esc2
        if escaped.any():
            e = idx[escaped]
            r = np.sqrt(z2[escaped])
            # Smooth iteration count: n + 1 - log_d(log|z| / log R).
            mu[e] = n + 1.0 - np.log(np.log(r) / log_esc) / log_d
            active[e] = False
            keep = ~escaped
            idx = idx[keep]
            if idx.size == 0:
                continue
            Zm = Zm[keep]
            z = z[keep]

        # Rebase (Zhuoran): pixel orbit closer to the origin than the
        # reference, or reference exhausted -> restart at Z_0 = 0.
        dd = delta[idx]
        rebase = (z.real * z.real + z.imag * z.imag
                  < dd.real * dd.real + dd.imag * dd.imag) \
            | (m[idx] + 1 >= n_ref)
        if rebase.any():
            rb = idx[rebase]
            delta[rb] = z[rebase]
            m[rb] = 0
            Zm = np.where(rebase, 0.0 + 0.0j, Zm)
            dd = delta[idx]

        # δ' = δ · Horner_q(δ) + δc, with q(δ) = Σ binom(d,k) Z^{d-k} δ^{k-1},
        # accumulated in ascending Z-power order (k = d-1 .. 1).
        q = np.full(idx.shape, binom[d], dtype=np.complex128)  # k = d term
        Zpow = np.ones_like(Zm)
        for j in range(1, d):
            Zpow = Zpow * Zm
            q = q * dd + binom[d - j] * Zpow
        delta[idx] = dd * q + dc[idx]
        m[idx] += 1

    return mu.reshape(shape)


def direct_grid(c: np.ndarray, power: int, max_iter: int,
                escape_radius: float = 64.0) -> np.ndarray:
    """Plain full-precision-free escape-time iteration (float64), for
    validation at shallow zooms."""
    z = np.zeros_like(c)
    mu = np.full(c.shape, -1.0, dtype=np.float64)
    active = np.ones(c.shape, dtype=bool)
    esc2 = float(escape_radius) ** 2
    log_d = math.log(power)
    log_esc = math.log(escape_radius)
    for n in range(max_iter):
        z[active] = z[active] ** power + c[active]
        z2 = z.real * z.real + z.imag * z.imag
        escaped = active & (z2 > esc2)
        if escaped.any():
            r = np.sqrt(z2[escaped])
            # The escaping value here is z_{n+1}; the smooth count is
            # (escape index) + 1 - correction, matching perturbation_grid.
            mu[escaped] = (n + 1) + 1.0 - np.log(np.log(r) / log_esc) / log_d
            active &= ~escaped
        if not active.any():
            break
    return mu


# ============================================================
# Dive geometry + colour
# ============================================================

def dive_scale(t: float, scale_start: float, scale_end: float) -> float:
    """Exponential zoom: constant apparent speed in log space."""
    t = max(0.0, min(1.0, t))
    return scale_start * (scale_end / scale_start) ** t


def pixel_deltas(width: int, height: int, scale: float,
                 rotation: float = 0.0,
                 sub_dx: float = 0.0, sub_dy: float = 0.0) -> np.ndarray:
    """δc grid (H, W) complex128, relative to the dive centre. `scale`
    is the half-height of the viewport in complex units; rows run
    top-down (row 0 = +imag). Sub-pixel offsets are in pixel units."""
    aspect = width / height
    xs = (np.arange(width, dtype=np.float64) + 0.5 + sub_dx) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5 + sub_dy) / height
    u = (xs * 2.0 - 1.0) * scale * aspect
    v = (1.0 - ys * 2.0) * scale
    du, dv = np.meshgrid(u, v)
    if rotation != 0.0:
        cr, sr = math.cos(rotation), math.sin(rotation)
        du, dv = du * cr - dv * sr, du * sr + dv * cr
    return du + 1j * dv


def iter_budget(scale: float, base: int = 600, per_decade: int = 800) -> int:
    """Iteration ceiling that grows with depth."""
    decades = max(0.0, -math.log10(max(scale, 1e-300)))
    return int(base + per_decade * decades)


def mu_span(mu: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """Robust (low, high) escape-count range of a frame, ignoring
    interior points. Both ends drift smoothly as the dive descends —
    which is what lets `colorize` normalise per frame without the
    palette flickering between frames."""
    esc = mu[mu >= 0.0]
    if esc.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(esc, lo_pct))
    hi = float(np.percentile(esc, hi_pct))
    return lo, max(hi, lo + 1e-6)


def colorize(mu: np.ndarray,
             anchors,
             color_cycles: float = 4.0,
             color_phase: float = 0.0,
             interior_color=(0.0, 0.0, 0.0),
             span=None) -> np.ndarray:
    """Map smooth iteration counts through cyclic palette anchors.
    Returns float32 (..., 3) scene-linear RGB.

    Escape counts are normalised against the frame's own robust range
    (`span`, defaulting to `mu_span(mu)`) before cycling, so a frame
    reads the same whether its counts run 20..300 near the surface or
    8500..14600 twenty decades down — without that, deep frames
    collapse to a single flat colour. `color_cycles` is how many times
    the palette repeats across that range.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    n = len(anchors)
    if n < 2:
        raise ValueError("need at least two palette anchors")
    interior = mu < 0.0
    lo, hi = mu_span(mu) if span is None else span
    norm = (np.maximum(mu, 0.0) - lo) / (hi - lo)
    t = color_cycles * norm + color_phase
    pos = (t % 1.0) * n
    i0 = pos.astype(np.int64) % n
    i1 = (i0 + 1) % n
    frac = (pos - np.floor(pos))[..., None]
    # Smoothstep between anchors keeps the bands from looking linear.
    frac = frac * frac * (3.0 - 2.0 * frac)
    out = anchors[i0] * (1.0 - frac) + anchors[i1] * frac
    out[interior] = np.asarray(interior_color, dtype=np.float64)
    return out.astype(np.float32)


def supersample_offsets(samples: int):
    """Fixed sub-pixel offset grid (deterministic — no RNG)."""
    s = max(1, int(samples))
    return [((sx + 0.5) / s - 0.5, (sy + 0.5) / s - 0.5)
            for sy in range(s) for sx in range(s)]


# ============================================================
# Taichi engine (production)
# ============================================================

if _HAS_TAICHI:

    @ti.data_oriented
    class FractalDiveEngine:
        """Taichi f64 perturbation iterator. Computes the smooth-mu
        grid on the backend; colouring stays on the CPU (cheap, and it
        keeps palette handling in one place)."""

        def __init__(self, width: int, height: int, orbit: ReferenceOrbit):
            self.width = int(width)
            self.height = int(height)
            self.set_orbit(orbit)
            self.mu = ti.field(dtype=ti.f32, shape=(self.height, self.width))
            self._binom = ti.field(dtype=ti.f64, shape=max(3, orbit.power + 1))

        def set_orbit(self, orbit: ReferenceOrbit) -> None:
            self.power = orbit.power
            self._n_ref = len(orbit)
            self._zr = ti.field(dtype=ti.f64, shape=self._n_ref)
            self._zi = ti.field(dtype=ti.f64, shape=self._n_ref)
            self._zr.from_numpy(orbit.z.real.copy())
            self._zi.from_numpy(orbit.z.imag.copy())

        def compute_mu(self, scale: float, rotation: float,
                       max_iter: int, escape_radius: float = 64.0,
                       sub_dx: float = 0.0, sub_dy: float = 0.0) -> np.ndarray:
            b = _binoms(self.power)
            for k in range(self.power + 1):
                self._binom[k] = float(b[k])
            self._kernel(float(scale), float(rotation), int(max_iter),
                         float(escape_radius) ** 2,
                         math.log(float(escape_radius)),
                         math.log(float(self.power)),
                         float(sub_dx), float(sub_dy))
            return self.mu.to_numpy()

        @ti.kernel
        def _kernel(self, scale: ti.f64, rotation: ti.f64, max_iter: ti.i32,
                    esc2: ti.f64, log_esc: ti.f64, log_d: ti.f64,
                    sub_dx: ti.f64, sub_dy: ti.f64):
            aspect = ti.cast(self.width, ti.f64) / ti.cast(self.height, ti.f64)
            for py, px in self.mu:
                u = ((ti.cast(px, ti.f64) + 0.5 + sub_dx)
                     / self.width * 2.0 - 1.0) * scale * aspect
                v = (1.0 - (ti.cast(py, ti.f64) + 0.5 + sub_dy)
                     / self.height * 2.0) * scale
                cr = ti.cos(rotation)
                sr = ti.sin(rotation)
                dcr = u * cr - v * sr
                dci = u * sr + v * cr

                dr = ti.cast(0.0, ti.f64)
                di = ti.cast(0.0, ti.f64)
                m = 0
                out = ti.cast(-1.0, ti.f64)
                for n in range(max_iter):
                    zr = self._zr[m] + dr
                    zi = self._zi[m] + di
                    z2 = zr * zr + zi * zi
                    if z2 > esc2:
                        r = ti.sqrt(z2)
                        out = ti.cast(n, ti.f64) + 1.0 \
                            - ti.log(ti.log(r) / log_esc) / log_d
                        break
                    # Rebase.
                    if z2 < dr * dr + di * di or m + 1 >= self._n_ref:
                        dr = zr
                        di = zi
                        m = 0
                    Zr = self._zr[m]
                    Zi = self._zi[m]
                    # Horner: q = Σ binom(d,k) Z^{d-k} δ^{k-1}, k = d..1.
                    qr = self._binom[self.power]
                    qi = ti.cast(0.0, ti.f64)
                    Zpr = ti.cast(1.0, ti.f64)
                    Zpi = ti.cast(0.0, ti.f64)
                    for j in range(1, self.power):
                        Zpr, Zpi = Zpr * Zr - Zpi * Zi, Zpr * Zi + Zpi * Zr
                        qr, qi = (qr * dr - qi * di
                                  + self._binom[self.power - j] * Zpr,
                                  qr * di + qi * dr
                                  + self._binom[self.power - j] * Zpi)
                    dr, di = (dr * qr - di * qi + dcr,
                              dr * qi + di * qr + dci)
                    m += 1
                self.mu[py, px] = ti.cast(out, ti.f32)


# ============================================================
# CLI (single still, for framing checks)
# ============================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fractal_dive")
    parser.add_argument("--center-re", default=DEFAULT_CENTER_RE)
    parser.add_argument("--center-im", default=DEFAULT_CENTER_IM)
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1e-6,
                        help="viewport half-height in complex units")
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--max-iter", type=int, default=0,
                        help="0 = auto from depth")
    parser.add_argument("--out", required=True,
                        help=".png (needs imageio/cv2) or .npy")
    args = parser.parse_args(argv)

    max_iter = args.max_iter or iter_budget(args.scale)
    orbit = ReferenceOrbit(args.center_re, args.center_im, power=args.power,
                           max_iter=max_iter,
                           precision=required_precision(args.scale))
    dc = pixel_deltas(args.width, args.height, args.scale,
                      math.radians(args.rotation_deg))
    mu = perturbation_grid(orbit, dc, max_iter)

    from scene.palette import get_palette
    rgb = colorize(mu, get_palette("trumbull_2001").anchors)
    img8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    if args.out.endswith(".npy"):
        np.save(args.out, rgb)
    else:
        try:
            import imageio.v3 as iio
            iio.imwrite(args.out, img8)
        except ImportError:
            import cv2
            cv2.imwrite(args.out, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out} (scale {args.scale:g}, {max_iter} max iters, "
          f"{len(orbit)} ref orbit points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
