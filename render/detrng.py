"""Deterministic counter-based RNG for Taichi kernels.

`ti.random()` draws from a per-thread stream whose sequence depends on
GPU scheduling, so two runs of the same frame can differ at sub-pixel
level — enough to seam chunked renders (see ODYSSEY_README's determinism
caveat). Every stochastic effect in the engine (AA jitter, film grain,
dye repainting, storm impulses) instead draws from these stateless hash
functions, keyed on *what* is being shaded — pixel, sample index, frame
tick, stream salt — so any frame renders bit-identically on any run,
any chunk boundary, and any thread schedule.

The hash is the single-round PCG output permutation over a chained key
(O'Neill 2014; Jarzynski & Olano 2020 rate it well on avalanche
quality at one round). Streams are decorrelated with distinct salts.

Usage inside a kernel:

    from render.detrng import rand01, rand_centered, tick, S_AA_X

    f = tick(time)                      # int tick from the kernel's time arg
    jx = rand_centered(i, j, s, f + S_AA_X)
    grain = rand_centered(i, j, f, S_GRAIN) * 0.04

Salts spread streams far apart in key space; adding the tick to a salt
is fine — they only need to differ between streams drawn with the same
(a, b, c) key.

`py_rand01` / `py_hash_u32` are bit-exact pure-Python mirrors of the
Taichi functions, used by CPU-side tests and available to any numpy
bake pipeline that wants the same streams.

(No `from __future__ import annotations` here: Taichi's @ti.func needs
real annotation objects, not strings.)
"""

# Stream salts. Arbitrary large odd constants, far apart. Kept below
# 2**31 so they stay valid Taichi i32 literals.
S_AA_X = 0x51ED2701
S_AA_Y = 0x63A90BB5
S_GRAIN = 0x2545F491
S_SPARKLE = 0x1B873593
S_DYE_HALO = 0x0BD4BCB5
S_DYE_SPOT = 0x7FEB352D
S_DYE_BANDS = 0x452821E7
S_STORM_GATE = 0x38D01377
S_STORM_VX = 0x2AB5C95F
S_STORM_VY = 0x6C62272E
S_POLAR_VX = 0x14057B7F
S_POLAR_VY = 0x30BE8698

_M32 = 0xFFFFFFFF

# The engine's frame times step by at least 1/60 s, so a 1024x
# quantisation separates consecutive frames while staying exact in f32
# for sequences hours long.
TICK_SCALE = 1024.0


def _py_pcg(x: int) -> int:
    state = (x * 747796405 + 2891336453) & _M32
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & _M32
    return ((word >> 22) ^ word) & _M32


def py_hash_u32(a: int, b: int, c: int, d: int) -> int:
    """Pure-Python mirror of `hash_u32` (bit-exact)."""
    h = _py_pcg(a & _M32)
    h = _py_pcg(h ^ (b & _M32))
    h = _py_pcg(h ^ (c & _M32))
    h = _py_pcg(h ^ (d & _M32))
    return h


def py_rand01(a: int, b: int, c: int, d: int) -> float:
    """Pure-Python mirror of `rand01` (same distribution; float64 here
    vs f32 in-kernel, so equality holds to f32 precision)."""
    return py_hash_u32(a, b, c, d) / 4294967296.0


def py_tick(time: float) -> int:
    """Pure-Python mirror of `tick`."""
    return int(time * TICK_SCALE + 0.5)


try:
    import taichi as ti
    _HAS_TAICHI = True
except Exception:  # pragma: no cover - GPU-only dep
    ti = None
    _HAS_TAICHI = False


if _HAS_TAICHI:

    @ti.func
    def _pcg(x: ti.u32) -> ti.u32:
        state = x * ti.u32(747796405) + ti.u32(2891336453)
        word = ((state >> ((state >> ti.u32(28)) + ti.u32(4))) ^ state) * ti.u32(277803737)
        return (word >> ti.u32(22)) ^ word

    @ti.func
    def hash_u32(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32) -> ti.u32:
        """Mix four ints into one well-avalanched u32."""
        h = _pcg(ti.cast(a, ti.u32))
        h = _pcg(h ^ ti.cast(b, ti.u32))
        h = _pcg(h ^ ti.cast(c, ti.u32))
        h = _pcg(h ^ ti.cast(d, ti.u32))
        return h

    @ti.func
    def rand01(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32) -> ti.f32:
        """Uniform in [0, 1), a pure function of the four key ints."""
        return ti.cast(hash_u32(a, b, c, d), ti.f32) * (1.0 / 4294967296.0)

    @ti.func
    def rand_centered(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32) -> ti.f32:
        """Uniform in [-0.5, 0.5), a pure function of the four key ints."""
        return rand01(a, b, c, d) - 0.5

    @ti.func
    def tick(time: ti.f32) -> ti.i32:
        """Distinct integer per frame from a kernel's time argument."""
        return ti.cast(time * TICK_SCALE + 0.5, ti.i32)

else:
    # CPU-only stubs so the symbols always exist. Calling raises
    # immediately rather than failing at GPU kernel compile time.
    def hash_u32(*a, **k):
        raise RuntimeError("hash_u32 requires Taichi (use py_hash_u32)")

    def rand01(*a, **k):
        raise RuntimeError("rand01 requires Taichi (use py_rand01)")

    def rand_centered(*a, **k):
        raise RuntimeError("rand_centered requires Taichi")

    def tick(*a, **k):
        raise RuntimeError("tick requires Taichi (use py_tick)")
