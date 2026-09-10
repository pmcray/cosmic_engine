# Fractal Dive — Mandelbrot / Multibrot deep zooms

`render/fractal_dive.py` renders exponential dives into the Mandelbrot set
and its power-*d* Multibrot cousins, far past the ~1e-13 scale where plain
float64 escape-time iteration dissolves into flat mush. It is the scenario
class the voyages were missing: a tunnel that is pure mathematics, with no
end to the structure at any depth.

Fully procedural, fully deterministic — no genAI, no RNG anywhere in the
path (supersampling uses a fixed sub-pixel grid), so the same manifest
renders the same frames bit-for-bit on every run and every chunk boundary.

## Method

Perturbation rendering (K.I. Martin, *SuperFractalThing*, 2013) with the
rebasing refinement (Zhuoran, fractalforums 2022):

1. **Reference orbit.** One orbit is iterated at the dive centre in
   arbitrary precision (stdlib `decimal`, precision auto-scaled to the
   target depth by `required_precision`), then kept as a float64 shadow:
   `Z₀ = 0`, `Z_{n+1} = Z_nᵈ + C`.
2. **Per-pixel deltas.** Every pixel iterates only its tiny offset δ from
   that reference, entirely in float64:
   `δ_{n+1} = Σ_{k=1..d} C(d,k)·Z_n^{d-k}·δᵏ + δc`
   — for *d*=2 the familiar `2Zδ + δ² + δc`.
3. **Rebasing.** Whenever `|Z_n + δ| < |δ|`, or the reference runs out,
   set `δ ← Z_n + δ` and restart at `Z₀ = 0`. This keeps δ small and
   removes the classic Pauldelbrot glitches without secondary references.

Escape values become smooth (fractional) iteration counts, coloured
through the shot palette's anchors.

Two engines share identical math: `perturbation_grid` (vectorised numpy,
the reference implementation the tests check) and `FractalDiveEngine`
(Taichi f64 kernel, for production zooms). They agree to under 0.01
iterations at the 99th percentile.

## Colouring

Deep frames have escape counts clustered far from zero (e.g. 8500–14600
twenty decades down). Mapping those absolutely collapses the frame to one
flat colour, so `colorize` normalises against the frame's own robust
range (1st–99th percentile, via `mu_span`) before cycling the palette.
Both ends of that range drift smoothly as the dive descends, so frames
stay consistent without flickering. `color_cycles` sets how many times the
palette repeats across that range.

## Single stills

```bash
# Shallow — the seahorse-valley spiral
python -m render.fractal_dive --scale 5e-3 --width 960 --height 540 --out shallow.png

# Deep — well past float64's reach
python -m render.fractal_dive --scale 1e-15 --width 960 --height 540 --out deep.png

# A Multibrot (d=3), rotated
python -m render.fractal_dive --power 3 --scale 1e-6 --rotation-deg 40 \
    --center-re 0.3575572475710187173097895550841719553 \
    --center-im 0.7101740570828986189270892820783667134 --out multi3.png
```

## As a graph shot

Registered as the `fractal_dive` renderer, so it drops into any manifest:

```json
{
  "id": "tunnel_00_fractal_dive",
  "renderer": "fractal_dive",
  "params": {
    "center_re": "-0.737751206161254491181307210762363450",
    "center_im": "0.1276207733591109064923511174750151381",
    "power": 2,
    "scale_start": 1.8,
    "scale_end": 1e-20,
    "rotation_deg_total": 180.0,
    "samples": 2,
    "color_cycles": 5.0
  },
  "duration_frames": 480,
  "palette": {"name": "trumbull_2001"}
}
```

The voyage composer places these automatically: `t_fractal_dive` (a long
plunge, in the tunnel phase alongside the slit-scan corridor) and
`t_fractal_dive_shallow` (a shorter descent, in the threshold and
resolution phases). Both draw from `FRACTAL_LOCATIONS` in
`studio/voyage_composer.py` — eight centres, each bisected at high
precision until its orbit survives thousands of iterations, so the dive
keeps finding structure all the way down.

## Choosing a centre

A dive centre must be a point that does *not* escape: an exterior point
bottoms out in a few dozen iterations and the dive hits featureless
colour almost immediately. The digit count of the centre bounds how deep
it can be pushed — roughly, you need as many decimal digits as decades of
zoom, plus guard digits (`required_precision` applies that rule). To add
a location, bisect at high precision between a nearby interior and
exterior point until the orbit survives several thousand iterations;
`tests/test_fractal_dive.py::test_fractal_locations_are_valid_and_parse`
enforces that property for every entry.

## Cost

Iteration budget grows with depth (`iter_budget`: ~800 more iterations per
decade by default), so deep frames cost proportionally more — a 1e-20
frame runs ~16,000 iterations per pixel against a reference orbit of the
same length. The Taichi engine is the one to use for anything past a
preview; the numpy engine is the CPU-host and test fallback.
