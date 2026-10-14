# Odyssey Director — "Jupiter Space and Beyond the Infinite"

Offline, headless renderer for long-form cinematic sequences in the spirit of
the *Jupiter Space* / *Beyond the Infinite* movements of *2001: A Space
Odyssey*, built from the Cosmic Engine's simulation stack. This is the
production path toward the project goal: **multi-minute UHD (3840×2160)
sequences** composed from the object classes the engine can simulate.

Unlike `stargate_sequence.py` (interactive `ti.GUI` window, square frame,
realtime), `odyssey_director.py` renders to MP4 or numbered PNG frames at any
resolution and duration, with no display required.

## The sequence

One continuous three-act timeline (fractions of total duration):

| Act | Span | Content |
|-----|------|---------|
| I — The Approach | 0.00 – 0.55 | Photoreal Jovian fluid dynamics (calmed, zonally constrained `FluidEngine`), slow dolly from deep space to the cloud deck, sun sinking behind the planet into a backlit crescent — ending in a white-out flash |
| II — The Stargate | 0.55 – 0.85 | Slit-scan light corridor (`render/stargate_corridor.py`): two converging luminous planes, streaming fBM light, four colour epochs (blue/violet → molten red → emerald → white-gold), a slow tumble from horizontal to vertical |
| III — The Infinite | 0.85 – 1.00 | Gravitationally lensed black hole (`physics/black_hole.py` geodesic integrator), a slow fall toward the event horizon, crossfaded in from the corridor, fading to black |

All acts share ACES tone mapping and the same film-grain pass, so the
sequence grades as one piece.

## Quick start

```bash
pip install taichi opencv-python-headless numpy

# 30-second preview straight to MP4 (fine on CPU)
python odyssey_director.py --preset preview --duration 30 --output preview.mp4

# Single frame to check framing/quality at any point on the timeline
python odyssey_director.py --preset hd --duration 240 --still 1400 --still-out check.png
```

## Rendering a multi-minute UHD master

A 4-minute UHD master at 24 fps is 5,760 frames of 3840×2160 — render it as
PNG frames, in chunks, on a GPU (Colab works; the engine initialises
`ti.gpu` and falls back to CPU):

```bash
# Chunk 1 (frames 0-1439) -- repeat with different ranges, in parallel
# sessions or sequentially. --seed makes every chunk agree on the planet.
python odyssey_director.py --preset uhd --duration 240 --seed 42 \
    --frames-dir frames_uhd --start-frame 0 --end-frame 1440

python odyssey_director.py --preset uhd --duration 240 --seed 42 \
    --frames-dir frames_uhd --start-frame 1440 --end-frame 2880
# ... 2880-4320, 4320-5760

# Encode the master
ffmpeg -framerate 24 -i frames_uhd/frame_%06d.png \
    -c:v libx264 -crf 16 -pix_fmt yuv420p odyssey_uhd_4min.mp4
```

Notes on chunking:

- **Resumable**: existing PNG frames are skipped (use `--overwrite` to force).
  A crashed chunk can simply be re-run.
- **Stateful simulation**: the Jovian atmosphere is a real fluid sim, so each
  chunk fast-forwards the simulation through the frames before its
  `--start-frame` (cheap relative to rendering — and skipped entirely once
  the planet is off screen after Act I).
- **Determinism**: `--seed` fixes the planet parameters, and every
  stochastic effect in the kernels (sub-pixel AA jitter, film grain, dye
  repainting, storm impulses) draws from the counter-based hash RNG in
  `render/detrng.py`, keyed on pixel/sample/frame — so any frame renders
  bit-identically on any run, chunk boundary, or thread schedule. Chunks
  can be split anywhere.

## Presets

| Preset | Resolution | Fluid grid | AA samples | fps |
|--------|-----------|------------|------------|-----|
| `preview` | 640×360 | 192² | 1 | 12 |
| `sd` | 960×540 | 256² | 2 | 24 |
| `hd` | 1920×1080 | 512² | 2 | 24 |
| `uhd` | 3840×2160 | 768² | 4 | 24 |

Every field is individually overridable (`--width --height --fluid-res
--samples --fps`). `--duration` is in seconds and defaults to 240.

## What changed in the engine to support this

- `render/camera.py` — `SphereCamera` accepts `render_w`/`render_h` for
  widescreen output with aspect-corrected rays; camera pose (pan/tilt/roll),
  a new dolly distance (`cam_dist`) and exposure now live in Taichi fields so
  per-frame animation actually reaches the kernels (plain Python attributes
  were baked in as compile-time constants); the cloud-wisp fBM uses smooth 3D
  value noise instead of per-sample white noise (which rendered as speckle).
- `render/stargate_corridor.py` — new widescreen slit-scan corridor renderer.
- `render/geodesic_engine.py` — rectangular output support.
- `physics/fluid_solver.py` — optional `meridional_damping` (default off)
  keeps circulation zonally constrained so the cloud bands survive long
  simulations; the director also runs the sim with a small `dt` and gentle
  turbulence multipliers for the stately 2001 pacing.

## Extending the sequence

Other simulated object classes can be spliced in as additional acts:
`encounters/exotic_physics.py` (hypernovae), `encounters/megastructure.py`
(L-system alien structures / mega-flora via the Factory pipeline), and the
`infinite_director.py` planetary waypoints. Each act only needs a
`render_<act>(progress, t)` method returning a `(W, H, 3)` float array and an
entry in `render_frame`'s timeline.
