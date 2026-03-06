# Chronos: L-System Growth Animation Engine

**Phase 4 of the Algorithmic Beauty of Plants Project**

Chronos implements smooth, temporally coherent growth animations from L-systems, building on Phases 1-3.

---

## Overview

Chronos extends the L-system pipeline with:
- **Timed DOL-systems** (Developmental L-systems with age tracking)
- **Smooth growth interpolation** (no "popping" effects)
- **Temporal coherence** for AI-generated videos
- **Multiple export formats** (OBJ sequences, depth maps, videos)

Based on ABOP Chapter 1, Section 1.10.1 (Timed L-systems).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Phase 1-2: L-System Core + Bio-Sim                 │
│  (Parametric rewriting, environmental simulation)   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│  PHASE 4: CHRONOS - ANIMATION ENGINE                │
│                                                      │
│  Timed Symbols  →  Interpolation  →  Frame Gen     │
│  (age tracking)    (smooth growth)   (trajectory)  │
│                                                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│  Phase 3: The Factory (Optional)                    │
│  (AI synthesis with temporal coherence)             │
└─────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Timed DOL-Systems

Each symbol has:
- **Age** (`age`): Current age of symbol
- **Terminal Age** (`terminal_age`): Age when symbol is fully mature
- **Growth Factor**: `age / terminal_age` (0.0 to 1.0)

Symbols grow smoothly over time instead of appearing instantly.

### 2. Temporal Interpolation

Between L-system iterations `n` and `n+1`, geometry is interpolated:
- Position interpolation
- Length/width scaling based on growth factor
- Smooth transitions using easing functions

### 3. Animation Timeline

```
Time:     0s ─────────── 5s ─────────── 10s ──────────
Iteration: 0            1             2
Alpha:    [0.0 ... 1.0] [0.0 ... 1.0]
```

At any time `t`, we determine:
- Which iteration (`n`)
- Interpolation factor (`α` between iterations `n` and `n+1`)

### 4. Temporal Coherence (AI Video)

For AI-synthesized videos:
- Previous frame influences next frame
- Maintains visual continuity
- Prevents flickering and inconsistency

---

## Modules

### Core Modules (6 files)

1. **`timed_symbol.py`** - Timed symbol data structures
   - `TimedSymbol`: Symbol with age and growth factor
   - `GrowthFunction`: Various growth curves
   - `TimedString`: Container for timed symbols

2. **`timer.py`** - Time management
   - `AnimationClock`: Global timeline management
   - `TimePoint`: Maps time to iteration + alpha
   - `EasingFunction`: Smooth interpolation curves

3. **`timed_turtle.py`** - Growth-aware turtle graphics
   - `TimedTurtle`: Interprets timed symbols
   - Geometry interpolation between states
   - `IterationCache`: Performance optimization

4. **`animation_generator.py`** - Frame generation
   - `AnimationGenerator`: Coordinates rewriting + interpolation
   - Frame-by-frame trajectory generation
   - OBJ sequence export

5. **`video_factory.py`** - Video synthesis
   - `VideoFactory`: Depth map sequences → video
   - Temporal coherence for AI synthesis
   - FFmpeg/OpenCV encoding

6. **`chronos.py`** - Command-line interface
   - User-friendly CLI
   - Multiple output modes
   - Integration with Factory (Phase 3)

---

## Installation

Chronos requires all dependencies from Phases 1-3:

```bash
# Phase 1-2 dependencies (required)
pip install numpy numba pandas torch matplotlib

# Phase 3 dependencies (optional, for AI video)
pip install Pillow opencv-python diffusers transformers accelerate

# Video encoding (at least one)
pip install opencv-python  # OpenCV
# OR install FFmpeg: https://ffmpeg.org/download.html
```

---

## Usage

### Quick Start

```bash
# Generate depth map sequence (no AI)
python chronos.py \
    --file examples/simple_tree.l \
    --iter 4 \
    --duration 10 \
    --maps-only \
    --output anim/

# Generate keyframes only (fast preview)
python chronos.py \
    --file examples/3d_tree.l \
    --iter 5 \
    --keyframes \
    --output keyframes/

# Export OBJ sequence for external rendering
python chronos.py \
    --file examples/tree.l \
    --iter 4 \
    --duration 8 \
    --obj-sequence \
    --output frames/
```

### AI Video Synthesis

```bash
# Generate AI video with temporal coherence
python chronos.py \
    --file examples/3d_tree.l \
    --iter 5 \
    --duration 15 \
    --video output.mp4 \
    --style prehistoric \
    --resolution 1024

# With custom settings
python chronos.py \
    --file examples/parametric_tree_anim.l \
    --iter 6 \
    --duration 20 \
    --video alien_growth.mp4 \
    --style bioluminescent \
    --temporal-strength 0.2 \
    --seed 42 \
    --camera iso \
    --fps 30
```

### Quick Preview

```bash
# Generate fast preview (limited frames, no AI)
python chronos.py \
    --file examples/tree.l \
    --iter 3 \
    --preview \
    --max-frames 60 \
    --output preview.mp4
```

---

## Command-Line Options

### Required
- `--file FILE` - L-system file (.l)

### Animation Parameters
- `--iter N` - Maximum iterations (default: 4)
- `--duration SECS` - Animation duration (default: 10.0s)
- `--fps FPS` - Frames per second (default: 30.0)
- `--terminal-age AGE` - Default terminal age (default: 1.0)

### Output Modes
- `--maps-only` - Generate depth maps only
- `--obj-sequence` - Export OBJ file sequence
- `--keyframes` - Export keyframes only
- `--video FILE` - Generate video file
- `--preview` - Quick preview (limited frames)

### AI Synthesis
- `--style PRESET` - Aesthetic style (default: prehistoric)
- `--seed N` - Random seed for AI
- `--temporal-strength F` - Temporal coherence (default: 0.15)
- `--list-styles` - List available styles

### Camera & Resolution
- `--camera PRESET` - Camera preset (iso, top, front, closeup)
- `--resolution N` - Output resolution (default: 1024)

### Utilities
- `--system-info` - Print system info
- `--verbose` - Verbose output
- `--stats` - Print statistics

---

## Examples

### Example 1: Growing Branch (2D)

```bash
python chronos.py \
    --file examples/timed/growing_branch.l \
    --iter 4 \
    --duration 12 \
    --maps-only \
    --output growing_branch/
```

**Result**: 360 depth map frames showing smooth branch growth

### Example 2: 3D Tree with AI

```bash
python chronos.py \
    --file examples/timed/simple_tree_anim.l \
    --iter 5 \
    --duration 15 \
    --video tree_growth.mp4 \
    --style botanical \
    --resolution 1024 \
    --verbose
```

**Result**: 15-second video with photorealistic tree growth

### Example 3: Spiral Growth

```bash
python chronos.py \
    --file examples/timed/spiral_growth.l \
    --iter 6 \
    --duration 18 \
    --obj-sequence \
    --output spiral_frames/
```

**Result**: OBJ sequence for external rendering (Blender, Maya, etc.)

### Example 4: Parametric Tree

```bash
python chronos.py \
    --file examples/timed/parametric_tree_anim.l \
    --iter 5 \
    --duration 10 \
    --video parametric.mp4 \
    --style alien \
    --seed 12345 \
    --camera closeup
```

**Result**: Alien-style parametric tree growth video

---

## Integration with Other Phases

### Phase 1 (L-System Core)

Chronos uses Phase 1 components:
- `lexer.py` - Parse L-system files
- `rewriter.py` - Apply production rules
- `turtle_3d.py` - Base turtle graphics

### Phase 2 (Bio-Sim)

**Future**: Chronos can be extended to animate bio-sim results:
- Time-varying tropism
- Resource allocation over time
- Shadow movement with sun position

### Phase 3 (The Factory)

Chronos integrates with Factory for AI video:
- `map_generator.py` - Depth/normal maps per frame
- `prompt_engine.py` - Aesthetic presets
- `diffusion_client.py` - AI synthesis with temporal coherence

---

## Output Formats

### 1. Depth Map Sequence

Directory containing:
- `depth_00000.png` to `depth_NNNNN.png`
- `video_metadata.json`

**Use case**: Import into video editing software

### 2. OBJ Sequence

Directory containing:
- `frame_00000.obj` to `frame_NNNNN.obj`
- `frame_metadata.json`

**Use case**: External rendering (Blender, Maya, Houdini)

### 3. Keyframes

Directory containing:
- `keyframe_000.obj` to `keyframe_NNN.obj`

**Use case**: Quick preview of growth stages

### 4. Video (MP4)

Single video file:
- H.264 encoded
- Configurable FPS
- Optional AI synthesis

**Use case**: Final output for presentations, papers, social media

---

## Performance

### Depth Maps Only
- **4 iterations, 10s @ 30fps**: ~5-10 seconds
- **5 iterations, 15s @ 30fps**: ~10-20 seconds

### OBJ Sequence
- **4 iterations, 10s @ 30fps**: ~10-20 seconds
- **5 iterations, 15s @ 30fps**: ~20-40 seconds

### AI Video (GPU Required)
- **1024px, 10s @ 30fps**: ~15-30 minutes (300 frames)
- **1024px, 15s @ 30fps**: ~20-45 minutes (450 frames)
- Requires NVIDIA GPU with 8GB+ VRAM

### Quick Preview
- **3 iterations, 60 frames**: ~2-5 seconds

---

## Technical Details

### Timed Symbol Growth

Each symbol's visual properties scale with growth factor:

```python
growth_factor = min(1.0, age / terminal_age)
current_length = max_length * growth_factor
current_width = max_width * growth_factor
```

### Growth Curves

Multiple growth functions available:
- **Linear**: `f(t) = t`
- **Smoothstep**: `f(t) = 3t² - 2t³`
- **Cubic**: Ken Perlin's smootherstep
- **Exponential**: Fast early growth
- **Sigmoid**: S-curve (slow-fast-slow)

### Interpolation Between Iterations

At time point with `iteration=n` and `alpha=0.3`:

```python
geometry_n = generate_geometry(iteration_n)
geometry_n1 = generate_geometry(iteration_n+1)

# Interpolate vertices
for v_n, v_n1 in zip(geometry_n, geometry_n1):
    v_interp = v_n + alpha * (v_n1 - v_n)
```

### Temporal Coherence (AI Video)

Each frame uses previous frame as reference:

```python
# Frame 0: Full synthesis from depth map
frame_0 = generate_from_depth(depth_0)

# Frame 1: Blend with frame 0
frame_1_new = generate_from_depth(depth_1)
frame_1 = blend(frame_0, frame_1_new, strength=0.15)
```

Lower strength = more coherence, less variation.

---

## Troubleshooting

### Issue: "No video encoding backend available"

**Solution**: Install OpenCV or FFmpeg

```bash
pip install opencv-python
# OR
# Install FFmpeg: https://ffmpeg.org/download.html
```

### Issue: AI video generation fails (Out of Memory)

**Solution 1**: Reduce resolution

```bash
--resolution 768  # or 512
```

**Solution 2**: Generate maps only, then AI synthesis separately

```bash
# Step 1: Generate maps
python chronos.py --file tree.l --iter 4 --maps-only --output maps/

# Step 2: AI synthesis on select frames
python factory.py --input maps/depth_00000.png --style prehistoric
```

### Issue: Slow frame generation

**Solution**: Use keyframes for preview

```bash
python chronos.py --file tree.l --iter 5 --keyframes --output preview/
```

### Issue: "Diffusers library not available"

**Solution**: Install Phase 3 dependencies

```bash
pip install diffusers transformers accelerate
```

Or generate maps only (no AI):

```bash
python chronos.py --file tree.l --iter 4 --maps-only --output maps/
```

---

## Advanced Usage

### Custom Growth Functions

Modify `timed_symbol.py` to add custom growth curves:

```python
def custom_growth(age, terminal_age):
    t = age / terminal_age
    # Custom curve here
    return custom_function(t)
```

### Batch Variants

Generate multiple animation variants:

```python
from animation_generator import BatchAnimationGenerator

batch = BatchAnimationGenerator(rewriter)
variants = batch.create_duration_variants([5, 10, 15, 20])
batch.export_all_variants('variants/', max_iterations=5)
```

### Integration with Blender

1. Export OBJ sequence
2. Import to Blender using Python script:

```python
import bpy

for i in range(300):
    bpy.ops.import_scene.obj(filepath=f"frame_{i:05d}.obj")
    # Render frame
```

---

## Future Enhancements

### Planned Features
- [ ] AnimateDiff integration for better temporal coherence
- [ ] Stable Video Diffusion support
- [ ] Interactive preview window
- [ ] Real-time animation scrubbing
- [ ] Camera path animation
- [ ] Time-varying L-system parameters
- [ ] Sound synthesis from growth patterns

### Phase 2 Integration
- [ ] Animate tropism forces
- [ ] Time-varying light direction
- [ ] Seasonal growth cycles
- [ ] Resource flow visualization

---

## Examples Directory

```
examples/timed/
├── growing_branch.l          # Simple 2D branching
├── simple_tree_anim.l        # 3D tree growth
├── spiral_growth.l           # Helical pattern
└── parametric_tree_anim.l    # Parametric growth
```

---

## API Reference

### AnimationGenerator

```python
from animation_generator import AnimationGenerator

gen = AnimationGenerator(
    rewriter,               # L-system rewriter
    duration=10.0,         # Animation duration (seconds)
    fps=30.0,              # Frames per second
    default_terminal_age=1.0
)

# Generate iterations
gen.generate_timed_iterations(max_iterations=5, verbose=True)

# Generate frame trajectory
trajectory = gen.generate_frame_trajectory(verbose=True)

# Export OBJ sequence
gen.export_trajectory_to_obj_sequence('output/', verbose=True)
```

### VideoFactory

```python
from video_factory import VideoFactory

factory = VideoFactory(
    animation_gen,           # AnimationGenerator instance
    resolution=1024,
    use_diffusion=True
)

# Generate depth maps
factory.generate_map_sequence('maps/', camera_preset='iso')

# Generate AI video
factory.generate_video_with_ai(
    'output.mp4',
    preset='prehistoric',
    temporal_strength=0.15,
    seed=42
)
```

---

## Citation

```bibtex
@software{chronos_lsystem_2026,
  title = {Chronos: L-System Growth Animation Engine},
  author = {L-System Project Contributors},
  year = {2026},
  note = {Phase 4: Timed DOL-systems with temporal coherence}
}
```

---

## References

1. **The Algorithmic Beauty of Plants**
   - Prusinkiewicz & Lindenmayer (1990)
   - Chapter 1, Section 1.10.1: Timed DOL-systems

2. **Temporal Coherence in AI Video**
   - AnimateDiff (2023)
   - Stable Video Diffusion (2023)

3. **Growth Animation**
   - Smooth interpolation techniques
   - Easing functions (Robert Penner)

---

## License

Educational/Research implementation based on "The Algorithmic Beauty of Plants"

---

**Phase 4 Complete: Smooth, Temporally Coherent L-System Growth Animations**

From static rules → dynamic growth → animated emergence
