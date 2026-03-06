# The Factory - Aesthetic Synthesis Engine

**Phase 3: AI-Driven Image Synthesis from L-System Geometry**

The Factory transforms algorithmic L-system structures into high-fidelity, photorealistic imagery using Stable Diffusion and ControlNet, achieving "uncanny" prehistoric and alien aesthetics.

## Overview

Phase 3 bridges the gap between algorithmic precision (Phases 1 & 2) and visual realism through:

1. **Geometric Conditioning** - Converting 3D geometry to depth/normal/edge maps
2. **AI-Guided Synthesis** - Using ControlNet to maintain structure while adding realistic textures
3. **Aesthetic Steering** - Prompt engineering for prehistoric, alien, and bioluminescent styles
4. **Material Hallucination** - AI-generated textures (bark, chitinous surfaces, bioluminescence)

## Architecture

```
L-System Geometry (OBJ)
         ↓
   Map Generator
    ├── Depth Map
    ├── Normal Map
    └── Edge Map
         ↓
   ControlNet Pipeline
    ├── Structure Preservation (from maps)
    └── Texture Generation (from prompts)
         ↓
   Stable Diffusion SDXL
         ↓
  High-Fidelity Image
         ↓
   Optional Refinement
         ↓
    Final Output
```

## Installation

### Basic Requirements

```bash
pip install -r requirements.txt
```

### For AI Generation (Phase 3)

Requires additional dependencies and GPU:

```bash
# Install diffusers and dependencies
pip install diffusers transformers accelerate

# For optimal performance
pip install xformers  # Optional, for memory efficiency

# OpenCV for advanced edge detection
pip install opencv-python
```

**System Requirements:**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended for SDXL)
- **Storage**: ~20GB for model downloads
- **RAM**: 16GB+ recommended

## Quick Start

### 1. Generate Maps Only (No GPU Required)

```bash
# Generate depth, normal, and edge maps from L-system geometry
python factory.py --input output/3d_tree.obj --maps-only --output maps/
```

### 2. Generate AI Images

```bash
# Prehistoric mega-flora style
python factory.py --input output/3d_tree.obj --style prehistoric --output gallery/

# Alien organism with bioluminescence
python factory.py --input output/alien.obj --style alien --output gallery/

# With refinement pass for higher quality
python factory.py --input output/tree.obj --style bioluminescent --refine --output gallery/
```

### 3. Multiple Camera Angles

```bash
python factory.py --input output/tree.obj \
    --style alien_creature \
    --angles front,iso,closeup \
    --output gallery/
```

## Available Styles

Use `python factory.py --list-styles` to see all options:

### Prehistoric
- **prehistoric** - Carboniferous mega-flora with ancient bark textures
- **botanical** - Scientific illustration style

### Alien/Otherworldly
- **alien** - Extraterrestrial vegetation with bioluminescence
- **alien_creature** - Animal-like alien organism with visible internal systems
- **abyssal** - Deep sea organism adaptations
- **crystalline** - Mineral-organic hybrid structures

### Organic/Biological
- **bioluminescent** - Glowing organisms in darkness
- **fungal** - Giant mushroom structures with mycelial networks
- **parasitic** - Symbiotic/parasitic relationships

## Command-Line Options

### Input/Output
```bash
--input FILE           # Input OBJ file from L-system
--output DIR          # Output directory
--batch PATTERN       # Batch process (glob pattern)
--batch-dir DIR       # Directory with OBJ files
```

### Style & Prompts
```bash
--style NAME          # Aesthetic preset (prehistoric, alien, etc.)
--custom-prompt TEXT  # Additional prompt text
--list-styles         # Show all available styles
```

### Map Generation
```bash
--maps-only           # Only generate maps, skip AI
--angles ANGLES       # Camera angles: front,iso,top,side,closeup
--map-resolution N    # Map resolution (default: 1024)
--save-maps           # Save intermediate maps
```

### Diffusion Parameters
```bash
--resolution N        # Output image resolution
--steps N            # Inference steps (default: 50)
--guidance FLOAT     # Guidance scale (default: 7.5)
--controlnet-scale F # ControlNet strength 0-1 (default: 0.8)
--seed N             # Random seed for reproducibility
```

### Post-Processing
```bash
--refine              # Apply img2img refinement pass
--refinement-strength # Refinement strength 0-1 (default: 0.25)
--upscale FACTOR     # Upscale by factor (e.g., 2.0)
```

### Model Selection
```bash
--model ID           # Stable Diffusion model ID
--controlnet ID      # ControlNet model ID
--device DEVICE      # cuda, cpu, or auto
--no-fp16            # Disable FP16 (use FP32)
```

### Output Options
```bash
--save-config        # Save generation config as JSON
--prefix PREFIX      # Output filename prefix
--verbose           # Verbose output
```

## Detailed Examples

### Example 1: Prehistoric Lepidodendron

Generate a prehistoric scale-tree with thick, ancient bark:

```bash
# First: Generate geometry with Bio-Sim
python biosim.py \
    --file examples/prehistoric_mega_flora.l \
    --iter 4 \
    --shadows \
    --out prehistoric_tree.obj

# Then: Synthesize image
python factory.py \
    --input prehistoric_tree.obj \
    --style prehistoric \
    --angles iso,closeup \
    --refine \
    --resolution 2048 \
    --output gallery/prehistoric/
```

### Example 2: Bioluminescent Alien Flora

Create glowing alien vegetation:

```bash
# Generate alien structure
python biosim.py \
    --file examples/alien_structure.l \
    --iter 4 \
    --gravity 0.1 \
    --photo -0.4 \
    --shadows \
    --out alien_plant.obj

# Synthesize with bioluminescence
python factory.py \
    --input alien_plant.obj \
    --style bioluminescent \
    --custom-prompt "pulsating glow patterns" \
    --angles iso \
    --refine \
    --seed 42 \
    --output gallery/alien/
```

### Example 3: Deep Sea Abyssal Organism

Simulate deep ocean life:

```bash
python factory.py \
    --input output/biosim/metabolic_tree.obj \
    --style abyssal \
    --custom-prompt "hydrothermal vent ecosystem" \
    --refine \
    --refinement-strength 0.3 \
    --output gallery/abyssal/
```

### Example 4: Batch Processing

Process multiple organisms:

```bash
python factory.py \
    --batch-dir output/biosim/ \
    --style alien \
    --angles iso \
    --output gallery/batch/ \
    --save-maps \
    --save-config
```

### Example 5: High-Resolution Production

Maximum quality output:

```bash
python factory.py \
    --input championship_tree.obj \
    --style crystalline \
    --resolution 2048 \
    --steps 75 \
    --guidance 8.5 \
    --refine \
    --refinement-strength 0.2 \
    --upscale 1.5 \
    --seed 12345 \
    --output final/
```

## Prompt Engineering

### Understanding Weighted Prompts

Styles use weighted keywords to control emphasis:

```
(keyword:1.2)  # 20% more emphasis
(keyword:0.8)  # 20% less emphasis
```

### Creating Custom Styles

Modify existing presets in Python:

```python
from prompt_engine import PromptEngine

engine = PromptEngine()

# Create custom variant
custom = engine.create_custom_prompt(
    base_style='alien',
    additional_keywords=[
        '(crystalline structures:1.3)',
        '(silicon-based life:1.2)'
    ],
    extra_negative=[
        'carbon-based',
        'familiar biology'
    ]
)

# Use in diffusion_client.py
```

### Blending Styles

Combine multiple aesthetics:

```python
blended = engine.blend_presets(
    ['prehistoric', 'alien'],
    weights=[0.6, 0.4]  # 60% prehistoric, 40% alien
)
```

## Technical Details

### Map Generation

**Depth Maps:**
- Rendered using ray-casting projection
- Near objects = white (255)
- Far objects = black (0)
- Used by ControlNet for structure preservation

**Normal Maps:**
- RGB encoding of surface normals
- R = X component, G = Y component, B = Z component
- Provides fine detail guidance

**Edge Maps:**
- Canny edge detection (with OpenCV)
- Or PIL edge filter (fallback)
- High contrast edges for sharp definition

### Camera Presets

- **front**: Front orthographic view
- **top**: Top-down view
- **side**: Side profile
- **iso**: Isometric 3D view (default)
- **closeup**: Closer perspective view

Camera distance automatically scaled based on model size.

### ControlNet Integration

Uses depth-based ControlNet to:
1. Lock geometry to L-system structure
2. Prevent AI from generating generic trees
3. Maintain branching topology
4. Allow texture and detail hallucination

**ControlNet Scale** (0.0 - 1.0):
- **0.8-1.0**: Strict adherence to geometry
- **0.5-0.7**: Balanced structure/creativity
- **0.3-0.5**: More artistic freedom
- **< 0.3**: May deviate from structure

### Model Requirements

**Default Models:**
- Base: `stabilityai/stable-diffusion-xl-base-1.0` (~13GB)
- ControlNet: `diffusers/controlnet-depth-sdxl-1.0` (~5GB)

**Alternative Models:**
- SD 1.5 (lower VRAM): `runwayml/stable-diffusion-v1-5`
- Community fine-tunes from HuggingFace

## Performance Optimization

### GPU Memory Management

```bash
# Use FP16 (default on CUDA)
python factory.py --input tree.obj --style alien --output gallery/

# Force FP32 if needed (more VRAM)
python factory.py --input tree.obj --style alien --no-fp16 --output gallery/
```

### Batch Size

For multiple images, process sequentially to avoid OOM:
- Pipeline unloads after each image
- Use `--batch` or `--batch-dir` for automatic handling

### Speed vs Quality Trade-offs

```bash
# Fast (20-30s per image)
--steps 25 --resolution 1024

# Balanced (40-60s per image)
--steps 50 --resolution 1024

# High Quality (90-120s per image)
--steps 75 --resolution 2048 --refine
```

## Integration with Phases 1 & 2

### Complete Workflow

```bash
# Phase 1: Generate L-system geometry
python lsystem.py \
    --file examples/3d_tree.l \
    --iter 5 \
    --out structure.obj

# Phase 2: Add environmental simulation
python biosim.py \
    --file examples/metabolic_tree.l \
    --iter 6 \
    --shadows \
    --metabolism \
    --out biosim_structure.obj

# Phase 3: Synthesize final image
python factory.py \
    --input biosim_structure.obj \
    --style prehistoric \
    --angles iso,closeup \
    --refine \
    --output final_gallery/
```

## Output Files

For each generation:

```
output/
├── tree_iso_prehistoric.png          # Main output
├── tree_iso_prehistoric_initial.png  # Pre-refinement (if --refine)
├── tree_iso_prehistoric_upscaled.png # Upscaled (if --upscale)
├── tree_iso_depth.png                # Depth map (if --save-maps)
├── tree_iso_normal.png               # Normal map (if --save-maps)
├── tree_iso_edge.png                 # Edge map (if --save-maps)
└── tree_iso_prehistoric_config.json  # Generation config (if --save-config)
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce resolution
--resolution 1024  # or 768

# Use FP16
# (default on CUDA)

# Reduce steps
--steps 30

# Process one at a time
# (automatic in batch mode)
```

### Low Quality Output

```bash
# Increase steps
--steps 75

# Add refinement
--refine --refinement-strength 0.3

# Adjust guidance
--guidance 8.5  # Higher = more prompt adherence

# Try different seed
--seed 42
```

### Structure Not Preserved

```bash
# Increase ControlNet strength
--controlnet-scale 0.9

# Check depth map quality
--save-maps  # Verify depth map looks correct

# Ensure high-contrast depth map
--map-resolution 1024  # or higher
```

### Model Download Issues

Models download automatically to:
`~/.cache/huggingface/`

If download fails:
- Check internet connection
- Check disk space (~20GB needed)
- Try manual download via HuggingFace website

## Advanced Usage

### Custom Models

```bash
# Use community fine-tuned model
python factory.py \
    --input tree.obj \
    --style alien \
    --model "some-user/custom-sdxl-model" \
    --output gallery/
```

### Python API

```python
from map_generator import MapGenerator
from diffusion_client import DiffusionClient

# Generate maps
generator = MapGenerator(1024, 1024)
maps = generator.generate_all_maps("tree.obj", "iso")

# Generate image
client = DiffusionClient()
image = client.generate_from_preset(
    maps['depth'],
    preset='prehistoric',
    seed=42
)
image.save("output.png")
```

## Known Limitations

1. **GPU Required**: AI generation requires CUDA GPU
2. **Model Size**: Initial download ~20GB
3. **Generation Time**: 30-120s per image depending on settings
4. **Memory**: 8GB+ VRAM recommended
5. **Map Quality**: Complex geometry may need manual depth map adjustment

## Future Enhancements

- [ ] Support for additional ControlNet types (Canny, Normal)
- [ ] Multiple diffusion models (SD1.5, SDXL, custom)
- [ ] Real-time preview mode
- [ ] Animation sequence generation
- [ ] Style transfer from reference images
- [ ] Automatic prompt optimization
- [ ] Integration with 3D rendering engines

## References

- **Stable Diffusion XL**: Podell et al. (2023)
- **ControlNet**: Zhang et al. (2023)
- **Diffusers Library**: HuggingFace Team
- **ABOP**: Prusinkiewicz & Lindenmayer (1990)

## Citation

If using The Factory for research or production:

```bibtex
@software{lsystem_factory_2026,
  title = {The Factory: Aesthetic Synthesis Engine for L-Systems},
  author = {L-System Core Project},
  year = {2026},
  note = {Phase 3: AI-Driven Image Synthesis}
}
```

---

**Phase 3 Status**: ✅ Complete

**Integration**: Full pipeline from L-system → Bio-Sim → AI Image Synthesis
