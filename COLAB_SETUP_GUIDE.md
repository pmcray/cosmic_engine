# Cosmic Engine - Local + Colab Hybrid Workflow

## Problem Solved

The "No geometry" issue in Phase 2 metabolic simulations has been **fixed**. The root cause was a parser bug in `lexer.py` where conditional rules with comparison operators (`>`, `<`, `>=`, `<=`) were being misidentified as context-sensitive rules.

### What Was Fixed

**File**: `lexer.py:151-198`

**Issue**: The parser checked for `<` or `>` characters in the entire rule string (including conditions like `r > 0.5`), causing conditional rules to be rejected as malformed context-sensitive rules.

**Solution**: Modified the parser to separate conditions from context checks before evaluating rule types.

### Verification

All metabolic examples now work correctly:

```bash
# Metabolic tree (resource-limited growth)
python biosim.py --file examples/metabolic_tree.l --iter 3 --out tree.obj --verbose
# Result: 52 vertices, 26 edges, 21.80 m ✓

# Alien structure (negative phototropism)
python biosim.py --file examples/alien_structure.l --iter 4 --out alien.obj --verbose
# Result: 108 vertices, 54 edges, 30.33 m ✓

# Prehistoric mega-flora (thick trunks)
python biosim.py --file examples/prehistoric_mega_flora.l --iter 4 --out prehistoric.obj --verbose
# Result: 90 vertices, 45 edges, 84.61 m ✓
```

---

## Hybrid Workflow Architecture

### Local Machine (Linux)

**Runs**: Phases 1, 2, 4 (CPU-only)

**Capabilities**:
- L-system generation (Phase 1)
- Metabolic simulation (Phase 2)
- Tropism and resource allocation
- Growth animations (Phase 4)

**Advantages**:
- Iterate quickly on L-system rules
- Generate hundreds of candidate structures
- Filter by geometry metrics (height, trunk diameter, branch count)
- No GPU required

### Google Colab (Cloud GPU)

**Runs**: Phase 3 (GPU required)

**Capabilities**:
- AI image synthesis with Stable Diffusion XL
- ControlNet-guided rendering
- Batch processing multiple structures
- 12GB+ VRAM for high-resolution outputs

**Advantages**:
- Free GPU access (T4, 16GB)
- No local CUDA installation needed
- Process selected structures at high quality

---

## Workflow Steps

### Step 1: Local Generation (Phases 1-2)

Generate candidate structures on your Linux machine:

```bash
# Example: Generate 10 alien structures with varying parameters
for i in {1..10}; do
    # Modify pipe_exponent programmatically or use different .l files
    python biosim.py \
        --file examples/alien_structure.l \
        --iter 5 \
        --out output/alien_candidate_$i.obj \
        --shadows
done
```

**Tip**: Create variations by editing `.l` files:
- `pipe_exponent`: 1.5-3.0 (controls trunk thickness)
- `phototropism`: -0.5 to 0.5 (positive = grows toward light, negative = away)
- `gravity`: 0.0-0.5 (gravitropic response strength)

### Step 2: Filter Candidates Locally

Use `biosim.py --stats` to analyze geometry:

```bash
python biosim.py --file examples/alien_structure.l --iter 5 --stats | grep "Total length"
```

Select the best 10-20 structures based on:
- Visual inspection (OBJ viewers: Blender, MeshLab, Online OBJ Viewer)
- Metrics: trunk diameter, total height, branching density

### Step 3: Upload to Colab (Phase 3)

1. Open `cosmic_engine_colab.ipynb` in Google Colab
2. Enable GPU: Runtime > Change runtime type > GPU
3. Upload selected `.obj` files
4. Run AI synthesis with preset styles:
   - `alien`: Biomechanical, crystalline, otherworldly
   - `prehistoric`: Carboniferous, mega-flora, ancient
   - `bioluminescent`: Deep-sea, glowing, mysterious

### Step 4: Download Results

Download rendered images from Colab:
- Individual downloads via `files.download()`
- Batch download as `.zip`
- Save directly to Google Drive

---

## Colab Notebook Features

### Included in `cosmic_engine_colab.ipynb`:

1. **GPU Setup & Verification**
   - Check CUDA availability
   - Monitor GPU memory usage

2. **Dependency Installation**
   - PyTorch with CUDA 11.8
   - Diffusers, Transformers, Accelerate
   - xformers (memory-efficient attention)

3. **Phase 3: The Factory**
   - Map generation (depth, normals, curvature)
   - AI image synthesis (SDXL + ControlNet)
   - Custom prompts for novel-specific aesthetics

4. **Batch Processing**
   - Process multiple structures with different styles
   - Automated iteration through `.obj` files

5. **Memory Optimization**
   - Handle OOM errors
   - Resolution/precision tradeoffs

---

## Novel Trilogy Recommendations

### Alien Structures (Book 1?)

**Characteristics**:
- **Negative phototropism** (`phototropism = -0.4`)
- **High pipe exponent** (`pipe_exponent = 2.8`)
- Spindly, top-heavy, unsettling geometry

**L-System Tips**:
```
phototropism = -0.4  # Grows AWAY from light
pipe_exponent = 2.8  # Very thin trunks, thick branches
gravity = 0.1        # Weak gravitropic response
```

**AI Styles**: `alien`, `biomechanical`, `crystalline`, `eldritch`

### Prehistoric Mega-Flora (Book 2?)

**Characteristics**:
- **Low pipe exponent** (`pipe_exponent = 1.5`)
- **Thick trunks**, sparse branching
- Modeled after Lepidodendron, Sigillaria

**L-System Tips**:
```
pipe_exponent = 1.5  # Unnaturally thick trunks
phototropism = 0.1   # Weak phototropism
gravity = 0.15       # Moderate gravitropism
```

**AI Styles**: `prehistoric`, `carboniferous`, `swamp`, `primordial`

### Post-Historic Entities (Book 3?)

**Characteristics**:
- Combine elements from Books 1 & 2
- Hybrid geometries (thick trunks + alien branching)
- Resource competition scenarios (`competitive_growth.l`)

**L-System Tips**:
```python
# Use examples/competitive_growth.l for multi-organism scenarios
python biosim.py --file examples/competitive_growth.l --iter 8 --shadows
```

**AI Styles**: `bioluminescent`, `fungal`, `hybrid`, custom prompts

---

## Advanced Techniques

### 1. Parametric Exploration

Generate a grid of structures with varying parameters:

```bash
#!/bin/bash
for pipe in 1.5 2.0 2.5 3.0; do
    for photo in -0.4 -0.2 0.0 0.2; do
        # Modify .l file programmatically
        sed "s/pipe_exponent = .*/pipe_exponent = $pipe/" examples/alien_structure.l > temp.l
        sed -i "s/phototropism = .*/phototropism = $photo/" temp.l

        python biosim.py --file temp.l --iter 5 --out output/param_${pipe}_${photo}.obj
    done
done
```

### 2. Pandas Filtering

Analyze batches of structures:

```python
import pandas as pd
import subprocess
import json

results = []
for obj_file in glob.glob("output/*.obj"):
    stats = subprocess.check_output(
        f"python biosim.py --file {obj_file} --stats --iter 0",
        shell=True
    )
    # Parse stats and add to dataframe
    results.append({...})

df = pd.DataFrame(results)
df.sort_values('total_length', ascending=False).head(10)  # Top 10 tallest
```

### 3. Custom Prompts on Colab

For specific novel scenes:

```python
# In Colab notebook
CUSTOM_PROMPT = """
Ancient bioluminescent forest, Carboniferous period,
towering Lepidodendron trees with bark texture like dragon scales,
volumetric fog, cinematic lighting, photorealistic, 8K
"""

!python factory.py \
    --input prehistoric.obj \
    --prompt "{CUSTOM_PROMPT}" \
    --output custom_renders/ \
    --resolution 2048
```

---

## Performance Notes

### Local (Phase 1-2)
- **CPU**: Any modern processor
- **RAM**: 8GB+ recommended
- **Time**: Seconds to minutes per structure
- **Bottleneck**: Iteration count (exponential growth)

### Colab (Phase 3)
- **GPU**: T4 (16GB), P100, V100, or A100
- **VRAM**: 12GB minimum for SDXL
- **Time**: 20-60 seconds per image (1024px)
- **Bottleneck**: Model download (first run), VRAM for high resolutions

### Optimization Tips

**Local**:
- Use `numba` JIT compilation for rewriter (10x speedup)
- Limit iterations (5-7 is usually sufficient)
- Disable `--shadows` if not using phototropism

**Colab**:
- Use xformers for memory-efficient attention
- Lower resolution if OOM (768px vs 1024px)
- Batch process to amortize model loading

---

## Troubleshooting

### Local Issues

**"No geometry" error** (FIXED):
- Ensure you're using the updated `lexer.py`
- Verify rules are being parsed: `python lsystem.py --file examples/metabolic_tree.l --print-string`

**Slow simulation**:
- Reduce `--voxel-res` (default 1.0, try 2.0)
- Disable `--shadows` if not needed
- Lower iteration count

### Colab Issues

**OOM (Out of Memory)**:
```bash
# Lower resolution
--resolution 768

# Use FP32 instead of FP16
--no-fp16

# Clear GPU cache between runs
import torch; torch.cuda.empty_cache()
```

**Slow downloads**:
- SDXL models are ~7GB, first run will be slow
- Use Colab Pro for faster downloads
- Models cache in `~/.cache/huggingface/`

**Session timeouts**:
- Free tier: 12 hour max
- Save outputs to Google Drive frequently
- Use Colab Pro for extended sessions

---

## File Organization

```
/home/pmc/abop/
├── examples/               # L-system definitions
│   ├── alien_structure.l
│   ├── prehistoric_mega_flora.l
│   ├── metabolic_tree.l
│   └── competitive_growth.l
├── output/                 # Local geometry output
│   ├── alien_candidate_*.obj
│   └── prehistoric_candidate_*.obj
├── cosmic_engine_colab.ipynb   # ← Colab notebook (upload this)
└── COLAB_SETUP_GUIDE.md    # ← This file
```

**Colab structure** (after setup):
```
/content/
├── input_geometries/       # Upload .obj files here
├── output_gallery/         # AI-rendered images
│   ├── alien_renders/
│   ├── prehistoric_renders/
│   └── custom_renders/
└── cosmic_engine_colab.ipynb
```

---

## Next Steps

1. ✓ **Fixed**: Parser bug resolved
2. ✓ **Verified**: All metabolic examples work
3. ✓ **Created**: Colab notebook for Phase 3

### Immediate Actions:

1. **Generate Structures Locally**:
   ```bash
   python biosim.py --file examples/alien_structure.l --iter 5 --out alien.obj --shadows --verbose
   python biosim.py --file examples/prehistoric_mega_flora.l --iter 5 --out prehistoric.obj --shadows --verbose
   ```

2. **Upload Notebook to Colab**:
   - Open https://colab.research.google.com
   - Upload `cosmic_engine_colab.ipynb`
   - Runtime > Change runtime type > GPU

3. **Run Phase 3 Rendering**:
   - Upload `.obj` files
   - Execute cells sequentially
   - Experiment with styles and prompts

### Long-term Strategy:

1. **Iterate Locally** (fast, CPU-only):
   - Generate 50-100 structures with parameter variations
   - Filter by metrics and visual inspection
   - Identify "uncanny" geometries for your trilogy

2. **Render on Colab** (slow, GPU-required):
   - Select best 10-20 structures
   - Generate 2-4 variations per structure
   - Fine-tune prompts for novel aesthetic

3. **Integrate into Writing**:
   - Use rendered images as visual references
   - Export high-res versions for cover art
   - Share L-system parameters in appendix (computational worldbuilding!)

---

## Resources

- **Phase 1 Docs**: `README.md`
- **Phase 2 Docs**: `BIOSIM_README.md`
- **Phase 3 Docs**: `FACTORY_README.md`
- **Phase 4 Docs**: `CHRONOS_README.md`
- **Installation**: `INSTALL.md`
- **Quickstart**: `QUICKSTART.md`

---

**Gemini's Assessment**: Validated ✓
**Parser Bug**: Fixed ✓
**Colab Notebook**: Ready ✓

Happy generating! 🌱→🌳→🖼️→📚
