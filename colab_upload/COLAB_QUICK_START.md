# Colab Quick Start Guide

## Files Ready for Upload

You have successfully generated two structures locally:

1. **`alien.obj`** (3.6 KB)
   - 108 vertices, 54 edges
   - 8.52m tall, spindly structure
   - Negative phototropism (-0.4)
   - High pipe exponent (2.8)

2. **`prehistoric.obj`** (5.7 KB)
   - 150 vertices, 75 edges
   - 38.82m tall, massive structure
   - Positive phototropism (+0.1)
   - Low pipe exponent (1.5)

---

## Step-by-Step Colab Workflow

### Step 1: Upload Notebook

1. Go to https://colab.research.google.com
2. Click **File > Upload notebook**
3. Upload `cosmic_engine_colab.ipynb`

### Step 2: Enable GPU

1. Click **Runtime > Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### Step 3: Verify GPU

Run the first cell:
```python
!nvidia-smi
```

You should see something like:
- Tesla T4 (16GB) - Free tier
- P100 (16GB) - Colab Pro
- V100 (16GB) or A100 (40GB) - Colab Pro+

### Step 4: Install Dependencies

Run the installation cells. This will take ~5 minutes:
- PyTorch with CUDA
- Diffusers, Transformers, Accelerate
- SDXL model download (~7GB, cached for future use)

### Step 5: Upload Your Project

**Option A - Upload Files Directly** (Recommended for first test):

```python
from google.colab import files

# Create directories
!mkdir -p input_geometries output_gallery

# Upload .obj files
uploaded = files.upload()

# Move to input directory
!mv *.obj input_geometries/
!ls -lh input_geometries/
```

Upload:
- `alien.obj`
- `prehistoric.obj`
- Python files from your project (if not using git)

**Option B - Use Google Drive**:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy your files
!cp /content/drive/MyDrive/cosmic_engine/*.obj input_geometries/
```

### Step 6: Test Map Generation (Fast, No GPU)

Start with map generation to verify everything works:

```python
!python factory.py --input input_geometries/alien.obj --maps-only --output output_gallery/alien_maps
```

This generates:
- Depth map
- Normal map
- Curvature map
- Edge detection map

### Step 7: AI Rendering (GPU Required)

Now for the main event - Stable Diffusion XL:

**Alien Structure**:
```python
!python factory.py \
    --input input_geometries/alien.obj \
    --style alien \
    --output output_gallery/alien_renders \
    --resolution 1024 \
    --samples 4
```

**Prehistoric Mega-Flora**:
```python
!python factory.py \
    --input input_geometries/prehistoric.obj \
    --style prehistoric \
    --output output_gallery/prehistoric_renders \
    --resolution 1024 \
    --samples 4
```

Expected time: ~30-60 seconds per image @ 1024px

### Step 8: View Results

```python
from IPython.display import Image, display
import os

for img in sorted(os.listdir('output_gallery/alien_renders')):
    if img.endswith('.png') and 'render' in img:
        display(Image(f'output_gallery/alien_renders/{img}', width=600))
```

### Step 9: Download Results

**Single file**:
```python
from google.colab import files
files.download('output_gallery/alien_renders/render_001.png')
```

**Batch download**:
```python
!zip -r cosmic_renders.zip output_gallery/
files.download('cosmic_renders.zip')
```

**Save to Google Drive**:
```python
!cp -r output_gallery /content/drive/MyDrive/cosmic_engine_outputs/
```

---

## Available Styles

The Factory comes with preset styles optimized for different aesthetics:

### For Alien Structure:
- **`alien`**: Biomechanical, crystalline, otherworldly textures
- **`bioluminescent`**: Glowing, deep-sea inspired
- **`crystalline`**: Mineral formations, geometric
- **`eldritch`**: Lovecraftian, incomprehensible geometry

### For Prehistoric Mega-Flora:
- **`prehistoric`**: Carboniferous period, ancient vegetation
- **`carboniferous`**: Swamp forests, scale trees
- **`primordial`**: Pre-Cambrian, alien-but-earthly
- **`swamp`**: Humid, dense vegetation

### Custom Prompts:

```python
CUSTOM_PROMPT = "bioluminescent crystalline trees growing in zero gravity, cinematic lighting, 8K, photorealistic"

!python factory.py \
    --input input_geometries/alien.obj \
    --prompt "{CUSTOM_PROMPT}" \
    --output output_gallery/custom \
    --resolution 1024 \
    --samples 2
```

---

## Troubleshooting

### "Out of Memory" Errors

**Solution 1** - Lower resolution:
```python
--resolution 768  # Instead of 1024
```

**Solution 2** - Use FP32:
```python
--no-fp16
```

**Solution 3** - Clear GPU memory:
```python
import torch
torch.cuda.empty_cache()
!nvidia-smi  # Check memory usage
```

### Slow First Run

The first run downloads SDXL models (~7GB). Subsequent runs will be much faster as models are cached.

### Session Timeout

Free Colab sessions timeout after:
- 12 hours of runtime
- 90 minutes of inactivity

**Solution**: Save outputs to Google Drive frequently.

---

## Batch Processing Script

Process both structures with multiple styles:

```python
structures = [
    ('alien.obj', ['alien', 'bioluminescent', 'crystalline']),
    ('prehistoric.obj', ['prehistoric', 'carboniferous', 'swamp'])
]

for obj_file, styles in structures:
    for style in styles:
        print(f"\n{'='*60}")
        print(f"Processing: {obj_file} - Style: {style}")
        print('='*60)

        !python factory.py \
            --input input_geometries/{obj_file} \
            --style {style} \
            --output output_gallery/{obj_file.replace('.obj', '')}_{style} \
            --resolution 1024 \
            --samples 2

        # Clear memory between runs
        import torch
        torch.cuda.empty_cache()

print("\n✓ Batch processing complete!")
```

---

## Expected Results

### Alien Structure
- 4 images per style
- Resolution: 1024x1024
- Style examples:
  - Translucent crystalline forms
  - Bioluminescent nodes
  - Biomechanical joints
  - Irregular surface textures

### Prehistoric Mega-Flora
- 4 images per style
- Resolution: 1024x1024
- Style examples:
  - Bark with diamond patterns (Lepidodendron)
  - Thick, columnar trunks
  - Sparse crown branching
  - Ancient forest atmosphere

---

## File Organization in Colab

After running everything:

```
/content/
├── input_geometries/
│   ├── alien.obj
│   └── prehistoric.obj
├── output_gallery/
│   ├── alien_maps/
│   │   ├── depth_map.png
│   │   ├── normal_map.png
│   │   └── curvature_map.png
│   ├── alien_renders/
│   │   ├── render_001.png
│   │   ├── render_002.png
│   │   ├── render_003.png
│   │   └── render_004.png
│   └── prehistoric_renders/
│       ├── render_001.png
│       ├── render_002.png
│       ├── render_003.png
│       └── render_004.png
└── cosmic_engine_colab.ipynb
```

---

## Next Steps After Colab

Once you have rendered images:

1. **Evaluate aesthetics**: Which style captures your novel's vision?
2. **Iterate locally**: Generate more structures with varied parameters
3. **Batch render**: Upload 10-20 structures for final selection
4. **High-res renders**: Use `--resolution 2048` for cover art quality

---

## Pro Tips

1. **Start small**: Test with 1-2 images before batch processing
2. **Monitor GPU**: Run `!nvidia-smi` between operations
3. **Save frequently**: Copy to Google Drive after each successful render
4. **Experiment with prompts**: Custom prompts often yield the most unique results
5. **Use Colab Pro**: If doing serious work, $10/month gets you:
   - Priority GPU access
   - Longer sessions
   - More memory
   - Faster GPUs (P100/V100)

---

## Ready to Upload

Your files are ready:
- ✓ `cosmic_engine_colab.ipynb` (Colab notebook)
- ✓ `alien.obj` (alien structure)
- ✓ `prehistoric.obj` (prehistoric mega-flora)
- ✓ All Python files in `/home/pmc/abop/`

**Next action**: Open https://colab.research.google.com and upload the notebook!

---

*Generated for the Cosmic Engine project - Phase 3: The Factory*
