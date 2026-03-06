# Colab Upload Checklist

## ✓ Files Ready for Upload

### Essential Files (Upload these to Colab):

1. **Colab Notebook** ✓
   - `cosmic_engine_colab.ipynb` (12 KB)
   - This is your main interface

2. **Geometry Files** ✓
   - `alien.obj` (3.6 KB) - Alien structure
   - `prehistoric.obj` (5.2 KB) - Prehistoric mega-flora

3. **Python Source Files** (needed for factory.py):
   - All `.py` files in `/home/pmc/abop/`
   - Or you can upload your entire project as a zip

### Documentation (Optional, for reference):
   - `COLAB_QUICK_START.md` - Step-by-step guide
   - `COLAB_SETUP_GUIDE.md` - Comprehensive documentation
   - `alien_structure_vis.png` - Local visualization
   - `prehistoric_vis.png` - Local visualization

---

## Quick Upload Instructions

### Option A: Upload Individual Files (Easiest for first test)

1. Go to https://colab.research.google.com
2. Click **File > Upload notebook**
3. Upload `cosmic_engine_colab.ipynb`
4. In the notebook, run the upload cell:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
5. Select and upload:
   - `alien.obj`
   - `prehistoric.obj`
   - All `.py` files

### Option B: Upload via Google Drive (Better for repeated use)

1. Upload entire `/home/pmc/abop/` folder to Google Drive
2. In Colab, mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Copy files:
   ```python
   !cp -r /content/drive/MyDrive/abop/* /content/
   ```

### Option C: Clone from Git (Best if you have a repo)

```python
!git clone YOUR_REPO_URL cosmic_engine
%cd cosmic_engine
```

---

## First Test Sequence

Once uploaded, run these cells in order:

### 1. Check GPU
```python
!nvidia-smi
```
**Expected**: Tesla T4 (16GB) or better

### 2. Install Dependencies (~5 minutes)
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers transformers accelerate xformers
```

### 3. Verify Python Files
```python
!ls -lh *.py
!python factory.py --list-styles
```
**Expected**: List of available styles (alien, prehistoric, etc.)

### 4. Test with Alien Structure (Maps Only, ~10 seconds)
```python
!python factory.py \
    --input alien.obj \
    --maps-only \
    --output test_output
```
**Expected**: 4 map images generated

### 5. View Test Results
```python
from IPython.display import Image, display
import os

for img in os.listdir('test_output'):
    if img.endswith('.png'):
        print(f"\n{img}:")
        display(Image(f'test_output/{img}', width=400))
```

### 6. Full AI Render Test (~30 seconds)
```python
!python factory.py \
    --input alien.obj \
    --style alien \
    --output alien_test \
    --resolution 1024 \
    --samples 1
```
**Expected**: 1 photorealistic AI-rendered image

### 7. View AI Render
```python
display(Image('alien_test/render_001.png', width=800))
```

**If this works**, you're ready to batch process! 🎉

---

## Troubleshooting Upload Issues

### "No such file or directory: factory.py"

**Problem**: Python files not uploaded

**Solution**:
```python
# Check current directory
!ls -lh

# If files missing, upload them
from google.colab import files
uploaded = files.upload()
```

### "CUDA out of memory"

**Problem**: GPU doesn't have enough VRAM

**Solution**:
```python
# Lower resolution
!python factory.py --input alien.obj --style alien --resolution 768

# Or clear GPU memory
import torch
torch.cuda.empty_cache()
```

### "No module named 'diffusers'"

**Problem**: Dependencies not installed

**Solution**: Re-run installation cell

---

## What to Expect

### Timeline:
- **First setup**: 10-15 minutes (includes SDXL download)
- **Map generation**: 5-10 seconds per structure
- **AI rendering**: 30-60 seconds per image @ 1024px
- **Batch processing** (2 structures × 3 styles × 2 samples): ~6 minutes

### Output Quality:
- **Resolution**: 1024×1024 (can go up to 2048×2048)
- **Style fidelity**: High (thanks to ControlNet)
- **Variety**: 4 samples per run give diverse results
- **Photorealism**: Depends on prompt quality

### GPU Memory Usage:
- **SDXL base**: ~5-6 GB VRAM
- **With ControlNet**: ~8-10 GB VRAM
- **T4 (16GB)**: Comfortable headroom
- **P100 (16GB)**: Same as T4
- **V100 (16GB)**: Slightly faster

---

## Your Two Structures

### Alien Structure (alien.obj)
**Characteristics**:
- Small, spindly (1.7m × 8.5m × 2.7m)
- Asymmetric, unsettling geometry
- Negative phototropism (grows away from light)
- Top-heavy (high pipe exponent)

**Recommended styles**:
1. `alien` - Biomechanical, crystalline
2. `bioluminescent` - Glowing, deep-sea
3. `crystalline` - Mineral, geometric

**Prompt ideas**:
- "alien plant life on distant exoplanet, bioluminescent, crystalline structure"
- "biomechanical tree-like organism, translucent, otherworldly"
- "parasitic xenoflora, uncanny valley, horror aesthetic"

### Prehistoric Mega-Flora (prehistoric.obj)
**Characteristics**:
- Massive (18m × 39m × 23m)
- Thick trunk, sparse branching
- Positive phototropism (Earth-like)
- Stable, imposing structure

**Recommended styles**:
1. `prehistoric` - Carboniferous period
2. `carboniferous` - Scale trees, swamp forests
3. `primordial` - Pre-Cambrian, ancient

**Prompt ideas**:
- "Lepidodendron scale tree, Carboniferous period, humid swamp atmosphere"
- "primordial mega-flora, 100 million years ago, photorealistic"
- "ancient tree with diamond-pattern bark, towering, prehistoric forest"

---

## Success Criteria

You'll know it's working when:

✓ GPU shows "Tesla T4" or better
✓ `factory.py --list-styles` shows available styles
✓ Map generation completes in <10 seconds
✓ First AI render completes in <60 seconds
✓ Output images show your structure with applied style

---

## After First Success

Once you've confirmed everything works:

1. **Batch process both structures** with 3-4 styles each
2. **Compare results** and pick best aesthetics
3. **Fine-tune prompts** for your novel's specific needs
4. **Generate high-res versions** (2048px) of favorites
5. **Download and save** to local machine or Drive

---

## Support Resources

- **COLAB_QUICK_START.md** - This guide in detail
- **COLAB_SETUP_GUIDE.md** - Full technical documentation
- **FACTORY_README.md** - Phase 3 documentation
- **Colab notebooks** - Built-in examples and documentation

---

## Ready to Go! 🚀

Open: https://colab.research.google.com

Upload: `cosmic_engine_colab.ipynb`

Follow: First Test Sequence (above)

Expected time to first AI render: **15 minutes**

Good luck with your novel trilogy! 📚✨
