# Installation Guide

## Quick Start (All Phases)

```bash
# Clone or navigate to project directory
cd /home/pmc/abop

# Install all dependencies
pip install -r requirements.txt
```

## Phase-by-Phase Installation

### Phase 1: L-System Core (Required)

Minimum dependencies for basic L-system generation:

```bash
pip install numpy>=1.24.0 numba>=0.57.0 pandas>=2.0.0
```

**Test:**
```bash
python lsystem.py --file examples/simple_tree.l --iter 4 --out test.obj
```

---

### Phase 2: Bio-Sim Module (Optional)

Additional dependencies for environmental simulation:

```bash
pip install torch>=2.0.0 matplotlib>=3.7.0
```

**Test:**
```bash
python biosim.py --file examples/tropism_tree.l --iter 3 --verbose
```

**Note:** PyTorch will use CPU if no GPU available. For GPU support:

```bash
# CUDA 11.8 (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Phase 3: The Factory (Optional)

Additional dependencies for AI image synthesis:

```bash
pip install Pillow>=10.0.0 opencv-python>=4.8.0
pip install diffusers>=0.21.0 transformers>=4.30.0 accelerate>=0.20.0
```

**Test Map Generation (No GPU):**
```bash
python factory.py --list-styles
python factory.py --input output/simple_tree.obj --maps-only --output test_maps/
```

**Test AI Generation (Requires GPU):**
```bash
python factory.py --system-info  # Check GPU availability
python factory.py --input output/simple_tree.obj --style prehistoric --output gallery/
```

---

## System Requirements

### Minimum (Phases 1-2)
- Python 3.8+
- 4GB RAM
- Any CPU
- 1GB storage

### Recommended (All Phases)
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 12GB+ VRAM
- 30GB storage (for SDXL models)

---

## Platform-Specific Notes

### Linux (Recommended)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Create virtual environment
python3 -m venv abop-env
source abop-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### macOS

```bash
# Install Python via Homebrew
brew install python@3.10

# Create virtual environment
python3 -m venv abop-env
source abop-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Phase 3 AI generation requires NVIDIA GPU (not available on macOS M1/M2). Map generation works fine.

### Windows

```powershell
# Create virtual environment
python -m venv abop-env
.\abop-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** For GPU support, install CUDA Toolkit first: https://developer.nvidia.com/cuda-downloads

---

## Troubleshooting

### Import Errors

```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM)

For Phase 3 image generation:

```bash
# Use lower resolution
python factory.py --input tree.obj --style alien --resolution 768

# Use FP32 instead of FP16 (paradoxically can help with some GPUs)
python factory.py --input tree.obj --style alien --no-fp16
```

### Diffusers Download Issues

Models download to: `~/.cache/huggingface/`

```bash
# Check disk space
df -h ~/.cache

# Manual download from https://huggingface.co if needed
```

---

## Verification

After installation, run all tests:

```bash
make test           # Phase 1
make test-biosim    # Phase 2
make test-factory   # Phase 3 (map generation only)
```

Expected output:
- Phase 1: OBJ files in `output/`
- Phase 2: Statistics printed, OBJ in `output/biosim/`
- Phase 3: Maps in `factory_output/test/`

---

## Selective Installation

### Minimal Install (Phase 1 Only)

```bash
pip install numpy numba pandas
```

**Use Case:** Generate L-system geometry only, no simulation or AI.

### Simulation Install (Phases 1-2)

```bash
pip install numpy numba pandas torch matplotlib
```

**Use Case:** Generate and simulate, but no AI rendering.

### Full Install (All Phases)

```bash
pip install -r requirements.txt
```

**Use Case:** Complete pipeline including AI synthesis.

---

## Optional Dependencies

### For Better Performance

```bash
# xformers for memory-efficient attention (CUDA only)
pip install xformers

# Numba for JIT compilation
pip install numba
```

### For Visualization

```bash
# Jupyter for interactive notebooks
pip install jupyter

# PyVista for 3D visualization
pip install pyvista
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf abop-env/

# Or just remove packages
pip uninstall -r requirements.txt -y
```

---

## Getting Help

1. Check documentation:
   - `README.md` - Main documentation
   - `BIOSIM_README.md` - Phase 2
   - `FACTORY_README.md` - Phase 3

2. Run with `--verbose` for detailed output:
   ```bash
   python lsystem.py --file examples/tree.l --iter 3 --verbose
   ```

3. Check system info:
   ```bash
   python factory.py --system-info
   ```

---

## Next Steps

After successful installation:

1. **Run examples:**
   ```bash
   make examples
   ```

2. **Read quickstart:**
   ```bash
   cat QUICKSTART.md
   ```

3. **Try each phase:**
   ```bash
   # Phase 1
   python lsystem.py --help

   # Phase 2
   python biosim.py --help

   # Phase 3
   python factory.py --help
   ```

Happy generating! 🌱→🌳→🖼️
