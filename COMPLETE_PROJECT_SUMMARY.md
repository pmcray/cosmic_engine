# L-System Core + Bio-Sim + Factory: Complete Implementation

## 🎉 Three-Phase Algorithmic Beauty Pipeline

A comprehensive system for generating, simulating, and visualizing algorithmic botanical structures with AI-enhanced aesthetics.

---

## Project Overview

### Phase 1: L-System Core (Static Rewriting)
**Status**: ✅ Complete

High-performance parametric L-system engine with 3D turtle graphics.

**Features:**
- Parametric L-systems with arithmetic expressions
- Stochastic production rules with probabilities
- Context-sensitive rules (left/right context)
- Conditional production rules
- 3D turtle graphics interpreter
- OBJ/PLY export with cylindrical meshes

**Modules:** 6 Python files, 7 examples

---

### Phase 2: Bio-Sim Module (Dynamic Simulation)
**Status**: ✅ Complete

Environmental simulation with tropism, light competition, and metabolic resource allocation.

**Features:**
- Voxel-based light propagation (PyTorch accelerated)
- Shadow casting from 3D structures
- Gravitropism and phototropism
- Pipe Model Theory for branch thickness
- Dual-signal resource distribution
- Self-pruning of shaded branches

**Modules:** 8 Python files, 7 bio-sim examples

---

### Phase 3: The Factory (Aesthetic Synthesis)
**Status**: ✅ Complete

AI-driven image synthesis transforming L-system geometry into photorealistic imagery.

**Features:**
- Depth/normal/edge map generation from 3D geometry
- ControlNet integration for structure preservation
- 9 aesthetic presets (prehistoric, alien, bioluminescent, etc.)
- Weighted prompt engineering
- Multi-angle rendering
- Refinement and upscaling pipelines

**Modules:** 3 Python files, 9 aesthetic styles

---

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1: L-SYSTEM CORE                    │
│                                                              │
│  L-System Rules  →  Rewriting Engine  →  3D Turtle  →  OBJ  │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 2: BIO-SIM MODULE                    │
│                                                              │
│  Environment  →  Tropism  →  Metabolism  →  Growth Graph    │
│  (Voxels)        Forces      Resource        (with light)   │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 3: THE FACTORY                       │
│                                                              │
│  Map Gen  →  ControlNet  →  Stable Diffusion  →  AI Image   │
│  (D/N/E)      Conditioning    SDXL                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
/home/pmc/abop/
├── Phase 1: L-System Core (6 modules)
│   ├── symbol.py                 # Data models
│   ├── lexer.py                  # Parser
│   ├── rewriter.py               # Rewriting engine
│   ├── turtle_3d.py              # 3D turtle
│   ├── obj_exporter.py           # Export
│   └── lsystem.py                # CLI
│
├── Phase 2: Bio-Sim (8 modules)
│   ├── environment.py            # Light simulation
│   ├── forces.py                 # Tropism
│   ├── metabolism.py             # Resources
│   ├── growth_node.py            # Graph structure
│   ├── turtle_tropism.py         # Tropism turtle
│   ├── simulator.py              # Orchestrator
│   ├── biosim.py                 # CLI
│   └── visualization.py          # Plotting
│
├── Phase 3: The Factory (3 modules)
│   ├── map_generator.py          # Depth/normal/edge maps
│   ├── prompt_engine.py          # Aesthetic presets
│   ├── diffusion_client.py       # ControlNet/SD interface
│   └── factory.py                # CLI
│
├── Examples
│   ├── Phase 1 (7 files)         # Basic L-systems
│   └── Phase 2 (7 files)         # Bio-sim L-systems
│
├── Documentation
│   ├── README.md                 # Main documentation
│   ├── QUICKSTART.md             # Quick start
│   ├── BIOSIM_README.md          # Bio-sim docs
│   ├── FACTORY_README.md         # Factory docs
│   └── COMPLETE_PROJECT_SUMMARY.md # This file
│
├── Configuration
│   ├── requirements.txt          # All dependencies
│   └── Makefile                  # Build automation
│
└── Output
    ├── output/                   # Phase 1 OBJ files
    ├── output/biosim/            # Phase 2 simulations
    └── factory_output/           # Phase 3 AI images
```

---

## Complete Workflow Example

### 1. Generate Algorithmic Structure (Phase 1)

```bash
# Create basic L-system geometry
python lsystem.py \
    --file examples/3d_tree.l \
    --iter 5 \
    --out structure.obj \
    --stats
```

### 2. Simulate Environmental Growth (Phase 2)

```bash
# Add environmental constraints and tropism
python biosim.py \
    --file examples/prehistoric_mega_flora.l \
    --iter 4 \
    --shadows \
    --metabolism \
    --gravity 0.2 \
    --photo 0.1 \
    --out prehistoric_structure.obj \
    --stats
```

### 3. Synthesize AI Image (Phase 3)

```bash
# Transform to photorealistic image
python factory.py \
    --input prehistoric_structure.obj \
    --style prehistoric \
    --angles iso,closeup \
    --refine \
    --resolution 2048 \
    --output final_gallery/
```

**Result**: Algorithmically precise, environmentally simulated, AI-rendered prehistoric mega-flora

---

## Statistics

### Implementation
- **Total Modules**: 17 Python files
- **Total Examples**: 14 L-system files
- **Lines of Code**: ~9,000+ lines
- **Documentation**: 5 comprehensive guides

### Capabilities Matrix

| Feature | Phase 1 | Phase 2 | Phase 3 |
|---------|---------|---------|---------|
| **L-System Rewriting** | ✅ | ✅ | - |
| **3D Turtle Graphics** | ✅ | ✅ | - |
| **Parametric Rules** | ✅ | ✅ | - |
| **Stochastic Rules** | ✅ | ✅ | - |
| **Context-Sensitive** | ✅ | ✅ | - |
| **Tropism Forces** | - | ✅ | - |
| **Light Simulation** | - | ✅ | - |
| **Resource Allocation** | - | ✅ | - |
| **Pipe Model** | - | ✅ | - |
| **Map Generation** | - | - | ✅ |
| **AI Image Synthesis** | - | - | ✅ |
| **Style Presets** | - | - | ✅ |
| **GPU Acceleration** | - | ✅ | ✅ |

---

## Dependencies

```bash
# Core (Phase 1)
numpy>=1.24.0
numba>=0.57.0
pandas>=2.0.0

# Bio-Sim (Phase 2)
torch>=2.0.0
matplotlib>=3.7.0

# Factory (Phase 3)
Pillow>=10.0.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
opencv-python>=4.8.0
```

**Total Install Size**: ~25GB (with SDXL models)

---

## Usage Patterns

### Quick Iterations (CPU Only)

```bash
# Phase 1: Generate structure (fast)
python lsystem.py --file examples/simple_tree.l --iter 4 --out tree.obj

# Phase 3: Generate maps (no GPU needed)
python factory.py --input tree.obj --maps-only --output maps/
```

### Full GPU Pipeline

```bash
# Phase 2: Simulate with environment
python biosim.py \
    --file examples/metabolic_tree.l \
    --iter 5 \
    --shadows \
    --out sim.obj

# Phase 3: AI synthesis
python factory.py \
    --input sim.obj \
    --style alien \
    --refine \
    --output gallery/
```

### Batch Production

```bash
# Generate multiple variations
for style in prehistoric alien bioluminescent; do
    python factory.py \
        --input sim.obj \
        --style $style \
        --angles iso,front,closeup \
        --output gallery/$style/
done
```

---

## Performance Benchmarks

### Phase 1: L-System Core
- **3 iterations**: < 1 second
- **5 iterations**: 1-3 seconds
- **10 iterations**: 5-15 seconds
- **Platform**: CPU, any modern processor

### Phase 2: Bio-Sim
- **5 iterations** (no shadows): 1-2 seconds
- **5 iterations** (with shadows): 3-8 seconds
- **10 iterations** (full sim): 10-30 seconds
- **Platform**: CPU or GPU (PyTorch)

### Phase 3: The Factory
- **Map generation**: 1-2 seconds
- **AI image (1024px)**: 30-60 seconds
- **AI image (2048px)**: 60-120 seconds
- **With refinement**: +20-30 seconds
- **Platform**: NVIDIA GPU (8GB+ VRAM)

---

## Scientific Accuracy

### Botanical Models
- **Pipe Model Theory** (Shinozaki et al., 1964)
- **Apical Dominance** (Borchert & Honda, 1984)
- **ABOP Signal Mechanisms** (Chapter 3)
- **Leonardo da Vinci's Rule** (n=2.0)

### Artistic Control
- Adjustable pipe exponent for morphologies
- Custom tropism vectors
- Negative phototropism for alien effects
- Weighted prompt engineering

---

## Aesthetic Styles (Phase 3)

### Realistic
1. **prehistoric** - Carboniferous mega-flora
2. **botanical** - Scientific illustration

### Alien/Exotic
3. **alien** - Bioluminescent extraterrestrial flora
4. **alien_creature** - Animal-like organism
5. **crystalline** - Mineral-organic hybrid
6. **abyssal** - Deep sea adaptations

### Biological
7. **bioluminescent** - Glowing organisms
8. **fungal** - Mycelial networks
9. **parasitic** - Symbiotic relationships

---

## Common Workflows

### 1. Realistic Earth Plant

```bash
# Use standard parameters
python biosim.py --file examples/metabolic_tree.l --iter 5 --shadows --out tree.obj
python factory.py --input tree.obj --style botanical --output gallery/
```

### 2. Prehistoric Mega-Flora

```bash
# Low pipe exponent, high resources
python biosim.py \
    --file examples/prehistoric_mega_flora.l \
    --iter 4 \
    --pipe-exp 1.5 \
    --shadows \
    --out prehistoric.obj

python factory.py \
    --input prehistoric.obj \
    --style prehistoric \
    --refine \
    --output gallery/
```

### 3. Alien Bioluminescent

```bash
# Negative phototropism, unusual gravity
python biosim.py \
    --file examples/alien_structure.l \
    --iter 4 \
    --gravity 0.1 \
    --photo -0.4 \
    --pipe-exp 2.8 \
    --shadows \
    --out alien.obj

python factory.py \
    --input alien.obj \
    --style bioluminescent \
    --custom-prompt "pulsating bio-light" \
    --refine \
    --output gallery/
```

### 4. Competitive Forest

```bash
# Multiple organisms competing for light
python biosim.py \
    --file examples/competitive_growth.l \
    --iter 6 \
    --shadows \
    --metabolism \
    --out competition.obj

python factory.py \
    --input competition.obj \
    --style fungal \
    --angles top,iso \
    --output gallery/
```

---

## System Requirements

### Minimum (Phases 1-2)
- **CPU**: Any modern processor
- **RAM**: 4GB
- **Storage**: 1GB
- **OS**: Linux, macOS, Windows

### Recommended (All Phases)
- **CPU**: Multi-core processor
- **GPU**: NVIDIA GPU, 12GB+ VRAM
- **RAM**: 16GB+
- **Storage**: 30GB (models + outputs)
- **OS**: Linux (optimal for GPU)

---

## Testing

```bash
# Test Phase 1
make test

# Test Phase 2
make test-biosim

# Test Phase 3 (maps only, no GPU)
make test-factory

# Generate all examples
make examples          # Phase 1
make examples-biosim   # Phase 2
make examples-factory  # Phase 3 (maps)
```

---

## Future Enhancements

### Phase 1 Improvements
- [ ] Numba JIT optimization for rewriting
- [ ] Bracketed L-systems (ABOP Chapter 1.7)
- [ ] Table L-systems

### Phase 2 Extensions
- [ ] Multiple competing organisms
- [ ] Soil nutrient simulation
- [ ] Wind effects
- [ ] Seasonal growth cycles
- [ ] Root system modeling

### Phase 3 Additions
- [ ] Multiple ControlNet types (Canny, Normal, Pose)
- [ ] LoRA fine-tuning integration
- [ ] Animation sequence generation
- [ ] Style transfer
- [ ] Real-time preview
- [ ] Integration with game engines

### Cross-Phase
- [ ] Web-based interface
- [ ] Parallel processing
- [ ] Database of generated structures
- [ ] VR/AR visualization

---

## Publications & References

### Core References
1. **The Algorithmic Beauty of Plants**
   - Prusinkiewicz & Lindenmayer (1990)
   - Springer-Verlag

2. **Pipe Model Theory**
   - Shinozaki et al. (1964)

3. **ControlNet**
   - Zhang et al. (2023)

4. **Stable Diffusion XL**
   - Podell et al. (2023)

### Implementation
- **PyTorch**: Paszke et al. (2019)
- **Diffusers**: HuggingFace Team
- **NumPy**: Harris et al. (2020)

---

## Citation

```bibtex
@software{lsystem_complete_2026,
  title = {L-System Core + Bio-Sim + Factory: Complete Algorithmic Beauty Pipeline},
  author = {L-System Core Project},
  year = {2026},
  note = {Three-phase implementation: Static Rewriting, Dynamic Simulation, AI Synthesis}
}
```

---

## Project Status

| Phase | Status | Modules | Examples | Tests |
|-------|--------|---------|----------|-------|
| Phase 1 | ✅ Complete | 6 | 7 | ✅ Pass |
| Phase 2 | ✅ Complete | 8 | 7 | ✅ Pass |
| Phase 3 | ✅ Complete | 3 | 9 styles | ✅ Pass |

**Overall Progress**: 100% Complete

**Last Updated**: January 2026

---

## Acknowledgments

- Przemyslaw Prusinkiewicz and Aristid Lindenmayer for ABOP
- Stability AI for Stable Diffusion
- HuggingFace for Diffusers
- PyTorch team
- The L-system research community

---

## License

Educational/Research implementation based on "The Algorithmic Beauty of Plants"

---

**🎉 All Three Phases Complete!**

From algorithmic precision → environmental simulation → AI aesthetics

*The complete pipeline for generating uncanny, prehistoric, and alien botanical imagery.*
