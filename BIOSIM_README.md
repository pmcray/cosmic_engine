# Bio-Sim Module

**Phase 2: Environmental Simulation & Metabolic L-Systems**

Bio-Sim extends the L-System Core with environmentally-aware growth simulation, including voxel-based light propagation, tropism forces, and resource allocation.

## Features

### Environmental Simulation
- **Voxel-Based Light Propagation**: 3D grid tracks light intensity and shadow casting
- **Shadow Dynamics**: Structures cast shadows that affect growth of lower branches
- **GPU Acceleration**: PyTorch-based computation for large-scale simulations

### Tropism (Bending Forces)
- **Gravitropism**: Bending toward or away from gravity
- **Phototropism**: Bending toward light gradients
- **Combined Effects**: Multiple tropism forces can interact

### Metabolic Resource Allocation
- **Pipe Model Theory**: Branch thickness follows hydraulic constraints
- **Signal-Based Distribution**: Basipetal demand + acropetal supply
- **Resource-Limited Growth**: Production rules conditional on available energy
- **Self-Pruning**: Shaded branches die back to redirect resources

## Installation

```bash
# Install all dependencies (including PyTorch)
pip install -r requirements.txt
```

## Quick Start

### Basic Tropism

```bash
# Tree with gravitropism (grows upward)
python biosim.py --file examples/tropism_tree.l --iter 4 --out tree.obj --stats
```

### Light Competition

```bash
# Plant that bends toward light with shadow casting
python biosim.py --file examples/phototropic_plant.l --iter 3 --shadows --out plant.obj --verbose
```

### Metabolic Simulation

```bash
# Resource-limited growth with self-pruning
python biosim.py --file examples/metabolic_tree.l --iter 5 --shadows --metabolism --out metabolic.obj --stats
```

## L-System File Format Extensions

Bio-Sim adds new context variables to L-system files:

```
# Tropism parameters
gravity = 0.3          # Gravitropism susceptibility (0.0 to 1.0)
phototropism = 0.2     # Phototropism susceptibility (0.0 to 1.0)

# Pipe model
pipe_exponent = 2.0    # Vascular scaling exponent
                       # 2.0 = Earth-like (Leonardo da Vinci)
                       # < 2.0 = Thick trunks (prehistoric)
                       # > 2.5 = Spindly (alien)

# L-system with resource parameter
axiom: A(10)

# Conditional growth based on resource level
A(r) : r > 1 -> F(r*0.3)[+A(r*0.4)][-A(r*0.4)]
A(r) : r <= 1 -> F(r*0.5)~

# ~ marks leaves (photosynthetic)
```

## Turtle Graphics Extensions

New symbols for bio-simulation:

| Symbol | Meaning |
|--------|---------|
| `~` | Mark as leaf (photosynthetic tissue) |
| `$` | Roll to vertical (align with gravity) |
| `%` | Cut off remainder (competitive pruning) |

## Command-Line Options

### Basic Options
```bash
--file FILE         # Input L-system file
--iter N            # Number of iterations
--out FILE          # Output OBJ file
```

### Environment Options
```bash
--bounds min_x,max_x,min_y,max_y,min_z,max_z  # Simulation bounds
--voxel-res FLOAT   # Voxel resolution in meters (default: 1.0)
```

### Simulation Flags
```bash
--shadows           # Enable shadow casting
--metabolism        # Enable metabolic simulation
--no-tropism        # Disable tropism forces
```

### Tropism Parameters
```bash
--gravity FLOAT     # Gravitropism susceptibility
--photo FLOAT       # Phototropism susceptibility
--pipe-exp FLOAT    # Pipe model exponent
```

### Output Options
```bash
--cylinders         # Export as cylindrical meshes
--save-history      # Save simulation history
--out-history FILE  # Export history as JSON
--export-light Z    # Export light field at z-level
--plot              # Plot final structure (requires matplotlib)
```

## Examples

### 1. Tropism Tree
Basic tree with gravitropism (branches grow upward):
```bash
python biosim.py --file examples/tropism_tree.l --iter 4 --out tropism.obj
```

### 2. Phototropic Plant
Plant that bends toward light:
```bash
python biosim.py --file examples/phototropic_plant.l --iter 3 --shadows --out photo.obj
```

### 3. Metabolic Tree
Resource-limited growth with self-pruning:
```bash
python biosim.py --file examples/metabolic_tree.l --iter 5 --shadows --metabolism --out metabolic.obj
```

### 4. Prehistoric Mega-Flora
Thick trunks (low pipe exponent):
```bash
python biosim.py --file examples/prehistoric_mega_flora.l --iter 4 --shadows --cylinders --out prehistoric.obj
```

### 5. Alien Structure
Spindly, top-heavy with negative phototropism:
```bash
python biosim.py --file examples/alien_structure.l --iter 4 --shadows --out alien.obj
```

### 6. Competitive Growth
Multiple iterations with aggressive light capture:
```bash
python biosim.py --file examples/competitive_growth.l --iter 6 --shadows --out competitive.obj
```

### 7. Self-Pruning Tree
Aggressive pruning of shaded branches:
```bash
python biosim.py --file examples/self_pruning_tree.l --iter 5 --shadows --metabolism --out pruning.obj
```

## Architecture

### Core Modules

- **environment.py**: Voxel-based light simulation with PyTorch
- **forces.py**: Tropism calculations (gravitropism, phototropism)
- **metabolism.py**: Resource allocation and pipe model
- **growth_node.py**: Graph structure for growth tracking
- **turtle_tropism.py**: Tropism-enabled turtle interpreter
- **simulator.py**: Simulation orchestrator
- **biosim.py**: Command-line interface
- **visualization.py**: Plotting and animation tools

### Data Flow

```
1. Parse L-system → Initial symbol string
2. Create environment → Voxel grid
3. Rewrite iteration:
   a. Apply production rules
   b. Interpret with tropism turtle
   c. Build growth graph
   d. Cast shadows in environment
   e. Calculate light at each node
   f. Distribute resources
   g. Update node states
4. Export geometry and statistics
```

## Pipe Model Theory

Branch thickness follows the constraint:

```
S_parent^n = Σ S_child_i^n
```

Where:
- S is cross-sectional area
- n is the pipe exponent

**Tuning for Different Morphologies:**
- **n = 2.0**: Standard Earth-like trees (Leonardo da Vinci's rule)
- **n < 2.0** (e.g., 1.5): Prehistoric mega-flora with thick, heavy trunks
- **n > 2.5** (e.g., 2.8): Alien structures that are spindly and top-heavy

## Resource Allocation

### Dual-Signal Mechanism

1. **Basipetal (Demand)**: From leaves to root
   - Each leaf generates demand based on light availability
   - Demand propagates down through parent nodes

2. **Acropetal (Supply)**: From root to tips
   - Resources distributed based on demand ratios
   - Transport resistance causes decay with distance
   - Apical dominance favors main branches

### Metabolic Feedback

```python
# Production (at leaves)
P = light_intensity × efficiency

# Allocation
resource = P × demand_ratio × exp(-resistance × distance)

# Growth condition
if resource > threshold:
    apply_production_rule()
else:
    become_dormant()
```

## Advanced Usage

### Custom Tropism Vectors

Create exotic growth forms by modifying gravity:

```python
from simulator import BioSimulator
import numpy as np

# Rotating gravity for spiral growth
simulator = load_biosim_from_file('myplant.l')
simulator.tropism_calc.set_gravity_vector(
    np.array([np.cos(t), np.sin(t), -0.5])  # Spiral!
)
```

### Time-Series Animation

```bash
# Generate animation data
python biosim.py --file examples/growth.l --iter 10 \
    --save-history --out-history growth.json

# Create animation (in Python)
from visualization import animate_growth
animate_growth('growth.json', 'growth.gif', fps=2)
```

### Light Field Visualization

```bash
# Export light slice at z=10m
python biosim.py --file examples/tree.l --iter 5 --shadows \
    --export-light 10 --light-output light_z10.npz

# Visualize (in Python)
from visualization import plot_light_field
plot_light_field('light_z10.npz')
```

## Performance Considerations

- **Voxel Resolution**: Lower resolution (larger voxels) = faster but less accurate
  - Fine detail: 0.1 - 0.5 m
  - Normal: 0.5 - 1.0 m
  - Fast: 1.0 - 2.0 m

- **PyTorch/GPU**: Automatically used if available
  - Significant speedup for large voxel grids (>100³)
  - CPU fallback if PyTorch not available

- **Shadow Casting**: Most expensive operation
  - Disable with `--no-shadows` for faster iteration
  - Enable only for final rendering

## Biological Realism vs Artistic Control

### For Realistic Plants
- Use n = 2.0 (pipe exponent)
- Moderate tropism (0.1 - 0.3)
- Enable metabolism and shadows
- Multiple iterations (6-10)

### For "Uncanny" Pre-Historic
- Use n = 1.3 - 1.8 (thick trunks)
- Moderate to high gravity (0.2 - 0.4)
- Low phototropism (0.0 - 0.2)
- Large initial resources

### For Alien Structures
- Use n = 2.5 - 3.0 (spindly)
- Negative phototropism (-0.3 to -0.5)
- Non-standard gravity direction
- Unusual branching angles

## Testing

```bash
# Run bio-sim examples
make test-biosim

# Generate all example outputs
make examples-biosim
```

## References

- **The Algorithmic Beauty of Plants** (ABOP) - Prusinkiewicz & Lindenmayer
  - Chapter 1: L-System fundamentals
  - Chapter 2: Tropism modeling
  - Chapter 3: Metabolic signals and resource allocation

- **Pipe Model Theory** - Shinozaki et al. (1964)
- **Apical Dominance** - Borchert & Honda (1984)

## Troubleshooting

**Problem**: PyTorch not found
```
Warning: PyTorch not available. Using NumPy (slower).
```
**Solution**: Install PyTorch: `pip install torch`

**Problem**: No geometry output
- Check that rules produce 'F' symbols (drawing)
- Use `--verbose` to see string length
- Verify resource levels aren't too low

**Problem**: Simulation very slow
- Reduce voxel resolution: `--voxel-res 2.0`
- Disable shadows: remove `--shadows`
- Reduce iterations

**Problem**: Unrealistic branching
- Adjust pipe exponent
- Check tropism parameters
- Verify production rules
