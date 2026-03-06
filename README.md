# L-System Core

A high-performance Python-based CLI tool and library for Parametric, Stochastic, and Context-Sensitive L-Systems with 3D geometry output.

## Features

- **Parametric L-Systems**: Support for numerical parameters and arithmetic expressions
- **Stochastic Rules**: Multiple productions per predecessor with probabilities
- **Context-Sensitive**: Left and right context matching
- **Conditional Rules**: Boolean conditions on parameters
- **3D Turtle Graphics**: Full 3D geometry interpretation
- **Multiple Output Formats**: OBJ, PLY, and cylindrical meshes
- **Performance Optimized**: Ready for Numba JIT optimization

## Installation

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python lsystem.py --file examples/tree.l --iter 5 --out tree.obj
```

### Options

- `--file, -f`: Input L-system file (required)
- `--iter, -i`: Number of iterations (default: 1)
- `--out, -o`: Output file (OBJ/PLY)
- `--format`: Output format: obj, ply, or cylinders (default: obj)
- `--angle`: Default rotation angle in degrees
- `--length`: Default step length (default: 1.0)
- `--width`: Default line width (default: 0.1)
- `--verbose, -v`: Verbose output
- `--stats, -s`: Show geometry statistics
- `--print-string, -p`: Print final L-system string

### Examples

```bash
# Simple 2D tree
python lsystem.py --file examples/simple_tree.l --iter 4 --out simple_tree.obj --stats

# Parametric tree with decreasing branch length
python lsystem.py --file examples/parametric_tree.l --iter 5 --out parametric_tree.obj

# Stochastic plant (different each time!)
python lsystem.py --file examples/stochastic_plant.l --iter 4 --out plant.obj

# 3D tree
python lsystem.py --file examples/3d_tree.l --iter 3 --out 3d_tree.obj

# Generate cylindrical meshes
python lsystem.py --file examples/simple_tree.l --iter 4 --out tree_cylinders.obj --format cylinders

# Run all examples
make examples
```

## L-System File Format

L-system files use a simple text format:

```
# Comments start with #

# Define constants
angle = 25.7
shrink = 0.9

# Define axiom (starting string)
axiom: F(10)

# Define production rules

# Simple rule
F -> FF

# Parametric rule
A(x) -> F(x)[+A(x*0.7)][-A(x*0.7)]

# Conditional rule
A(x) : x > 5 -> B(x/2)C(x/2)

# Stochastic rules (probabilities must sum to ~1.0)
F : 0.5 -> F[+F]F
F : 0.5 -> F[-F]F

# Context-sensitive rule (A < B means "B with A to the left")
A < B -> C
B > C -> D
A < B > C -> D
```

### Turtle Graphics Symbols

- `F(len)`: Move forward and draw (default length if no parameter)
- `f(len)`: Move forward without drawing
- `+, -`: Rotate around Up axis (yaw)
- `&, ^`: Rotate around Left axis (pitch)
- `/, \`: Rotate around Heading axis (roll)
- `[`: Push current state onto stack
- `]`: Pop state from stack
- `!(width)`: Set line width

### Coordinate System

The 3D turtle uses a right-handed coordinate system:
- **H** (Heading): Forward direction (initially +Y)
- **L** (Left): Left direction (initially +X)
- **U** (Up): Up direction (initially +Z)

All rotations follow the right-hand rule around their respective axes.

## Module Structure

- `symbol.py`: Core data models (Symbol, ProductionRule, LSystemContext)
- `lexer.py`: Parser for L-system files
- `rewriter.py`: Rewriting engine with stochastic support
- `turtle_3d.py`: 3D turtle graphics interpreter
- `obj_exporter.py`: OBJ/PLY export functionality
- `lsystem.py`: CLI interface

## Examples Included

1. **simple_tree.l**: Basic branching tree (0L-system)
2. **parametric_tree.l**: Tree with decreasing branch lengths
3. **stochastic_plant.l**: Plant with random branching
4. **3d_tree.l**: Full 3D tree using pitch and yaw
5. **algae.l**: Lindenmayer's original algae model
6. **context_sensitive.l**: Signal propagation
7. **conditional.l**: Conditional branching based on parameters

## Testing

```bash
# Run quick tests
make test

# Generate all example outputs
make examples

# Clean generated files
make clean
```

## Performance Notes

- The rewriting engine is optimized for large-scale L-systems
- For simple non-parametric systems, consider using the OptimizedRewriter
- Numba JIT compilation can be added to critical loops for 10+ iterations
- Memory-efficient symbol handling supports very long strings

## Output Formats

### OBJ (lines)
Simple line-based geometry, compatible with most 3D software.

### PLY
Includes edge width information.

### Cylinders
Generates cylindrical meshes for each edge, creating solid geometry suitable for rendering.

## References

This implementation follows the formalism described in:
*"The Algorithmic Beauty of Plants"* by Przemyslaw Prusinkiewicz and Aristid Lindenmayer

## Project Phases

This is part of a complete 4-phase L-system implementation:

### Phase 1: L-System Core (This Module)
Parametric, stochastic, and context-sensitive L-systems with 3D turtle graphics.

**See this file for Phase 1 documentation.**

### Phase 2: Bio-Sim Module
Environmental simulation with tropism, light competition, and metabolic resource allocation.

**Documentation**: `BIOSIM_README.md`

```bash
python biosim.py --file examples/metabolic_tree.l --iter 5 --shadows
```

### Phase 3: The Factory
AI-driven image synthesis using Stable Diffusion and ControlNet.

**Documentation**: `FACTORY_README.md`

```bash
python factory.py --input tree.obj --style prehistoric --output gallery/
```

### Phase 4: Chronos Animation Engine
Smooth growth animations with temporal coherence.

**Documentation**: `CHRONOS_README.md`

```bash
python chronos.py --file examples/tree.l --iter 5 --duration 15 --video output.mp4
```

---

## Complete Pipeline Example

```bash
# 1. Generate L-system geometry (Phase 1)
python lsystem.py --file examples/3d_tree.l --iter 5 --out tree.obj

# 2. Add environmental simulation (Phase 2)
python biosim.py --file examples/metabolic_tree.l --iter 5 --shadows --out bio_tree.obj

# 3. AI-synthesize imagery (Phase 3)
python factory.py --input bio_tree.obj --style prehistoric --output gallery/

# 4. Create growth animation (Phase 4)
python chronos.py --file examples/tree.l --iter 5 --duration 15 --video growth.mp4 --style alien
```

**Full Documentation**: `COMPLETE_PROJECT_SUMMARY.md`

**Installation Guide**: `INSTALL.md`

**Quick Start**: `QUICKSTART.md`

---

## License

Educational/Research implementation based on "The Algorithmic Beauty of Plants"
