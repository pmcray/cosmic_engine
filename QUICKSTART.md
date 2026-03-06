# L-System Core - Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Run Your First L-System

```bash
# Simple tree (4 iterations)
python lsystem.py --file examples/simple_tree.l --iter 4 --out tree.obj --stats

# Parametric tree with decreasing branch lengths
python lsystem.py --file examples/parametric_tree.l --iter 5 --out param_tree.obj --stats

# 3D tree with pitch and yaw rotations
python lsystem.py --file examples/3d_tree.l --iter 3 --out 3d_tree.obj --stats
```

## View the Results

The generated `.obj` files can be opened in:
- Blender
- MeshLab
- Online viewers like https://3dviewer.net

## Create Your Own L-System

Create a text file (e.g., `my_plant.l`):

```
# Set rotation angle
angle = 22.5

# Starting symbol
axiom: F

# Production rule
F -> F[+F][-F]F
```

Run it:

```bash
python lsystem.py --file my_plant.l --iter 5 --out my_plant.obj --verbose
```

## Examples Included

1. **simple_tree.l** - Basic branching structure
2. **parametric_tree.l** - Branches get smaller (uses parameters)
3. **stochastic_plant.l** - Random variations (different each time!)
4. **3d_tree.l** - Full 3D with pitch, yaw, and roll
5. **conditional.l** - Conditional branching based on size
6. **context_sensitive.l** - Rules that depend on neighboring symbols
7. **algae.l** - Lindenmayer's original model (string rewriting only)

## Turtle Graphics Commands

In your L-system rules, use these symbols:

| Symbol | Meaning |
|--------|---------|
| `F` | Move forward and draw a line |
| `f` | Move forward without drawing |
| `+` | Turn left (yaw) |
| `-` | Turn right (yaw) |
| `&` | Pitch down |
| `^` | Pitch up |
| `/` | Roll left |
| `\` | Roll right |
| `[` | Save position/direction |
| `]` | Restore position/direction |
| `!` | Set line width |

## Parameters

Add numbers in parentheses:

```
F(2.5)     # Move forward 2.5 units
+(45)      # Turn 45 degrees
A(10)      # Custom symbol with parameter 10
```

## Advanced Features

### Parametric Rules

```
axiom: A(10)

# Branch length decreases by 30% each time
A(len) -> F(len)[+A(len*0.7)][-A(len*0.7)]
```

### Stochastic (Random) Rules

```
# 50% chance for each branch pattern
F : 0.5 -> F[+F]F
F : 0.5 -> F[-F]F
```

### Conditional Rules

```
# Only branch if length > 1
A(x) : x > 1 -> F(x)[+A(x*0.7)][-A(x*0.7)]
A(x) : x <= 1 -> F(x)
```

### Context-Sensitive Rules

```
# B becomes C only if preceded by A
A < B -> C

# B becomes D only if followed by C
B > C -> D

# B becomes E only if between A and C
A < B > C -> E
```

## Tips

- Start with 3-5 iterations and increase gradually
- Higher iterations = exponentially more complex geometry
- Use `--verbose` to see what's happening
- Use `--stats` to see geometry size
- Use `--print-string` to see the symbol string (good for debugging)

## Common Issues

**Too many vertices?**
- Reduce iterations (each iteration multiplies complexity)
- Simplify your rules

**Nothing rendered?**
- Make sure you have `F` symbols (they draw lines)
- Check your axiom has at least one symbol
- Use `--verbose --print-string` to debug

**Want thicker branches?**
- Use `!(width)` in your rules
- Export with `--format cylinders` for solid geometry

## Next Steps

1. Read README.md for full documentation
2. Experiment with the examples
3. Modify existing examples
4. Create your own plant models!

## Reference

Based on: *The Algorithmic Beauty of Plants* (ABOP)
by Przemyslaw Prusinkiewicz and Aristid Lindenmayer
