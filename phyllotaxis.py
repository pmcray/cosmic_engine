"""
Phyllotaxis Models (ABOP Chapter 4).
Generates patterns for spiral phyllotaxis (sunflower heads, pine cones, etc).
"""
import numpy as np
import argparse
from turtle_3d import Turtle3D
from obj_exporter import export_to_obj


def generate_planar_phyllotaxis(n_elements: int = 100, c: float = 1.0, alpha: float = 137.5):
    """
    Vogel's formula for planar phyllotaxis (sunflower head).
    phi = n * alpha
    r = c * sqrt(n)
    
    Args:
        n_elements: Number of elements to generate
        c: Scaling factor
        alpha: Divergence angle in degrees. Default is golden angle 137.5.
        
    Returns:
        List of 3D points (x, y, 0)
    """
    points = []
    alpha_rad = np.radians(alpha)
    for n in range(1, n_elements + 1):
        phi = n * alpha_rad
        r = c * np.sqrt(n)
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append(np.array([x, y, 0.0]))
        
    return points


def generate_cylindrical_phyllotaxis(n_elements: int = 100, r: float = 1.0, h: float = 0.5, alpha: float = 137.5):
    """
    Cylindrical phyllotaxis model (pine cone, spiral stem).
    phi = n * alpha
    r = constant
    H = n * h
    
    Args:
        n_elements: Number of elements
        r: Cylinder radius
        h: Vertical displacement per element
        alpha: Divergence angle
        
    Returns:
        List of 3D points (x, z, y) where y is the vertical axis.
    """
    points = []
    alpha_rad = np.radians(alpha)
    for n in range(1, n_elements + 1):
        phi = n * alpha_rad
        height = n * h
        
        x = r * np.cos(phi)
        z = r * np.sin(phi)
        points.append(np.array([x, height, z]))
        
    return points


def render_phyllotaxis_to_obj(filename: str, points: list, size: float = 0.2):
    """
    Render points as simple diamond-shaped polygons using turtle symbols.
    """
    turtle = Turtle3D(default_angle=90, default_length=size)
    
    # We'll construct a mock sequence of symbols to just place polygons at these points
    import symbol
    
    # Manually inject polygons
    for p in points:
        turtle.state.position = p.copy()
        
        # Create a small diamond at position p
        turtle._process_symbol(symbol.Symbol('{', []))
        
        turtle._process_symbol(symbol.Symbol('.', []))
        
        turtle.state.position += np.array([size, 0, 0])
        turtle._process_symbol(symbol.Symbol('.', []))
        
        turtle.state.position += np.array([-size, size, 0])
        turtle._process_symbol(symbol.Symbol('.', []))
        
        turtle.state.position += np.array([-size, -size, 0])
        turtle._process_symbol(symbol.Symbol('.', []))
        
        turtle.state.position = p.copy()
        turtle._process_symbol(symbol.Symbol('}', []))

    turtle.to_obj(filename)
    print(f"Exported phyllotaxis geometry to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phyllotaxis patterns")
    parser.add_argument("--type", choices=['planar', 'cylindrical'], default='planar', help="Type of model")
    parser.add_argument("--n", type=int, default=200, help="Number of elements")
    parser.add_argument("--alpha", type=float, default=137.508, help="Divergence angle (degrees)")
    parser.add_argument("--c", type=float, default=0.5, help="Scaling factor (planar)")
    parser.add_argument("--r", type=float, default=2.0, help="Radius (cylindrical)")
    parser.add_argument("--h", type=float, default=0.2, help="Vertical step (cylindrical)")
    parser.add_argument("--out", type=str, default="phyllotaxis.obj", help="Output OBJ file")
    
    args = parser.parse_args()
    
    if args.type == 'planar':
        pts = generate_planar_phyllotaxis(args.n, args.c, args.alpha)
    else:
        pts = generate_cylindrical_phyllotaxis(args.n, args.r, args.h, args.alpha)
        
    render_phyllotaxis_to_obj(args.out, pts)
