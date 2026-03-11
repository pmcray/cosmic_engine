"""
Iterated Function Systems (ABOP Chapter 8).
Models plant-like structures and mathematical fractals using affine transformations.
"""
import numpy as np
import random
import argparse


class IFSTransform:
    """An affine transformation with an associated probability."""
    def __init__(self, a, b, c, d, e, f, prob=1.0):
        self.matrix = np.array([
            [a, b, e],
            [c, d, f],
            [0, 0, 1]
        ])
        self.prob = prob

    def apply(self, point: np.ndarray) -> np.ndarray:
        """Applies transformation to a 2D point [x, y]."""
        p_homog = np.array([point[0], point[1], 1.0])
        p_new = self.matrix @ p_homog
        return p_new[:2]


class IFSSystem:
    def __init__(self):
        self.transforms = []
        self.probs = []

    def add_transform(self, t: IFSTransform):
        self.transforms.append(t)
        self.probs.append(t.prob)
        # Normalize probabilities
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]

    def generate_chaos_game(self, iterations: int = 10000, start_point=(0.0, 0.0)):
        """
        Generate IFS fractal points using the Chaos Game algorithm (stochastic).
        """
        points = []
        current = np.array(start_point, dtype=float)
        
        # Discard first few iterations to reach attractor
        for _ in range(20):
            t = random.choices(self.transforms, weights=self.probs)[0]
            current = t.apply(current)
            
        for _ in range(iterations):
            t = random.choices(self.transforms, weights=self.probs)[0]
            current = t.apply(current)
            points.append(current.copy())
            
        return points

def create_barnsley_fern():
    """Definition of Barnsley's Fern IFS (ABOP Fig 8.3)"""
    ifs = IFSSystem()
    # Stem
    ifs.add_transform(IFSTransform(0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01))
    # Successive leaf
    ifs.add_transform(IFSTransform(0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85))
    # Largest left-hand leaflet
    ifs.add_transform(IFSTransform(0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07))
    # Largest right-hand leaflet
    ifs.add_transform(IFSTransform(-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07))
    return ifs

def export_pts_to_obj(filename, points):
    """
    Export point cloud to OBJ file.
    """
    with open(filename, 'w') as f:
        f.write("# IFS Fractal Point Cloud\n")
        f.write(f"# Points: {len(points)}\n\n")
        for p in points:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} 0.000000\n")
            
    print(f"Exported IFS point cloud to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFS Fractal Generator")
    parser.add_argument("--preset", choices=['fern'], default='fern', help="IFS preset to use")
    parser.add_argument("--iter", type=int, default=50000, help="Number of points to generate via Chaos Game")
    parser.add_argument("--out", type=str, default="ifs_fractal.obj", help="Output OBJ file")
    
    args = parser.parse_args()
    
    if args.preset == 'fern':
        ifs = create_barnsley_fern()
        
    pts = ifs.generate_chaos_game(iterations=args.iter)
    export_pts_to_obj(args.out, pts)
