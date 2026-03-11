"""
Plant Organs and Surfaces (ABOP Chapter 5).
Provide predefined surfaces and structural strings for compound leaves and petals.
"""
import numpy as np
from symbol import Symbol


def generate_cordate_leaf(length: float = 1.0, width: float = 0.8) -> str:
    """
    Generate an L-system string for a heart-shaped (cordate) leaf using polygon symbols.
    
    Args:
        length: Overall sequence length of the leaf
        width: Maximum width of the leaf
        
    Returns:
        L-system string to be rendered by the 3D turtle.
    """
    # A simplified cordate leaf approximation using polygons.
    # The string uses { to start, . to record vertices, } to end.
    # We trace the perimeter of the leaf.
    # We can use branching [ ... ] to draw to points and record them.
    
    # We trace from base to left tip, to apex, to right tip, to base.
    
    seq = "{"
    seq += "."  # Base vertex
    
    # Left lobe
    seq += f"[ +(60) f({length * 0.4}) . ]"
    # Left mid
    seq += f"[ +(30) f({length * 0.7}) . ]"
    # Apex
    seq += f"[ f({length}) . ]"
    # Right mid
    seq += f"[ -(30) f({length * 0.7}) . ]"
    # Right lobe
    seq += f"[ -(60) f({length * 0.4}) . ]"
    
    seq += "}"
    
    return seq


def generate_simple_petal(length: float = 1.0, width: float = 0.5) -> str:
    """
    Generate an L-system string for a simple petal.
    """
    seq = "{"
    seq += "."  # Base
    
    seq += f"[ +(40) f({length * 0.3}) . ]"
    seq += f"[ +(20) f({length * 0.8}) . ]"
    seq += f"[ f({length}) . ]"
    seq += f"[ -(20) f({length * 0.8}) . ]"
    seq += f"[ -(40) f({length * 0.3}) . ]"
    
    seq += "}"
    return seq
