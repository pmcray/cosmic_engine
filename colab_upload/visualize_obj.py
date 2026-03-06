#!/usr/bin/env python3
"""
Simple OBJ visualizer - saves matplotlib plot to file.
"""
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_obj(filename):
    """Read vertices and lines from OBJ file."""
    vertices = []
    lines = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # Vertex
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
            elif line.startswith('l '):
                # Line
                parts = line.split()
                start = int(parts[1]) - 1  # OBJ is 1-indexed
                end = int(parts[2]) - 1
                lines.append([start, end])

    return np.array(vertices), lines

def visualize_obj(obj_file, output_file='visualization.png', title=None):
    """Visualize OBJ file and save to PNG."""
    print(f"Reading {obj_file}...")
    vertices, lines = read_obj(obj_file)

    if len(vertices) == 0:
        print("Error: No vertices found")
        return

    print(f"  Vertices: {len(vertices)}")
    print(f"  Lines: {len(lines)}")

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))

    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_3d_view(ax1, vertices, lines, "3D View")

    # Top view (X-Y)
    ax2 = fig.add_subplot(2, 2, 2)
    plot_2d_view(ax2, vertices, lines, 'X', 'Y', "Top View (X-Y)")

    # Front view (Y-Z)
    ax3 = fig.add_subplot(2, 2, 3)
    plot_2d_view(ax3, vertices, lines, 'Y', 'Z', "Front View (Y-Z)")

    # Side view (X-Z)
    ax4 = fig.add_subplot(2, 2, 4)
    plot_2d_view(ax4, vertices, lines, 'X', 'Z', "Side View (X-Z)")

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_file}")

    # Also save statistics
    stats_file = output_file.replace('.png', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"OBJ File: {obj_file}\n")
        f.write(f"Vertices: {len(vertices)}\n")
        f.write(f"Lines: {len(lines)}\n")
        f.write(f"\nBounds:\n")
        f.write(f"  X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]\n")
        f.write(f"  Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]\n")
        f.write(f"  Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]\n")
        f.write(f"\nDimensions:\n")
        f.write(f"  Width (X):  {vertices[:, 0].max() - vertices[:, 0].min():.2f} m\n")
        f.write(f"  Depth (Y):  {vertices[:, 1].max() - vertices[:, 1].min():.2f} m\n")
        f.write(f"  Height (Z): {vertices[:, 2].max() - vertices[:, 2].min():.2f} m\n")
    print(f"✓ Saved statistics to {stats_file}")

def plot_3d_view(ax, vertices, lines, title):
    """Plot 3D view."""
    # Plot lines
    for start_idx, end_idx in lines:
        if start_idx < len(vertices) and end_idx < len(vertices):
            start = vertices[start_idx]
            end = vertices[end_idx]
            ax.plot([start[0], end[0]],
                   [start[1], end[1]],
                   [start[2], end[2]],
                   'g-', linewidth=1.0, alpha=0.7)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

def plot_2d_view(ax, vertices, lines, axis1, axis2, title):
    """Plot 2D projection."""
    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    idx1 = axis_map[axis1]
    idx2 = axis_map[axis2]

    # Plot lines
    for start_idx, end_idx in lines:
        if start_idx < len(vertices) and end_idx < len(vertices):
            start = vertices[start_idx]
            end = vertices[end_idx]
            ax.plot([start[idx1], end[idx1]],
                   [start[idx2], end[idx2]],
                   'g-', linewidth=0.5, alpha=0.7)

    ax.set_xlabel(f'{axis1} (m)')
    ax.set_ylabel(f'{axis2} (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_obj.py <input.obj> [output.png] [title]")
        sys.exit(1)

    obj_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else obj_file.replace('.obj', '_visualization.png')
    title = sys.argv[3] if len(sys.argv) > 3 else obj_file

    visualize_obj(obj_file, output_file, title)
