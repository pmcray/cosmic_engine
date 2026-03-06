"""
Visualization utilities for Bio-Sim.

Provides plotting and animation capabilities using matplotlib.
"""
import numpy as np
from typing import Optional, List
import json

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")


def plot_structure(turtle, title: str = "L-System Structure", show: bool = True):
    """
    Plot 3D structure using matplotlib.

    Args:
        turtle: Turtle3D or TropismTurtle3D instance
        title: Plot title
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return

    vertices = turtle.get_vertices()
    edges = turtle.get_edges()

    if len(vertices) == 0:
        print("No geometry to plot")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot edges
    for start_idx, end_idx, width in edges:
        if start_idx < len(vertices) and end_idx < len(vertices):
            start = vertices[start_idx]
            end = vertices[end_idx]

            ax.plot([start[0], end[0]],
                   [start[1], end[1]],
                   [start[2], end[2]],
                   'g-', linewidth=max(0.5, width * 10))

    # Set labels and title
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

    if show:
        plt.show()

    return fig, ax


def plot_light_field(light_slice_file: str, title: str = "Light Field"):
    """
    Plot a horizontal light field slice.

    Args:
        light_slice_file: NPZ file from environment.export_light_slice()
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return

    data = np.load(light_slice_file)
    light_slice = data['slice']
    z_level = data['z_level']
    bounds = data['bounds']
    resolution = data['resolution']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot light intensity
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    im = ax.imshow(light_slice.T, origin='lower', extent=extent,
                  cmap='viridis', vmin=0, vmax=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"{title} at Z={z_level:.1f}m")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Light Intensity')

    plt.show()

    return fig, ax


def plot_statistics(stats_list: List[dict], title: str = "Simulation Statistics"):
    """
    Plot simulation statistics over time.

    Args:
        stats_list: List of statistics dictionaries from simulation
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return

    iterations = [s['iteration'] for s in stats_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: String length and vertices
    ax = axes[0, 0]
    ax.plot(iterations, [s.get('string_length', 0) for s in stats_list],
           'b-', label='String length')
    ax.plot(iterations, [s.get('num_vertices', 0) for s in stats_list],
           'r-', label='Vertices')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Count')
    ax.set_title('Growth Metrics')
    ax.legend()
    ax.grid(True)

    # Plot 2: Total length
    ax = axes[0, 1]
    ax.plot(iterations, [s.get('total_length', 0) for s in stats_list], 'g-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Length (m)')
    ax.set_title('Structural Length')
    ax.grid(True)

    # Plot 3: Metabolic balance (if available)
    ax = axes[1, 0]
    if 'production' in stats_list[0]:
        ax.plot(iterations, [s.get('production', 0) for s in stats_list],
               'g-', label='Production')
        ax.plot(iterations, [s.get('respiration', 0) for s in stats_list],
               'r-', label='Respiration')
        ax.plot(iterations, [s.get('net_balance', 0) for s in stats_list],
               'b-', label='Net Balance')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.set_title('Metabolic Balance')
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No metabolic data', ha='center', va='center')

    # Plot 4: Light statistics (if available)
    ax = axes[1, 1]
    if 'env_stats' in stats_list[0]:
        mean_light = [s['env_stats'].get('mean_light', 1.0) for s in stats_list]
        shadowed = [s['env_stats'].get('shadowed_voxels', 0) for s in stats_list]

        ax2 = ax.twinx()
        ax.plot(iterations, mean_light, 'y-', label='Mean light')
        ax2.plot(iterations, shadowed, 'k--', label='Shadowed voxels')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Light Intensity', color='y')
        ax2.set_ylabel('Shadowed Voxels', color='k')
        ax.set_title('Light Environment')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No environmental data', ha='center', va='center')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig, axes


def animate_growth(history_file: str, output_file: Optional[str] = None,
                  fps: int = 5):
    """
    Create animation from simulation history.

    Args:
        history_file: JSON file from simulator.export_history()
        output_file: Optional output video file (.mp4, .gif)
        fps: Frames per second
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    except ImportError:
        print("Error: animation support not available")
        return

    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)

    if not history:
        print("No history to animate")
        return

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate global bounds
    all_vertices = []
    for frame in history:
        if frame['vertices']:
            all_vertices.extend(frame['vertices'])

    all_vertices = np.array(all_vertices)

    max_range = np.array([
        all_vertices[:, 0].max() - all_vertices[:, 0].min(),
        all_vertices[:, 1].max() - all_vertices[:, 1].min(),
        all_vertices[:, 2].max() - all_vertices[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_vertices[:, 0].max() + all_vertices[:, 0].min()) * 0.5
    mid_y = (all_vertices[:, 1].max() + all_vertices[:, 1].min()) * 0.5
    mid_z = (all_vertices[:, 2].max() + all_vertices[:, 2].min()) * 0.5

    def update(frame_idx):
        ax.clear()

        frame = history[frame_idx]
        vertices = np.array(frame['vertices'])
        edges = frame['edges']

        if len(vertices) > 0:
            for start_idx, end_idx, width in edges:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    start = vertices[start_idx]
                    end = vertices[end_idx]

                    ax.plot([start[0], end[0]],
                           [start[1], end[1]],
                           [start[2], end[2]],
                           'g-', linewidth=max(0.5, width * 10))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Iteration {frame['iteration']}")

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000/fps)

    if output_file:
        if output_file.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps)

        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
    else:
        plt.show()

    return anim


def plot_growth_graph(graph, title: str = "Growth Graph"):
    """
    Plot growth graph with resource levels.

    Args:
        graph: GrowthGraph instance
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return

    nodes = graph.get_all_nodes()

    if not nodes:
        print("No nodes to plot")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Get resource levels for coloring
    resources = np.array([n.resource for n in nodes])
    max_resource = max(resources.max(), 0.1)

    # Plot nodes
    for node in nodes:
        # Color based on resource level
        color_val = node.resource / max_resource
        color = plt.cm.viridis(color_val)

        # Plot node
        ax.scatter([node.position[0]], [node.position[1]], [node.position[2]],
                  c=[color], s=100 * node.width, marker='o')

        # Plot connections to children
        for child in node.children:
            ax.plot([node.position[0], child.position[0]],
                   [node.position[1], child.position[1]],
                   [node.position[2], child.position[2]],
                   'k-', linewidth=node.width * 2, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                              norm=plt.Normalize(vmin=0, vmax=max_resource))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Resource Level')

    plt.show()

    return fig, ax
