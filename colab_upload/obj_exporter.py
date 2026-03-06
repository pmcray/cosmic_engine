"""
OBJ file exporter for 3D geometry.
"""
import numpy as np
from typing import List, Tuple


def export_to_obj(filename: str, vertices: List[np.ndarray], edges: List[Tuple[int, int, float]]):
    """
    Export vertices and edges to OBJ file format.

    Args:
        filename: Output filename
        vertices: List of vertex positions
        edges: List of (start_idx, end_idx, width) tuples
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# L-System Core OBJ Export\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Edges: {len(edges)}\n\n")

        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        f.write("\n")

        # Write edges as lines (OBJ line elements)
        for start_idx, end_idx, width in edges:
            # OBJ indices are 1-based
            f.write(f"l {start_idx + 1} {end_idx + 1}\n")


def export_to_ply(filename: str, vertices: List[np.ndarray], edges: List[Tuple[int, int, float]]):
    """
    Export vertices and edges to PLY file format.

    Args:
        filename: Output filename
        vertices: List of vertex positions
        edges: List of (start_idx, end_idx, width) tuples
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property float width\n")
        f.write("end_header\n")

        # Write vertices
        for vertex in vertices:
            f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        # Write edges
        for start_idx, end_idx, width in edges:
            f.write(f"{start_idx} {end_idx} {width:.6f}\n")


def export_cylinders_to_obj(filename: str, vertices: List[np.ndarray],
                            edges: List[Tuple[int, int, float]],
                            segments: int = 8):
    """
    Export edges as cylindrical meshes to OBJ file.

    Args:
        filename: Output filename
        vertices: List of vertex positions
        edges: List of (start_idx, end_idx, width) tuples
        segments: Number of segments around cylinder circumference
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# L-System Core OBJ Export (Cylinders)\n")
        f.write(f"# Original Vertices: {len(vertices)}\n")
        f.write(f"# Edges: {len(edges)}\n\n")

        vertex_offset = 0

        for start_idx, end_idx, width in edges:
            start_pos = vertices[start_idx]
            end_pos = vertices[end_idx]

            # Create cylinder mesh
            cylinder_vertices, cylinder_faces = create_cylinder(
                start_pos, end_pos, width / 2, segments
            )

            # Write vertices
            for v in cylinder_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (with offset)
            for face in cylinder_faces:
                f.write(f"f {face[0] + vertex_offset + 1} {face[1] + vertex_offset + 1} {face[2] + vertex_offset + 1}\n")

            vertex_offset += len(cylinder_vertices)


def create_cylinder(start: np.ndarray, end: np.ndarray, radius: float, segments: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Create a cylinder mesh between two points.

    Args:
        start: Start position
        end: End position
        radius: Cylinder radius
        segments: Number of segments around circumference

    Returns:
        Tuple of (vertices, faces)
    """
    vertices = []
    faces = []

    # Calculate cylinder direction
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return [], []

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))

    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # Create ring vertices
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)

        # Start ring
        vertices.append(start + offset)
        # End ring
        vertices.append(end + offset)

    # Create faces
    for i in range(segments):
        next_i = (i + 1) % segments

        start_idx1 = i * 2
        end_idx1 = i * 2 + 1
        start_idx2 = next_i * 2
        end_idx2 = next_i * 2 + 1

        # Two triangles per segment
        faces.append((start_idx1, end_idx1, start_idx2))
        faces.append((start_idx2, end_idx1, end_idx2))

    return vertices, faces


def export_segments_to_obj(segments, filename: str):
    """
    Export Segment objects to OBJ file.

    Args:
        segments: List of Segment objects
        filename: Output filename
    """
    # Convert segments to vertices and edges
    vertices = []
    edges = []

    for seg in segments:
        start_idx = len(vertices)
        vertices.append(seg.start)
        vertices.append(seg.end)
        edges.append((start_idx, start_idx + 1, seg.width))

    # Export using existing function
    export_to_obj(filename, vertices, edges)
