r"""
3D Turtle graphics interpreter for L-systems.

Coordinate system:
- H (Heading): Forward direction (initially +Y)
- L (Left): Left direction (initially +X)
- U (Up): Up direction (initially +Z)

Rotation symbols (from ABOP Chapter 1, Section 1.3):
- +, -: Rotation around U axis (yaw)
- &, ^: Rotation around L axis (pitch)
- /, \: Rotation around H axis (roll)
- [ : Push state onto stack
- ] : Pop state from stack
- { : Start a polygon
- . : Record a vertex for the current polygon
- } : End a polygon
- F(length): Move forward and draw
- f(length): Move forward without drawing
- ! (width): Set line width
- ~ (surface): Draw a predefined surface (TBD)
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from symbol import Symbol


@dataclass
class TurtleState:
    """State of the 3D turtle."""
    position: np.ndarray
    heading: np.ndarray  # H vector
    left: np.ndarray     # L vector
    up: np.ndarray       # U vector
    width: float

    def copy(self):
        """Create a copy of this state."""
        return TurtleState(
            position=self.position.copy(),
            heading=self.heading.copy(),
            left=self.left.copy(),
            up=self.up.copy(),
            width=self.width
        )


@dataclass
class Segment:
    """A line segment with start point, end point, and width."""
    start: np.ndarray
    end: np.ndarray
    width: float

    def copy(self):
        """Create a copy of this segment."""
        return Segment(
            start=self.start.copy(),
            end=self.end.copy(),
            width=self.width
        )


class Turtle3D:
    """3D turtle graphics interpreter."""

    def __init__(self, default_angle: float = 25.7, default_length: float = 1.0, default_width: float = 0.1):
        """
        Initialize the 3D turtle.

        Args:
            default_angle: Default rotation angle in degrees
            default_length: Default step length in meters
            default_width: Default line width
        """
        self.default_angle = np.radians(default_angle)
        self.default_length = default_length
        self.default_width = default_width

        # Initialize state
        self.state = TurtleState(
            position=np.array([0.0, 0.0, 0.0]),
            heading=np.array([0.0, 1.0, 0.0]),   # +Y
            left=np.array([1.0, 0.0, 0.0]),      # +X
            up=np.array([0.0, 0.0, 1.0]),        # +Z
            width=default_width
        )

        self.stack = []
        self.vertices = []
        self.edges = []
        self.segments = []  # List of Segment objects
        self.polygons = []  # List of lists of vertex indices
        self.current_polygon = None  # None or list of vertex indices
        self.current_vertex_index = 0

    def reset(self):
        """Reset the turtle to initial state."""
        self.state = TurtleState(
            position=np.array([0.0, 0.0, 0.0]),
            heading=np.array([0.0, 1.0, 0.0]),
            left=np.array([1.0, 0.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
            width=self.default_width
        )
        self.stack = []
        self.vertices = []
        self.edges = []
        self.segments = []
        self.polygons = []
        self.current_polygon = None
        self.current_vertex_index = 0

    def push(self):
        """Push current state onto stack."""
        self.stack.append(self.state.copy())

    def pop(self):
        """Pop state from stack."""
        if self.stack:
            self.state = self.stack.pop()

    def yaw(self, angle: float):
        """Rotate around Up axis (yaw)."""
        self._rotate_u(np.radians(angle))

    def pitch(self, angle: float):
        """Rotate around Left axis (pitch)."""
        self._rotate_l(np.radians(angle))

    def roll(self, angle: float):
        """Rotate around Heading axis (roll)."""
        self._rotate_h(np.radians(angle))

    def _rotate_to_vertical(self):
        """Rotate heading to align with the up direction (vertical)."""
        # Make heading point in the direction of the up vector
        self.state.heading = np.array([0.0, 0.0, 1.0])
        self.state.left = np.array([1.0, 0.0, 0.0])
        self.state.up = np.array([0.0, 1.0, 0.0])
        self._normalize_vectors()

    def interpret(self, symbols: List[Symbol]):
        """
        Interpret a list of symbols and generate geometry.

        Args:
            symbols: List of Symbol objects to interpret
        """
        for symbol in symbols:
            self._process_symbol(symbol)

    def _process_symbol(self, symbol: Symbol):
        """Process a single symbol."""
        char = symbol.char
        params = symbol.params if symbol.params else []

        if char == 'F':
            # Move forward and draw
            length = params[0] if params else self.default_length
            self._draw_forward(length)

        elif char == 'f':
            # Move forward without drawing
            length = params[0] if params else self.default_length
            self._move_forward(length)

        elif char == '+':
            # Rotate around U (yaw left)
            angle = np.radians(params[0]) if params else self.default_angle
            self._rotate_u(angle)

        elif char == '-':
            # Rotate around U (yaw right)
            angle = np.radians(params[0]) if params else self.default_angle
            self._rotate_u(-angle)

        elif char == '&':
            # Rotate around L (pitch down)
            angle = np.radians(params[0]) if params else self.default_angle
            self._rotate_l(angle)

        elif char == '^':
            # Rotate around L (pitch up)
            angle = np.radians(params[0]) if params else self.default_angle
            self._rotate_l(-angle)

        elif char == '/':
            # Rotate around H (roll left)
            angle = np.radians(params[0]) if params else self.default_angle
            self._rotate_h(angle)

        elif char == '\\':
            # Rotate around H (roll right)
            angle = np.radians(params[0]) if params else self.default_angle
            self._rotate_h(-angle)

        elif char == '[':
            # Push state
            self.push()

        elif char == ']':
            # Pop state
            self.pop()

        elif char == '!':
            # Set width
            if params:
                self.state.width = params[0]

        elif char == '{':
            # Start polygon
            self.current_polygon = []

        elif char == '.':
            # Record vertex for polygon
            if self.current_polygon is not None:
                self.vertices.append(self.state.position.copy())
                self.current_polygon.append(self.current_vertex_index)
                self.current_vertex_index += 1

        elif char == '}':
            # End polygon
            if self.current_polygon is not None:
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon)
                self.current_polygon = None

    def _draw_forward(self, length: float):
        """Move forward and draw a line."""
        start_pos = self.state.position.copy()
        start_idx = self.current_vertex_index

        # Add start vertex
        self.vertices.append(start_pos)
        self.current_vertex_index += 1

        # Move forward
        self.state.position += self.state.heading * length

        # Add end vertex
        self.vertices.append(self.state.position.copy())
        end_idx = self.current_vertex_index
        self.current_vertex_index += 1

        # Add edge
        self.edges.append((start_idx, end_idx, self.state.width))

        # Add segment
        self.segments.append(Segment(start=start_pos, end=self.state.position.copy(), width=self.state.width))

    def _move_forward(self, length: float):
        """Move forward without drawing."""
        self.state.position += self.state.heading * length

    def _rotate_u(self, angle: float):
        """Rotate around U axis (yaw)."""
        rotation_matrix = self._rotation_matrix(self.state.up, angle)
        self.state.heading = rotation_matrix @ self.state.heading
        self.state.left = rotation_matrix @ self.state.left
        self._normalize_vectors()

    def _rotate_l(self, angle: float):
        """Rotate around L axis (pitch)."""
        rotation_matrix = self._rotation_matrix(self.state.left, angle)
        self.state.heading = rotation_matrix @ self.state.heading
        self.state.up = rotation_matrix @ self.state.up
        self._normalize_vectors()

    def _rotate_h(self, angle: float):
        """Rotate around H axis (roll)."""
        rotation_matrix = self._rotation_matrix(self.state.heading, angle)
        self.state.left = rotation_matrix @ self.state.left
        self.state.up = rotation_matrix @ self.state.up
        self._normalize_vectors()

    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Create a rotation matrix for rotation around an arbitrary axis.

        Uses Rodrigues' rotation formula.

        Args:
            axis: Rotation axis (normalized)
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle

        x, y, z = axis

        rotation = np.array([
            [cos_angle + x*x*one_minus_cos,     x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
            [y*x*one_minus_cos + z*sin_angle,   cos_angle + y*y*one_minus_cos,   y*z*one_minus_cos - x*sin_angle],
            [z*x*one_minus_cos - y*sin_angle,   z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos]
        ])

        return rotation

    def _normalize_vectors(self):
        """Normalize the H, L, U vectors to prevent drift."""
        self.state.heading = self.state.heading / np.linalg.norm(self.state.heading)
        self.state.left = self.state.left / np.linalg.norm(self.state.left)
        self.state.up = self.state.up / np.linalg.norm(self.state.up)

    def get_vertices(self) -> np.ndarray:
        """Get all vertices as a numpy array."""
        if not self.vertices:
            return np.array([])
        return np.array(self.vertices)

    def get_edges(self) -> List[Tuple[int, int, float]]:
        """Get all edges as a list of (start_idx, end_idx, width) tuples."""
        return self.edges

    def get_polygons(self) -> List[List[int]]:
        """Get all polygons as a list of lists of vertex indices."""
        return self.polygons

    def to_obj(self, filename: str):
        """Export geometry to OBJ file."""
        from obj_exporter import export_to_obj
        export_to_obj(filename, self.vertices, self.edges, self.polygons)

    def get_statistics(self) -> dict:
        """Get statistics about the generated geometry."""
        vertices = self.get_vertices()

        if len(vertices) == 0:
            return {
                'num_vertices': 0,
                'num_edges': 0,
                'total_length': 0.0,
                'bounds': None
            }

        # Calculate bounds
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)

        # Calculate total length
        total_length = 0.0
        for start_idx, end_idx, width in self.edges:
            start = vertices[start_idx]
            end = vertices[end_idx]
            total_length += np.linalg.norm(end - start)

        return {
            'num_vertices': len(vertices),
            'num_edges': len(self.edges),
            'num_polygons': len(self.polygons),
            'total_length': total_length,
            'bounds': {
                'min': min_bounds,
                'max': max_bounds,
                'size': max_bounds - min_bounds
            }
        }
