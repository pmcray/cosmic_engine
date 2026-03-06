"""
3D Turtle with tropism support for bio-simulation.

Extends the basic Turtle3D with:
- Gravitropism and phototropism
- Growth graph construction
- Resource-aware interpretation
"""
import numpy as np
from typing import List, Optional
from turtle_3d import Turtle3D, TurtleState
from symbol import Symbol
from growth_node import GrowthNode, GrowthGraph
from forces import TropismCalculator, apply_tropism_to_frame
from environment import Environment


class TropismTurtle3D(Turtle3D):
    """3D turtle with tropism and growth graph generation."""

    def __init__(self, default_angle: float = 25.7, default_length: float = 1.0,
                 default_width: float = 0.1, environment: Optional[Environment] = None):
        """
        Initialize tropism-enabled turtle.

        Args:
            default_angle: Default rotation angle in degrees
            default_length: Default step length
            default_width: Default line width
            environment: Optional environment for phototropism
        """
        super().__init__(default_angle, default_length, default_width)

        # Tropism calculator
        self.tropism_calc = TropismCalculator(environment)
        self.environment = environment

        # Tropism parameters
        self.gravity_susceptibility = 0.0    # e_gravity
        self.photo_susceptibility = 0.0      # e_photo
        self.positive_gravitropism = False   # Grow toward vs away from gravity

        # Growth graph construction
        self.build_graph = False
        self.growth_graph = None
        self.current_node = None
        self.node_stack = []

        # Tropism application frequency
        self.tropism_segments = 1  # Apply tropism every N segments

    def enable_graph_building(self):
        """Enable growth graph construction during interpretation."""
        self.build_graph = True

        # Create root node
        root = GrowthNode(
            symbol=Symbol('ROOT', []),
            position=self.state.position.copy(),
            heading=self.state.heading.copy(),
            left=self.state.left.copy(),
            up=self.state.up.copy(),
            width=self.state.width
        )
        self.growth_graph = GrowthGraph(root)
        self.current_node = root

    def set_tropism(self, gravity: float = 0.0, photo: float = 0.0,
                   positive_gravity: bool = False):
        """
        Set tropism parameters.

        Args:
            gravity: Gravitropism susceptibility (0.0 to 1.0)
            photo: Phototropism susceptibility (0.0 to 1.0)
            positive_gravity: If True, grow toward gravity (roots)
        """
        self.gravity_susceptibility = gravity
        self.photo_susceptibility = photo
        self.positive_gravitropism = positive_gravity

    def _apply_tropism(self):
        """Apply tropism to current heading."""
        if (abs(self.gravity_susceptibility) < 1e-6 and
            abs(self.photo_susceptibility) < 1e-6):
            return

        # Calculate new heading
        new_heading = self.tropism_calc.calculate_combined_tropism(
            heading=self.state.heading,
            position=self.state.position,
            gravity_susceptibility=self.gravity_susceptibility,
            photo_susceptibility=self.photo_susceptibility,
            positive_gravitropism=self.positive_gravitropism
        )

        # Update frame to maintain orthogonality
        self.state.heading, self.state.left, self.state.up = apply_tropism_to_frame(
            self.state.heading, self.state.left, self.state.up, new_heading
        )

    def _create_growth_node(self, symbol: Symbol, length: float) -> Optional[GrowthNode]:
        """Create a growth node for this segment."""
        if not self.build_graph:
            return None

        # Create node
        node = GrowthNode(
            symbol=symbol,
            position=self.state.position.copy(),
            heading=self.state.heading.copy(),
            left=self.state.left.copy(),
            up=self.state.up.copy(),
            width=self.state.width,
            length=length
        )

        # Determine if this is a leaf/growth tip
        # (Will be refined during graph analysis)
        node.is_leaf = symbol.char in ['A', 'B', 'X', 'Y']  # Typical growth symbols

        # Get light intensity if environment available
        if self.environment:
            intensity = self.environment.get_light_intensity(
                node.position.reshape(1, 3)
            )
            node.light_intensity = intensity[0]

        # Add to graph
        if self.current_node is not None:
            self.current_node.add_child(node)

        return node

    def _draw_forward(self, length: float, symbol: Optional[Symbol] = None):
        """Override to add tropism and graph building."""
        # Apply tropism before moving
        self._apply_tropism()

        # Create growth node if building graph
        if self.build_graph and symbol is not None:
            node = self._create_growth_node(symbol, length)
            prev_node = self.current_node
            self.current_node = node

        # Call parent implementation
        super()._draw_forward(length)

        # Restore current node for graph building
        if self.build_graph and symbol is not None:
            self.current_node = prev_node

    def _process_symbol(self, symbol: Symbol):
        """Override to handle tropism symbols and graph building."""
        char = symbol.char
        params = symbol.params if symbol.params else []

        # Standard turtle commands (from parent)
        if char in ['F', 'f', '+', '-', '&', '^', '/', '\\', '!']:
            super()._process_symbol(symbol)

        # Stack operations with graph support
        elif char == '[':
            self.stack.append(self.state.copy())
            if self.build_graph:
                self.node_stack.append(self.current_node)

        elif char == ']':
            if self.stack:
                self.state = self.stack.pop()
            if self.build_graph and self.node_stack:
                self.current_node = self.node_stack.pop()

        # Tropism control symbols (new)
        elif char == '$':
            # Roll to vertical (align U with gravity)
            up_target = -self.tropism_calc.gravity_vector
            self.state.up = up_target / np.linalg.norm(up_target)
            self.state.left = np.cross(self.state.up, self.state.heading)
            self.state.left = self.state.left / np.linalg.norm(self.state.left)

        elif char == '%':
            # Cut off remainder of branch (for competitive growth)
            if self.build_graph:
                # Mark subsequent nodes as pruned
                pass

        # Leaf marker
        elif char == '~':
            # Mark as leaf for metabolism
            if self.build_graph and self.current_node:
                self.current_node.is_leaf = True

        # Resource parameter (for metabolic L-systems)
        elif char in ['A', 'B', 'C', 'X', 'Y']:
            # These are typically growth symbols
            # Just mark in graph, don't draw
            if self.build_graph:
                node = GrowthNode(
                    symbol=symbol,
                    position=self.state.position.copy(),
                    heading=self.state.heading.copy(),
                    left=self.state.left.copy(),
                    up=self.state.up.copy(),
                    width=self.state.width,
                    is_leaf=True
                )

                if params:
                    node.resource = params[0]  # First param is resource

                if self.environment:
                    intensity = self.environment.get_light_intensity(
                        node.position.reshape(1, 3)
                    )
                    node.light_intensity = intensity[0]

                if self.current_node:
                    self.current_node.add_child(node)

    def interpret_with_tropism(self, symbols: List[Symbol],
                              gravity: float = 0.0, photo: float = 0.0):
        """
        Interpret symbols with tropism enabled.

        Args:
            symbols: List of symbols to interpret
            gravity: Gravitropism susceptibility
            photo: Phototropism susceptibility
        """
        self.set_tropism(gravity, photo)
        self.interpret(symbols)

    def interpret_and_build_graph(self, symbols: List[Symbol],
                                  gravity: float = 0.0, photo: float = 0.0) -> GrowthGraph:
        """
        Interpret symbols and build growth graph.

        Args:
            symbols: List of symbols to interpret
            gravity: Gravitropism susceptibility
            photo: Phototropism susceptibility

        Returns:
            Growth graph
        """
        self.enable_graph_building()
        self.set_tropism(gravity, photo)
        self.interpret(symbols)

        # Mark leaf nodes (nodes with no children)
        if self.growth_graph:
            for node in self.growth_graph.get_all_nodes():
                if not node.children:
                    node.is_leaf = True

        return self.growth_graph

    def get_growth_graph(self) -> Optional[GrowthGraph]:
        """Get the constructed growth graph."""
        return self.growth_graph


def interpret_with_environment(symbols: List[Symbol], environment: Environment,
                               gravity_susceptibility: float = 0.3,
                               photo_susceptibility: float = 0.2,
                               default_angle: float = 25.7) -> tuple:
    """
    Convenience function to interpret symbols with environment and tropism.

    Args:
        symbols: L-system symbols
        environment: Environment for light and tropism
        gravity_susceptibility: Gravitropism strength
        photo_susceptibility: Phototropism strength
        default_angle: Default rotation angle

    Returns:
        Tuple of (turtle, growth_graph)
    """
    turtle = TropismTurtle3D(
        default_angle=default_angle,
        environment=environment
    )

    graph = turtle.interpret_and_build_graph(
        symbols,
        gravity=gravity_susceptibility,
        photo=photo_susceptibility
    )

    return turtle, graph
