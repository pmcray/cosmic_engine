"""
Growth node graph structure for bio-simulation.

Represents the L-system string as a directed acyclic graph (DAG)
with 3D spatial information and resource allocation data.
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from symbol import Symbol


@dataclass
class GrowthNode:
    """A node in the growth graph representing a symbol with spatial data."""

    symbol: Symbol
    position: np.ndarray  # 3D position in space
    heading: np.ndarray   # H vector (forward direction)
    left: np.ndarray      # L vector (left direction)
    up: np.ndarray        # U vector (up direction)

    # Graph structure
    parent: Optional['GrowthNode'] = None
    children: List['GrowthNode'] = field(default_factory=list)

    # Resource allocation
    resource: float = 1.0         # Current resource level
    demand_signal: float = 0.0    # Basipetal demand signal
    supply_signal: float = 0.0    # Acropetal supply signal

    # Physical properties
    width: float = 0.1            # Branch width/diameter
    length: float = 1.0           # Segment length

    # Environmental interaction
    light_intensity: float = 1.0  # Local light level (0.0 to 1.0)

    # Metabolic state
    is_active: bool = True        # Whether this node can grow
    is_leaf: bool = False         # Whether this is a leaf/growth tip

    def add_child(self, child: 'GrowthNode'):
        """Add a child node."""
        self.children.append(child)
        child.parent = self

    def get_demand(self) -> float:
        """
        Calculate total demand from this node and all descendants.
        Based on light availability and active status.
        """
        if not self.is_active:
            return 0.0

        # Own demand (if leaf/growth tip)
        own_demand = self.light_intensity if self.is_leaf else 0.0

        # Sum demand from children
        child_demand = sum(child.get_demand() for child in self.children)

        self.demand_signal = own_demand + child_demand
        return self.demand_signal

    def receive_resource(self, amount: float):
        """Receive resource from parent."""
        self.resource = amount
        self.supply_signal = amount

    def calculate_pipe_area(self, exponent: float = 2.0) -> float:
        """
        Calculate required cross-sectional area based on Pipe Model Theory.

        Args:
            exponent: Pipe model exponent (n)
                n = 2.0: Standard Earth-like trees (Leonardo da Vinci's rule)
                n < 2.0: Pre-historic mega-flora (thick, heavy trunks)
                n > 2.5: Alien structures (spindly, top-heavy)

        Returns:
            Required cross-sectional area
        """
        if not self.children:
            # Leaf node - minimum area
            return np.pi * (0.01 ** 2)

        # Sum of children's areas (raised to power n)
        child_areas = sum(child.calculate_pipe_area(exponent) for child in self.children)

        # Parent area from pipe model: A_parent^n = sum(A_child^n)
        parent_area = child_areas ** (1.0 / exponent)

        return parent_area

    def update_width_from_pipe_model(self, exponent: float = 2.0):
        """Update width based on pipe model (call after tree is built)."""
        area = self.calculate_pipe_area(exponent)
        self.width = 2.0 * np.sqrt(area / np.pi)  # Diameter from area

        # Recursively update children
        for child in self.children:
            child.update_width_from_pipe_model(exponent)

    def get_all_nodes(self) -> List['GrowthNode']:
        """Get all nodes in subtree (depth-first)."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_leaves(self) -> List['GrowthNode']:
        """Get all leaf nodes in subtree."""
        if self.is_leaf or not self.children:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def prune_inactive(self):
        """Remove inactive branches from the tree."""
        # Recursively prune children first
        for child in self.children[:]:  # Copy list to allow modification
            child.prune_inactive()

            # Remove child if it's inactive and has no active descendants
            if not child.is_active and not child.children:
                self.children.remove(child)

        # Mark self as inactive if no active children and no resources
        if not self.children and self.resource < 0.01:
            self.is_active = False


class GrowthGraph:
    """Manages the entire growth graph structure."""

    def __init__(self, root: GrowthNode):
        self.root = root

    def get_all_nodes(self) -> List[GrowthNode]:
        """Get all nodes in the graph."""
        return self.root.get_all_nodes()

    def get_leaves(self) -> List[GrowthNode]:
        """Get all leaf/growth tip nodes."""
        return self.root.get_leaves()

    def distribute_resources(self, total_supply: float, resistance: float = 0.01,
                            dominance_lambda: float = 1.0):
        """
        Distribute resources through the tree using signal-based partitioning.

        Args:
            total_supply: Total resource to distribute
            resistance: Transport resistance (ρ) - controls decay with distance
            dominance_lambda: Apical dominance factor (λ)
        """
        # First, calculate demand signals (basipetal - from tips to root)
        self.root.get_demand()

        # Then distribute supply (acropetal - from root to tips)
        self._distribute_from_node(self.root, total_supply, resistance, dominance_lambda)

    def _distribute_from_node(self, node: GrowthNode, supply: float,
                             resistance: float, dominance_lambda: float):
        """Recursively distribute resources from a node to its children."""
        node.receive_resource(supply)

        if not node.children:
            return

        # Calculate total demand from all children
        total_demand = sum(child.demand_signal for child in node.children)

        if total_demand < 1e-6:
            return

        # Distribute to children based on demand ratio
        for i, child in enumerate(node.children):
            # Base allocation by demand ratio
            ratio = child.demand_signal / total_demand

            # Apply apical dominance (first/main branch gets more)
            if i == 0:
                ratio *= dominance_lambda

            allocated = supply * ratio

            # Apply transport resistance (decay with distance)
            decayed_supply = allocated * np.exp(-resistance * child.length)

            # Recursively distribute to subtree
            self._distribute_from_node(child, decayed_supply, resistance, dominance_lambda)

    def update_widths(self, pipe_exponent: float = 2.0):
        """Update all node widths based on pipe model."""
        self.root.update_width_from_pipe_model(pipe_exponent)

    def prune_dead_branches(self):
        """Remove inactive/dead branches from the graph."""
        self.root.prune_inactive()

    def get_statistics(self) -> dict:
        """Get statistics about the growth graph."""
        nodes = self.get_all_nodes()
        leaves = self.get_leaves()
        active_nodes = [n for n in nodes if n.is_active]

        total_biomass = sum(n.resource * n.width * n.length for n in nodes)
        total_length = sum(n.length for n in nodes)

        return {
            'total_nodes': len(nodes),
            'active_nodes': len(active_nodes),
            'leaf_nodes': len(leaves),
            'total_biomass': total_biomass,
            'total_length': total_length,
            'avg_resource': np.mean([n.resource for n in active_nodes]) if active_nodes else 0.0,
            'avg_light': np.mean([n.light_intensity for n in nodes])
        }
