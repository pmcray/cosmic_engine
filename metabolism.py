"""
Metabolic resource allocation for L-system growth.

Implements:
- Pipe Model Theory for vascular scaling
- Signal-based resource distribution (ABOP Chapter 3)
- Resource-limited growth with feedback
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from growth_node import GrowthNode, GrowthGraph
from environment import Environment


class MetabolicModel:
    """Manages resource production, distribution, and consumption."""

    def __init__(self, environment: Optional[Environment] = None):
        """
        Initialize metabolic model.

        Args:
            environment: Optional environment for light-based resource production
        """
        self.environment = environment

        # Model parameters
        self.photosynthetic_efficiency = 1.0  # η - conversion efficiency
        self.respiration_cost = 0.05          # Maintenance cost per unit biomass
        self.growth_threshold = 0.1           # Minimum resource for growth
        self.transport_resistance = 0.01      # ρ - decay with distance
        self.apical_dominance = 1.2           # λ - main vs lateral branch preference
        self.pipe_exponent = 2.0              # n - vascular scaling exponent

    def calculate_photosynthesis(self, node: GrowthNode) -> float:
        """
        Calculate photosynthetic production for a node.

        P = L × η where L is light intensity and η is efficiency

        Args:
            node: Growth node (typically a leaf)

        Returns:
            Photosynthetic production
        """
        if not node.is_active:
            return 0.0

        # Production based on light and efficiency
        production = node.light_intensity * self.photosynthetic_efficiency

        # Leaves produce more than stems
        if node.is_leaf:
            production *= 2.0

        return production

    def calculate_respiration(self, node: GrowthNode) -> float:
        """
        Calculate maintenance respiration cost.

        Cost proportional to biomass (width × length)

        Args:
            node: Growth node

        Returns:
            Respiration cost
        """
        if not node.is_active:
            return 0.0

        biomass = node.width * node.length
        return biomass * self.respiration_cost

    def update_light_levels(self, graph: GrowthGraph):
        """
        Update light intensity for all nodes based on environment.

        Args:
            graph: Growth graph
        """
        if self.environment is None:
            return

        nodes = graph.get_all_nodes()

        for node in nodes:
            # Get light at node position
            intensities = self.environment.get_light_intensity(
                node.position.reshape(1, 3)
            )
            node.light_intensity = intensities[0]

    def calculate_resource_balance(self, graph: GrowthGraph) -> dict:
        """
        Calculate total resource production and consumption.

        Args:
            graph: Growth graph

        Returns:
            Dictionary with production, consumption, and net balance
        """
        nodes = graph.get_all_nodes()

        total_production = sum(self.calculate_photosynthesis(node) for node in nodes)
        total_respiration = sum(self.calculate_respiration(node) for node in nodes)

        net_balance = total_production - total_respiration

        return {
            'production': total_production,
            'respiration': total_respiration,
            'net_balance': net_balance,
            'active_nodes': len([n for n in nodes if n.is_active])
        }

    def distribute_resources(self, graph: GrowthGraph, total_supply: Optional[float] = None):
        """
        Distribute resources through the growth graph.

        Uses dual-signal mechanism:
        1. Basipetal (up): Demand signals from leaves to root
        2. Acropetal (down): Supply distribution from root to tips

        Args:
            graph: Growth graph
            total_supply: Total resource to distribute (if None, calculated from photosynthesis)
        """
        # Calculate total production if not provided
        if total_supply is None:
            balance = self.calculate_resource_balance(graph)
            total_supply = max(0.0, balance['net_balance'])

        # Distribute using the graph's method
        graph.distribute_resources(
            total_supply=total_supply,
            resistance=self.transport_resistance,
            dominance_lambda=self.apical_dominance
        )

    def apply_growth_rules(self, graph: GrowthGraph) -> List[GrowthNode]:
        """
        Determine which nodes can grow based on resource levels.

        Args:
            graph: Growth graph

        Returns:
            List of nodes with sufficient resources to grow
        """
        nodes = graph.get_all_nodes()
        can_grow = []

        for node in nodes:
            if not node.is_active:
                continue

            # Check if node has enough resources to grow
            if node.resource >= self.growth_threshold:
                can_grow.append(node)
            elif node.resource < 0.01:
                # Mark as inactive if starved
                node.is_active = False

        return can_grow

    def prune_shaded_branches(self, graph: GrowthGraph, light_threshold: float = 0.15):
        """
        Remove branches that are too shaded to survive.

        Args:
            graph: Growth graph
            light_threshold: Minimum light level for survival
        """
        nodes = graph.get_all_nodes()

        for node in nodes:
            if node.light_intensity < light_threshold:
                # Check if this node and all descendants are shaded
                descendants = node.get_all_nodes()
                all_shaded = all(n.light_intensity < light_threshold for n in descendants)

                if all_shaded:
                    node.is_active = False

        # Remove dead branches
        graph.prune_dead_branches()

    def update_pipe_widths(self, graph: GrowthGraph):
        """
        Update branch widths based on Pipe Model Theory.

        S_parent^n = sum(S_child^n)

        Args:
            graph: Growth graph
        """
        graph.update_widths(self.pipe_exponent)

    def set_pipe_exponent(self, exponent: float):
        """
        Set pipe model exponent for different morphologies.

        Args:
            exponent: Pipe model exponent
                n = 2.0: Earth-like trees (Leonardo da Vinci)
                n < 2.0: Pre-historic mega-flora (thick trunks)
                n > 2.5: Alien structures (spindly)
        """
        self.pipe_exponent = exponent

    def simulate_step(self, graph: GrowthGraph) -> dict:
        """
        Perform one metabolic simulation step.

        1. Update light levels from environment
        2. Calculate production and consumption
        3. Distribute resources
        4. Apply growth rules
        5. Update widths
        6. Prune shaded branches

        Args:
            graph: Growth graph

        Returns:
            Statistics for this step
        """
        # Update light levels
        self.update_light_levels(graph)

        # Calculate resource balance
        balance = self.calculate_resource_balance(graph)

        # Distribute resources
        self.distribute_resources(graph, balance['net_balance'])

        # Determine which nodes can grow
        growable_nodes = self.apply_growth_rules(graph)

        # Update widths based on pipe model
        self.update_pipe_widths(graph)

        # Prune heavily shaded branches
        self.prune_shaded_branches(graph)

        return {
            **balance,
            'growable_nodes': len(growable_nodes),
            'total_nodes': len(graph.get_all_nodes())
        }


class ResourceLimitedGrowth:
    """Helper for resource-limited L-system production rules."""

    def __init__(self, growth_graph: GrowthGraph, metabolic_model: MetabolicModel):
        self.graph = growth_graph
        self.model = metabolic_model
        self.node_map = {}  # Map symbols to nodes

    def can_produce(self, symbol_index: int, min_resource: float = 0.1) -> bool:
        """
        Check if a symbol has enough resources to produce.

        Args:
            symbol_index: Index of symbol in L-system string
            min_resource: Minimum resource threshold

        Returns:
            True if symbol can produce
        """
        if symbol_index not in self.node_map:
            return True  # Default to allowing production

        node = self.node_map[symbol_index]
        return node.is_active and node.resource >= min_resource

    def get_resource_for_symbol(self, symbol_index: int) -> float:
        """Get resource level for a symbol."""
        if symbol_index not in self.node_map:
            return 1.0

        return self.node_map[symbol_index].resource

    def allocate_to_offspring(self, parent_index: int, num_offspring: int,
                            ratios: Optional[List[float]] = None) -> List[float]:
        """
        Allocate parent's resources to offspring.

        Args:
            parent_index: Parent symbol index
            num_offspring: Number of offspring symbols
            ratios: Optional allocation ratios (must sum to ~1.0)

        Returns:
            List of resource allocations for each offspring
        """
        parent_resource = self.get_resource_for_symbol(parent_index)

        if ratios is None:
            # Equal distribution
            ratios = [1.0 / num_offspring] * num_offspring
        else:
            # Normalize ratios
            total = sum(ratios)
            ratios = [r / total for r in ratios]

        return [parent_resource * r for r in ratios]


def create_metabolic_rule_condition(graph: GrowthGraph, metabolic_model: MetabolicModel,
                                   min_resource: float = 0.1):
    """
    Create a condition function for resource-limited production rules.

    Usage in L-system:
        condition = create_metabolic_rule_condition(graph, model)
        rule = ProductionRule(
            predecessor='A',
            successor=lambda ctx, params: [...],
            condition=condition
        )

    Args:
        graph: Growth graph
        metabolic_model: Metabolic model
        min_resource: Minimum resource for growth

    Returns:
        Condition function for use in ProductionRule
    """
    def condition(context: dict, params: List[float]) -> bool:
        # Get resource from params (if provided)
        if len(params) > 0:
            resource = params[0]
            return resource >= min_resource

        # Default to allowing growth
        return True

    return condition
