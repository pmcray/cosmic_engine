"""
Bio-Sim simulation orchestrator.

Integrates L-system rewriting with environmental simulation,
tropism, and metabolic resource allocation.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

from symbol import Symbol
from rewriter import LSystemRewriter
from lexer import LSystemParser
from environment import Environment
from metabolism import MetabolicModel
from turtle_tropism import TropismTurtle3D
from growth_node import GrowthGraph
from forces import TropismCalculator
from obj_exporter import export_to_obj, export_cylinders_to_obj


class BioSimulator:
    """Main simulation orchestrator for environmentally-aware L-system growth."""

    def __init__(self, axiom: List[Symbol], rules: List, context: dict,
                 bounds: Tuple[float, float, float, float, float, float] = (-50, 50, -50, 50, 0, 100),
                 voxel_resolution: float = 1.0,
                 fluid_engine=None):
        """
        Initialize the bio-simulator.

        Args:
            axiom: Initial L-system string
            rules: Production rules
            context: L-system context
            bounds: Simulation bounds (min_x, max_x, min_y, max_y, min_z, max_z)
            voxel_resolution: Voxel size for light simulation
            fluid_engine: Optional fluid solver for fluid tropism
        """
        # L-system components
        self.axiom = axiom
        self.rules = rules
        self.context = context
        self.rewriter = LSystemRewriter(rules, context)

        # Current state
        self.current_string = axiom
        self.iteration = 0

        # Environment
        self.environment = Environment(bounds, voxel_resolution)
        self.fluid_engine = fluid_engine

        # Metabolic model
        self.metabolic_model = MetabolicModel(self.environment)

        # Simulation parameters
        self.default_angle = context.get('angle', 25.7)
        self.default_length = context.get('length', 1.0)
        self.gravity_susceptibility = context.get('gravity', 0.0)
        self.photo_susceptibility = context.get('phototropism', 0.0)
        self.fluid_susceptibility = context.get('fluid', 0.0)
        self.pipe_exponent = context.get('pipe_exponent', 2.0)

        # Set pipe exponent
        self.metabolic_model.set_pipe_exponent(self.pipe_exponent)

        # History for animation
        self.history = []
        self.save_history = False

    def step(self, apply_tropism: bool = True, update_shadows: bool = True) -> dict:
        """
        Perform one simulation step.

        1. Rewrite L-system
        2. Interpret with turtle (optionally with tropism)
        3. Update environment (cast shadows)
        4. Calculate resource allocation
        5. Update growth graph

        Args:
            apply_tropism: Whether to apply tropism forces
            update_shadows: Whether to update shadow casting

        Returns:
            Statistics for this step
        """
        # Step fluid simulation if available
        if self.fluid_engine is not None:
            self.fluid_engine.step(self.iteration)

        # Rewrite L-system
        self.current_string = self.rewriter.rewrite(self.current_string, 1)
        self.iteration += 1

        # Interpret with tropism-enabled turtle
        turtle = TropismTurtle3D(
            default_angle=self.default_angle,
            default_length=self.default_length,
            environment=self.environment,
            fluid_engine=self.fluid_engine
        )

        # Build growth graph
        if apply_tropism:
            growth_graph = turtle.interpret_and_build_graph(
                self.current_string,
                gravity=self.gravity_susceptibility,
                photo=self.photo_susceptibility,
                fluid=self.fluid_susceptibility
            )
        else:
            turtle.enable_graph_building()
            turtle.interpret(self.current_string)
            growth_graph = turtle.get_growth_graph()

        # Update environment
        if update_shadows and growth_graph:
            # Reset shadows
            self.environment.reset_light()

            # Cast shadows from all segments
            vertices = turtle.get_vertices()
            edges = turtle.get_edges()

            segments = []
            for start_idx, end_idx, width in edges:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    segments.append((
                        vertices[start_idx],
                        vertices[end_idx],
                        width
                    ))

            self.environment.cast_shadows_from_segments(segments)

        # Run metabolic simulation
        if growth_graph:
            metabolic_stats = self.metabolic_model.simulate_step(growth_graph)
        else:
            metabolic_stats = {}

        # Get geometry statistics
        geom_stats = turtle.get_statistics()

        # Combine statistics
        stats = {
            'iteration': self.iteration,
            'string_length': len(self.current_string),
            **geom_stats,
            **metabolic_stats,
            'env_stats': self.environment.get_statistics()
        }

        # Save to history if enabled
        if self.save_history:
            self.history.append({
                'iteration': self.iteration,
                'stats': stats,
                'vertices': turtle.get_vertices().tolist() if len(turtle.get_vertices()) > 0 else [],
                'edges': turtle.get_edges()
            })

        return stats

    def simulate(self, iterations: int, apply_tropism: bool = True,
                update_shadows: bool = True, verbose: bool = False) -> List[dict]:
        """
        Run simulation for multiple iterations.

        Args:
            iterations: Number of iterations to run
            apply_tropism: Whether to apply tropism
            update_shadows: Whether to update shadows
            verbose: Print progress

        Returns:
            List of statistics for each iteration
        """
        all_stats = []

        for i in range(iterations):
            if verbose:
                print(f"Iteration {self.iteration + 1}...")

            stats = self.step(apply_tropism, update_shadows)
            all_stats.append(stats)

            if verbose:
                print(f"  String length: {stats['string_length']}")
                print(f"  Vertices: {stats.get('num_vertices', 0)}")
                if 'production' in stats:
                    print(f"  Production: {stats['production']:.2f}")
                    print(f"  Net balance: {stats['net_balance']:.2f}")

        return all_stats

    def get_final_geometry(self) -> Tuple[TropismTurtle3D, Optional[GrowthGraph]]:
        """
        Get final turtle and growth graph.

        Returns:
            Tuple of (turtle, growth_graph)
        """
        turtle = TropismTurtle3D(
            default_angle=self.default_angle,
            default_length=self.default_length,
            environment=self.environment,
            fluid_engine=self.fluid_engine
        )

        growth_graph = turtle.interpret_and_build_graph(
            self.current_string,
            gravity=self.gravity_susceptibility,
            photo=self.photo_susceptibility,
            fluid=self.fluid_susceptibility
        )

        return turtle, growth_graph

    def export_obj(self, filename: str, use_cylinders: bool = False):
        """Export final geometry to OBJ file."""
        turtle, _ = self.get_final_geometry()

        vertices = turtle.get_vertices()
        edges = turtle.get_edges()

        if len(vertices) == 0:
            print("Warning: No geometry to export")
            return

        if use_cylinders:
            export_cylinders_to_obj(filename, vertices, edges)
        else:
            export_to_obj(filename, vertices, edges)

    def export_history(self, filename: str):
        """Export simulation history for animation."""
        if not self.history:
            print("Warning: No history to export. Enable save_history first.")
            return

        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)

    def export_light_slice(self, z_level: float, filename: str):
        """Export horizontal light slice."""
        self.environment.export_light_slice(z_level, filename)

    def enable_history_saving(self):
        """Enable saving simulation history for animation."""
        self.save_history = True
        self.history = []


def load_biosim_from_file(filename: str, bounds: Optional[Tuple] = None,
                          voxel_resolution: float = 1.0,
                          fluid_engine = None) -> BioSimulator:
    """
    Load L-system and create bio-simulator.

    Args:
        filename: L-system file
        bounds: Optional custom bounds
        voxel_resolution: Voxel resolution
        fluid_engine: Optional fluid solver instance

    Returns:
        Initialized BioSimulator
    """
    parser = LSystemParser()
    axiom, rules, context = parser.parse_file(filename)

    # Use default bounds if not provided
    if bounds is None:
        bounds = (-50, 50, -50, 50, 0, 100)

    simulator = BioSimulator(axiom, rules, context, bounds, voxel_resolution, fluid_engine)

    # Update parameters from context
    if 'gravity' in context.variables:
        simulator.gravity_susceptibility = context.variables['gravity']
    if 'phototropism' in context.variables:
        simulator.photo_susceptibility = context.variables['phototropism']
    if 'fluid' in context.variables:
        simulator.fluid_susceptibility = context.variables['fluid']
    if 'pipe_exponent' in context.variables:
        simulator.pipe_exponent = context.variables['pipe_exponent']
        simulator.metabolic_model.set_pipe_exponent(simulator.pipe_exponent)

    return simulator


class CompetitiveGrowthSimulation:
    """Simulate multiple L-systems competing for light in shared environment."""

    def __init__(self, bounds: Tuple, voxel_resolution: float = 1.0):
        self.environment = Environment(bounds, voxel_resolution)
        self.organisms = []
        self.bounds = bounds
        self.iteration = 0

    def add_organism(self, axiom: List[Symbol], rules: List, context: dict,
                    start_position: np.ndarray):
        """Add an organism to the simulation."""
        # Modify context to include start position
        context_copy = context.copy()

        simulator = BioSimulator(
            axiom, rules, context_copy,
            self.bounds,
            voxel_resolution=1.0
        )

        # Share environment
        simulator.environment = self.environment

        self.organisms.append({
            'simulator': simulator,
            'start_position': start_position
        })

    def step(self) -> List[dict]:
        """Simulate one step for all organisms."""
        # Reset environment
        self.environment.reset_light()

        # Step each organism
        all_stats = []
        for org in self.organisms:
            stats = org['simulator'].step(
                apply_tropism=True,
                update_shadows=False  # Do shadows after all organisms
            )
            all_stats.append(stats)

        # Now cast all shadows
        for org in self.organisms:
            turtle, _ = org['simulator'].get_final_geometry()
            vertices = turtle.get_vertices()
            edges = turtle.get_edges()

            segments = []
            for start_idx, end_idx, width in edges:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    segments.append((
                        vertices[start_idx],
                        vertices[end_idx],
                        width
                    ))

            self.environment.cast_shadows_from_segments(segments)

        self.iteration += 1
        return all_stats

    def simulate(self, iterations: int, verbose: bool = False) -> List[List[dict]]:
        """Run competitive simulation."""
        history = []

        for i in range(iterations):
            if verbose:
                print(f"\nCompetitive iteration {self.iteration + 1}...")

            stats = self.step()
            history.append(stats)

            if verbose:
                for j, s in enumerate(stats):
                    print(f"  Organism {j+1}: {s.get('string_length', 0)} symbols")

        return history
