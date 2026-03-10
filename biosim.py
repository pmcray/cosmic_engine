#!/usr/bin/env python3
"""
Bio-Sim - Environmentally-aware L-system simulation.

Extends the basic L-system Core with:
- Voxel-based light simulation
- Gravitropism and phototropism
- Resource allocation and metabolic constraints
- Competitive growth simulation
"""
import argparse
import sys
from pathlib import Path
import json

from simulator import load_biosim_from_file, BioSimulator, CompetitiveGrowthSimulation
from lexer import LSystemParser
import taichi as ti

# Try to import fluid solver
try:
    from physics.fluid_solver import FluidEngine
    HAS_FLUIDS = True
except ImportError:
    HAS_FLUIDS = False


def main():
    parser = argparse.ArgumentParser(
        description='Bio-Sim - Environmentally-aware L-system simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation with tropism
  python biosim.py --file examples/tropism_tree.l --iter 5 --out tree.obj

  # Enable shadow casting and resource allocation
  python biosim.py --file examples/metabolic_plant.l --iter 7 --shadows --metabolism

  # Export animation sequence
  python biosim.py --file examples/growth.l --iter 10 --save-history --out-history anim.json

  # High-resolution voxel grid
  python biosim.py --file examples/tree.l --iter 5 --voxel-res 0.25 --shadows

  # Fluid-based Tropism
  python biosim.py --file examples/plant.l --iter 5 --fluid-tropism --planet-type jupiter
        """
    )

    parser.add_argument('--file', '-f', required=True, help='Input L-system file')
    parser.add_argument('--iter', '-i', type=int, default=5, help='Number of iterations')
    parser.add_argument('--out', '-o', help='Output OBJ file')

    # Environment options
    parser.add_argument('--bounds', help='Simulation bounds: min_x,max_x,min_y,max_y,min_z,max_z')
    parser.add_argument('--voxel-res', type=float, default=1.0, help='Voxel resolution (meters)')

    # Simulation options
    parser.add_argument('--shadows', action='store_true', help='Enable shadow casting')
    parser.add_argument('--metabolism', action='store_true', help='Enable metabolic simulation')
    parser.add_argument('--no-tropism', action='store_true', help='Disable tropism forces')

    # Tropism parameters
    parser.add_argument('--gravity', type=float, help='Gravitropism susceptibility (0-1)')
    parser.add_argument('--photo', type=float, help='Phototropism susceptibility (0-1)')
    parser.add_argument('--pipe-exp', type=float, help='Pipe model exponent (default 2.0)')
    
    # Fluid parameters
    parser.add_argument('--fluid-tropism', action='store_true', help='Enable fluid-based tropism (wind/currents)')
    parser.add_argument('--fluid', type=float, help='Fluid tropism susceptibility (0-1)')
    parser.add_argument('--planet-type', type=str, default='jupiter', help='Planet type for fluid simulation')

    # Output options
    parser.add_argument('--cylinders', action='store_true', help='Export as cylindrical meshes')
    parser.add_argument('--save-history', action='store_true', help='Save simulation history')
    parser.add_argument('--out-history', help='Output history JSON file')
    parser.add_argument('--export-light', help='Export light field slice (z-level)')
    parser.add_argument('--light-output', help='Light slice output file')

    # Display options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--stats', '-s', action='store_true', help='Show detailed statistics')
    parser.add_argument('--plot', action='store_true', help='Plot final structure (requires matplotlib)')

    args = parser.parse_args()

    # Check input file
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"Error: Input file '{args.file}' not found", file=sys.stderr)
        return 1

    try:
        # Initialize Taichi and FluidEngine if requested
        fluid_engine = None
        if args.fluid_tropism or args.fluid is not None:
            if not HAS_FLUIDS:
                print("Warning: FluidEngine could not be imported. Ensure Taichi is installed.")
            else:
                ti.init(arch=ti.gpu) # try GPU first
                fluid_engine = FluidEngine(planet_type=args.planet_type)
                if args.verbose:
                    print(f"Initialized FluidEngine with planet type: {args.planet_type}")

        # Parse bounds if provided
        bounds = None
        if args.bounds:
            parts = [float(x) for x in args.bounds.split(',')]
            if len(parts) != 6:
                print("Error: Bounds must be 6 values: min_x,max_x,min_y,max_y,min_z,max_z",
                     file=sys.stderr)
                return 1
            bounds = tuple(parts)

        # Load simulator
        if args.verbose:
            print(f"Loading L-system from {args.file}...")

        simulator = load_biosim_from_file(
            str(input_file),
            bounds=bounds,
            voxel_resolution=args.voxel_res,
            fluid_engine=fluid_engine
        )

        # Override tropism parameters if provided
        if args.gravity is not None:
            simulator.gravity_susceptibility = args.gravity
        if args.photo is not None:
            simulator.photo_susceptibility = args.photo
        if args.fluid is not None:
            simulator.fluid_susceptibility = args.fluid
        elif args.fluid_tropism and simulator.fluid_susceptibility == 0.0:
            # Enable with default strength if flag is passed but not specified in L-system or args
            simulator.fluid_susceptibility = 0.5
            
        if args.pipe_exp is not None:
            simulator.pipe_exponent = args.pipe_exp
            simulator.metabolic_model.set_pipe_exponent(args.pipe_exp)

        if args.verbose:
            print(f"  Gravitropism: {simulator.gravity_susceptibility}")
            print(f"  Phototropism: {simulator.photo_susceptibility}")
            print(f"  Fluid Tropism: {simulator.fluid_susceptibility}")
            print(f"  Pipe exponent: {simulator.pipe_exponent}")
            print(f"  Voxel resolution: {args.voxel_res} m")

        # Enable history saving if requested
        if args.save_history:
            simulator.enable_history_saving()

        # Run simulation
        if args.verbose:
            print(f"\nRunning simulation for {args.iter} iterations...")

        apply_tropism = not args.no_tropism
        update_shadows = args.shadows

        all_stats = simulator.simulate(
            iterations=args.iter,
            apply_tropism=apply_tropism,
            update_shadows=update_shadows,
            verbose=args.verbose
        )

        # Show final statistics
        if args.stats or args.verbose:
            final_stats = all_stats[-1] if all_stats else {}

            print(f"\nFinal Statistics:")
            print(f"  Iteration: {final_stats.get('iteration', 0)}")
            print(f"  String length: {final_stats.get('string_length', 0)}")
            print(f"  Vertices: {final_stats.get('num_vertices', 0)}")
            print(f"  Edges: {final_stats.get('num_edges', 0)}")
            print(f"  Total length: {final_stats.get('total_length', 0):.2f} m")

            if 'production' in final_stats:
                print(f"\n  Metabolic:")
                print(f"    Production: {final_stats['production']:.2f}")
                print(f"    Respiration: {final_stats['respiration']:.2f}")
                print(f"    Net balance: {final_stats['net_balance']:.2f}")
                print(f"    Growable nodes: {final_stats.get('growable_nodes', 0)}")

            if 'env_stats' in final_stats:
                env = final_stats['env_stats']
                print(f"\n  Environment:")
                print(f"    Mean light: {env.get('mean_light', 0):.3f}")
                print(f"    Shadowed voxels: {env.get('shadowed_voxels', 0)}")

        # Export geometry
        if args.out:
            if args.verbose:
                print(f"\nExporting geometry to {args.out}...")

            simulator.export_obj(args.out, use_cylinders=args.cylinders)

            if args.verbose:
                print(f"  Exported to {args.out}")

        # Export history
        if args.save_history and args.out_history:
            if args.verbose:
                print(f"\nExporting history to {args.out_history}...")

            simulator.export_history(args.out_history)

            if args.verbose:
                print(f"  Exported {len(simulator.history)} frames")

        # Export light field
        if args.export_light:
            z_level = float(args.export_light)
            output_file = args.light_output or f"light_z{z_level}.npz"

            if args.verbose:
                print(f"\nExporting light field at z={z_level} to {output_file}...")

            simulator.export_light_slice(z_level, output_file)

        # Plot if requested
        if args.plot:
            try:
                from visualization import plot_structure
                if args.verbose:
                    print("\nPlotting structure...")

                turtle, graph = simulator.get_final_geometry()
                plot_structure(turtle, title=f"Bio-Sim: {input_file.name}")

            except ImportError:
                print("Warning: matplotlib not available for plotting", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
