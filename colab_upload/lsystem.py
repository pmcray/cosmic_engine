#!/usr/bin/env python3
"""
L-System Core - A high-performance parametric L-system engine.

Usage:
    python lsystem.py --file tree.l --iter 7 --out result.obj
"""
import argparse
import sys
import time
from pathlib import Path

from lexer import LSystemParser
from rewriter import LSystemRewriter
from turtle_3d import Turtle3D
from obj_exporter import export_to_obj, export_to_ply, export_cylinders_to_obj


def main():
    parser = argparse.ArgumentParser(
        description='L-System Core - Parametric L-system engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lsystem.py --file examples/tree.l --iter 5 --out tree.obj
  python lsystem.py --file examples/plant.l --iter 7 --out plant.ply --format ply
  python lsystem.py --file examples/algae.l --iter 3 --verbose
        """
    )

    parser.add_argument('--file', '-f', required=True, help='Input L-system file')
    parser.add_argument('--iter', '-i', type=int, default=1, help='Number of iterations')
    parser.add_argument('--out', '-o', help='Output file (OBJ/PLY)')
    parser.add_argument('--format', choices=['obj', 'ply', 'cylinders'], default='obj',
                       help='Output format (default: obj)')
    parser.add_argument('--angle', type=float, help='Default rotation angle in degrees')
    parser.add_argument('--length', type=float, default=1.0, help='Default step length')
    parser.add_argument('--width', type=float, default=0.1, help='Default line width')
    parser.add_argument('--cylinder-segments', type=int, default=8,
                       help='Number of segments for cylinder output')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--stats', '-s', action='store_true', help='Show statistics')
    parser.add_argument('--print-string', '-p', action='store_true',
                       help='Print final L-system string')

    args = parser.parse_args()

    # Check input file
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"Error: Input file '{args.file}' not found", file=sys.stderr)
        return 1

    try:
        # Parse L-system
        if args.verbose:
            print(f"Parsing L-system from {args.file}...")

        parser_obj = LSystemParser()
        axiom, rules, context = parser_obj.parse_file(str(input_file))

        if args.verbose:
            print(f"  Axiom: {string_to_str(axiom)}")
            print(f"  Rules: {len(rules)}")
            print(f"  Context variables: {context.variables}")

        # Get angle from context or arguments
        if args.angle:
            default_angle = args.angle
        else:
            default_angle = context.get('angle', 25.7)

        # Rewrite L-system
        if args.verbose:
            print(f"\nRewriting for {args.iter} iterations...")

        start_time = time.time()
        rewriter = LSystemRewriter(rules, context)
        result = rewriter.rewrite(axiom, args.iter)
        rewrite_time = time.time() - start_time

        if args.verbose:
            print(f"  Time: {rewrite_time:.3f}s")
            print(f"  String length: {len(result)} symbols")

        if args.print_string:
            print(f"\nFinal string:")
            print(string_to_str(result))

        # Interpret with 3D turtle
        if args.verbose:
            print(f"\nInterpreting with 3D turtle...")

        start_time = time.time()
        turtle = Turtle3D(default_angle, args.length, args.width)
        turtle.interpret(result)
        interpret_time = time.time() - start_time

        if args.verbose:
            print(f"  Time: {interpret_time:.3f}s")

        # Show statistics
        if args.stats or args.verbose:
            stats = turtle.get_statistics()
            print(f"\nGeometry Statistics:")
            print(f"  Vertices: {stats['num_vertices']}")
            print(f"  Edges: {stats['num_edges']}")
            print(f"  Total length: {stats['total_length']:.2f} m")

            if stats['bounds']:
                bounds = stats['bounds']
                print(f"  Bounds:")
                print(f"    Min: [{bounds['min'][0]:.2f}, {bounds['min'][1]:.2f}, {bounds['min'][2]:.2f}]")
                print(f"    Max: [{bounds['max'][0]:.2f}, {bounds['max'][1]:.2f}, {bounds['max'][2]:.2f}]")
                print(f"    Size: [{bounds['size'][0]:.2f}, {bounds['size'][1]:.2f}, {bounds['size'][2]:.2f}]")

        # Export to file
        if args.out:
            if args.verbose:
                print(f"\nExporting to {args.out}...")

            vertices = turtle.get_vertices()
            edges = turtle.get_edges()

            if len(vertices) == 0:
                print("Warning: No geometry to export", file=sys.stderr)
            else:
                if args.format == 'ply':
                    export_to_ply(args.out, vertices, edges)
                elif args.format == 'cylinders':
                    export_cylinders_to_obj(args.out, vertices, edges, args.cylinder_segments)
                else:
                    export_to_obj(args.out, vertices, edges)

                if args.verbose:
                    print(f"  Exported to {args.out}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


def string_to_str(symbols):
    """Helper function to convert symbol list to string."""
    return ''.join(str(s) for s in symbols)


if __name__ == '__main__':
    sys.exit(main())
