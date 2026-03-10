#!/usr/bin/env python3
"""
Chronos: L-System Growth Animation Engine

Command-line interface for generating smooth growth animations
from L-systems with temporal coherence and AI synthesis.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from lexer import LSystemParser
from rewriter import LSystemRewriter
from animation_generator import AnimationGenerator
from video_factory import VideoFactory, check_video_dependencies, print_video_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chronos: L-System Growth Animation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate depth map sequence (no AI)
  python chronos.py --file examples/simple_tree.l --iter 4 --duration 10 --maps-only --output anim/

  # Generate AI-synthesized video
  python chronos.py --file examples/3d_tree.l --iter 5 --duration 15 --style prehistoric --video output.mp4

  # Export OBJ sequence for external rendering
  python chronos.py --file examples/tree.l --iter 4 --duration 8 --obj-sequence --output frames/

  # Quick preview (limited frames, no AI)
  python chronos.py --file examples/simple_tree.l --iter 3 --preview --max-frames 60 --output preview.mp4

  # Generate keyframes only
  python chronos.py --file examples/tree.l --iter 5 --keyframes --output keyframes/
        """
    )

    # Input
    parser.add_argument('--file', '-f',
                       help='L-system file (.l)')

    # Animation parameters
    parser.add_argument('--iter', '-n', type=int, default=4,
                       help='Maximum iterations (default: 4)')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                       help='Animation duration in seconds (default: 10.0)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frames per second (default: 30.0)')
    parser.add_argument('--terminal-age', type=float, default=1.0,
                       help='Default terminal age for symbols (default: 1.0)')

    # Output modes
    parser.add_argument('--output', '-o', default='chronos_output/',
                       help='Output directory or filename (default: chronos_output/)')
    parser.add_argument('--maps-only', action='store_true',
                       help='Generate depth maps only (no AI synthesis)')
    parser.add_argument('--obj-sequence', action='store_true',
                       help='Export OBJ file sequence')
    parser.add_argument('--video', metavar='FILE',
                       help='Generate video file')
    parser.add_argument('--keyframes', action='store_true',
                       help='Export keyframes only (one per iteration)')
    parser.add_argument('--preview', action='store_true',
                       help='Generate quick preview (limited frames, no AI)')
    parser.add_argument('--stabilize-video', metavar='FILE',
                       help='Stabilize an existing video file (e.g., Stargate simulation)')
    parser.add_argument('--use-svd', action='store_true',
                       help='Use Stable Video Diffusion for stabilization')

    # Preview options
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames for preview mode (default: 100)')

    # AI synthesis options
    parser.add_argument('--style', default='prehistoric',
                       help='Aesthetic style preset (default: prehistoric)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for AI generation')
    parser.add_argument('--temporal-strength', type=float, default=0.15,
                       help='Temporal coherence strength (default: 0.15)')
    parser.add_argument('--list-styles', action='store_true',
                       help='List available aesthetic styles')

    # Camera
    parser.add_argument('--camera', default='iso',
                       help='Camera preset: iso, top, front, closeup (default: iso)')

    # Resolution
    parser.add_argument('--resolution', type=int, default=1024,
                       help='Output resolution (default: 1024)')

    # System info
    parser.add_argument('--system-info', action='store_true',
                       help='Print system information and exit')

    # Verbose
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics')

    return parser.parse_args()


def load_lsystem(filename: str, verbose: bool = False):
    """Load L-system from file."""
    if verbose:
        print(f"Loading L-system from {filename}...")

    parser = LSystemParser()
    axiom, rules, context = parser.parse_file(filename)

    if verbose:
        print(f"Loaded L-system: {len(rules)} rules")
        print(f"Axiom: {len(axiom)} symbols")

    # Add axiom and rules to context for easy access
    context.axiom = axiom
    context.rules = rules

    return context


def create_animation_generator(lsystem, args) -> AnimationGenerator:
    """Create animation generator from L-system."""
    rewriter = LSystemRewriter(lsystem.rules, lsystem)

    gen = AnimationGenerator(
        rewriter,
        lsystem.axiom,
        duration=args.duration,
        fps=args.fps,
        default_terminal_age=args.terminal_age,
        max_iterations=args.iter
    )

    return gen


def generate_maps_only(gen: AnimationGenerator, args):
    """Generate depth map sequence only."""
    print(f"\n=== Generating Depth Map Sequence ===")
    print(f"Duration: {args.duration}s @ {args.fps} fps")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Camera: {args.camera}")

    # Generate iterations
    gen.generate_timed_iterations(args.iter, verbose=args.verbose)

    # Create video factory
    video_factory = VideoFactory(gen, resolution=args.resolution, use_diffusion=False)

    # Generate maps
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_factory.generate_map_sequence(
        str(output_dir),
        camera_preset=args.camera,
        verbose=args.verbose
    )

    print(f"\nDepth maps saved to {output_dir}/")


def generate_obj_sequence(gen: AnimationGenerator, args):
    """Generate OBJ file sequence."""
    print(f"\n=== Generating OBJ Sequence ===")
    print(f"Duration: {args.duration}s @ {args.fps} fps")
    print(f"Total frames: {gen.clock.total_frames}")

    # Generate iterations
    gen.generate_timed_iterations(args.iter, verbose=args.verbose)

    # Export OBJ sequence
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen.export_trajectory_to_obj_sequence(
        str(output_dir),
        prefix="frame",
        verbose=args.verbose
    )

    print(f"\nOBJ sequence saved to {output_dir}/")


def generate_keyframes(gen: AnimationGenerator, args):
    """Generate keyframes only."""
    print(f"\n=== Generating Keyframes ===")
    print(f"Iterations: {args.iter}")

    # Generate iterations
    gen.generate_timed_iterations(args.iter, verbose=args.verbose)

    # Export keyframes
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen.preview_keyframes(str(output_dir), verbose=args.verbose)

    print(f"\nKeyframes saved to {output_dir}/")


def generate_video(gen: AnimationGenerator, args):
    """Generate video with AI synthesis."""
    print(f"\n=== Generating AI Video ===")
    print(f"Duration: {args.duration}s @ {args.fps} fps")
    print(f"Style: {args.style}")
    print(f"Resolution: {args.resolution}x{args.resolution}")

    # Check dependencies
    deps = check_video_dependencies()
    if not (deps['opencv'] or deps['ffmpeg']):
        print("\nERROR: No video encoding backend available!")
        print("Install OpenCV: pip install opencv-python")
        print("Or install FFmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)

    # Generate iterations
    gen.generate_timed_iterations(args.iter, verbose=args.verbose)

    # Create video factory
    video_factory = VideoFactory(gen, resolution=args.resolution, use_diffusion=True)

    # Generate video
    output_file = args.video if args.video else str(Path(args.output) / "animation.mp4")

    try:
        video_factory.generate_video_with_ai(
            output_file,
            preset=args.style,
            camera_preset=args.camera,
            temporal_strength=args.temporal_strength,
            seed=args.seed,
            verbose=args.verbose
        )

        print(f"\nVideo saved to {output_file}")

    except Exception as e:
        print(f"\nERROR generating AI video: {e}")
        print("\nTrying map-only video instead...")

        # Fallback to map-only video
        video_factory.generate_map_video_only(
            output_file,
            camera_preset=args.camera,
            verbose=args.verbose
        )


def generate_preview(gen: AnimationGenerator, args):
    """Generate quick preview video."""
    print(f"\n=== Generating Preview ===")
    print(f"Max frames: {args.max_frames}")

    # Generate iterations
    gen.generate_timed_iterations(args.iter, verbose=args.verbose)

    # Create video factory
    video_factory = VideoFactory(gen, resolution=args.resolution, use_diffusion=False)

    # Generate preview
    output_file = str(Path(args.output) / "preview.mp4")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    video_factory.export_preview_video(
        output_file,
        max_frames=args.max_frames,
        verbose=args.verbose
    )

    print(f"\nPreview saved to {output_file}")


def generate_stabilized_video(gen: Optional[AnimationGenerator], args):
    """Stabilize an existing video using SVD/img2img."""
    print(f"\n=== Stabilizing Video ===")
    print(f"Input Video: {args.stabilize_video}")
    print(f"Style: {args.style}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Use SVD: {args.use_svd}")

    # Check dependencies
    deps = check_video_dependencies()
    if not (deps['opencv'] or deps['ffmpeg']):
        print("\nERROR: No video encoding backend available!")
        sys.exit(1)

    # Create video factory
    video_factory = VideoFactory(gen, resolution=args.resolution, use_diffusion=True)

    # Generate video
    output_file = args.video if args.video else str(Path(args.output) / "stabilized_output.mp4")

    try:
        video_factory.stabilize_existing_video(
            input_video=args.stabilize_video,
            output_file=output_file,
            preset=args.style,
            temporal_strength=args.temporal_strength,
            seed=args.seed,
            use_svd=args.use_svd,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\nERROR stabilizing video: {e}")


def list_styles():
    """List available aesthetic styles."""
    from prompt_engine import PromptEngine

    engine = PromptEngine()
    presets = engine.list_presets()

    print("\n=== Available Aesthetic Styles ===\n")

    for name in presets:
        preset = engine.get_preset(name)
        print(f"{name:20s} - {preset.positive_prompt[:60]}...")

    print(f"\nTotal: {len(presets)} styles\n")


def print_statistics(gen: AnimationGenerator):
    """Print animation statistics."""
    stats = gen.get_statistics()

    print("\n=== Animation Statistics ===")
    print(f"Duration: {stats['duration']:.2f}s")
    print(f"FPS: {stats['fps']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Iterations: {stats['iterations']}")

    print("\nSymbols per iteration:")
    for i, count in enumerate(stats['symbols_per_iteration']):
        print(f"  Iteration {i}: {count} symbols")

    if 'cache_stats' in stats:
        cache = stats['cache_stats']
        print(f"\nCache: {cache['cached_iterations']} iterations, "
              f"{cache['cached_geometries']} geometries")

    print("=" * 30 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle special commands
    if args.system_info:
        print_video_system_info()
        sys.exit(0)

    if args.list_styles:
        list_styles()
        sys.exit(0)

    # Validate input file (not required for system-info, list-styles, or stabilize-video)
    if not args.file and not args.stabilize_video:
        print("ERROR: --file or --stabilize-video argument is required")
        sys.exit(1)

    if args.file and not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)
        
    if args.stabilize_video and not Path(args.stabilize_video).exists():
        print(f"ERROR: Video file not found: {args.stabilize_video}")
        sys.exit(1)

    print("=" * 50)
    print("  CHRONOS: L-System Growth Animation Engine")
    print("=" * 50)

    gen = None
    if args.file:
        # Load L-system
        lsystem = load_lsystem(args.file, args.verbose)
    
        # Create animation generator
        gen = create_animation_generator(lsystem, args)

    # Determine output mode
    if args.stabilize_video:
        generate_stabilized_video(gen, args)

    elif args.keyframes:
        generate_keyframes(gen, args)

    elif args.obj_sequence:
        generate_obj_sequence(gen, args)

    elif args.preview:
        generate_preview(gen, args)

    elif args.video:
        generate_video(gen, args)

    elif args.maps_only:
        generate_maps_only(gen, args)

    else:
        # Default: generate maps
        print("\nNo output mode specified. Generating depth maps.")
        print("Use --help to see available options.")
        generate_maps_only(gen, args)

    # Print statistics if requested
    if args.stats:
        print_statistics(gen)

    print("\n" + "=" * 50)
    print("  Animation generation complete!")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
