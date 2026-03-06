#!/usr/bin/env python3
"""
The Factory - Aesthetic Synthesis Engine CLI.

Phase 3: Transform L-system geometry into high-fidelity AI-generated imagery
using Stable Diffusion and ControlNet.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json

from map_generator import MapGenerator, create_multiple_views
from diffusion_client import DiffusionClient, check_diffusers_availability, print_system_info
from prompt_engine import PromptEngine


def main():
    parser = argparse.ArgumentParser(
        description='The Factory - Aesthetic Synthesis Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate depth/normal/edge maps only
  python factory.py --input tree.obj --maps-only --output maps/

  # Generate with prehistoric style
  python factory.py --input tree.obj --style prehistoric --output gallery/

  # Multiple camera angles
  python factory.py --input tree.obj --style alien --angles front,iso,closeup --output gallery/

  # With refinement pass
  python factory.py --input tree.obj --style bioluminescent --refine --output gallery/

  # High resolution
  python factory.py --input tree.obj --style alien_creature --resolution 2048 --output gallery/

  # Custom prompt
  python factory.py --input tree.obj --custom-prompt "ancient organism" --style prehistoric --output gallery/

  # List available styles
  python factory.py --list-styles

  # Check system capabilities
  python factory.py --system-info
        """
    )

    # Input/Output
    parser.add_argument('--input', '-i', help='Input OBJ file from L-system')
    parser.add_argument('--output', '-o', help='Output directory')

    # Style and prompts
    parser.add_argument('--style', '-s', help='Aesthetic preset style')
    parser.add_argument('--list-styles', action='store_true', help='List available styles')
    parser.add_argument('--custom-prompt', help='Custom base description to add to style')

    # Map generation
    parser.add_argument('--maps-only', action='store_true', help='Only generate maps, skip AI generation')
    parser.add_argument('--angles', help='Camera angles (comma-separated: front,iso,top,side,closeup)')
    parser.add_argument('--map-resolution', type=int, default=1024, help='Map resolution (default: 1024)')

    # Diffusion parameters
    parser.add_argument('--resolution', type=int, help='Output image resolution')
    parser.add_argument('--steps', type=int, help='Number of inference steps')
    parser.add_argument('--guidance', type=float, help='Guidance scale')
    parser.add_argument('--controlnet-scale', type=float, help='ControlNet conditioning scale')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    # Post-processing
    parser.add_argument('--refine', action='store_true', help='Apply refinement pass')
    parser.add_argument('--refinement-strength', type=float, default=0.25, help='Refinement strength (0-1)')
    parser.add_argument('--upscale', type=float, help='Upscale factor (e.g., 2.0)')

    # Model options
    parser.add_argument('--model', help='Stable Diffusion model ID')
    parser.add_argument('--controlnet', help='ControlNet model ID')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto', help='Device to use')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 (use FP32)')

    # Batch processing
    parser.add_argument('--batch', help='Batch process multiple OBJ files (glob pattern)')
    parser.add_argument('--batch-dir', help='Directory containing OBJ files')

    # Output options
    parser.add_argument('--save-maps', action='store_true', help='Save intermediate maps')
    parser.add_argument('--save-config', action='store_true', help='Save generation config JSON')
    parser.add_argument('--prefix', help='Output filename prefix')

    # System info
    parser.add_argument('--system-info', action='store_true', help='Print system information')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # System info
    if args.system_info:
        print_system_info()
        return 0

    # List styles
    if args.list_styles:
        prompt_engine = PromptEngine()
        print("\n=== Available Aesthetic Styles ===\n")

        for name, description in prompt_engine.get_preset_descriptions().items():
            print(f"{name:20s} - {description}")

        print("\nUse --style <name> to apply a style")
        return 0

    # Validate input
    if not args.input and not args.batch and not args.batch_dir:
        print("Error: Must specify --input, --batch, or --batch-dir", file=sys.stderr)
        parser.print_help()
        return 1

    if not args.maps_only and not args.style:
        print("Error: Must specify --style or use --maps-only", file=sys.stderr)
        return 1

    # Check diffusers availability
    if not args.maps_only and not check_diffusers_availability():
        print("Error: Diffusers library required for AI generation", file=sys.stderr)
        print("Install with: pip install diffusers transformers accelerate", file=sys.stderr)
        return 1

    try:
        # Determine input files
        input_files = []

        if args.input:
            input_files.append(args.input)
        elif args.batch:
            import glob
            input_files = glob.glob(args.batch)
        elif args.batch_dir:
            input_files = list(Path(args.batch_dir).glob("*.obj"))

        if not input_files:
            print("Error: No input files found", file=sys.stderr)
            return 1

        # Parse camera angles
        if args.angles:
            angles = [a.strip() for a in args.angles.split(',')]
        else:
            angles = ['iso']  # Default

        # Create output directory
        output_dir = Path(args.output) if args.output else Path('factory_output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each input file
        for input_file in input_files:
            print(f"\n{'='*60}")
            print(f"Processing: {input_file}")
            print(f"{'='*60}\n")

            input_path = Path(input_file)
            if not input_path.exists():
                print(f"Warning: {input_file} not found, skipping", file=sys.stderr)
                continue

            # Generate maps for each angle
            for angle in angles:
                print(f"\nRendering from {angle} view...")

                # Map generation
                map_gen = MapGenerator(args.map_resolution, args.map_resolution)

                prefix = args.prefix or input_path.stem
                maps_prefix = output_dir / f"{prefix}_{angle}"

                if args.save_maps or args.maps_only:
                    save_prefix = str(maps_prefix)
                else:
                    save_prefix = None

                maps = map_gen.generate_all_maps(
                    str(input_file),
                    angle,
                    save_prefix
                )

                print(f"  Maps generated: {maps['metadata']['num_vertices']} vertices, "
                      f"{maps['metadata']['num_edges']} edges")

                # Skip AI generation if maps-only
                if args.maps_only:
                    continue

                # Initialize diffusion client
                print("\nInitializing diffusion client...")

                model_id = args.model or "stabilityai/stable-diffusion-xl-base-1.0"
                controlnet_id = args.controlnet or "diffusers/controlnet-depth-sdxl-1.0"

                client = DiffusionClient(
                    model_id=model_id,
                    controlnet_id=controlnet_id,
                    device=args.device,
                    use_fp16=not args.no_fp16
                )

                # Get preset
                prompt_engine = PromptEngine()
                preset = prompt_engine.get_preset(args.style)

                if preset is None:
                    print(f"Error: Style '{args.style}' not found", file=sys.stderr)
                    print(f"Available styles: {', '.join(prompt_engine.list_presets())}")
                    return 1

                # Override parameters if specified
                if args.steps:
                    preset.num_inference_steps = args.steps
                if args.guidance:
                    preset.guidance_scale = args.guidance
                if args.controlnet_scale:
                    preset.controlnet_scale = args.controlnet_scale

                # Generate image
                print(f"\nGenerating with '{args.style}' style...")

                if args.refine:
                    initial, refined = client.generate_with_refinement(
                        depth_map=maps['depth'],
                        preset=preset,
                        base_description=args.custom_prompt or "",
                        refinement_strength=args.refinement_strength,
                        seed=args.seed,
                        width=args.resolution,
                        height=args.resolution
                    )

                    # Save initial
                    initial_file = output_dir / f"{prefix}_{angle}_{args.style}_initial.png"
                    initial.save(initial_file)
                    print(f"  Initial saved: {initial_file}")

                    # Save refined
                    output_file = output_dir / f"{prefix}_{angle}_{args.style}.png"
                    refined.save(output_file)
                    final_image = refined

                else:
                    generated = client.generate_from_preset(
                        depth_map=maps['depth'],
                        preset=preset,
                        base_description=args.custom_prompt or "",
                        seed=args.seed,
                        width=args.resolution,
                        height=args.resolution
                    )

                    output_file = output_dir / f"{prefix}_{angle}_{args.style}.png"
                    generated.save(output_file)
                    final_image = generated

                print(f"  Generated image saved: {output_file}")

                # Upscale if requested
                if args.upscale:
                    print(f"\nUpscaling by {args.upscale}x...")
                    upscaled = client.upscale_image(final_image, args.upscale)

                    upscaled_file = output_dir / f"{prefix}_{angle}_{args.style}_upscaled.png"
                    upscaled.save(upscaled_file)
                    print(f"  Upscaled saved: {upscaled_file}")

                # Save config
                if args.save_config:
                    config = {
                        'input_file': str(input_file),
                        'style': args.style,
                        'angle': angle,
                        'preset': preset.to_dict(),
                        'custom_prompt': args.custom_prompt,
                        'seed': args.seed,
                        'resolution': args.resolution,
                        'refine': args.refine,
                        'upscale': args.upscale
                    }

                    config_file = output_dir / f"{prefix}_{angle}_{args.style}_config.json"
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)

                # Cleanup
                client.unload()

        print(f"\n{'='*60}")
        print(f"Processing complete! Output in: {output_dir}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
