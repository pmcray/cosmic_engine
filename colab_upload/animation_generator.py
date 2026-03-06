"""
Animation Trajectory Generator for L-System Growth.

Coordinates L-system rewriting, time management, and frame generation
for smooth growth animations.
"""
import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from pathlib import Path
import json

from symbol import Symbol
from timed_symbol import TimedSymbol, TimedString, create_timed_axiom
from rewriter import LSystemRewriter
from timer import AnimationClock, TimePoint, create_animation_timeline
from timed_turtle import TimedTurtle, IterationCache
from turtle_3d import Segment
from obj_exporter import export_segments_to_obj


class AnimationGenerator:
    """
    Generates animation trajectories from L-systems.

    Coordinates rewriting, time management, and frame generation.
    """

    def __init__(self, rewriter: LSystemRewriter,
                 axiom: List[Symbol],
                 duration: float = 10.0,
                 fps: float = 30.0,
                 default_terminal_age: float = 1.0,
                 max_iterations: int = 5):
        """
        Initialize animation generator.

        Args:
            rewriter: L-system rewriter
            axiom: Starting axiom symbols
            duration: Animation duration (seconds)
            fps: Frames per second
            default_terminal_age: Default terminal age for symbols
            max_iterations: Maximum iterations for animation
        """
        self.rewriter = rewriter
        self.axiom = axiom
        self.duration = duration
        self.fps = fps
        self.default_terminal_age = default_terminal_age

        # Animation clock
        max_iterations = max_iterations
        self.clock = create_animation_timeline(duration, max_iterations, fps)

        # Turtle for interpretation
        self.turtle = TimedTurtle(clock=self.clock)

        # Cache for iterations
        self.cache = IterationCache()

        # Generated iterations
        self.timed_iterations: List[TimedString] = []

        print(f"AnimationGenerator: {duration}s @ {fps} fps, {self.clock.total_frames} frames")

    def generate_timed_iterations(self, max_iterations: int,
                                  verbose: bool = False) -> List[TimedString]:
        """
        Generate all timed L-system iterations.

        Args:
            max_iterations: Maximum iterations to generate
            verbose: Print progress

        Returns:
            List of TimedString for each iteration
        """
        # Convert axiom to timed symbols
        timed_axiom = create_timed_axiom(
            self.axiom,
            default_terminal_age=self.default_terminal_age
        )

        self.timed_iterations = [timed_axiom]

        if verbose:
            print(f"Iteration 0: {len(timed_axiom)} symbols")

        # Perform rewriting
        current = timed_axiom

        for i in range(max_iterations):
            # Rewrite to get next iteration
            next_iteration = self._rewrite_timed_string(current)

            # Update ages - symbols born at this iteration start at age 0
            for symbol in next_iteration.symbols:
                symbol.age = 0.0

            self.timed_iterations.append(next_iteration)

            if verbose:
                print(f"Iteration {i+1}: {len(next_iteration)} symbols")

            # Cache this iteration
            self.cache.store_iteration(i + 1, next_iteration)

            current = next_iteration

        return self.timed_iterations

    def _rewrite_timed_string(self, timed_string: TimedString) -> TimedString:
        """
        Rewrite a timed string using production rules.

        Args:
            timed_string: Input timed string

        Returns:
            Rewritten timed string
        """
        # Convert to base symbols
        base_symbols = timed_string.to_base_symbols()

        # Apply rewriting (one iteration)
        rewritten_symbols = self.rewriter.rewrite(base_symbols, iterations=1)

        # Convert back to timed symbols
        timed_symbols = []
        for sym in rewritten_symbols:
            timed_sym = TimedSymbol(
                char=sym.char,
                age=0.0,
                terminal_age=self.default_terminal_age,
                params=sym.params.copy() if sym.params else []
            )
            timed_symbols.append(timed_sym)

        return TimedString(timed_symbols)

    def generate_frame_trajectory(self, verbose: bool = False) -> List[Tuple[int, TimePoint, List[Segment]]]:
        """
        Generate complete frame-by-frame trajectory.

        Args:
            verbose: Print progress

        Returns:
            List of (frame, time_point, segments) tuples
        """
        if not self.timed_iterations:
            raise ValueError("No iterations generated. Call generate_timed_iterations() first.")

        trajectory = []
        total_frames = self.clock.total_frames

        for frame, time_point in self.clock.iter_frames():
            if verbose and frame % 10 == 0:
                print(f"Generating frame {frame}/{total_frames} "
                      f"(t={time_point.time:.2f}s, iter={time_point.iteration}, α={time_point.alpha:.2f})")

            # Get segments for this frame
            segments = self._generate_frame_geometry(time_point)

            trajectory.append((frame, time_point, segments))

        return trajectory

    def _generate_frame_geometry(self, time_point: TimePoint) -> List[Segment]:
        """
        Generate geometry for a specific frame.

        Args:
            time_point: Time point for this frame

        Returns:
            List of segments
        """
        iter_n = time_point.iteration
        iter_n1 = min(iter_n + 1, len(self.timed_iterations) - 1)

        # Update symbol ages based on current time
        string_n = self._update_symbol_ages(self.timed_iterations[iter_n], time_point.time, iter_n)
        string_n1 = self._update_symbol_ages(self.timed_iterations[iter_n1], time_point.time, iter_n1)

        # Interpolate geometry
        if iter_n == iter_n1:
            # At final iteration, no interpolation needed
            self.turtle.reset()
            self.turtle.set_time_point(time_point)
            self.turtle.interpret_timed_string(string_n)
            return self.turtle.segments.copy()
        else:
            # Interpolate between iterations
            return self.turtle.get_interpolated_geometry(time_point, string_n, string_n1)

    def _update_symbol_ages(self, timed_string: TimedString, current_time: float,
                           birth_iteration: int) -> TimedString:
        """
        Update symbol ages based on current time.

        Args:
            timed_string: String to update
            current_time: Current global time
            birth_iteration: Iteration when symbols were born

        Returns:
            Updated TimedString
        """
        # Create copy
        updated = timed_string.copy()

        # Calculate birth time
        birth_time = birth_iteration * self.clock.time_per_iteration

        # Update each symbol's age
        for symbol in updated.symbols:
            symbol.age = max(0.0, current_time - birth_time)

            # Update maturity status
            symbol.is_mature = symbol.age >= symbol.terminal_age

        return updated

    def export_trajectory_to_obj_sequence(self, output_dir: str,
                                         prefix: str = "frame",
                                         verbose: bool = False):
        """
        Export trajectory as sequence of OBJ files.

        Args:
            output_dir: Output directory
            prefix: Filename prefix
            verbose: Print progress
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trajectory = self.generate_frame_trajectory(verbose=verbose)

        for frame, time_point, segments in trajectory:
            filename = output_path / f"{prefix}_{frame:05d}.obj"

            # Export segments
            export_segments_to_obj(segments, str(filename))

            if verbose and frame % 30 == 0:
                print(f"Exported frame {frame} to {filename}")

        print(f"Exported {len(trajectory)} frames to {output_dir}/")

        # Export metadata
        metadata = {
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': self.clock.total_frames,
            'iterations': len(self.timed_iterations)
        }

        metadata_file = output_path / f"{prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def export_single_frame(self, frame: int, output_file: str):
        """
        Export a single frame to OBJ.

        Args:
            frame: Frame number
            output_file: Output filename
        """
        time_point = self.clock.get_frame_time_point(frame)
        segments = self._generate_frame_geometry(time_point)

        export_segments_to_obj(segments, output_file)

        print(f"Exported frame {frame} to {output_file}")

    def get_frame_geometry(self, frame: int) -> List[Segment]:
        """
        Get geometry for a specific frame.

        Args:
            frame: Frame number

        Returns:
            List of segments
        """
        time_point = self.clock.get_frame_time_point(frame)
        return self._generate_frame_geometry(time_point)

    def preview_keyframes(self, output_dir: str, verbose: bool = False):
        """
        Export only keyframes for quick preview.

        Args:
            output_dir: Output directory
            verbose: Print progress
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate geometry for each iteration (keyframe)
        for i, timed_string in enumerate(self.timed_iterations):
            # Set time to end of iteration
            time = (i + 1) * self.clock.time_per_iteration
            time_point = self.clock.get_time_point(time)

            # Update ages
            updated_string = self._update_symbol_ages(timed_string, time, i)

            # Generate geometry
            self.turtle.reset()
            self.turtle.set_time_point(time_point)
            self.turtle.interpret_timed_string(updated_string)

            # Export
            filename = output_path / f"keyframe_{i:03d}.obj"
            export_segments_to_obj(self.turtle.segments, str(filename))

            if verbose:
                print(f"Exported keyframe {i} (iteration {i}) to {filename}")

        print(f"Exported {len(self.timed_iterations)} keyframes to {output_dir}/")

    def get_statistics(self) -> Dict:
        """Get animation statistics."""
        return {
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': self.clock.total_frames,
            'iterations': len(self.timed_iterations),
            'symbols_per_iteration': [len(ts) for ts in self.timed_iterations],
            'cache_stats': self.cache.get_stats()
        }


class BatchAnimationGenerator:
    """
    Generates multiple animation variants from a single L-system.

    Useful for exploring parameter spaces or generating variations.
    """

    def __init__(self, rewriter: LSystemRewriter):
        self.rewriter = rewriter
        self.generators: List[AnimationGenerator] = []

    def create_duration_variants(self, durations: List[float],
                                fps: float = 30.0) -> List[AnimationGenerator]:
        """
        Create variants with different durations.

        Args:
            durations: List of durations to generate
            fps: Frames per second

        Returns:
            List of AnimationGenerator instances
        """
        variants = []

        for duration in durations:
            gen = AnimationGenerator(self.rewriter, duration=duration, fps=fps)
            variants.append(gen)

        self.generators = variants
        return variants

    def create_fps_variants(self, duration: float,
                           fps_values: List[float]) -> List[AnimationGenerator]:
        """
        Create variants with different FPS values.

        Args:
            duration: Animation duration
            fps_values: List of FPS values

        Returns:
            List of AnimationGenerator instances
        """
        variants = []

        for fps in fps_values:
            gen = AnimationGenerator(self.rewriter, duration=duration, fps=fps)
            variants.append(gen)

        self.generators = variants
        return variants

    def export_all_variants(self, output_dir: str, max_iterations: int = 5,
                           verbose: bool = False):
        """
        Export all variants.

        Args:
            output_dir: Base output directory
            max_iterations: Maximum iterations
            verbose: Print progress
        """
        for i, gen in enumerate(self.generators):
            variant_dir = Path(output_dir) / f"variant_{i:02d}"

            if verbose:
                print(f"\nGenerating variant {i}...")

            gen.generate_timed_iterations(max_iterations, verbose=verbose)
            gen.export_trajectory_to_obj_sequence(str(variant_dir), verbose=verbose)


def create_simple_animation(rewriter: LSystemRewriter, duration: float = 10.0,
                           fps: float = 30.0, max_iterations: int = 5,
                           output_dir: Optional[str] = None) -> AnimationGenerator:
    """
    Quick helper to create a simple animation.

    Args:
        rewriter: L-system rewriter
        duration: Duration in seconds
        fps: Frames per second
        max_iterations: Maximum iterations
        output_dir: Optional output directory for OBJ sequence

    Returns:
        AnimationGenerator
    """
    gen = AnimationGenerator(rewriter, duration=duration, fps=fps)
    gen.generate_timed_iterations(max_iterations, verbose=True)

    if output_dir:
        gen.export_trajectory_to_obj_sequence(output_dir, verbose=True)

    return gen
