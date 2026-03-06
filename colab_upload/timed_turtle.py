"""
Timed Turtle Graphics for L-System Growth Animations.

Extends the base Turtle3D to support interpolation between states
for smooth growth animations without "popping" effects.
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from turtle_3d import Turtle3D, TurtleState, Segment
from timed_symbol import TimedSymbol, TimedString
from timer import TimePoint, AnimationClock, interpolate_value, interpolate_vector


@dataclass
class TimedSegment(Segment):
    """Segment with timing information for interpolation."""

    # Timing
    birth_iteration: int = 0  # Iteration when this segment was created
    age: float = 0.0  # Current age of segment

    # Growth interpolation
    growth_factor: float = 1.0  # Current growth (0.0 to 1.0)
    target_length: float = 1.0  # Fully grown length
    target_width: float = 0.1  # Fully grown width

    def get_interpolated_length(self) -> float:
        """Get length at current growth factor."""
        return self.target_length * self.growth_factor

    def get_interpolated_width(self) -> float:
        """Get width at current growth factor."""
        return self.target_width * self.growth_factor


class TimedTurtle(Turtle3D):
    """
    Turtle graphics interpreter with time-based growth interpolation.

    Supports smooth animations by interpolating between growth states.
    """

    def __init__(self, clock: Optional[AnimationClock] = None,
                 default_angle: float = 25.7,
                 default_length: float = 1.0,
                 default_width: float = 0.1):
        """
        Initialize timed turtle.

        Args:
            clock: Animation clock for time management
            default_angle: Default rotation angle (degrees)
            default_length: Default segment length
            default_width: Default segment width
        """
        super().__init__(default_angle, default_length, default_width)

        self.clock = clock
        self.current_iteration = 0
        self.current_time_point: Optional[TimePoint] = None

        # Track timed segments
        self.timed_segments: List[TimedSegment] = []

        # Symbol birth tracking
        self.symbol_birth_iterations: Dict[int, int] = {}  # symbol_id -> iteration

    def set_time_point(self, time_point: TimePoint):
        """
        Set current time point for interpolation.

        Args:
            time_point: Current time point
        """
        self.current_time_point = time_point
        self.current_iteration = time_point.iteration

    def interpret_timed_string(self, timed_string: TimedString,
                               time_point: Optional[TimePoint] = None):
        """
        Interpret a timed L-system string with growth interpolation.

        Args:
            timed_string: TimedString to interpret
            time_point: Optional time point for interpolation
        """
        if time_point:
            self.set_time_point(time_point)

        # Reset state
        self.reset()

        # Interpret each timed symbol
        for symbol in timed_string:
            self._interpret_timed_symbol(symbol)

    def _interpret_timed_symbol(self, symbol: TimedSymbol):
        """
        Interpret a single timed symbol.

        Args:
            symbol: TimedSymbol to interpret
        """
        char = symbol.char

        # Get growth factor for this symbol
        growth_factor = symbol.get_growth_factor()

        # Get interpolated dimensions
        length = symbol.get_current_length()
        width = symbol.get_current_width()

        if char == 'F':
            # Move forward and draw with interpolated length
            self._draw_forward_timed(symbol, length, width, growth_factor)

        elif char == 'f':
            # Move forward without drawing
            self._move_forward(length)

        elif char == '+':
            # Turn left (yaw)
            angle = symbol.params[0] if symbol.params else self.default_angle
            self.yaw(angle)

        elif char == '-':
            # Turn right (yaw)
            angle = symbol.params[0] if symbol.params else self.default_angle
            self.yaw(-angle)

        elif char == '&':
            # Pitch down
            angle = symbol.params[0] if symbol.params else self.default_angle
            self.pitch(angle)

        elif char == '^':
            # Pitch up
            angle = symbol.params[0] if symbol.params else self.default_angle
            self.pitch(-angle)

        elif char == '\\':
            # Roll left
            angle = symbol.params[0] if symbol.params else self.default_angle
            self.roll(angle)

        elif char == '/':
            # Roll right
            angle = symbol.params[0] if symbol.params else self.default_angle
            self.roll(-angle)

        elif char == '|':
            # Turn around (180 degrees)
            self.yaw(180.0)

        elif char == '[':
            # Push state
            self.push()

        elif char == ']':
            # Pop state
            self.pop()

        elif char == '!':
            # Decrease width
            if symbol.params:
                self.state.width *= symbol.params[0]
            else:
                self.state.width *= 0.7

        elif char == '$':
            # Rotate to vertical
            self._rotate_to_vertical()

    def _draw_forward_timed(self, symbol: TimedSymbol, length: float,
                           width: float, growth_factor: float):
        """
        Draw forward with timing information.

        Args:
            symbol: TimedSymbol
            length: Interpolated length
            width: Interpolated width
            growth_factor: Current growth factor
        """
        start = self.state.position.copy()
        end = start + self.state.heading * length

        # Create timed segment
        segment = TimedSegment(
            start=start,
            end=end,
            width=width,
            birth_iteration=0,  # Will be set externally
            age=symbol.age,
            growth_factor=growth_factor,
            target_length=symbol.max_length,
            target_width=symbol.max_width
        )

        self.segments.append(segment)
        self.timed_segments.append(segment)

        # Update position
        self.state.position = end

    def _move_forward(self, length: float):
        """Move forward without drawing."""
        self.state.position += self.state.heading * length

    def interpolate_between_iterations(self, string_iter_n: TimedString,
                                      string_iter_n1: TimedString,
                                      alpha: float) -> List[Segment]:
        """
        Interpolate geometry between two iterations.

        Args:
            string_iter_n: String at iteration n
            string_iter_n1: String at iteration n+1
            alpha: Interpolation factor (0.0 to 1.0)

        Returns:
            List of interpolated segments
        """
        # Generate geometry for both iterations
        self.reset()
        self.interpret_timed_string(string_iter_n)
        segments_n = [seg for seg in self.segments]

        self.reset()
        self.interpret_timed_string(string_iter_n1)
        segments_n1 = [seg for seg in self.segments]

        # Interpolate segments
        # Simple approach: if counts match, interpolate directly
        # If counts differ, blend visibility

        if len(segments_n) == len(segments_n1):
            # Direct interpolation
            interpolated = []
            for seg_n, seg_n1 in zip(segments_n, segments_n1):
                interp_seg = self._interpolate_segments(seg_n, seg_n1, alpha)
                interpolated.append(interp_seg)
            return interpolated

        else:
            # Counts differ - show growing/shrinking segments
            # Keep all segments from iteration n
            # Fade in new segments from iteration n+1

            interpolated = []

            # Full opacity for segments from iteration n
            for seg in segments_n:
                interpolated.append(seg)

            # Fade in new segments (simplified)
            if len(segments_n1) > len(segments_n):
                new_segments = segments_n1[len(segments_n):]
                for seg in new_segments:
                    # Scale new segments by alpha (growth effect)
                    scaled_seg = self._scale_segment(seg, alpha)
                    interpolated.append(scaled_seg)

            return interpolated

    def _interpolate_segments(self, seg1: Segment, seg2: Segment,
                             alpha: float) -> Segment:
        """
        Interpolate between two segments.

        Args:
            seg1: First segment
            seg2: Second segment
            alpha: Interpolation factor

        Returns:
            Interpolated segment
        """
        start = interpolate_vector(seg1.start, seg2.start, alpha)
        end = interpolate_vector(seg1.end, seg2.end, alpha)
        width = interpolate_value(seg1.width, seg2.width, alpha)

        return Segment(start=start, end=end, width=width)

    def _scale_segment(self, segment: Segment, scale: float) -> Segment:
        """
        Scale a segment by a factor.

        Args:
            segment: Segment to scale
            scale: Scale factor

        Returns:
            Scaled segment
        """
        # Scale from start point
        direction = segment.end - segment.start
        scaled_end = segment.start + direction * scale

        return Segment(
            start=segment.start,
            end=scaled_end,
            width=segment.width * scale
        )

    def get_interpolated_geometry(self, time_point: TimePoint,
                                  string_n: TimedString,
                                  string_n1: TimedString) -> List[Segment]:
        """
        Get interpolated geometry at a specific time point.

        Args:
            time_point: Time point
            string_n: String at iteration n
            string_n1: String at iteration n+1

        Returns:
            List of interpolated segments
        """
        self.set_time_point(time_point)
        return self.interpolate_between_iterations(string_n, string_n1, time_point.alpha)

    def get_timed_segments(self) -> List[TimedSegment]:
        """Get all timed segments."""
        return self.timed_segments

    def update_segment_ages(self, current_time: float):
        """
        Update ages of all timed segments.

        Args:
            current_time: Current global time
        """
        for segment in self.timed_segments:
            # Calculate time since birth
            if self.clock:
                birth_time = segment.birth_iteration * self.clock.time_per_iteration
                segment.age = max(0.0, current_time - birth_time)


class IterationCache:
    """
    Caches L-system iteration results for efficient interpolation.

    Avoids re-computing iterations when interpolating between frames.
    """

    def __init__(self):
        self.cache: Dict[int, TimedString] = {}
        self.geometry_cache: Dict[int, List[Segment]] = {}

    def store_iteration(self, iteration: int, string: TimedString):
        """
        Store iteration result.

        Args:
            iteration: Iteration number
            string: TimedString result
        """
        self.cache[iteration] = string.copy()

    def get_iteration(self, iteration: int) -> Optional[TimedString]:
        """
        Get cached iteration.

        Args:
            iteration: Iteration number

        Returns:
            TimedString if cached, None otherwise
        """
        return self.cache.get(iteration)

    def store_geometry(self, iteration: int, segments: List[Segment]):
        """
        Store geometry for an iteration.

        Args:
            iteration: Iteration number
            segments: List of segments
        """
        self.geometry_cache[iteration] = [seg for seg in segments]

    def get_geometry(self, iteration: int) -> Optional[List[Segment]]:
        """
        Get cached geometry.

        Args:
            iteration: Iteration number

        Returns:
            List of segments if cached, None otherwise
        """
        return self.geometry_cache.get(iteration)

    def clear(self):
        """Clear all caches."""
        self.cache.clear()
        self.geometry_cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cached_iterations': len(self.cache),
            'cached_geometries': len(self.geometry_cache)
        }


def create_smooth_trajectory(turtle: TimedTurtle,
                            timed_strings: List[TimedString],
                            clock: AnimationClock) -> List[Tuple[int, List[Segment]]]:
    """
    Create smooth animation trajectory from timed strings.

    Args:
        turtle: TimedTurtle instance
        timed_strings: List of TimedString for each iteration
        clock: Animation clock

    Returns:
        List of (frame_number, segments) tuples
    """
    trajectory = []

    for frame, time_point in clock.iter_frames():
        iter_n = time_point.iteration
        iter_n1 = min(iter_n + 1, len(timed_strings) - 1)

        if iter_n >= len(timed_strings):
            # Past available iterations, use last
            turtle.reset()
            turtle.interpret_timed_string(timed_strings[-1], time_point)
            segments = turtle.segments.copy()
        else:
            # Interpolate between iterations
            string_n = timed_strings[iter_n]
            string_n1 = timed_strings[iter_n1]

            segments = turtle.get_interpolated_geometry(time_point, string_n, string_n1)

        trajectory.append((frame, segments))

    return trajectory


def compute_growth_trajectory_single(turtle: TimedTurtle,
                                    timed_string: TimedString,
                                    duration: float,
                                    fps: float = 30.0) -> List[List[Segment]]:
    """
    Compute growth trajectory for a single timed string.

    Shows smooth growth of symbols based on age.

    Args:
        turtle: TimedTurtle instance
        timed_string: TimedString with age information
        duration: Duration of growth animation
        fps: Frames per second

    Returns:
        List of segment lists (one per frame)
    """
    num_frames = int(duration * fps)
    trajectory = []

    for frame in range(num_frames):
        t = (frame / num_frames) * duration

        # Advance ages of all symbols
        timed_string_copy = timed_string.copy()
        for symbol in timed_string_copy.symbols:
            symbol.age = t

        # Generate geometry
        turtle.reset()
        turtle.interpret_timed_string(timed_string_copy)

        trajectory.append(turtle.segments.copy())

    return trajectory
