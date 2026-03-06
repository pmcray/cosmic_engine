"""
Timer and Time Management for L-System Growth Animations.

Manages the global clock, maps time to iteration depths, and calculates
interpolation factors for smooth growth animations.
"""
import numpy as np
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class TimePoint:
    """A specific point in time with associated iteration and interpolation."""

    time: float  # Global time value
    iteration: int  # Which iteration this time maps to
    alpha: float  # Interpolation factor (0.0 to 1.0)

    def __repr__(self):
        return f"TimePoint(t={self.time:.2f}, iter={self.iteration}, α={self.alpha:.3f})"


@dataclass
class Keyframe:
    """A keyframe in the animation timeline."""

    time: float
    iteration: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Keyframe(t={self.time:.2f}, iter={self.iteration})"


class AnimationClock:
    """
    Manages the global animation clock and time progression.

    Maps continuous time values to discrete L-system iterations with
    interpolation factors for smooth animations.
    """

    def __init__(self, duration: float, max_iterations: int,
                 time_per_iteration: Optional[float] = None,
                 fps: float = 30.0):
        """
        Initialize animation clock.

        Args:
            duration: Total animation duration (seconds)
            max_iterations: Maximum number of L-system iterations
            time_per_iteration: Time allocated per iteration (None = auto)
            fps: Target frames per second
        """
        self.duration = duration
        self.max_iterations = max_iterations
        self.fps = fps
        self.total_frames = int(duration * fps)

        # Calculate time per iteration if not provided
        if time_per_iteration is None:
            # Distribute time evenly across all iterations
            self.time_per_iteration = duration / max_iterations
        else:
            self.time_per_iteration = time_per_iteration

        # Keyframes
        self.keyframes: List[Keyframe] = []
        self._auto_generate_keyframes()

        print(f"AnimationClock: {duration}s, {max_iterations} iterations, "
              f"{self.total_frames} frames @ {fps} fps")

    def _auto_generate_keyframes(self):
        """Automatically generate keyframes at iteration boundaries."""
        self.keyframes = []

        for i in range(self.max_iterations + 1):
            time = i * self.time_per_iteration
            if time <= self.duration:
                self.keyframes.append(Keyframe(time=time, iteration=i))

        # Ensure final keyframe at duration
        if not self.keyframes or self.keyframes[-1].time < self.duration:
            self.keyframes.append(
                Keyframe(time=self.duration, iteration=self.max_iterations)
            )

    def add_keyframe(self, time: float, iteration: int, metadata: dict = None):
        """
        Add a custom keyframe.

        Args:
            time: Time value
            iteration: Iteration depth at this keyframe
            metadata: Optional metadata
        """
        kf = Keyframe(time, iteration, metadata or {})
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.time)

    def get_time_point(self, time: float) -> TimePoint:
        """
        Map continuous time to discrete iteration with interpolation factor.

        Args:
            time: Current time value

        Returns:
            TimePoint with iteration and alpha
        """
        # Clamp time to valid range
        time = max(0.0, min(time, self.duration))

        # Find which iteration this time falls into
        iteration = int(time / self.time_per_iteration)
        iteration = min(iteration, self.max_iterations - 1)

        # Calculate interpolation factor within this iteration
        iteration_start_time = iteration * self.time_per_iteration
        iteration_end_time = (iteration + 1) * self.time_per_iteration

        if iteration_end_time > iteration_start_time:
            alpha = (time - iteration_start_time) / (iteration_end_time - iteration_start_time)
        else:
            alpha = 0.0

        # Clamp alpha to [0, 1]
        alpha = max(0.0, min(1.0, alpha))

        return TimePoint(time=time, iteration=iteration, alpha=alpha)

    def get_frame_time(self, frame: int) -> float:
        """
        Get time value for a specific frame number.

        Args:
            frame: Frame index (0 to total_frames-1)

        Returns:
            Time value
        """
        if self.total_frames <= 1:
            return 0.0

        return (frame / (self.total_frames - 1)) * self.duration

    def get_frame_time_point(self, frame: int) -> TimePoint:
        """
        Get TimePoint for a specific frame.

        Args:
            frame: Frame index

        Returns:
            TimePoint
        """
        time = self.get_frame_time(frame)
        return self.get_time_point(time)

    def iter_frames(self) -> List[Tuple[int, TimePoint]]:
        """
        Iterate through all frames.

        Yields:
            Tuples of (frame_number, time_point)
        """
        frames = []
        for frame in range(self.total_frames):
            time_point = self.get_frame_time_point(frame)
            frames.append((frame, time_point))
        return frames

    def get_iteration_time_range(self, iteration: int) -> Tuple[float, float]:
        """
        Get the time range for a specific iteration.

        Args:
            iteration: Iteration number

        Returns:
            Tuple of (start_time, end_time)
        """
        start_time = iteration * self.time_per_iteration
        end_time = (iteration + 1) * self.time_per_iteration

        return (start_time, min(end_time, self.duration))

    def get_keyframe_at_time(self, time: float, tolerance: float = 0.01) -> Optional[Keyframe]:
        """
        Get keyframe near a specific time.

        Args:
            time: Time value
            tolerance: Time tolerance for matching

        Returns:
            Keyframe if found, None otherwise
        """
        for kf in self.keyframes:
            if abs(kf.time - time) < tolerance:
                return kf
        return None


class IterationTimer:
    """
    Manages timing for individual L-system iterations.

    Tracks symbol ages and growth progression within an iteration.
    """

    def __init__(self, iteration: int, duration: float,
                 default_terminal_age: float = 1.0):
        """
        Initialize iteration timer.

        Args:
            iteration: Iteration number
            duration: Duration of this iteration
            default_terminal_age: Default terminal age for symbols
        """
        self.iteration = iteration
        self.duration = duration
        self.default_terminal_age = default_terminal_age
        self.elapsed_time = 0.0

    def reset(self):
        """Reset elapsed time."""
        self.elapsed_time = 0.0

    def advance(self, dt: float):
        """
        Advance time by dt.

        Args:
            dt: Time delta
        """
        self.elapsed_time += dt

    def get_progress(self) -> float:
        """
        Get progress through this iteration (0.0 to 1.0).

        Returns:
            Progress factor
        """
        if self.duration <= 0:
            return 1.0

        return min(1.0, self.elapsed_time / self.duration)

    def is_complete(self) -> bool:
        """Check if iteration is complete."""
        return self.elapsed_time >= self.duration


class GrowthScheduler:
    """
    Schedules growth events and symbol maturation.

    Coordinates symbol age advancement with the global clock.
    """

    def __init__(self, clock: AnimationClock):
        """
        Initialize growth scheduler.

        Args:
            clock: Animation clock
        """
        self.clock = clock
        self.iteration_timers: dict = {}

    def get_or_create_timer(self, iteration: int) -> IterationTimer:
        """
        Get or create timer for an iteration.

        Args:
            iteration: Iteration number

        Returns:
            IterationTimer
        """
        if iteration not in self.iteration_timers:
            duration = self.clock.time_per_iteration
            self.iteration_timers[iteration] = IterationTimer(iteration, duration)

        return self.iteration_timers[iteration]

    def calculate_symbol_age(self, iteration_born: int, current_time: float) -> float:
        """
        Calculate symbol age based on birth iteration and current time.

        Args:
            iteration_born: Iteration when symbol was created
            current_time: Current global time

        Returns:
            Symbol age
        """
        # Time when symbol was born
        birth_time = iteration_born * self.clock.time_per_iteration

        # Age is time since birth
        age = max(0.0, current_time - birth_time)

        return age

    def calculate_maturity_time(self, iteration_born: int, terminal_age: float) -> float:
        """
        Calculate when a symbol reaches maturity.

        Args:
            iteration_born: Iteration when symbol was created
            terminal_age: Terminal age

        Returns:
            Time when symbol reaches maturity
        """
        birth_time = iteration_born * self.clock.time_per_iteration
        maturity_time = birth_time + terminal_age

        return maturity_time


class EasingFunction:
    """
    Easing functions for smooth animations.

    Provides various interpolation curves beyond linear.
    """

    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation."""
        return t

    @staticmethod
    def ease_in_quad(t: float) -> float:
        """Quadratic ease-in."""
        return t * t

    @staticmethod
    def ease_out_quad(t: float) -> float:
        """Quadratic ease-out."""
        return t * (2.0 - t)

    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Quadratic ease-in-out."""
        if t < 0.5:
            return 2.0 * t * t
        else:
            return -1.0 + (4.0 - 2.0 * t) * t

    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in."""
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out."""
        t1 = t - 1.0
        return t1 * t1 * t1 + 1.0

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out."""
        if t < 0.5:
            return 4.0 * t * t * t
        else:
            t1 = 2.0 * t - 2.0
            return 0.5 * t1 * t1 * t1 + 1.0

    @staticmethod
    def smoothstep(t: float) -> float:
        """Smoothstep interpolation."""
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def smootherstep(t: float) -> float:
        """Smoother step (Ken Perlin)."""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def get_easing(name: str) -> Callable[[float], float]:
        """
        Get easing function by name.

        Args:
            name: Easing function name

        Returns:
            Easing function
        """
        easings = {
            'linear': EasingFunction.linear,
            'ease_in_quad': EasingFunction.ease_in_quad,
            'ease_out_quad': EasingFunction.ease_out_quad,
            'ease_in_out_quad': EasingFunction.ease_in_out_quad,
            'ease_in_cubic': EasingFunction.ease_in_cubic,
            'ease_out_cubic': EasingFunction.ease_out_cubic,
            'ease_in_out_cubic': EasingFunction.ease_in_out_cubic,
            'smoothstep': EasingFunction.smoothstep,
            'smootherstep': EasingFunction.smootherstep,
        }

        return easings.get(name, EasingFunction.linear)


def interpolate_value(v1: float, v2: float, alpha: float,
                     easing: str = 'linear') -> float:
    """
    Interpolate between two values with optional easing.

    Args:
        v1: Start value
        v2: End value
        alpha: Interpolation factor (0.0 to 1.0)
        easing: Easing function name

    Returns:
        Interpolated value
    """
    easing_func = EasingFunction.get_easing(easing)
    t = easing_func(alpha)

    return v1 + t * (v2 - v1)


def interpolate_vector(v1: np.ndarray, v2: np.ndarray, alpha: float,
                       easing: str = 'linear') -> np.ndarray:
    """
    Interpolate between two vectors with optional easing.

    Args:
        v1: Start vector
        v2: End vector
        alpha: Interpolation factor
        easing: Easing function name

    Returns:
        Interpolated vector
    """
    easing_func = EasingFunction.get_easing(easing)
    t = easing_func(alpha)

    return v1 + t * (v2 - v1)


def create_animation_timeline(duration: float, iterations: int, fps: float = 30.0) -> AnimationClock:
    """
    Create a standard animation timeline.

    Args:
        duration: Total duration (seconds)
        iterations: Number of L-system iterations
        fps: Frames per second

    Returns:
        AnimationClock
    """
    return AnimationClock(duration=duration, max_iterations=iterations, fps=fps)


# Convenience functions for common timing patterns

def exponential_timing(iterations: int, base_time: float = 1.0,
                       growth_factor: float = 1.5) -> List[float]:
    """
    Generate exponential timing for iterations.

    Early iterations are quick, later ones slow down.

    Args:
        iterations: Number of iterations
        base_time: Base time for first iteration
        growth_factor: Time multiplier per iteration

    Returns:
        List of time durations per iteration
    """
    return [base_time * (growth_factor ** i) for i in range(iterations)]


def logarithmic_timing(iterations: int, total_time: float) -> List[float]:
    """
    Generate logarithmic timing (fast at start, slow at end).

    Args:
        iterations: Number of iterations
        total_time: Total time to distribute

    Returns:
        List of time durations per iteration
    """
    if iterations <= 1:
        return [total_time]

    # Use log scale for time distribution
    log_weights = [np.log(i + 2) for i in range(iterations)]
    total_weight = sum(log_weights)

    return [(w / total_weight) * total_time for w in log_weights]


def uniform_timing(iterations: int, total_time: float) -> List[float]:
    """
    Generate uniform timing (equal time per iteration).

    Args:
        iterations: Number of iterations
        total_time: Total time to distribute

    Returns:
        List of time durations per iteration
    """
    time_per_iter = total_time / iterations
    return [time_per_iter] * iterations
