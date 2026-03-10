"""
Tropism forces for plant growth simulation.

Implements gravitropism, phototropism, and other environmental response mechanisms
based on ABOP Section 2.1.
"""
import numpy as np
from typing import Tuple, Optional
from environment import Environment


class TropismCalculator:
    """Calculates bending forces for plant growth."""

    def __init__(self, environment: Optional[Environment] = None, fluid_engine=None):
        """
        Initialize tropism calculator.

        Args:
            environment: Optional environment for phototropism
            fluid_engine: Optional fluid solver for fluid-based tropism (wind/currents)
        """
        self.environment = environment
        self.fluid_engine = fluid_engine

        # Standard gravity vector (downward)
        self.gravity_vector = np.array([0.0, 0.0, -1.0])

    def calculate_fluid_tropism(self, heading: np.ndarray, position: np.ndarray,
                                susceptibility: float) -> np.ndarray:
        """
        Calculate bending due to fluid velocity (wind/current).

        Args:
            heading: Current heading vector (H)
            position: Current position in world space
            susceptibility: Bending strength (e)

        Returns:
            New heading vector after fluid tropism
        """
        if self.fluid_engine is None:
            return heading

        # Map 3D position to 2D fluid grid (assuming X-Y plane maps to fluid grid)
        # Assuming position coordinates are roughly centered around 0, scale them
        # to fit within the fluid grid resolution.
        res = self.fluid_engine.RES
        grid_x = int((position[0] * 5.0 + res / 2) % res)
        grid_y = int((position[1] * 5.0 + res / 2) % res)

        # Retrieve fluid velocity at this point
        # The velocity field is a Taichi field, so we access it as [grid_x, grid_y]
        vel = self.fluid_engine.velocity[grid_x, grid_y]

        # Convert 2D velocity to 3D vector (assuming flow in X-Y plane)
        fluid_vector = np.array([vel[0], vel[1], 0.0])

        norm_fluid = np.linalg.norm(fluid_vector)
        if norm_fluid < 1e-6:
            return heading

        # Normalize heading and fluid vector
        heading = heading / np.linalg.norm(heading)
        fluid_vector_norm = fluid_vector / norm_fluid

        # The strength of the bend is proportional to both susceptibility and fluid speed
        effective_susceptibility = susceptibility * min(norm_fluid * 0.01, 1.0)

        # Apply tropism formula toward fluid flow direction
        correction = np.cross(np.cross(heading, fluid_vector_norm), heading)
        new_heading = heading + effective_susceptibility * correction

        # Normalize result
        norm = np.linalg.norm(new_heading)
        if norm > 1e-6:
            new_heading = new_heading / norm
        else:
            new_heading = heading

        return new_heading

    def calculate_gravitropism(self, heading: np.ndarray, susceptibility: float,
                              positive: bool = False) -> np.ndarray:
        """
        Calculate gravitropic bending.

        Args:
            heading: Current heading vector (H)
            susceptibility: Bending strength (e) - how much the plant responds
            positive: If True, grow toward gravity; if False, grow away (default)

        Returns:
            New heading vector after tropism
        """
        # Normalize heading
        heading = heading / np.linalg.norm(heading)

        # Tropism vector (gravity or anti-gravity)
        tropism_vec = self.gravity_vector if positive else -self.gravity_vector

        # Apply ABOP Section 2.1 formula:
        # H_next = Normalize(H + e * (H × T) × H)
        # This steers H toward T
        correction = np.cross(np.cross(heading, tropism_vec), heading)
        new_heading = heading + susceptibility * correction

        # Normalize result
        norm = np.linalg.norm(new_heading)
        if norm > 1e-6:
            new_heading = new_heading / norm
        else:
            new_heading = heading

        return new_heading

    def calculate_phototropism(self, heading: np.ndarray, position: np.ndarray,
                               susceptibility: float) -> np.ndarray:
        """
        Calculate phototropic bending (toward light).

        Args:
            heading: Current heading vector (H)
            position: Current position in world space
            susceptibility: Bending strength (e)

        Returns:
            New heading vector after tropism
        """
        if self.environment is None:
            return heading

        # Get light gradient at current position
        light_gradient = self.environment.calculate_light_gradient(position)

        # If no gradient, no bending
        if np.linalg.norm(light_gradient) < 1e-6:
            return heading

        # Normalize heading
        heading = heading / np.linalg.norm(heading)

        # Apply tropism formula toward light gradient
        correction = np.cross(np.cross(heading, light_gradient), heading)
        new_heading = heading + susceptibility * correction

        # Normalize result
        norm = np.linalg.norm(new_heading)
        if norm > 1e-6:
            new_heading = new_heading / norm
        else:
            new_heading = heading

        return new_heading

    def calculate_combined_tropism(self, heading: np.ndarray, position: np.ndarray,
                                  gravity_susceptibility: float = 0.0,
                                  photo_susceptibility: float = 0.0,
                                  fluid_susceptibility: float = 0.0,
                                  positive_gravitropism: bool = False) -> np.ndarray:
        """
        Calculate combined gravitropic, phototropic, and fluid effects.

        Args:
            heading: Current heading vector (H)
            position: Current position in world space
            gravity_susceptibility: Gravitropism strength
            photo_susceptibility: Phototropism strength
            fluid_susceptibility: Fluid/Wind tropism strength
            positive_gravitropism: Grow toward vs away from gravity

        Returns:
            New heading vector after all tropisms
        """
        new_heading = heading.copy()

        # Apply gravitropism
        if abs(gravity_susceptibility) > 1e-6:
            new_heading = self.calculate_gravitropism(
                new_heading, gravity_susceptibility, positive_gravitropism
            )

        # Apply phototropism
        if abs(photo_susceptibility) > 1e-6 and self.environment is not None:
            new_heading = self.calculate_phototropism(
                new_heading, position, photo_susceptibility
            )

        # Apply fluid tropism
        if abs(fluid_susceptibility) > 1e-6 and self.fluid_engine is not None:
            new_heading = self.calculate_fluid_tropism(
                new_heading, position, fluid_susceptibility
            )

        return new_heading

    def set_gravity_vector(self, vector: np.ndarray):
        """
        Set custom gravity vector.

        For creating alien/pre-historic effects:
        - Rotating gravity for spiral growth
        - Non-vertical gravity for unusual forms
        """
        self.gravity_vector = vector / np.linalg.norm(vector)

    def get_effective_tropism_vector(self, position: np.ndarray,
                                    gravity_weight: float = 1.0,
                                    photo_weight: float = 1.0) -> np.ndarray:
        """
        Get the combined effective tropism vector at a position.

        Args:
            position: World position
            gravity_weight: Weight for gravity component
            photo_weight: Weight for phototropism component

        Returns:
            Combined tropism vector (normalized)
        """
        tropism = np.zeros(3)

        # Add gravity component (negative = grow up)
        if abs(gravity_weight) > 1e-6:
            tropism -= self.gravity_vector * gravity_weight

        # Add photo component
        if abs(photo_weight) > 1e-6 and self.environment is not None:
            light_grad = self.environment.calculate_light_gradient(position)
            tropism += light_grad * photo_weight

        # Normalize
        norm = np.linalg.norm(tropism)
        if norm > 1e-6:
            tropism = tropism / norm

        return tropism


def apply_tropism_to_frame(heading: np.ndarray, left: np.ndarray, up: np.ndarray,
                           new_heading: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update the turtle's coordinate frame (H, L, U) when heading changes.

    Maintains orthogonality of the frame while smoothly transitioning heading.

    Args:
        heading: Current H vector
        left: Current L vector
        up: Current U vector
        new_heading: New H vector after tropism

    Returns:
        Tuple of (new_heading, new_left, new_up)
    """
    # Normalize new heading
    new_heading = new_heading / np.linalg.norm(new_heading)

    # Reconstruct orthogonal frame
    # Keep U perpendicular to H
    # Use Gram-Schmidt orthogonalization

    # Project U onto plane perpendicular to new_heading
    new_up = up - np.dot(up, new_heading) * new_heading
    norm_up = np.linalg.norm(new_up)

    if norm_up < 1e-6:
        # U and new_heading are parallel - need to reconstruct frame
        # Use old left as reference
        new_up = np.cross(left, new_heading)
        norm_up = np.linalg.norm(new_up)

        if norm_up < 1e-6:
            # Still parallel - use arbitrary perpendicular
            if abs(new_heading[0]) < 0.9:
                new_up = np.cross(np.array([1, 0, 0]), new_heading)
            else:
                new_up = np.cross(np.array([0, 1, 0]), new_heading)
            norm_up = np.linalg.norm(new_up)

    new_up = new_up / norm_up

    # Left is perpendicular to both H and U
    new_left = np.cross(new_up, new_heading)
    new_left = new_left / np.linalg.norm(new_left)

    return new_heading, new_left, new_up


class TimeVaryingTropism:
    """Tropism that changes over time (for dynamic alien effects)."""

    def __init__(self, base_gravity: np.ndarray = np.array([0, 0, -1])):
        self.base_gravity = base_gravity
        self.time = 0.0

    def get_gravity_at_time(self, t: float, mode: str = 'static') -> np.ndarray:
        """
        Get gravity vector at time t.

        Args:
            t: Current time
            mode: 'static', 'rotating', 'pulsing', 'random'

        Returns:
            Gravity vector at time t
        """
        if mode == 'static':
            return self.base_gravity

        elif mode == 'rotating':
            # Rotate gravity around vertical axis
            angle = t * 0.1  # Slow rotation
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Rotate in XY plane
            gx = self.base_gravity[0] * cos_a - self.base_gravity[1] * sin_a
            gy = self.base_gravity[0] * sin_a + self.base_gravity[1] * cos_a
            gz = self.base_gravity[2]

            return np.array([gx, gy, gz])

        elif mode == 'pulsing':
            # Vary gravity strength with time
            strength = 1.0 + 0.3 * np.sin(t * 0.5)
            return self.base_gravity * strength

        elif mode == 'random':
            # Add random perturbations
            noise = np.random.randn(3) * 0.1
            perturbed = self.base_gravity + noise
            return perturbed / np.linalg.norm(perturbed)

        else:
            return self.base_gravity

    def step(self, dt: float):
        """Advance time."""
        self.time += dt
