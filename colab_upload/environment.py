"""
Environmental simulation using voxel-based light propagation.

Implements a 3D occupancy grid for tracking light density and shadow casting.
Uses PyTorch for GPU acceleration when available.
"""
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using NumPy (slower).")

from typing import Tuple, List, Optional


class Environment:
    """3D voxelized environment for light and resource simulation."""

    def __init__(self, bounds: Tuple[float, float, float, float, float, float],
                 resolution: float = 0.5, use_torch: bool = True):
        """
        Initialize the environment.

        Args:
            bounds: (min_x, max_x, min_y, max_y, min_z, max_z) in meters
            resolution: Voxel size in meters
            use_torch: Use PyTorch if available
        """
        self.bounds = bounds
        self.resolution = resolution
        self.use_torch = use_torch and TORCH_AVAILABLE

        # Calculate grid dimensions
        self.dims = self._calc_dims(bounds, resolution)
        self.origin = np.array([bounds[0], bounds[2], bounds[4]])

        # Initialize light grid (1.0 = full light, 0.0 = complete shadow)
        if self.use_torch:
            self.light_grid = torch.ones(self.dims, dtype=torch.float32)
            # Move to GPU if available
            if torch.cuda.is_available():
                self.light_grid = self.light_grid.cuda()
        else:
            self.light_grid = np.ones(self.dims, dtype=np.float32)

        # Light direction (default: from above)
        self.light_direction = np.array([0.0, 0.0, -1.0])  # Downward
        self.ambient_light = 0.2  # Minimum light level

    def _calc_dims(self, bounds: Tuple, resolution: float) -> Tuple[int, int, int]:
        """Calculate grid dimensions from bounds."""
        dx = int(np.ceil((bounds[1] - bounds[0]) / resolution))
        dy = int(np.ceil((bounds[3] - bounds[2]) / resolution))
        dz = int(np.ceil((bounds[5] - bounds[4]) / resolution))
        return (dx, dy, dz)

    def world_to_voxel(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to voxel indices.

        Args:
            positions: Array of shape (N, 3) with world coordinates

        Returns:
            Array of shape (N, 3) with voxel indices
        """
        positions = np.atleast_2d(positions)
        voxel_coords = (positions - self.origin) / self.resolution
        indices = np.floor(voxel_coords).astype(np.int32)

        # Clamp to valid range
        indices[:, 0] = np.clip(indices[:, 0], 0, self.dims[0] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, self.dims[1] - 1)
        indices[:, 2] = np.clip(indices[:, 2], 0, self.dims[2] - 1)

        return indices

    def voxel_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates (center of voxel)."""
        indices = np.atleast_2d(indices)
        positions = self.origin + (indices + 0.5) * self.resolution
        return positions

    def get_light_intensity(self, positions: np.ndarray) -> np.ndarray:
        """
        Get light intensity at world positions.

        Args:
            positions: Array of shape (N, 3) with world coordinates

        Returns:
            Array of shape (N,) with light intensities (0.0 to 1.0)
        """
        indices = self.world_to_voxel(positions)

        if self.use_torch:
            # Convert to tensor
            if isinstance(self.light_grid, torch.Tensor):
                grid_cpu = self.light_grid.cpu().numpy()
            else:
                grid_cpu = self.light_grid
            intensities = grid_cpu[indices[:, 0], indices[:, 1], indices[:, 2]]
        else:
            intensities = self.light_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

        return np.maximum(intensities, self.ambient_light)

    def cast_shadow(self, start_pos: np.ndarray, end_pos: np.ndarray,
                   occlusion_strength: float = 0.3, radius: float = 0.1):
        """
        Cast shadow from a line segment.

        Args:
            start_pos: Start position of segment
            end_pos: End position of segment
            occlusion_strength: How much light is blocked (0.0 to 1.0)
            radius: Radius of shadow in meters
        """
        # Calculate segment center and direction
        center = (start_pos + end_pos) / 2.0
        segment_dir = end_pos - start_pos
        segment_length = np.linalg.norm(segment_dir)

        if segment_length < 1e-6:
            return

        # Get voxel coordinates
        center_voxel = self.world_to_voxel(center.reshape(1, 3))[0]

        # Calculate shadow volume (cast along light direction)
        # For simplicity, cast shadows in the light direction from the segment
        shadow_direction = -self.light_direction  # Opposite to light
        shadow_length = 100.0  # Cast shadow far down

        # Sample points along shadow ray
        num_samples = int(shadow_length / self.resolution)
        for i in range(num_samples):
            offset = shadow_direction * (i * self.resolution)
            shadow_pos = center + offset

            # Get voxel index
            voxel_idx = self.world_to_voxel(shadow_pos.reshape(1, 3))[0]

            # Apply occlusion in a radius around the shadow ray
            radius_voxels = int(np.ceil(radius / self.resolution))

            for dx in range(-radius_voxels, radius_voxels + 1):
                for dy in range(-radius_voxels, radius_voxels + 1):
                    for dz in range(-radius_voxels, radius_voxels + 1):
                        ix = voxel_idx[0] + dx
                        iy = voxel_idx[1] + dy
                        iz = voxel_idx[2] + dz

                        # Check bounds
                        if (0 <= ix < self.dims[0] and
                            0 <= iy < self.dims[1] and
                            0 <= iz < self.dims[2]):

                            # Distance-based falloff
                            dist = np.sqrt(dx**2 + dy**2 + dz**2) * self.resolution
                            if dist <= radius:
                                falloff = 1.0 - (dist / radius)
                                reduction = occlusion_strength * falloff

                                if self.use_torch:
                                    current = float(self.light_grid[ix, iy, iz])
                                    self.light_grid[ix, iy, iz] = max(0.0, current - reduction)
                                else:
                                    self.light_grid[ix, iy, iz] = max(
                                        0.0, self.light_grid[ix, iy, iz] - reduction
                                    )

    def cast_shadows_from_segments(self, segments: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Cast shadows from multiple line segments.

        Args:
            segments: List of (start_pos, end_pos, width) tuples
        """
        for start_pos, end_pos, width in segments:
            self.cast_shadow(start_pos, end_pos, radius=width/2.0)

    def calculate_light_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate light gradient at a position (for phototropism).

        Args:
            position: World position (x, y, z)

        Returns:
            Gradient vector pointing toward increasing light
        """
        # Sample light in a small neighborhood
        delta = self.resolution
        gradient = np.zeros(3)

        # Finite difference approximation
        for i, d in enumerate([
            np.array([delta, 0, 0]),
            np.array([0, delta, 0]),
            np.array([0, 0, delta])
        ]):
            pos_plus = position + d
            pos_minus = position - d

            light_plus = self.get_light_intensity(pos_plus.reshape(1, 3))[0]
            light_minus = self.get_light_intensity(pos_minus.reshape(1, 3))[0]

            gradient[i] = (light_plus - light_minus) / (2 * delta)

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm

        return gradient

    def reset_light(self):
        """Reset light grid to full illumination."""
        if self.use_torch:
            self.light_grid.fill_(1.0)
        else:
            self.light_grid.fill(1.0)

    def set_light_direction(self, direction: np.ndarray):
        """Set the direction of incoming light."""
        self.light_direction = direction / np.linalg.norm(direction)

    def get_statistics(self) -> dict:
        """Get statistics about the light environment."""
        if self.use_torch:
            grid_cpu = self.light_grid.cpu().numpy() if isinstance(self.light_grid, torch.Tensor) else self.light_grid
        else:
            grid_cpu = self.light_grid

        return {
            'grid_dims': self.dims,
            'resolution': self.resolution,
            'total_voxels': np.prod(self.dims),
            'mean_light': float(np.mean(grid_cpu)),
            'min_light': float(np.min(grid_cpu)),
            'max_light': float(np.max(grid_cpu)),
            'shadowed_voxels': int(np.sum(grid_cpu < 0.5)),
            'using_torch': self.use_torch
        }

    def export_light_slice(self, z_level: float, filename: str):
        """
        Export a horizontal slice of the light field for visualization.

        Args:
            z_level: Z coordinate in world space
            filename: Output file (numpy .npz format)
        """
        z_index = int((z_level - self.bounds[4]) / self.resolution)
        z_index = np.clip(z_index, 0, self.dims[2] - 1)

        if self.use_torch:
            grid_cpu = self.light_grid.cpu().numpy() if isinstance(self.light_grid, torch.Tensor) else self.light_grid
        else:
            grid_cpu = self.light_grid

        slice_data = grid_cpu[:, :, z_index]

        np.savez(filename,
                slice=slice_data,
                z_level=z_level,
                bounds=self.bounds,
                resolution=self.resolution)
