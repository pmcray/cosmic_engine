"""
Map Generator for Aesthetic Synthesis Engine.

Generates depth maps, normal maps, and edge maps from 3D geometry
for use with ControlNet-based image synthesis.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from typing import Tuple, Optional, List
import json
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Edge detection will use PIL.")


class Camera:
    """Simple camera for rendering views."""

    def __init__(self, position: np.ndarray, target: np.ndarray, up: np.ndarray,
                 fov: float = 60.0, aspect: float = 1.0):
        """
        Initialize camera.

        Args:
            position: Camera position (x, y, z)
            target: Look-at target (x, y, z)
            up: Up vector (x, y, z)
            fov: Field of view in degrees
            aspect: Aspect ratio (width/height)
        """
        self.position = np.array(position)
        self.target = np.array(target)
        self.up = np.array(up) / np.linalg.norm(up)
        self.fov = fov
        self.aspect = aspect

        # Compute view matrix
        self._compute_view_matrix()

    def _compute_view_matrix(self):
        """Compute camera view matrix."""
        # Forward (from camera to target)
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        # Right
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        # Recompute up
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        self.forward = forward
        self.right = right
        self.up = up

    def project(self, point: np.ndarray, width: int, height: int) -> Tuple[int, int, float]:
        """
        Project 3D point to 2D screen coordinates.

        Args:
            point: 3D point (x, y, z)
            width: Image width
            height: Image height

        Returns:
            Tuple of (screen_x, screen_y, depth)
        """
        # Transform to camera space
        relative = point - self.position

        # Project onto camera axes
        x = np.dot(relative, self.right)
        y = np.dot(relative, self.up)
        z = np.dot(relative, self.forward)

        if z <= 0:
            return None, None, None

        # Perspective projection
        fov_rad = np.radians(self.fov)
        scale = np.tan(fov_rad / 2)

        screen_x = (x / (z * scale)) / self.aspect
        screen_y = y / (z * scale)

        # Convert to pixel coordinates
        px = int((screen_x + 1) * width / 2)
        py = int((1 - screen_y) * height / 2)

        return px, py, z


class MapGenerator:
    """Generates conditioning maps from 3D geometry."""

    def __init__(self, width: int = 1024, height: int = 1024):
        """
        Initialize map generator.

        Args:
            width: Output image width
            height: Output image height
        """
        self.width = width
        self.height = height

    def load_obj(self, filename: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Load vertices and edges from OBJ file.

        Args:
            filename: Path to OBJ file

        Returns:
            Tuple of (vertices, edges)
        """
        vertices = []
        edges = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('v '):
                    # Vertex
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])

                elif line.startswith('l '):
                    # Line (edge)
                    parts = line.split()
                    # OBJ indices are 1-based
                    v1 = int(parts[1]) - 1
                    v2 = int(parts[2]) - 1
                    edges.append((v1, v2))

        return np.array(vertices), edges

    def compute_bounds(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute bounding box and size."""
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        size = (max_bounds - min_bounds).max()
        center = (min_bounds + max_bounds) / 2

        return min_bounds, max_bounds, center, size

    def create_camera_preset(self, vertices: np.ndarray, preset: str = "front") -> Camera:
        """
        Create camera from preset angle.

        Args:
            vertices: Vertex array
            preset: Camera preset ("front", "top", "side", "iso", "closeup")

        Returns:
            Camera object
        """
        min_b, max_b, center, size = self.compute_bounds(vertices)

        # Camera distance based on size
        distance = size * 2.5

        if preset == "front":
            position = center + np.array([0, -distance, 0])
            up = np.array([0, 0, 1])

        elif preset == "top":
            position = center + np.array([0, 0, distance])
            up = np.array([0, 1, 0])

        elif preset == "side":
            position = center + np.array([distance, 0, 0])
            up = np.array([0, 0, 1])

        elif preset == "iso":
            # Isometric view
            offset = distance / np.sqrt(3)
            position = center + np.array([offset, -offset, offset])
            up = np.array([0, 0, 1])

        elif preset == "closeup":
            position = center + np.array([0, -distance * 0.8, size * 0.3])
            up = np.array([0, 0, 1])

        else:
            # Default front view
            position = center + np.array([0, -distance, 0])
            up = np.array([0, 0, 1])

        return Camera(position, center, up, fov=60.0, aspect=self.width/self.height)

    def render_depth_map(self, vertices: np.ndarray, edges: List[Tuple[int, int]],
                        camera: Camera, line_width: int = 2) -> Image.Image:
        """
        Render depth map from camera viewpoint.

        Args:
            vertices: Vertex array
            edges: Edge list
            camera: Camera object
            line_width: Line width for rendering

        Returns:
            PIL Image with depth values (0=far, 255=near)
        """
        # Create depth buffer
        depth_buffer = np.full((self.height, self.width), np.inf)

        # Project all vertices
        projected = []
        for v in vertices:
            px, py, depth = camera.project(v, self.width, self.height)
            projected.append((px, py, depth))

        # Render edges
        for v1_idx, v2_idx in edges:
            p1 = projected[v1_idx]
            p2 = projected[v2_idx]

            if p1[0] is None or p2[0] is None:
                continue

            # Draw line with depth interpolation
            self._draw_line_depth(depth_buffer, p1, p2, line_width)

        # Normalize depth to 0-255
        valid_depths = depth_buffer[depth_buffer != np.inf]

        if len(valid_depths) == 0:
            return Image.new('L', (self.width, self.height), 0)

        min_depth = valid_depths.min()
        max_depth = valid_depths.max()

        # Normalize (near=white, far=black for ControlNet)
        normalized = np.zeros_like(depth_buffer)
        valid_mask = depth_buffer != np.inf

        if max_depth > min_depth:
            normalized[valid_mask] = 255 * (1 - (depth_buffer[valid_mask] - min_depth) / (max_depth - min_depth))
        else:
            normalized[valid_mask] = 128

        return Image.fromarray(normalized.astype(np.uint8), mode='L')

    def _draw_line_depth(self, depth_buffer: np.ndarray, p1: Tuple, p2: Tuple, width: int):
        """Draw line into depth buffer with depth interpolation."""
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        if z1 is None or z2 is None:
            return

        # Bresenham's line algorithm with depth
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        err = dx - dy

        x, y = x1, y1
        steps = max(dx, dy)

        for step in range(steps + 1):
            # Interpolate depth
            t = step / max(steps, 1)
            z = z1 + t * (z2 - z1)

            # Draw with thickness
            for dy_offset in range(-width//2, width//2 + 1):
                for dx_offset in range(-width//2, width//2 + 1):
                    px = x + dx_offset
                    py = y + dy_offset

                    if 0 <= px < self.width and 0 <= py < self.height:
                        if z < depth_buffer[py, px]:
                            depth_buffer[py, px] = z

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def render_normal_map(self, vertices: np.ndarray, edges: List[Tuple[int, int]],
                         camera: Camera) -> Image.Image:
        """
        Render normal map (simplified - edge directions).

        Args:
            vertices: Vertex array
            edges: Edge list
            camera: Camera object

        Returns:
            PIL Image with normal vectors encoded as RGB
        """
        # Create RGB image
        img = Image.new('RGB', (self.width, self.height), (128, 128, 255))
        draw = ImageDraw.Draw(img)

        # Project and draw edges with direction-based colors
        for v1_idx, v2_idx in edges:
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]

            px1, py1, z1 = camera.project(v1, self.width, self.height)
            px2, py2, z2 = camera.project(v2, self.width, self.height)

            if px1 is None or px2 is None:
                continue

            # Calculate edge direction in camera space
            edge_dir = v2 - v1
            edge_dir = edge_dir / (np.linalg.norm(edge_dir) + 1e-6)

            # Project direction to camera axes
            nx = np.dot(edge_dir, camera.right)
            ny = np.dot(edge_dir, camera.up)
            nz = np.dot(edge_dir, camera.forward)

            # Encode as RGB (normalized to 0-255)
            r = int((nx + 1) * 127.5)
            g = int((ny + 1) * 127.5)
            b = int((nz + 1) * 127.5)

            color = (r, g, b)

            draw.line([(px1, py1), (px2, py2)], fill=color, width=2)

        return img

    def generate_edge_map(self, depth_map: Image.Image, threshold: int = 50) -> Image.Image:
        """
        Generate edge map using edge detection.

        Args:
            depth_map: Input depth map
            threshold: Edge detection threshold

        Returns:
            PIL Image with edges (white on black)
        """
        if CV2_AVAILABLE:
            # Use Canny edge detection with OpenCV
            img_array = np.array(depth_map)
            edges = cv2.Canny(img_array, threshold, threshold * 2)
            return Image.fromarray(edges, mode='L')
        else:
            # Use PIL edge detection
            edges = depth_map.filter(ImageFilter.FIND_EDGES)
            # Threshold
            edges = edges.point(lambda x: 255 if x > threshold else 0)
            return edges

    def generate_all_maps(self, obj_file: str, camera_preset: str = "iso",
                         output_prefix: Optional[str] = None) -> dict:
        """
        Generate all conditioning maps from OBJ file.

        Args:
            obj_file: Path to OBJ file
            camera_preset: Camera preset name
            output_prefix: Optional prefix for output files

        Returns:
            Dictionary with map images and metadata
        """
        # Load geometry
        vertices, edges = self.load_obj(obj_file)

        if len(vertices) == 0:
            raise ValueError(f"No vertices found in {obj_file}")

        # Create camera
        camera = self.create_camera_preset(vertices, camera_preset)

        # Generate maps
        depth_map = self.render_depth_map(vertices, edges, camera)
        normal_map = self.render_normal_map(vertices, edges, camera)
        edge_map = self.generate_edge_map(depth_map)

        results = {
            'depth': depth_map,
            'normal': normal_map,
            'edge': edge_map,
            'metadata': {
                'camera_preset': camera_preset,
                'width': self.width,
                'height': self.height,
                'num_vertices': len(vertices),
                'num_edges': len(edges)
            }
        }

        # Save if prefix provided
        if output_prefix:
            depth_map.save(f"{output_prefix}_depth.png")
            normal_map.save(f"{output_prefix}_normal.png")
            edge_map.save(f"{output_prefix}_edge.png")

            with open(f"{output_prefix}_metadata.json", 'w') as f:
                json.dump(results['metadata'], f, indent=2)

        return results

class FlightPathGenerator:
    """Generates procedural camera flight paths for seamless transitions."""

    def __init__(self, start_camera: Camera, end_camera: Camera, num_frames: int):
        """
        Initialize the flight path generator.

        Args:
            start_camera: Initial camera state (e.g., planet surface)
            end_camera: Final camera state (e.g., inside Stargate)
            num_frames: Total number of frames for the transition
        """
        self.start_camera = start_camera
        self.end_camera = end_camera
        self.num_frames = num_frames

    def _ease_in_out(self, t: float) -> float:
        """Smooth ease-in-out interpolation curve."""
        return t * t * (3.0 - 2.0 * t)

    def get_camera_at_frame(self, frame_index: int) -> Camera:
        """
        Get the interpolated camera for a specific frame.

        Args:
            frame_index: Current frame number (0 to num_frames - 1)

        Returns:
            Interpolated Camera object
        """
        if self.num_frames <= 1:
            return self.start_camera

        # Calculate progress (0.0 to 1.0)
        t = max(0.0, min(1.0, frame_index / float(self.num_frames - 1)))
        
        # Apply easing function for smoother motion
        eased_t = self._ease_in_out(t)

        # Spherical linear interpolation (Slerp) for direction/target, linear for position
        # For simplicity in this procedural flight path, we'll use linear interpolation 
        # combined with easing for all components, which gives a cinematic tracking shot feel.
        pos = self.start_camera.position + (self.end_camera.position - self.start_camera.position) * eased_t
        target = self.start_camera.target + (self.end_camera.target - self.start_camera.target) * eased_t
        
        # Slerp for the up vector to avoid flipping
        start_up = self.start_camera.up
        end_up = self.end_camera.up
        dot = np.dot(start_up, end_up)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot) * eased_t
        
        relative_up = end_up - start_up * dot
        if np.linalg.norm(relative_up) > 1e-6:
            relative_up = relative_up / np.linalg.norm(relative_up)
            up = start_up * np.cos(theta) + relative_up * np.sin(theta)
        else:
            up = start_up
            
        up = up / np.linalg.norm(up)

        # Interpolate FOV if needed
        fov = self.start_camera.fov + (self.end_camera.fov - self.start_camera.fov) * eased_t

        return Camera(pos, target, up, fov=fov, aspect=self.start_camera.aspect)

    def generate_path(self) -> List[Camera]:
        """
        Generate the full sequence of cameras for the flight path.

        Returns:
            List of Camera objects representing the transition
        """
        return [self.get_camera_at_frame(i) for i in range(self.num_frames)]


def create_multiple_views(obj_file: str, output_dir: str, presets: List[str] = None,
                         width: int = 1024, height: int = 1024):
    """
    Generate maps from multiple camera angles.

    Args:
        obj_file: Path to OBJ file
        output_dir: Output directory
        presets: List of camera presets
        width: Image width
        height: Image height
    """
    if presets is None:
        presets = ["front", "iso", "closeup"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    generator = MapGenerator(width, height)

    results = {}
    for preset in presets:
        print(f"Rendering {preset} view...")
        output_prefix = f"{output_dir}/{Path(obj_file).stem}_{preset}"
        maps = generator.generate_all_maps(obj_file, preset, output_prefix)
        results[preset] = maps

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Map Generator & Flight Path Tool")
    parser.add_argument('--obj', type=str, help='Input OBJ file for single map generation')
    parser.add_argument('--preset', type=str, default='iso', help='Camera preset')
    parser.add_argument('--out', type=str, default='maps_out', help='Output directory')
    
    # Flight path arguments
    parser.add_argument('--flight-path', action='store_true', help='Generate a flight path transition')
    parser.add_argument('--obj-planet', type=str, help='Planet OBJ file for the start of transition')
    parser.add_argument('--obj-stargate', type=str, help='Stargate OBJ file for the end of transition')
    parser.add_argument('--frames', type=int, default=120, help='Number of frames for transition')
    
    args = parser.parse_args()
    
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    if args.flight_path:
        if not args.obj_planet or not args.obj_stargate:
            print("Error: --obj-planet and --obj-stargate required for flight path.")
            return
            
        generator = MapGenerator(1024, 1024)
        
        # Load geometries
        print(f"Loading planet geometry: {args.obj_planet}")
        planet_v, planet_e = generator.load_obj(args.obj_planet)
        print(f"Loading stargate geometry: {args.obj_stargate}")
        stargate_v, stargate_e = generator.load_obj(args.obj_stargate)
        
        # Create start camera (e.g., distant 'iso' view of planet)
        start_camera = generator.create_camera_preset(planet_v, "iso")
        
        # Create end camera (e.g., 'closeup' view diving into stargate)
        end_camera = generator.create_camera_preset(stargate_v, "closeup")
        
        flight_gen = FlightPathGenerator(start_camera, end_camera, args.frames)
        
        print(f"Rendering {args.frames} transition frames...")
        for i in range(args.frames):
            cam = flight_gen.get_camera_at_frame(i)
            
            # Blend the vertices for a morphing transition (simple linear blend)
            # A more advanced version would just render the active object based on distance
            t = i / float(max(1, args.frames - 1))
            
            # For simplicity in this demo, we'll render the stargate geometry throughout the flight
            # but start from the planet's camera position.
            depth_map = generator.render_depth_map(stargate_v, stargate_e, cam)
            out_file = Path(args.out) / f"transition_{i:04d}.png"
            depth_map.save(out_file)
            
            if i % 10 == 0:
                print(f"Rendered frame {i}/{args.frames}")
                
        print(f"Flight path transition saved to {args.out}/")
        
    elif args.obj:
        create_multiple_views(args.obj, args.out, presets=[args.preset])
        print(f"Maps generated in {args.out}/")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
