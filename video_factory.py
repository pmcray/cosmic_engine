"""
Video Factory with Temporal Coherence for L-System Animations.

Extends the Factory module to generate temporally coherent videos
from animation trajectories.
"""
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import json
import subprocess
import warnings

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Install with: pip install opencv-python")

from map_generator import MapGenerator
from prompt_engine import PromptEngine, AestheticPreset
from animation_generator import AnimationGenerator
from turtle_3d import Segment

# Try to import diffusion client
try:
    from diffusion_client import DiffusionClient
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False


class VideoFactory:
    """
    Generates temporally coherent videos from L-system animations.

    Uses depth map sequences and AI synthesis with temporal coherence.
    """

    def __init__(self, animation_gen: AnimationGenerator,
                 resolution: int = 1024,
                 use_diffusion: bool = True):
        """
        Initialize video factory.

        Args:
            animation_gen: Animation generator
            resolution: Output resolution
            use_diffusion: Whether to use AI synthesis
        """
        self.animation_gen = animation_gen
        self.resolution = resolution
        self.use_diffusion = use_diffusion and DIFFUSION_AVAILABLE

        # Map generator
        self.map_gen = MapGenerator(width=resolution, height=resolution)

        # Diffusion client (lazy load)
        self.diffusion_client: Optional[DiffusionClient] = None

        # Prompt engine
        self.prompt_engine = PromptEngine()

        print(f"VideoFactory initialized (resolution={resolution}, diffusion={self.use_diffusion})")

    def _load_diffusion_client(self):
        """Load diffusion client (lazy)."""
        if self.diffusion_client is None and DIFFUSION_AVAILABLE:
            self.diffusion_client = DiffusionClient()

    def generate_map_sequence(self, output_dir: str,
                             camera_preset: str = "iso",
                             verbose: bool = False) -> List[str]:
        """
        Generate depth/normal/edge map sequence.

        Args:
            output_dir: Output directory
            camera_preset: Camera preset
            verbose: Print progress

        Returns:
            List of depth map filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate trajectory
        trajectory = self.animation_gen.generate_frame_trajectory(verbose=verbose)

        depth_map_files = []

        # Create camera once for all frames
        camera = None

        for frame, time_point, segments in trajectory:
            if verbose and frame % 30 == 0:
                print(f"Generating maps for frame {frame}/{len(trajectory)}")

            # Get vertices and edges from segments
            vertices, edges = self._segments_to_geometry(segments)

            if len(vertices) == 0:
                # Empty frame, create blank map
                depth_map = Image.new('L', (self.resolution, self.resolution), 255)
            else:
                # Create camera from preset if not yet created
                if camera is None:
                    camera = self.map_gen.create_camera_preset(vertices, camera_preset)

                # Render depth map
                depth_map = self.map_gen.render_depth_map(
                    vertices, edges, camera, line_width=2
                )

            # Save depth map
            depth_file = output_path / f"depth_{frame:05d}.png"
            depth_map.save(depth_file)
            depth_map_files.append(str(depth_file))

        print(f"Generated {len(depth_map_files)} depth maps in {output_dir}/")

        # Save metadata
        metadata = {
            'total_frames': len(depth_map_files),
            'fps': self.animation_gen.fps,
            'duration': self.animation_gen.duration,
            'resolution': self.resolution,
            'camera': str(camera_preset)
        }

        metadata_file = output_path / "video_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return depth_map_files

    def _segments_to_geometry(self, segments: List[Segment]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Convert segments to vertices and edges.

        Args:
            segments: List of segments

        Returns:
            Tuple of (vertices, edges)
        """
        vertices = []
        edges = []

        for seg in segments:
            start_idx = len(vertices)
            vertices.append(seg.start)
            vertices.append(seg.end)
            edges.append((start_idx, start_idx + 1))

        if vertices:
            return np.array(vertices), edges
        else:
            return np.array([]).reshape(0, 3), []

    def generate_video_with_ai(self, output_file: str,
                              preset: Union[str, AestheticPreset] = "prehistoric",
                              camera_preset: str = "iso",
                              temporal_strength: float = 0.15,
                              seed: Optional[int] = None,
                              verbose: bool = False) -> str:
        """
        Generate AI-synthesized video with temporal coherence.

        Args:
            output_file: Output video filename
            preset: Aesthetic preset
            camera_preset: Camera preset
            temporal_strength: Temporal coherence strength (lower = more coherent)
            seed: Random seed
            verbose: Print progress

        Returns:
            Output video path
        """
        if not self.use_diffusion or not DIFFUSION_AVAILABLE:
            raise RuntimeError("Diffusion not available. Install required packages.")

        self._load_diffusion_client()

        # Create temp directory for frames
        temp_dir = Path(output_file).parent / "temp_frames"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate depth maps
        print("Generating depth map sequence...")
        depth_maps = self.generate_map_sequence(str(temp_dir / "maps"), camera_preset, verbose)

        # Generate AI frames with temporal coherence
        print("Generating AI frames with temporal coherence...")
        ai_frames = self._generate_temporally_coherent_frames(
            depth_maps, preset, temporal_strength, seed, temp_dir, verbose
        )

        # Encode to video
        print("Encoding video...")
        video_path = self._encode_video(ai_frames, output_file, verbose)

        print(f"Video saved to {video_path}")

        return video_path

    def _generate_temporally_coherent_frames(self, depth_maps: List[str],
                                            preset: Union[str, AestheticPreset],
                                            temporal_strength: float,
                                            seed: Optional[int],
                                            output_dir: Path,
                                            verbose: bool) -> List[str]:
        """
        Generate AI frames with temporal coherence.

        Uses img2img on previous frame to maintain consistency.

        Args:
            depth_maps: List of depth map filenames
            preset: Aesthetic preset
            temporal_strength: Temporal coherence strength
            seed: Random seed
            output_dir: Output directory
            verbose: Print progress

        Returns:
            List of AI frame filenames
        """
        ai_frames = []
        previous_frame: Optional[Image.Image] = None

        for i, depth_map_file in enumerate(depth_maps):
            if verbose:
                print(f"Generating AI frame {i+1}/{len(depth_maps)}")

            depth_map = Image.open(depth_map_file)

            if previous_frame is None:
                # First frame: generate from depth map
                frame = self.diffusion_client.generate_from_preset(
                    depth_map, preset, seed=seed,
                    width=self.resolution, height=self.resolution
                )
            else:
                # Subsequent frames: use img2img for temporal coherence
                # Generate new frame from depth map
                new_frame = self.diffusion_client.generate_from_preset(
                    depth_map, preset, seed=seed,
                    width=self.resolution, height=self.resolution
                )

                # Blend with previous frame for temporal coherence
                frame = self._blend_frames(previous_frame, new_frame, temporal_strength)

            # Save frame
            frame_file = output_dir / f"frame_{i:05d}.png"
            frame.save(frame_file)
            ai_frames.append(str(frame_file))

            previous_frame = frame

        return ai_frames

    def _blend_frames(self, prev_frame: Image.Image, new_frame: Image.Image,
                     strength: float) -> Image.Image:
        """
        Blend two frames for temporal coherence.

        Args:
            prev_frame: Previous frame
            new_frame: New frame
            strength: Blending strength (0 = all previous, 1 = all new)

        Returns:
            Blended frame
        """
        # Convert to numpy
        prev_array = np.array(prev_frame).astype(np.float32)
        new_array = np.array(new_frame).astype(np.float32)

        # Blend
        blended = prev_array * (1.0 - strength) + new_array * strength
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def stabilize_existing_video(self, input_video: str, output_file: str,
                                preset: Union[str, AestheticPreset] = "prehistoric",
                                temporal_strength: float = 0.85,
                                seed: Optional[int] = None,
                                use_svd: bool = True,
                                verbose: bool = False) -> str:
        """
        Stabilize and stylize an existing video (like Stargate) using SVD and temporal blending.
        
        Args:
            input_video: Path to input mp4
            output_file: Path to output mp4
            preset: Aesthetic preset for styling
            temporal_strength: How much of the previous frame to keep (for img2img blending)
            seed: Random seed
            use_svd: Whether to use Stable Video Diffusion for chunked coherent generation
            verbose: Print progress
            
        Returns:
            Output video path
        """
        if not self.use_diffusion or not DIFFUSION_AVAILABLE:
            raise RuntimeError("Diffusion not available. Install required packages.")
            
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required to read input videos.")

        self._load_diffusion_client()

        # Create temp directory for frames
        temp_dir = Path(output_file).parent / "temp_stabilize"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Read video frames
        cap = cv2.VideoCapture(input_video)
        input_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_frames.append(Image.fromarray(frame))
        cap.release()
        
        if not input_frames:
            raise ValueError(f"Could not read any frames from {input_video}")
            
        print(f"Read {len(input_frames)} frames from {input_video}")
        
        # Resize to standard resolution to save VRAM
        input_frames = [f.resize((self.resolution, self.resolution), Image.LANCZOS) for f in input_frames]
        
        ai_frames = []
        
        if use_svd and len(input_frames) > 0:
            print("Using Stable Video Diffusion for temporal coherence...")
            # SVD generates videos in chunks (e.g., 25 frames at a time).
            # For a 600 frame video, we would process it in chunks, using the stylized
            # last frame of the previous chunk as the init_image for the next.
            # To keep it grounded to the original animation, we can blend SVD output with img2img.
            
            chunk_size = 25
            for i in range(0, len(input_frames), chunk_size):
                chunk = input_frames[i:i+chunk_size]
                if verbose:
                    print(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} frames)")
                
                # Stylize the first frame of the chunk using img2img
                init_frame = self.diffusion_client.refine_image(
                    chunk[0], 
                    prompt=preset if isinstance(preset, str) else preset.positive_prompt,
                    strength=0.6,
                    seed=seed
                )
                
                # Generate SVD frames
                svd_frames = self.diffusion_client.generate_video_from_image(
                    init_frame,
                    num_frames=len(chunk),
                    fps=int(self.animation_gen.fps) if self.animation_gen else 30,
                    seed=seed
                )
                
                # Blend SVD frames with the original structured frames to keep the Stargate shape
                for j, (svd_frame, orig_frame) in enumerate(zip(svd_frames, chunk)):
                    # Optional: apply img2img on the original frame to stylize it
                    stylized_orig = self.diffusion_client.refine_image(
                        orig_frame,
                        prompt=preset if isinstance(preset, str) else preset.positive_prompt,
                        strength=0.4,
                        seed=seed
                    )
                    
                    # Blend the temporally smooth SVD frame with the structured stylized frame
                    final_frame = self._blend_frames(svd_frame, stylized_orig, strength=0.5)
                    
                    frame_file = temp_dir / f"frame_{i+j:05d}.png"
                    final_frame.save(frame_file)
                    ai_frames.append(str(frame_file))
                    
        else:
            print("Using img2img with temporal blending...")
            previous_frame = None
            for i, frame in enumerate(input_frames):
                if verbose and i % 10 == 0:
                    print(f"Stylizing frame {i+1}/{len(input_frames)}")
                
                # Apply img2img
                new_frame = self.diffusion_client.refine_image(
                    frame,
                    prompt=preset if isinstance(preset, str) else preset.positive_prompt,
                    strength=0.55,
                    seed=seed
                )
                
                if previous_frame is not None:
                    # Blend with previous frame (temporal_strength: higher = more of previous frame)
                    final_frame = self._blend_frames(previous_frame, new_frame, strength=1.0 - temporal_strength)
                else:
                    final_frame = new_frame
                    
                frame_file = temp_dir / f"frame_{i:05d}.png"
                final_frame.save(frame_file)
                ai_frames.append(str(frame_file))
                previous_frame = final_frame

        # Encode to video
        print("Encoding stabilized video...")
        video_path = self._encode_video(ai_frames, output_file, verbose)

        print(f"Video saved to {video_path}")
        return video_path

    def _encode_video(self, frame_files: List[str], output_file: str,
                     verbose: bool = False) -> str:
        """
        Encode frames to video using FFmpeg.

        Args:
            frame_files: List of frame filenames
            output_file: Output video filename
            verbose: Print progress

        Returns:
            Output video path
        """
        if not CV2_AVAILABLE:
            # Try using FFmpeg directly
            return self._encode_video_ffmpeg(frame_files, output_file, verbose)
        else:
            # Use OpenCV
            return self._encode_video_opencv(frame_files, output_file, verbose)

    def _encode_video_opencv(self, frame_files: List[str], output_file: str,
                            verbose: bool) -> str:
        """Encode video using OpenCV."""
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.animation_gen.fps,
                             (width, height))

        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            out.write(frame)

        out.release()

        if verbose:
            print(f"Encoded {len(frame_files)} frames to {output_file}")

        return output_file

    def _encode_video_ffmpeg(self, frame_files: List[str], output_file: str,
                            verbose: bool) -> str:
        """Encode video using FFmpeg command."""
        # Get input pattern from first file
        first_file = Path(frame_files[0])
        input_pattern = str(first_file.parent / first_file.name.replace(
            first_file.name.split('_')[-1], "%05d.png"
        ))

        # FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite
            '-framerate', str(self.animation_gen.fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_file
        ]

        if verbose:
            print(f"Running FFmpeg: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=not verbose)
            return output_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg encoding failed: {e}")

    def generate_map_video_only(self, output_file: str,
                                camera_preset: str = "iso",
                                colorize: bool = True,
                                verbose: bool = False) -> str:
        """
        Generate video from depth maps only (no AI synthesis).

        Args:
            output_file: Output video filename
            camera_preset: Camera preset
            colorize: Colorize depth maps
            verbose: Print progress

        Returns:
            Output video path
        """
        # Create temp directory
        temp_dir = Path(output_file).parent / "temp_maps"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate depth maps
        print("Generating depth map sequence...")
        depth_maps = self.generate_map_sequence(str(temp_dir), camera_preset, verbose)

        # Optionally colorize
        if colorize and CV2_AVAILABLE:
            print("Colorizing depth maps...")
            depth_maps = self._colorize_depth_maps(depth_maps, temp_dir, verbose)

        # Encode to video
        print("Encoding video...")
        video_path = self._encode_video(depth_maps, output_file, verbose)

        print(f"Depth map video saved to {video_path}")

        return video_path

    def _colorize_depth_maps(self, depth_map_files: List[str],
                            output_dir: Path, verbose: bool) -> List[str]:
        """
        Colorize depth maps using a color map.

        Args:
            depth_map_files: List of depth map filenames
            output_dir: Output directory
            verbose: Print progress

        Returns:
            List of colorized depth map filenames
        """
        colorized_dir = output_dir / "colorized"
        colorized_dir.mkdir(exist_ok=True)

        colorized_files = []

        for i, depth_file in enumerate(depth_map_files):
            # Read depth map
            depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)

            # Apply colormap
            colored = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)

            # Save
            colored_file = colorized_dir / f"colored_{i:05d}.png"
            cv2.imwrite(str(colored_file), colored)
            colorized_files.append(str(colored_file))

        return colorized_files

    def export_preview_video(self, output_file: str, max_frames: int = 100,
                            verbose: bool = False) -> str:
        """
        Export a preview video (limited frames, no AI).

        Args:
            output_file: Output filename
            max_frames: Maximum frames to render
            verbose: Print progress

        Returns:
            Output video path
        """
        # Limit frames
        original_duration = self.animation_gen.duration
        self.animation_gen.duration = min(
            original_duration,
            max_frames / self.animation_gen.fps
        )
        self.animation_gen.clock.duration = self.animation_gen.duration
        self.animation_gen.clock.total_frames = min(
            max_frames,
            self.animation_gen.clock.total_frames
        )

        # Generate map-only video
        video_path = self.generate_map_video_only(output_file, verbose=verbose)

        # Restore duration
        self.animation_gen.duration = original_duration

        return video_path


def check_video_dependencies() -> Dict[str, bool]:
    """Check which video encoding dependencies are available."""
    deps = {
        'opencv': CV2_AVAILABLE,
        'ffmpeg': False
    }

    # Check for FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        deps['ffmpeg'] = True
    except:
        pass

    return deps


def print_video_system_info():
    """Print video system information."""
    print("=== Video Factory System Info ===")

    deps = check_video_dependencies()
    print(f"OpenCV available: {deps['opencv']}")
    print(f"FFmpeg available: {deps['ffmpeg']}")

    if not (deps['opencv'] or deps['ffmpeg']):
        print("\nWARNING: No video encoding backend available!")
        print("Install OpenCV: pip install opencv-python")
        print("Or install FFmpeg: https://ffmpeg.org/download.html")

    print("=" * 35)
