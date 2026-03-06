"""
Diffusion Client for Aesthetic Synthesis Engine.

Interfaces with Stable Diffusion and ControlNet for image generation.
"""
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
import warnings

try:
    import torch
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
        AutoencoderKL
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn(
        "Diffusers library not available. Install with: pip install diffusers transformers accelerate"
    )

from prompt_engine import PromptEngine, AestheticPreset


class DiffusionClient:
    """Client for running Stable Diffusion with ControlNet conditioning."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 controlnet_id: str = "diffusers/controlnet-depth-sdxl-1.0",
                 device: str = "auto",
                 use_fp16: bool = True):
        """
        Initialize diffusion client.

        Args:
            model_id: Stable Diffusion model identifier
            controlnet_id: ControlNet model identifier
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_fp16: Use half precision (FP16) for faster inference
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers library required. Install with: "
                "pip install diffusers transformers accelerate"
            )

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # Pipelines (loaded on demand)
        self.controlnet_pipeline = None
        self.img2img_pipeline = None

        # Prompt engine
        self.prompt_engine = PromptEngine()

        print(f"DiffusionClient initialized on {self.device}")
        if self.use_fp16:
            print("Using FP16 for faster inference")

    def _load_controlnet_pipeline(self):
        """Load ControlNet pipeline (lazy loading)."""
        if self.controlnet_pipeline is not None:
            return

        print(f"Loading ControlNet model from {self.controlnet_id}...")

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=self.dtype
        )

        print(f"Loading base model from {self.model_id}...")

        # Determine which pipeline to use
        if "xl" in self.model_id.lower():
            pipeline_class = StableDiffusionXLControlNetPipeline
        else:
            pipeline_class = StableDiffusionControlNetPipeline

        self.controlnet_pipeline = pipeline_class.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype
        )

        self.controlnet_pipeline.to(self.device)

        # Enable optimizations
        if self.device == "cuda":
            try:
                self.controlnet_pipeline.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except:
                pass

        print("ControlNet pipeline loaded successfully")

    def _load_img2img_pipeline(self):
        """Load img2img pipeline for refinement."""
        if self.img2img_pipeline is not None:
            return

        print(f"Loading img2img pipeline...")

        self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype
        )

        self.img2img_pipeline.to(self.device)

        if self.device == "cuda":
            try:
                self.img2img_pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass

        print("Img2img pipeline loaded successfully")

    def generate_from_depth(self, depth_map: Image.Image, prompt: str,
                           negative_prompt: str = "",
                           controlnet_scale: float = 0.8,
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5,
                           seed: Optional[int] = None,
                           width: Optional[int] = None,
                           height: Optional[int] = None) -> Image.Image:
        """
        Generate image from depth map using ControlNet.

        Args:
            depth_map: Input depth map (PIL Image)
            prompt: Text prompt
            negative_prompt: Negative prompt
            controlnet_scale: ControlNet conditioning strength (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            width: Output width (None = auto from depth map)
            height: Output height (None = auto from depth map)

        Returns:
            Generated PIL Image
        """
        self._load_controlnet_pipeline()

        # Prepare depth map
        if width and height:
            depth_map = depth_map.resize((width, height), Image.LANCZOS)

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)

        # Generate
        print(f"Generating image (steps={num_inference_steps}, guidance={guidance_scale})...")

        result = self.controlnet_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_map,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        return result.images[0]

    def generate_from_preset(self, depth_map: Image.Image, preset: Union[str, AestheticPreset],
                            base_description: str = "",
                            seed: Optional[int] = None,
                            width: Optional[int] = None,
                            height: Optional[int] = None) -> Image.Image:
        """
        Generate image using an aesthetic preset.

        Args:
            depth_map: Input depth map
            preset: Preset name or AestheticPreset object
            base_description: Optional base description to prepend
            seed: Random seed
            width: Output width
            height: Output height

        Returns:
            Generated PIL Image
        """
        # Get preset
        if isinstance(preset, str):
            preset_obj = self.prompt_engine.get_preset(preset)
            if preset_obj is None:
                raise ValueError(f"Preset '{preset}' not found")
        else:
            preset_obj = preset

        # Build full prompt
        full_prompt = preset_obj.get_full_positive_prompt(base_description)

        print(f"Using preset: {preset_obj.name}")
        print(f"Prompt: {full_prompt[:100]}...")

        # Generate
        return self.generate_from_depth(
            depth_map=depth_map,
            prompt=full_prompt,
            negative_prompt=preset_obj.negative_prompt,
            controlnet_scale=preset_obj.controlnet_scale,
            num_inference_steps=preset_obj.num_inference_steps,
            guidance_scale=preset_obj.guidance_scale,
            seed=seed,
            width=width,
            height=height
        )

    def refine_image(self, image: Image.Image, prompt: str,
                    strength: float = 0.3,
                    num_inference_steps: int = 30,
                    guidance_scale: float = 7.5,
                    seed: Optional[int] = None) -> Image.Image:
        """
        Refine image using img2img pipeline.

        Args:
            image: Input image to refine
            prompt: Refinement prompt
            strength: How much to change (0-1, higher = more change)
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            seed: Random seed

        Returns:
            Refined PIL Image
        """
        self._load_img2img_pipeline()

        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)

        print(f"Refining image (strength={strength})...")

        result = self.img2img_pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        return result.images[0]

    def generate_with_refinement(self, depth_map: Image.Image, preset: Union[str, AestheticPreset],
                                 base_description: str = "",
                                 refinement_prompt: Optional[str] = None,
                                 refinement_strength: float = 0.25,
                                 seed: Optional[int] = None,
                                 width: Optional[int] = None,
                                 height: Optional[int] = None) -> Tuple[Image.Image, Image.Image]:
        """
        Generate image with automatic refinement pass.

        Args:
            depth_map: Input depth map
            preset: Aesthetic preset
            base_description: Base description
            refinement_prompt: Custom refinement prompt (None = use preset prompt)
            refinement_strength: Refinement strength
            seed: Random seed
            width: Output width
            height: Output height

        Returns:
            Tuple of (initial_image, refined_image)
        """
        # Generate initial image
        initial = self.generate_from_preset(
            depth_map, preset, base_description, seed, width, height
        )

        # Determine refinement prompt
        if refinement_prompt is None:
            if isinstance(preset, str):
                preset_obj = self.prompt_engine.get_preset(preset)
            else:
                preset_obj = preset

            refinement_prompt = (
                f"{preset_obj.positive_prompt}, "
                "fine details, intricate textures, biological weathering, "
                "organic complexity, naturalistic detail"
            )

        # Refine
        refined = self.refine_image(
            initial,
            prompt=refinement_prompt,
            strength=refinement_strength,
            seed=seed
        )

        return initial, refined

    def generate_batch(self, depth_maps: List[Image.Image], preset: Union[str, AestheticPreset],
                      base_descriptions: Optional[List[str]] = None,
                      seeds: Optional[List[int]] = None,
                      output_dir: Optional[str] = None) -> List[Image.Image]:
        """
        Generate batch of images.

        Args:
            depth_maps: List of depth maps
            preset: Aesthetic preset
            base_descriptions: Optional list of base descriptions
            seeds: Optional list of seeds
            output_dir: Optional output directory

        Returns:
            List of generated images
        """
        if base_descriptions is None:
            base_descriptions = [""] * len(depth_maps)

        if seeds is None:
            seeds = [None] * len(depth_maps)

        results = []

        for i, (depth_map, desc, seed) in enumerate(zip(depth_maps, base_descriptions, seeds)):
            print(f"\nGenerating image {i+1}/{len(depth_maps)}...")

            image = self.generate_from_preset(depth_map, preset, desc, seed)
            results.append(image)

            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_file = f"{output_dir}/generated_{i:03d}.png"
                image.save(output_file)
                print(f"Saved to {output_file}")

        return results

    def upscale_image(self, image: Image.Image, scale: float = 2.0) -> Image.Image:
        """
        Upscale image (simple Lanczos resampling).

        For more advanced upscaling, consider using separate upscaling models.

        Args:
            image: Input image
            scale: Upscale factor

        Returns:
            Upscaled image
        """
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def list_presets(self) -> List[str]:
        """List available aesthetic presets."""
        return self.prompt_engine.list_presets()

    def get_preset_info(self, preset_name: str) -> Optional[Dict]:
        """Get information about a preset."""
        preset = self.prompt_engine.get_preset(preset_name)
        if preset:
            return preset.to_dict()
        return None

    def unload(self):
        """Unload models to free memory."""
        if self.controlnet_pipeline is not None:
            del self.controlnet_pipeline
            self.controlnet_pipeline = None

        if self.img2img_pipeline is not None:
            del self.img2img_pipeline
            self.img2img_pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Models unloaded")


def check_diffusers_availability() -> bool:
    """Check if diffusers library is available."""
    return DIFFUSERS_AVAILABLE


def print_system_info():
    """Print system information for diffusion."""
    print("=== Diffusion System Info ===")
    print(f"Diffusers available: {DIFFUSERS_AVAILABLE}")

    if DIFFUSERS_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 30)
