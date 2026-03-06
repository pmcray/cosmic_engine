"""
Prompt Engine for Aesthetic Synthesis.

Library of prompt templates and aesthetic recipes for generating
uncanny, prehistoric, and alien biological imagery.
"""
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class AestheticPreset:
    """An aesthetic preset with prompts and parameters."""
    name: str
    description: str
    positive_prompt: str
    negative_prompt: str
    style_keywords: List[str]
    controlnet_scale: float = 0.8
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None

    def get_full_positive_prompt(self, base_description: str = "") -> str:
        """Combine base description with style keywords."""
        keywords = ", ".join(self.style_keywords)

        if base_description:
            return f"{base_description}, {self.positive_prompt}, {keywords}"
        else:
            return f"{self.positive_prompt}, {keywords}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class PromptEngine:
    """Manages aesthetic presets and prompt generation."""

    def __init__(self):
        self.presets = {}
        self._load_default_presets()

    def _load_default_presets(self):
        """Load default aesthetic presets."""

        # Prehistoric / Carboniferous
        self.presets['prehistoric'] = AestheticPreset(
            name='prehistoric',
            description='Carboniferous period mega-flora with ancient, massive scale',
            positive_prompt=(
                "prehistoric carboniferous forest, giant scale-tree bark texture, "
                "lepidodendron trunk, ancient fossilized wood patterns, "
                "massive prehistoric plant structure, carboniferous atmosphere, "
                "primordial vegetation, thick ribbed bark, "
                "photorealistic, highly detailed, dramatic lighting"
            ),
            negative_prompt=(
                "modern leaves, oak leaves, regular green leaves, "
                "standard tree bark, contemporary forest, "
                "smooth bark, small scale, miniature, "
                "cartoon, anime, illustration, painting, "
                "oversaturated, artificial colors"
            ),
            style_keywords=[
                "(petrified wood texture:1.2)",
                "(giant scale-tree bark:1.3)",
                "(carboniferous forest atmosphere:1.2)",
                "(ancient moss and lichen:1.1)",
                "(ribbed prehistoric bark:1.2)",
                "(massive trunk structure:1.1)"
            ],
            controlnet_scale=0.85,
            num_inference_steps=50,
            guidance_scale=8.0
        )

        # Alien Flora
        self.presets['alien'] = AestheticPreset(
            name='alien',
            description='Otherworldly alien plant structure with bioluminescence',
            positive_prompt=(
                "alien plant organism, extraterrestrial vegetation, "
                "bioluminescent structures, translucent bio-material, "
                "non-terrestrial flora, exotic alien biology, "
                "chitinous surface texture, iridescent sheen, "
                "otherworldly atmosphere, strange organic patterns, "
                "photorealistic, ultra detailed, cinematic lighting"
            ),
            negative_prompt=(
                "earth plants, terrestrial vegetation, "
                "normal trees, regular leaves, earth-like, "
                "standard green, familiar biology, "
                "cartoon, illustration, painting, "
                "oversaturated, artificial"
            ),
            style_keywords=[
                "(translucent skin:1.3)",
                "(pulsating bioluminescence:1.4)",
                "(chitinous exoskeleton:1.2)",
                "(iridescent bio-material:1.3)",
                "(exposed organic systems:1.2)",
                "(alien atmospheric glow:1.2)",
                "(non-Euclidean branching:1.1)"
            ],
            controlnet_scale=0.75,
            num_inference_steps=60,
            guidance_scale=7.0
        )

        # Alien Animal/Creature
        self.presets['alien_creature'] = AestheticPreset(
            name='alien_creature',
            description='Alien animal-like organism with visible biological systems',
            positive_prompt=(
                "alien creature, extraterrestrial organism, "
                "translucent skin revealing internal structures, "
                "visible circulatory system, pulsating organs, "
                "bioluminescent veins, semi-transparent tissue, "
                "exotic anatomy, otherworldly biology, "
                "biomechanical details, organic machinery, "
                "photorealistic, highly detailed, dramatic mood lighting"
            ),
            negative_prompt=(
                "earth animals, terrestrial creatures, mammals, "
                "normal skin, opaque tissue, standard anatomy, "
                "familiar biology, recognizable species, "
                "cartoon, anime, illustration, "
                "oversaturated, flat lighting"
            ),
            style_keywords=[
                "(translucent skin:1.4)",
                "(exposed circulatory system:1.3)",
                "(pulsating bioluminescence:1.4)",
                "(visible internal organs:1.2)",
                "(biomechanical fusion:1.2)",
                "(alien atmospheric effects:1.3)",
                "(wet organic surface:1.1)"
            ],
            controlnet_scale=0.7,
            num_inference_steps=60,
            guidance_scale=7.5
        )

        # Bioluminescent
        self.presets['bioluminescent'] = AestheticPreset(
            name='bioluminescent',
            description='Glowing bioluminescent organism in darkness',
            positive_prompt=(
                "bioluminescent organism, glowing bio-structures, "
                "natural bioluminescence, luminescent patterns, "
                "ethereal glow, phosphorescent tissues, "
                "glowing veins and nodes, bio-light emission, "
                "dark environment, night scene, dramatic contrast, "
                "photorealistic, ultra detailed, cinematic"
            ),
            negative_prompt=(
                "artificial lights, LED, electric lights, neon, "
                "bright daylight, standard lighting, "
                "cartoon, illustration, oversaturated, "
                "fake glow, photoshop glow effect"
            ),
            style_keywords=[
                "(natural bioluminescence:1.4)",
                "(glowing organic structures:1.3)",
                "(phosphorescent patterns:1.3)",
                "(dark atmospheric environment:1.2)",
                "(ethereal bio-light:1.3)",
                "(luminescent veins:1.2)"
            ],
            controlnet_scale=0.75,
            num_inference_steps=55,
            guidance_scale=8.0
        )

        # Deep Ocean / Abyssal
        self.presets['abyssal'] = AestheticPreset(
            name='abyssal',
            description='Deep sea organism with bioluminescence and pressure adaptations',
            positive_prompt=(
                "deep sea organism, abyssal zone creature, "
                "bioluminescent deep ocean life, pressure-adapted biology, "
                "translucent abyssal tissue, bio-luminous lures, "
                "deep water environment, hydrothermal vent ecosystem, "
                "extreme depth adaptation, alien-like appearance, "
                "photorealistic, ultra detailed, volumetric lighting"
            ),
            negative_prompt=(
                "shallow water, surface dweller, bright colors, "
                "tropical fish, coral reef, daylight, "
                "cartoon, illustration, unrealistic"
            ),
            style_keywords=[
                "(deep sea bioluminescence:1.4)",
                "(translucent tissue:1.3)",
                "(pressure-adapted morphology:1.2)",
                "(abyssal darkness:1.3)",
                "(bio-luminous patterns:1.3)",
                "(alien deep-sea biology:1.2)"
            ],
            controlnet_scale=0.8,
            num_inference_steps=50,
            guidance_scale=7.5
        )

        # Fungal / Mycological
        self.presets['fungal'] = AestheticPreset(
            name='fungal',
            description='Fungal organism with spores and mycelial networks',
            positive_prompt=(
                "giant fungal organism, massive mushroom structure, "
                "mycelial network, spore-bearing structures, "
                "fungal fruiting body, bioluminescent fungi, "
                "decomposer organism, forest floor ecosystem, "
                "intricate gill structures, sporulating caps, "
                "photorealistic, macro photography style, detailed textures"
            ),
            negative_prompt=(
                "plants, trees, leaves, flowers, "
                "standard vegetation, green photosynthetic tissue, "
                "cartoon, illustration, artificial"
            ),
            style_keywords=[
                "(fungal tissue texture:1.3)",
                "(mycelial networks:1.2)",
                "(spore dispersal:1.2)",
                "(gill structure details:1.3)",
                "(bioluminescent mushroom:1.2)",
                "(decomposition processes:1.1)"
            ],
            controlnet_scale=0.8,
            num_inference_steps=50,
            guidance_scale=7.0
        )

        # Crystalline / Mineral
        self.presets['crystalline'] = AestheticPreset(
            name='crystalline',
            description='Organism with crystalline or mineral-like structures',
            positive_prompt=(
                "crystalline biological structure, mineral-organic hybrid, "
                "crystallized tissue, biomineralization, "
                "translucent crystal formations, organic crystal growth, "
                "geometric bio-structures, silicon-based life, "
                "refractive organic materials, prismatic effects, "
                "photorealistic, highly detailed, dramatic lighting"
            ),
            negative_prompt=(
                "standard organic tissue, soft tissue only, "
                "regular plants, normal biology, "
                "cartoon, illustration, fake crystals"
            ),
            style_keywords=[
                "(crystalline bio-structures:1.4)",
                "(biomineralization:1.3)",
                "(translucent crystal tissue:1.3)",
                "(prismatic light effects:1.2)",
                "(geometric organic patterns:1.2)",
                "(mineral-organic fusion:1.2)"
            ],
            controlnet_scale=0.75,
            num_inference_steps=55,
            guidance_scale=7.5
        )

        # Parasitic / Symbiotic
        self.presets['parasitic'] = AestheticPreset(
            name='parasitic',
            description='Parasitic or symbiotic organism with host integration',
            positive_prompt=(
                "parasitic organism, symbiotic relationship, "
                "host-parasite interaction, biological integration, "
                "invasive growth patterns, organic fusion, "
                "parasitic tendrils, nutrient extraction structures, "
                "complex symbiosis, merged biological systems, "
                "photorealistic, detailed macro, organic horror aesthetic"
            ),
            negative_prompt=(
                "isolated organism, independent growth, "
                "clean separation, healthy tissue only, "
                "cartoon, illustration, sanitized"
            ),
            style_keywords=[
                "(parasitic integration:1.3)",
                "(organic fusion:1.3)",
                "(invasive tendrils:1.2)",
                "(symbiotic structures:1.2)",
                "(host-parasite boundary:1.2)",
                "(biological merger:1.2)"
            ],
            controlnet_scale=0.8,
            num_inference_steps=55,
            guidance_scale=8.0
        )

        # Botanical Realism
        self.presets['botanical'] = AestheticPreset(
            name='botanical',
            description='Realistic botanical illustration style',
            positive_prompt=(
                "botanical specimen, scientific illustration style, "
                "detailed plant morphology, natural history documentation, "
                "accurate botanical details, taxonomic precision, "
                "herbarium quality, field guide aesthetic, "
                "photorealistic rendering, neutral background, "
                "highly detailed, professional photography"
            ),
            negative_prompt=(
                "fantasy elements, impossible biology, "
                "artistic liberties, stylized, "
                "cartoon, anime, oversaturated"
            ),
            style_keywords=[
                "(scientific accuracy:1.2)",
                "(botanical detail:1.3)",
                "(natural morphology:1.2)",
                "(specimen photography:1.2)",
                "(taxonomic features:1.1)"
            ],
            controlnet_scale=0.9,
            num_inference_steps=40,
            guidance_scale=7.0
        )

    def get_preset(self, name: str) -> Optional[AestheticPreset]:
        """Get preset by name."""
        return self.presets.get(name.lower())

    def list_presets(self) -> List[str]:
        """List available preset names."""
        return list(self.presets.keys())

    def get_preset_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all presets."""
        return {name: preset.description for name, preset in self.presets.items()}

    def add_preset(self, preset: AestheticPreset):
        """Add a custom preset."""
        self.presets[preset.name.lower()] = preset

    def save_presets(self, filename: str):
        """Save presets to JSON file."""
        data = {name: preset.to_dict() for name, preset in self.presets.items()}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_presets(self, filename: str):
        """Load presets from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        for name, preset_dict in data.items():
            preset = AestheticPreset(**preset_dict)
            self.presets[name] = preset

    def create_custom_prompt(self, base_style: str, additional_keywords: List[str],
                            extra_negative: List[str] = None) -> AestheticPreset:
        """
        Create custom preset by modifying an existing one.

        Args:
            base_style: Name of base preset
            additional_keywords: Additional style keywords to add
            extra_negative: Additional negative prompts

        Returns:
            Modified AestheticPreset
        """
        base = self.get_preset(base_style)
        if base is None:
            raise ValueError(f"Base style '{base_style}' not found")

        # Copy and modify
        custom = AestheticPreset(
            name=f"{base_style}_custom",
            description=f"Custom variant of {base_style}",
            positive_prompt=base.positive_prompt,
            negative_prompt=base.negative_prompt,
            style_keywords=base.style_keywords + additional_keywords,
            controlnet_scale=base.controlnet_scale,
            num_inference_steps=base.num_inference_steps,
            guidance_scale=base.guidance_scale
        )

        if extra_negative:
            custom.negative_prompt += ", " + ", ".join(extra_negative)

        return custom

    def blend_presets(self, preset_names: List[str], weights: Optional[List[float]] = None) -> AestheticPreset:
        """
        Blend multiple presets together.

        Args:
            preset_names: List of preset names to blend
            weights: Optional weights for each preset

        Returns:
            Blended AestheticPreset
        """
        if not preset_names:
            raise ValueError("Must provide at least one preset to blend")

        if weights is None:
            weights = [1.0 / len(preset_names)] * len(preset_names)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Get presets
        presets = [self.get_preset(name) for name in preset_names]

        # Combine prompts
        positive_parts = []
        negative_parts = []
        all_keywords = []

        for preset, weight in zip(presets, weights):
            if preset is None:
                continue

            # Weight-based inclusion (only include if weight > 0.3)
            if weight > 0.3:
                positive_parts.append(preset.positive_prompt)
                negative_parts.append(preset.negative_prompt)

            all_keywords.extend(preset.style_keywords)

        # Average parameters
        avg_controlnet = sum(p.controlnet_scale * w for p, w in zip(presets, weights) if p)
        avg_steps = int(sum(p.num_inference_steps * w for p, w in zip(presets, weights) if p))
        avg_guidance = sum(p.guidance_scale * w for p, w in zip(presets, weights) if p)

        blended = AestheticPreset(
            name="_".join(preset_names),
            description=f"Blend of {', '.join(preset_names)}",
            positive_prompt=", ".join(positive_parts),
            negative_prompt=", ".join(set(negative_parts)),  # Deduplicate
            style_keywords=list(set(all_keywords)),  # Deduplicate
            controlnet_scale=avg_controlnet,
            num_inference_steps=avg_steps,
            guidance_scale=avg_guidance
        )

        return blended


def create_prompt_variations(base_prompt: str, num_variations: int = 3) -> List[str]:
    """
    Create variations of a prompt for diverse generation.

    Args:
        base_prompt: Base prompt string
        num_variations: Number of variations to create

    Returns:
        List of prompt variations
    """
    # Variation suffixes
    variations = [
        ", dramatic lighting, cinematic composition",
        ", soft natural lighting, organic atmosphere",
        ", volumetric lighting, atmospheric depth",
        ", high contrast lighting, stark shadows",
        ", diffused lighting, ethereal mood"
    ]

    # Combine base with variations
    results = [base_prompt]

    for i in range(min(num_variations - 1, len(variations))):
        results.append(base_prompt + variations[i])

    return results
