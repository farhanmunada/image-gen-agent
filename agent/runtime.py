"""Runtime options used to wire Colab inputs into the Agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from .config import DriveConfig, TrendScraperConfig
from .image_generator import ImageGenerationConfig
from .prompt_generator import PromptGeneratorConfig


class ImageSize(Enum):
    """Preset image sizes for generation."""

    FHD_LANDSCAPE = (1920, 1080)  # 16:9 landscape
    SQUARE = (2000, 2000)  # Square
    PORTRAIT = (1080, 1920)  # 9:16 portrait

    @property
    def width(self) -> int:
        return self[0]

    @property
    def height(self) -> int:
        return self[1]


@dataclass(slots=True)
class PipelineRuntimeOptions:
    """User-editable options for one generation session."""

    category: str = "Design/Arts"
    geo: str = "International"  # General worldwide for trend scraping
    num_images: int = 1
    image_size: ImageSize = ImageSize.FHD_LANDSCAPE
    max_keywords: int = 50  # Maximize trend keywords
    seed: int | None = None
    model_name: str = "stabilityai/sdxl-turbo"
    num_inference_steps: int = 50  # Maximize steps for quality
    guidance_scale: float = 7.5  # Maximize guidance for prompt adherence
    drive_root_folder_name: str = "AI_Microstock_Agent"
    use_cpu_offload: bool = True  # Maintain for Colab stability
    prompt_min_keywords: int = 30  # Keep defaults
    prompt_max_keywords: int = 50

    def build_drive_config(self) -> DriveConfig:
        return DriveConfig(root_folder_name=self.drive_root_folder_name)

    def build_trend_config(self) -> TrendScraperConfig:
        return TrendScraperConfig(
            category=self.category,
            geo=self.geo,
            max_keywords=self.max_keywords,
        )

    def build_prompt_config(self) -> PromptGeneratorConfig:
        return PromptGeneratorConfig(
            min_keywords=self.prompt_min_keywords,
            max_keywords=self.prompt_max_keywords,
        )

    def build_image_config(self) -> ImageGenerationConfig:
        return ImageGenerationConfig(
            model_name=self.model_name,
            height=self.image_size.height,
            width=self.image_size.width,
            num_images=self.num_images,
            seed=self.seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            use_cpu_offload=self.use_cpu_offload,
        )
