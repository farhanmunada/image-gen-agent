"""Runtime options used to wire Colab inputs into the MAMG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import DriveConfig, TrendScraperConfig
from .image_generator import ImageGenerationConfig
from .prompt_generator import PromptGeneratorConfig


@dataclass(slots=True)
class PipelineRuntimeOptions:
    """User-editable options for one generation session."""

    category: str = "Design/Arts"
    geo: str = "US"
    max_keywords: int = 10
    num_images: int = 1
    seed: int | None = None
    model_name: str = "stabilityai/sdxl-turbo"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    upscale_target: int = 4000
    use_cpu_offload: bool = True
    enable_lora: bool = False
    lora_paths: List[str] = field(default_factory=list)
    visual_type_lora_map: dict[str, str] = field(default_factory=dict)
    prompt_min_keywords: int = 30
    prompt_max_keywords: int = 50
    drive_root_folder_name: str = "AI_Microstock_Agent"

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
            height=self.height,
            width=self.width,
            num_images=self.num_images,
            seed=self.seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            upscale_target=self.upscale_target,
            use_cpu_offload=self.use_cpu_offload,
            enable_lora=self.enable_lora,
            lora_paths=self.lora_paths,
            visual_type_lora_map=self.visual_type_lora_map,
        )
