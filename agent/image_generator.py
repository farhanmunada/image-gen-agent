"""Agent 3: image generation and upscaling."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

from .prompt_generator import PromptBundle
from .utils import save_image_with_fallback, sanitize_filename


@dataclass(slots=True)
class ImageGenerationConfig:
    """Settings for the diffusion pipeline and upscale stage."""

    model_name: str = "stabilityai/sdxl-turbo"
    device: str = "cuda"
    height: int = 1080
    width: int = 1920
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    use_cpu_offload: bool = True
    upscale_method: str = "lanczos"

    def validate(self) -> None:
        if self.height < 1080 or self.width < 1920:
            raise ValueError("Image dimensions must be at least 1920x1080 pixels.")
        if self.num_images < 1:
            raise ValueError("num_images must be at least 1.")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be zero or greater.")

    def calculate_upscale_target(self) -> int:
        """Calculate upscale target: minimum 2000px on the shortest side."""
        min_side = min(self.height, self.width)
        if min_side >= 2000:
            return max(self.height, self.width)
        scale = 2000 / min_side
        return int(max(self.height, self.width) * scale)


class ImageGenerator:
    """Generate and upscale images for stock production."""

    def __init__(self, config: Optional[ImageGenerationConfig] = None) -> None:
        self.config = config or ImageGenerationConfig()
        self.config.validate()
        self._pipeline = None

    def load(self) -> None:
        """Load the diffusion pipeline if the runtime has the required libraries."""

        try:
            import torch
            from diffusers import AutoPipelineForText2Image  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "diffusers and torch are required for image generation. Install them in Colab first."
            ) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._pipeline = self._load_pipeline(dtype)

        if torch.cuda.is_available():
            if self.config.use_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline.to("cuda")
        else:
            self._pipeline.to("cpu")

    def generate(
        self,
        bundle: PromptBundle,
        output_dir: Path,
        file_prefix: str,
    ) -> List[Path]:
        """Generate one or more images and return the saved file paths."""

        self._ensure_loaded(bundle)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting image generation: {self.config.num_images} image(s) for {file_prefix}")
        generated_paths: List[Path] = []
        for index in range(self.config.num_images):
            logger.info(f"Generating image {index + 1}/{self.config.num_images}...")
            image = self._generate_single(bundle.prompt, index)
            upscale_image = self.upscale(image)
            filename = f"{sanitize_filename(file_prefix)}_{index + 1:03d}.png"
            image_path = output_dir / filename
            saved_path = save_image_with_fallback(upscale_image, image_path)
            logger.info(f"Saved image {index + 1} to {saved_path.name}")
            generated_paths.append(saved_path)

        logger.info(f"Completed generation: {len(generated_paths)} image(s) created")
        return generated_paths

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale to maintain minimum 2000px on shortest side."""

        target = self.config.calculate_upscale_target()
        if image.width >= target and image.height >= target:
            return image

        try:
            return self._upscale_with_realesrgan(image, target)
        except Exception:
            return self._upscale_with_pil(image, target)

    def _generate_single(self, prompt: str, index: int) -> Image.Image:
        """Run the diffusion pipeline for a single image."""

        import torch

        if self._pipeline is None:
            self.load()

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(self.config.seed + index)

        logger.debug(f"Generating image {index + 1} with prompt (truncated): {prompt[:100]}...")
        
        result = self._pipeline(
            prompt=prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            height=self.config.height,
            width=self.config.width,
            num_images_per_prompt=1,  # Explicitly single image per prompt
            generator=generator,
        )
        
        image = result.images[0]
        logger.debug(f"Generated image {index + 1}: {image.size}")
        return image

    def _load_pipeline(self, dtype):
        """Choose the right diffusers pipeline for the configured model."""

        import torch

        model_name = self.config.model_name.lower()
        common_kwargs = {
            "torch_dtype": dtype,
            "variant": "fp16" if dtype == torch.float16 else None,
        }
        if "flux" in model_name:
            try:
                from diffusers import FluxPipeline  # type: ignore

                return FluxPipeline.from_pretrained(self.config.model_name, **common_kwargs)
            except Exception:
                pass

        from diffusers import AutoPipelineForText2Image  # type: ignore

        return AutoPipelineForText2Image.from_pretrained(self.config.model_name, **common_kwargs)

    def _upscale_with_realesrgan(self, image: Image.Image, target: int) -> Image.Image:
        """Try a real ESRGAN-based upscale when available."""

        try:
            import torch
            from realesrgan import RealESRGAN  # type: ignore
        except ImportError:
            logger.debug("Real-ESRGAN not installed, using PIL upscaling.")
            raise RuntimeError("Real-ESRGAN unavailable") from None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = RealESRGAN(device, scale=4)
            model.load_weights("RealESRGAN_x4plus.pth", download=True)
            return model.predict(image)
        except Exception as exc:
            logger.debug("Real-ESRGAN upscale failed: %s, using PIL.", exc)
            raise

    def _upscale_with_pil(self, image: Image.Image, target: int) -> Image.Image:
        """Fallback upscale using high-quality PIL resampling."""

        scale = max(target / image.width, target / image.height)
        new_width = math.ceil(image.width * scale)
        new_height = math.ceil(image.height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _load_lora_weights(self, lora_path: str) -> None:
        """Load LoRA weights when the diffusers pipeline supports it."""

        if self._pipeline is None:
            return

        try:
            self._pipeline.load_lora_weights(lora_path)
        except Exception as exc:
            logger.warning("Failed to load LoRA weights from %s: %s", lora_path, exc)

    def _apply_visual_lora(self, bundle: PromptBundle) -> None:
        """Load a style-specific LoRA when one is configured."""

        if not self.config.enable_lora or self._pipeline is None:
            return

        selected_paths: list[str] = list(self.config.lora_paths)
        if self.config.auto_lora_by_visual_type:
            visual_path = self.config.visual_type_lora_map.get(bundle.visual_type)
            if visual_path:
                selected_paths.insert(0, visual_path)

        for lora_path in selected_paths:
            if Path(lora_path).exists():
                self._load_lora_weights(lora_path)

    def _ensure_loaded(self, bundle: PromptBundle | None = None) -> None:
        """Ensure the diffusion pipeline is available."""

        if self._pipeline is None:
            self.load()
        if bundle is not None:
            self._apply_visual_lora(bundle)
