"""End-to-end orchestration for the MAMG workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import DriveConfig, TrendScraperConfig
from .drive_setup import DriveManager, DrivePaths
from .image_generator import ImageGenerationConfig, ImageGenerator
from .metadata_export import MetadataExporter
from .prompt_generator import PromptBundle, PromptGenerator, PromptGeneratorConfig
from .trend_scraper import TrendResult, TrendScraper
from .utils import sanitize_filename
from .runtime import PipelineRuntimeOptions


@dataclass(slots=True)
class PipelineResult:
    """Summary of one completed run."""

    drive_paths: DrivePaths
    trend_result: TrendResult
    prompt_bundle: PromptBundle
    generated_files: list[Path]
    metadata_csv_path: Path


class MAMGPipeline:
    """Coordinate the full microstock generation workflow."""

    def __init__(
        self,
        drive_config: Optional[DriveConfig] = None,
        trend_config: Optional[TrendScraperConfig] = None,
        prompt_config: Optional[PromptGeneratorConfig] = None,
        image_config: Optional[ImageGenerationConfig] = None,
    ) -> None:
        self.drive_manager = DriveManager(drive_config)
        self.trend_scraper = TrendScraper(trend_config)
        self.prompt_generator = PromptGenerator(prompt_config)
        self.image_generator = ImageGenerator(image_config)
        self.metadata_exporter = MetadataExporter()

    @classmethod
    def from_runtime_options(cls, runtime: PipelineRuntimeOptions) -> "MAMGPipeline":
        """Build a pipeline from user-facing runtime options."""

        return cls(
            drive_config=runtime.build_drive_config(),
            trend_config=runtime.build_trend_config(),
            prompt_config=runtime.build_prompt_config(),
            image_config=runtime.build_image_config(),
        )

    def run(self) -> PipelineResult:
        """Run the workflow from trend scraping through export."""

        self.drive_manager.mount()
        trend_result = self.trend_scraper.scrape()
        trend_name = trend_result.keywords[0] if trend_result.keywords else trend_result.category
        drive_paths = self.drive_manager.ensure_structure(trend_name=trend_name)
        prompt_bundle = self.prompt_generator.generate(trend_result)

        file_prefix = sanitize_filename(
            f"{trend_result.category}_{prompt_bundle.trend_keyword}_{trend_result.fetched_at}"
        )
        generated_files = self.image_generator.generate(
            bundle=prompt_bundle,
            output_dir=drive_paths.session_images,
            file_prefix=file_prefix,
        )
        metadata_csv_path = self.metadata_exporter.export(
            csv_path=drive_paths.session_metadata,
            bundle=prompt_bundle,
            filenames=generated_files,
        )

        return PipelineResult(
            drive_paths=drive_paths,
            trend_result=trend_result,
            prompt_bundle=prompt_bundle,
            generated_files=generated_files,
            metadata_csv_path=metadata_csv_path,
        )
