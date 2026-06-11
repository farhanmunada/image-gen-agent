"""Colab entrypoint for the full MAMG pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict

from mamg import MAMGPipeline, PipelineRuntimeOptions


def main() -> None:
    runtime = PipelineRuntimeOptions()
    pipeline = MAMGPipeline.from_runtime_options(runtime)
    result = pipeline.run()

    print("Drive root:", result.drive_paths.root)
    print("Session folder:", result.drive_paths.session_output)
    print("Images folder:", result.drive_paths.session_images)
    print("Metadata file:", result.metadata_csv_path)
    print("Trend source:", result.trend_result.source)
    print("Generated files:", [path.name for path in result.generated_files])
    print(json.dumps(asdict(result.trend_result), ensure_ascii=False, indent=2))
    print(json.dumps(asdict(result.prompt_bundle), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
