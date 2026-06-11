"""Colab workflow template for the MAMG pipeline.

This file is valid Python, but it is arranged like notebook cells so you can
copy each section into Google Colab with minimal changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mamg import MAMGPipeline, PipelineRuntimeOptions, display_runtime_widgets, runtime_from_widgets


# Cell 1: Mount Google Drive
def mount_drive() -> None:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


# Cell 2: Clone the repository
def clone_repo(repo_url: str, target_dir: str = "image-gen-agent") -> Path:
    import subprocess

    workspace = Path("/content")
    repo_path = workspace / target_dir
    if repo_path.exists():
        return repo_path

    subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
    return repo_path


# Cell 3: Install dependencies
def install_requirements(repo_path: Path) -> None:
    import subprocess

    requirements_path = repo_path / "requirements_colab.txt"
    subprocess.run(["pip", "-q", "install", "-r", str(requirements_path)], check=True)


# Cell 4: Collect user options
def collect_runtime_options(defaults: Optional[PipelineRuntimeOptions] = None) -> PipelineRuntimeOptions:
    widgets = display_runtime_widgets(defaults or PipelineRuntimeOptions())
    return runtime_from_widgets(widgets)


# Cell 5: Run the pipeline
def run_pipeline(runtime: PipelineRuntimeOptions) -> None:
    pipeline = MAMGPipeline.from_runtime_options(runtime)
    result = pipeline.run()

    print("Drive root:", result.drive_paths.root)
    print("Session folder:", result.drive_paths.session_output)
    print("Images folder:", result.drive_paths.session_images)
    print("Metadata CSV:", result.metadata_csv_path)
    print("Trend source:", result.trend_result.source)
    print("Generated files:", [path.name for path in result.generated_files])
    print("Prompt title:", result.prompt_bundle.title)
    print("Prompt:", result.prompt_bundle.prompt)


if __name__ == "__main__":
    print("Import this module in Colab and run the cell functions in order.")
