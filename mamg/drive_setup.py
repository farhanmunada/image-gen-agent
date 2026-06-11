"""Google Drive initialization and project folder management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import DriveConfig, build_session_folder_name


@dataclass(slots=True)
class DrivePaths:
    """Resolved paths used by the project."""

    root: Path
    models: Path
    outputs: Path
    logs: Path
    session_output: Path
    session_images: Path
    session_metadata: Path


class DriveManager:
    """Utility for mounting Drive and creating the folder structure."""

    def __init__(self, config: Optional[DriveConfig] = None) -> None:
        self.config = config or DriveConfig()

    def mount(self) -> None:
        """Mount Google Drive when running inside Colab."""

        try:
            from google.colab import drive  # type: ignore

            mount_point = self.config.drive_mount_point
            if not os.path.exists(mount_point) or not os.listdir(mount_point):
                drive.mount(mount_point)
        except Exception as exc:  # pragma: no cover - Colab-only behavior
            raise RuntimeError(
                "Failed to mount Google Drive. This script is intended for Google Colab."
            ) from exc

    def ensure_structure(self, trend_name: str | None = None) -> DrivePaths:
        """Create the base folder layout and return resolved paths."""

        root = self.config.root_path()
        models = root / self.config.models_folder_name
        outputs = root / self.config.outputs_folder_name
        logs = root / self.config.logs_folder_name

        session_folder = outputs / build_session_folder_name(trend_name)
        session_images = session_folder / "images"
        session_metadata = session_folder / "metadata.csv"

        for path in [root, models, outputs, logs, session_folder, session_images]:
            path.mkdir(parents=True, exist_ok=True)

        return DrivePaths(
            root=root,
            models=models,
            outputs=outputs,
            logs=logs,
            session_output=session_folder,
            session_images=session_images,
            session_metadata=session_metadata,
        )

    def initialize(self, trend_name: str | None = None) -> DrivePaths:
        """Mount Drive and prepare the project folders."""

        self.mount()
        return self.ensure_structure(trend_name=trend_name)
