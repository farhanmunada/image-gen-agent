"""Configuration objects for the MAMG Colab workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass(slots=True)
class DriveConfig:
    """Google Drive folder layout used by the project."""

    root_folder_name: str = "AI_Microstock_Agent"
    models_folder_name: str = "models"
    outputs_folder_name: str = "outputs"
    logs_folder_name: str = "logs"
    drive_mount_point: str = "/content/drive"
    drive_my_drive_path: str = "/content/drive/MyDrive"

    def root_path(self) -> Path:
        return Path(self.drive_my_drive_path) / self.root_folder_name


@dataclass(slots=True)
class TrendScraperConfig:
    """Input parameters for the trend scraper."""

    category: str = "Design/Arts"
    geo: str = "US"
    hl: str = "en-US"
    tz: int = 420
    timeframe: str = "now 7-d"
    max_keywords: int = 10
    fallback_region: str = "US"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    supported_categories: List[str] = field(
        default_factory=lambda: ["Design/Arts", "Business", "Technology"]
    )

    def validate(self) -> None:
        if self.max_keywords <= 0:
            raise ValueError("max_keywords must be greater than zero.")
        if self.category not in self.supported_categories:
            raise ValueError(
                f"Unsupported category '{self.category}'. "
                f"Choose one of: {', '.join(self.supported_categories)}."
            )


def build_session_folder_name(trend_name: str | None = None) -> str:
    """Return the output folder name for the current session."""

    date_part = datetime.now().strftime("%Y-%m-%d")
    safe_trend = (trend_name or "general").strip().replace("/", "-").replace("\\", "-")
    safe_trend = "_".join(segment for segment in safe_trend.split() if segment)
    safe_trend = safe_trend[:60] if safe_trend else "general"
    return f"{date_part}_{safe_trend}"
