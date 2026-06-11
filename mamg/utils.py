"""Shared helper utilities for the MAMG pipeline."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from PIL import Image


def sanitize_filename(value: str, max_length: int = 120) -> str:
    """Return a filesystem-safe filename stem."""

    value = value.strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value[:max_length] or "image").lower()


def flatten_text(value: str) -> str:
    """Normalize whitespace for prompt and metadata fields."""

    return re.sub(r"\s+", " ", value).strip()


def ensure_parent(path: Path) -> None:
    """Create the parent folder for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def append_csv_rows(csv_path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Append rows to a CSV file and create the header if needed."""

    ensure_parent(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_rows_with_fallback(
    csv_path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
    backup_root: Path | None = None,
) -> Path:
    """Write CSV rows and fall back to local storage if the target fails."""

    try:
        append_csv_rows(csv_path, rows, fieldnames)
        return csv_path
    except Exception:
        fallback_root = backup_root or Path("/content/AI_Microstock_Agent_backup")
        fallback_path = fallback_root / csv_path.name
        append_csv_rows(fallback_path, rows, fieldnames)
        return fallback_path


def save_image_with_fallback(image: Image.Image, image_path: Path, backup_root: Path | None = None) -> Path:
    """Save an image and fall back to local storage if Drive write fails."""

    ensure_parent(image_path)
    try:
        image.save(image_path)
        return image_path
    except Exception:
        fallback_root = backup_root or Path("/content/AI_Microstock_Agent_backup/images")
        fallback_path = fallback_root / image_path.name
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(fallback_path)
        return fallback_path


def to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert dataclasses or mappings into plain dictionaries."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported value type: {type(value)!r}")


def join_keywords(keywords: Iterable[str]) -> str:
    """Join keywords into a microstock-friendly tag string."""

    cleaned = [flatten_text(keyword) for keyword in keywords if flatten_text(keyword)]
    return ", ".join(cleaned)
