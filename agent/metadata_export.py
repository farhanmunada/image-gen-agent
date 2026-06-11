"""CSV metadata export for microstock submissions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

from .prompt_generator import PromptBundle
from .utils import append_csv_rows_with_fallback, join_keywords


@dataclass(slots=True)
class MetadataRow:
    """One microstock metadata row."""

    Filename: str
    Title: str
    Description: str
    Keywords: str


class MetadataExporter:
    """Write metadata.csv files for each trend session."""

    fieldnames = ["Filename", "Title", "Description", "Keywords"]

    def export(self, csv_path: Path, bundle: PromptBundle, filenames: Sequence[Path]) -> Path:
        """Append metadata rows to the session CSV."""

        rows = []
        for filename in filenames:
            row = MetadataRow(
                Filename=filename.name,
                Title=bundle.title,
                Description=bundle.description,
                Keywords=join_keywords(bundle.keywords),
            )
            rows.append(asdict(row))

        return append_csv_rows_with_fallback(csv_path, rows, self.fieldnames)
