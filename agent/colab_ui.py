"""Colab widgets for collecting generation options from the user."""

from __future__ import annotations

import json
from typing import Dict

from .config import TrendScraperConfig
from .runtime import ImageSize, PipelineRuntimeOptions


def build_runtime_widgets(defaults: PipelineRuntimeOptions | None = None) -> Dict[str, object]:
    """Return a dictionary of ipywidgets for interactive Colab input (simplified)."""

    defaults = defaults or PipelineRuntimeOptions()

    try:
        import ipywidgets as widgets  # type: ignore
    except Exception as exc:  # pragma: no cover - notebook-only dependency
        raise RuntimeError("ipywidgets is required for interactive Colab controls.") from exc

    category = widgets.Dropdown(
        options=TrendScraperConfig().supported_categories,
        value=defaults.category,
        description="Category",
    )
    
    num_images = widgets.IntSlider(
        value=defaults.num_images,
        min=1,
        max=10,
        step=1,
        description="Images",
    )
    
    image_size = widgets.Dropdown(
        options=[(f"{s.value[0]}x{s.value[1]}", s) for s in ImageSize],
        value=defaults.image_size,
        description="Size",
    )

    box = widgets.VBox([category, num_images, image_size])
    
    return {
        "box": box,
        "category": category,
        "num_images": num_images,
        "image_size": image_size,
    }


def display_runtime_widgets(defaults: PipelineRuntimeOptions | None = None) -> Dict[str, object]:
    """Display the widgets and return them for later value collection."""

    from IPython.display import display  # type: ignore

    widgets = build_runtime_widgets(defaults)
    display(widgets["box"])
    return widgets


def runtime_from_widgets(widgets_map: Dict[str, object]) -> PipelineRuntimeOptions:
    """Build runtime options from simplified widget values."""

    return PipelineRuntimeOptions(
        category=str(widgets_map["category"].value),  # type: ignore[attr-defined]
        num_images=int(widgets_map["num_images"].value),  # type: ignore[attr-defined]
        image_size=ImageSize(widgets_map["image_size"].value),  # type: ignore[attr-defined]
    )
