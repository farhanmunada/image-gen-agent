"""Colab widgets for collecting generation options from the user."""

from __future__ import annotations

import json
from typing import Dict

from .runtime import PipelineRuntimeOptions


def build_runtime_widgets(defaults: PipelineRuntimeOptions | None = None) -> Dict[str, object]:
    """Return a dictionary of ipywidgets for interactive Colab input."""

    defaults = defaults or PipelineRuntimeOptions()

    try:
        import ipywidgets as widgets  # type: ignore
    except Exception as exc:  # pragma: no cover - notebook-only dependency
        raise RuntimeError("ipywidgets is required for interactive Colab controls.") from exc

    category = widgets.Dropdown(
        options=["Design/Arts", "Business", "Technology"],
        value=defaults.category,
        description="Category",
    )
    geo = widgets.Text(value=defaults.geo, description="Geo")
    max_keywords = widgets.IntSlider(
        value=defaults.max_keywords,
        min=3,
        max=25,
        step=1,
        description="Trend tags",
    )
    num_images = widgets.IntSlider(
        value=defaults.num_images,
        min=1,
        max=10,
        step=1,
        description="Images",
    )
    model_name = widgets.Dropdown(
        options=[
            "stabilityai/sdxl-turbo",
            "black-forest-labs/FLUX.1-schnell",
        ],
        value=defaults.model_name,
        description="Model",
    )
    height = widgets.IntText(value=defaults.height, description="Height")
    width = widgets.IntText(value=defaults.width, description="Width")
    seed = widgets.IntText(value=defaults.seed or 0, description="Seed")
    use_cpu_offload = widgets.Checkbox(value=defaults.use_cpu_offload, description="CPU offload")
    enable_lora = widgets.Checkbox(value=defaults.enable_lora, description="LoRA")
    lora_paths = widgets.Textarea(
        value=", ".join(defaults.lora_paths),
        description="LoRA paths",
        layout=widgets.Layout(width="100%", height="70px"),
    )
    visual_type_lora_map = widgets.Textarea(
        value=json.dumps(defaults.visual_type_lora_map, ensure_ascii=False, indent=2),
        description="LoRA map",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    upscale_target = widgets.IntText(value=defaults.upscale_target, description="Upscale")
    num_inference_steps = widgets.IntText(value=defaults.num_inference_steps, description="Steps")
    guidance_scale = widgets.FloatText(value=defaults.guidance_scale, description="Guidance")
    prompt_min_keywords = widgets.IntText(value=defaults.prompt_min_keywords, description="Min tags")
    prompt_max_keywords = widgets.IntText(value=defaults.prompt_max_keywords, description="Max tags")

    box = widgets.VBox(
        [
            category,
            geo,
            max_keywords,
            num_images,
            model_name,
            height,
            width,
            seed,
            use_cpu_offload,
            enable_lora,
            lora_paths,
            visual_type_lora_map,
            upscale_target,
            num_inference_steps,
            guidance_scale,
            prompt_min_keywords,
            prompt_max_keywords,
        ]
    )
    return {
        "box": box,
        "category": category,
        "geo": geo,
        "max_keywords": max_keywords,
        "num_images": num_images,
        "model_name": model_name,
        "height": height,
        "width": width,
        "seed": seed,
        "use_cpu_offload": use_cpu_offload,
        "enable_lora": enable_lora,
        "lora_paths": lora_paths,
        "visual_type_lora_map": visual_type_lora_map,
        "upscale_target": upscale_target,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "prompt_min_keywords": prompt_min_keywords,
        "prompt_max_keywords": prompt_max_keywords,
    }


def display_runtime_widgets(defaults: PipelineRuntimeOptions | None = None) -> Dict[str, object]:
    """Display the widgets and return them for later value collection."""

    from IPython.display import display  # type: ignore

    widgets = build_runtime_widgets(defaults)
    display(widgets["box"])
    return widgets


def runtime_from_widgets(widgets_map: Dict[str, object]) -> PipelineRuntimeOptions:
    """Build runtime options from widget values."""

    seed_value = int(widgets_map["seed"].value)  # type: ignore[attr-defined]
    lora_paths_value = str(widgets_map["lora_paths"].value).strip()  # type: ignore[attr-defined]
    raw_map_value = str(widgets_map["visual_type_lora_map"].value).strip()  # type: ignore[attr-defined]
    parsed_map = {}
    if raw_map_value:
        try:
            parsed_map = json.loads(raw_map_value)
        except json.JSONDecodeError as exc:
            raise ValueError("LoRA map must be valid JSON, for example: {\"3d_render\": \"/path/to/lora.safetensors\"}") from exc
    return PipelineRuntimeOptions(
        category=str(widgets_map["category"].value),  # type: ignore[attr-defined]
        geo=str(widgets_map["geo"].value),  # type: ignore[attr-defined]
        max_keywords=int(widgets_map["max_keywords"].value),  # type: ignore[attr-defined]
        num_images=int(widgets_map["num_images"].value),  # type: ignore[attr-defined]
        model_name=str(widgets_map["model_name"].value),  # type: ignore[attr-defined]
        height=int(widgets_map["height"].value),  # type: ignore[attr-defined]
        width=int(widgets_map["width"].value),  # type: ignore[attr-defined]
        seed=None if seed_value <= 0 else seed_value,
        use_cpu_offload=bool(widgets_map["use_cpu_offload"].value),  # type: ignore[attr-defined]
        enable_lora=bool(widgets_map["enable_lora"].value),  # type: ignore[attr-defined]
        lora_paths=[path.strip() for path in lora_paths_value.split(",") if path.strip()],
        visual_type_lora_map={str(key): str(value) for key, value in parsed_map.items()},
        upscale_target=int(widgets_map["upscale_target"].value),  # type: ignore[attr-defined]
        num_inference_steps=int(widgets_map["num_inference_steps"].value),  # type: ignore[attr-defined]
        guidance_scale=float(widgets_map["guidance_scale"].value),  # type: ignore[attr-defined]
        prompt_min_keywords=int(widgets_map["prompt_min_keywords"].value),  # type: ignore[attr-defined]
        prompt_max_keywords=int(widgets_map["prompt_max_keywords"].value),  # type: ignore[attr-defined]
    )
