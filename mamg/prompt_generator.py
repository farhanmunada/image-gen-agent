"""Agent 2: prompt and metadata generation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .trend_scraper import TrendResult
from .utils import flatten_text


@dataclass(slots=True)
class PromptBundle:
    """Commercial image prompt plus microstock metadata."""

    trend_keyword: str
    title: str
    description: str
    prompt: str
    keywords: List[str]
    style: str
    visual_type: str
    category: str


@dataclass(slots=True)
class PromptGeneratorConfig:
    """Heuristic settings for prompt generation."""

    min_keywords: int = 30
    max_keywords: int = 50
    title_max_words: int = 7
    forbidden_terms: List[str] = field(
        default_factory=lambda: [
            "disney",
            "apple",
            "iphone",
            "elon musk",
            "tesla",
            "nike",
            "adidas",
            "spotify",
            "netflix",
            "microsoft",
            "openai",
        ]
    )


class PromptGenerator:
    """Convert trend keywords into stock-friendly image instructions."""

    def __init__(self, config: Optional[PromptGeneratorConfig] = None) -> None:
        self.config = config or PromptGeneratorConfig()

    def generate(self, trend_result: TrendResult) -> PromptBundle:
        """Build a commercial prompt bundle from a trend result."""

        keyword = self._pick_primary_keyword(trend_result.keywords)
        style = self._pick_style(trend_result.category, keyword)
        title = self._build_title(keyword, style)
        description = self._build_description(keyword, style, trend_result.category)
        prompt = self._build_prompt(keyword, style, trend_result.category)
        keywords = self._build_keywords(trend_result.keywords, keyword, style, trend_result.category)
        visual_type = self._pick_visual_type(trend_result.category, keyword)

        return PromptBundle(
            trend_keyword=keyword,
            title=title,
            description=description,
            prompt=prompt,
            keywords=keywords,
            style=style,
            visual_type=visual_type,
            category=trend_result.category,
        )

    def _pick_primary_keyword(self, keywords: List[str]) -> str:
        """Choose a safe, clean primary trend keyword."""

        for keyword in keywords:
            clean = self._clean_term(keyword)
            if clean:
                return clean
        return "abstract concept"

    def _pick_style(self, category: str, keyword: str) -> str:
        """Pick a visual direction based on category and trend content."""

        seed = f"{category} {keyword}".lower()
        if any(token in seed for token in ["ai", "tech", "software", "data", "cyber", "digital"]):
            return "futuristic commercial technology scene"
        if any(token in seed for token in ["office", "finance", "business", "startup", "strategy"]):
            return "clean corporate workspace"
        if any(token in seed for token in ["eco", "green", "nature", "sustainable", "environment"]):
            return "modern sustainable lifestyle"
        if any(token in seed for token in ["health", "medical", "wellness", "fitness"]):
            return "bright wellness editorial scene"
        if category == "Design/Arts":
            return "premium editorial design composition"
        return "high-demand commercial stock photography"

    def _pick_visual_type(self, category: str, keyword: str) -> str:
        """Select the visual family used for model and LoRA routing."""

        seed = f"{category} {keyword}".lower()
        if any(token in seed for token in ["3d", "render", "illustration", "cgi", "animation"]):
            return "3d_render"
        if any(token in seed for token in ["photo", "portrait", "lifestyle", "office", "business", "stock"]):
            return "photorealistic"
        if any(token in seed for token in ["vector", "icon", "graphic", "design", "poster", "layout"]):
            return "graphic_design"
        return "editorial"

    def _build_title(self, keyword: str, style: str) -> str:
        """Generate a concise title with the core keyword."""

        base = self._title_case(keyword)
        style_hint = self._title_case(style.split()[0])
        title = f"{style_hint} {base}"
        return self._limit_words(title, self.config.title_max_words)

    def _build_description(self, keyword: str, style: str, category: str) -> str:
        """Create an SEO-friendly stock description."""

        return flatten_text(
            f"High-quality {style} featuring {keyword} in a {category.lower()} context, "
            "designed for commercial stock use, clean composition, realistic details, and broad licensing appeal."
        )

    def _build_prompt(self, keyword: str, style: str, category: str) -> str:
        """Create a detailed image generation prompt."""

        negative_prompt = (
            "no brand logos, no watermark, no text, no copyrighted character, no celebrity, "
            "no distorted anatomy, no low resolution, no blurry details"
        )
        return flatten_text(
            f"{style}, centered composition, polished commercial stock image, {category.lower()} theme, "
            f"subject: {keyword}, soft natural lighting, balanced color palette, high detail, sharp focus, "
            f"minimal clutter, professional photography look, {negative_prompt}"
        )

    def _build_keywords(
        self,
        trend_keywords: List[str],
        primary_keyword: str,
        style: str,
        category: str,
    ) -> List[str]:
        """Expand seed keywords into a microstock keyword list."""

        base_keywords = [
            primary_keyword,
            category.lower(),
            "commercial",
            "stock",
            "high resolution",
            "photorealistic",
            "editorial",
            "modern",
            "professional",
            "clean background",
            "copy space",
            "creative",
            "visual trend",
            "concept",
            "marketing",
            "branding",
            "advertising",
            "minimalist",
            "high detail",
            "soft light",
            "studio lighting",
            "trend",
            "business concept",
            "visual storytelling",
            "digital asset",
            "commercial use",
            "fresh idea",
            "content creation",
            "premium quality",
            "art direction",
            "global trend",
            "future",
            "innovation",
            "design",
            "concept art",
            style,
        ]

        expanded: List[str] = []
        for keyword in trend_keywords + base_keywords:
            clean = self._clean_term(keyword)
            if not clean:
                continue
            if self._is_forbidden(clean):
                continue
            if clean.lower() in {item.lower() for item in expanded}:
                continue
            expanded.append(clean)
            if len(expanded) >= self.config.max_keywords:
                break

        while len(expanded) < self.config.min_keywords:
            filler = random.choice(
                [
                    "clean composition",
                    "copy space",
                    "modern aesthetics",
                    "commercial appeal",
                    "visual trend",
                    "high quality",
                    "professional image",
                ]
            )
            if filler.lower() not in {item.lower() for item in expanded}:
                expanded.append(filler)

        return expanded[: self.config.max_keywords]

    def _clean_term(self, value: str) -> str:
        """Normalize a keyword and remove prohibited characters."""

        value = flatten_text(value)
        value = re.sub(r"[^\w\s-]", "", value)
        value = flatten_text(value)
        return value

    def _is_forbidden(self, value: str) -> bool:
        """Avoid trademarks, brands, and celebrity names."""

        lowered = value.lower()
        return any(term in lowered for term in self.config.forbidden_terms)

    def _title_case(self, value: str) -> str:
        """Title-case a phrase while keeping it compact."""

        return " ".join(word.capitalize() for word in flatten_text(value).split())

    def _limit_words(self, value: str, max_words: int) -> str:
        """Limit a phrase to a specific word count."""

        words = flatten_text(value).split()
        return " ".join(words[:max_words])
