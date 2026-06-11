"""Trend scraping agent for Google Trends and related sources."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

from .config import TrendScraperConfig


@dataclass(slots=True)
class TrendResult:
    """Structured result produced by the trend scraper."""

    category: str
    keywords: List[str]
    source: str
    fetched_at: str
    raw: Dict[str, Any] = field(default_factory=dict)


class TrendScraper:
    """Fetches trend keywords using pytrends first, then HTML/RSS fallback."""

    def __init__(self, config: Optional[TrendScraperConfig] = None) -> None:
        self.config = config or TrendScraperConfig()
        self.config.validate()

    def scrape(self) -> TrendResult:
        """Return the strongest trend keywords for the configured category."""

        keywords = self._from_pytrends()
        source = "pytrends"

        if not keywords:
            logger.warning("Pytrends did not return keywords, using RSS fallback.")
            keywords = self._from_google_trends_rss()
            source = "google_trends_rss"

        if not keywords:
            logger.warning("RSS fallback did not return keywords, using HTML fallback.")
            keywords = self._from_google_trends_html()
            source = "google_trends_html"

        keywords = self._normalize_keywords(keywords)
        keywords = keywords[: self.config.max_keywords]

        if not keywords:
            logger.warning("No trend keywords were found for category '%s'.", self.config.category)

        return TrendResult(
            category=self.config.category,
            keywords=keywords,
            source=source,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            raw={"category": self.config.category, "geo": self.config.geo},
        )

    def to_json(self) -> str:
        """Serialize the current scrape result as JSON."""

        return json.dumps(asdict(self.scrape()), ensure_ascii=False, indent=2)

    def _from_pytrends(self) -> List[str]:
        """Fetch related queries and rising trends through pytrends."""

        try:
            from pytrends.request import TrendReq  # type: ignore
        except Exception:
            return []

        try:
            pytrends = TrendReq(
                hl=self.config.hl,
                tz=self.config.tz,
                timeout=(10, 25),
                retries=2,
                backoff_factor=0.2,
            )

            trending = pytrends.trending_searches(pn=self.config.fallback_region)
            if trending is not None and not trending.empty:
                values = trending.iloc[: self.config.max_keywords, 0].astype(str).tolist()
                if values:
                    return values

            kw_list = [self._category_seed(self.config.category)]
            pytrends.build_payload(kw_list, timeframe=self.config.timeframe, geo=self.config.geo)
            related = pytrends.related_queries()
            related_data = related.get(kw_list[0], {})

            rising = related_data.get("rising")
            if rising is not None and not rising.empty:
                return rising["query"].astype(str).tolist()

            top = related_data.get("top")
            if top is not None and not top.empty:
                return top["query"].astype(str).tolist()
        except Exception as exc:
            logger.debug("Pytrends fetch failed: %s", exc, exc_info=True)
            return []

        return []

    def _from_google_trends_rss(self) -> List[str]:
        """Fetch daily trending searches via the public RSS endpoint."""

        url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={self.config.geo}"
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.config.user_agent},
                timeout=20,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "xml")
            items = soup.find_all("item")
            keywords: List[str] = []
            for item in items:
                title_tag = item.find("title")
                if title_tag and title_tag.text:
                    keywords.append(title_tag.text.strip())
                if len(keywords) >= self.config.max_keywords:
                    break
            return keywords
        except Exception as exc:
            logger.warning("Google Trends RSS fetch failed: %s", exc)
            return []

    def _from_google_trends_html(self) -> List[str]:
        """Fallback scraper for the Google Trends trending page."""

        url = "https://trends.google.com/trends/trendingsearches/daily"
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.config.user_agent},
                timeout=20,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            keywords: List[str] = []
            for element in soup.select("a, span, div"):
                text = self._clean_text(element.get_text(" ", strip=True))
                if self._looks_like_keyword(text):
                    keywords.append(text)
                if len(keywords) >= self.config.max_keywords:
                    break
            return keywords
        except Exception as exc:
            logger.warning("Google Trends HTML fetch failed: %s", exc)
            return []

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """Remove duplicates and scrub unusable values."""

        seen = set()
        normalized: List[str] = []
        for keyword in keywords:
            clean = self._clean_text(keyword)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(clean)
        return normalized

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove obvious junk."""

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
        return text

    def _looks_like_keyword(self, text: str) -> bool:
        """Heuristic filter for HTML fallback content."""

        if not text:
            return False
        if len(text) < 3 or len(text) > 80:
            return False
        if text.lower() in {"google trends", "trending searches"}:
            return False
        return any(char.isalpha() for char in text)

    def _category_seed(self, category: str) -> str:
        """Create a single seed keyword for pytrends payload building."""

        mapping = {
            "Design/Arts": "design",
            "Business": "business",
            "Technology": "technology",
        }
        return mapping.get(category, "design")
