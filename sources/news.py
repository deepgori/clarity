"""
Clarity API - News Source Fetcher

Fetches recent news about a company.
Primary: Google News RSS (free, no API key, reliable, no rate limits)
Fallback 1: DuckDuckGo search
Fallback 2: NewsAPI.org (free tier, 100 req/day)
"""

import os
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import httpx
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10.0


async def _fetch_via_google_news_rss(company_name: str, domain: str) -> str | None:
    """Primary: Fetch news via Google News RSS feed. Free, no API key, reliable."""
    try:
        query = quote_plus(f"{company_name} {domain}")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "ClarityBot/1.0 (company-research)"},
            )

        if response.status_code != 200:
            logger.warning(f"Google News RSS returned {response.status_code}")
            return None

        root = ET.fromstring(response.text)
        items = root.findall(".//item")

        if not items:
            return None

        formatted = []
        for item in items[:8]:  # Top 8 recent articles
            title = item.find("title")
            pub_date = item.find("pubDate")
            source = item.find("source")
            link = item.find("link")
            description = item.find("description")

            title_text = title.text if title is not None else ""
            date_text = pub_date.text if pub_date is not None else ""
            source_text = source.text if source is not None else ""
            link_text = link.text if link is not None else ""
            # Description from Google News RSS is HTML, extract text
            desc_text = ""
            if description is not None and description.text:
                # Strip HTML tags from description
                import re
                desc_text = re.sub(r"<[^>]+>", "", description.text).strip()

            formatted.append(
                f"HEADLINE: {title_text}\n"
                f"SOURCE: {source_text}\n"
                f"DATE: {date_text}\n"
                f"URL: {link_text}\n"
                f"SUMMARY: {desc_text}\n"
            )

        logger.info(f"Google News RSS: {len(formatted)} articles found for {company_name}")
        return "\n---\n".join(formatted)

    except Exception as e:
        logger.warning(f"Google News RSS failed: {e}")
        return None


async def _fetch_via_duckduckgo(company_name: str, domain: str) -> str | None:
    """Fallback 1: Search DuckDuckGo for recent news about the company."""
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        news_results = ddgs.news(
            f"{company_name} {domain}",
            max_results=5,
        )

        if not news_results:
            news_results = ddgs.text(
                f"{company_name} news funding launch 2024 2025 2026",
                max_results=5,
            )

        if news_results:
            formatted = []
            for item in news_results:
                title = item.get("title", "")
                body = item.get("body", item.get("snippet", ""))
                source = item.get("source", item.get("href", ""))
                date = item.get("date", "")
                formatted.append(
                    f"HEADLINE: {title}\n"
                    f"SOURCE: {source}\n"
                    f"DATE: {date}\n"
                    f"SUMMARY: {body}\n"
                )
            return "\n---\n".join(formatted)

        return None

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return None


async def _fetch_via_newsapi(company_name: str) -> str | None:
    """Fallback 2: Use NewsAPI.org free tier."""
    api_key = os.getenv("NEWS_API_KEY", "").strip()
    if not api_key:
        logger.info("No NEWS_API_KEY set, skipping NewsAPI fallback")
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": company_name,
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": api_key,
                },
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                if articles:
                    formatted = []
                    for article in articles:
                        formatted.append(
                            f"HEADLINE: {article.get('title', '')}\n"
                            f"SOURCE: {article.get('source', {}).get('name', '')}\n"
                            f"DATE: {article.get('publishedAt', '')}\n"
                            f"SUMMARY: {article.get('description', '')}\n"
                        )
                    return "\n---\n".join(formatted)
        return None

    except Exception as e:
        logger.warning(f"NewsAPI fallback failed: {e}")
        return None


async def fetch_news(company_name: str, domain: str) -> SourceResult:
    """
    Fetch recent news about a company.
    Priority: Google News RSS > DuckDuckGo > NewsAPI.org.
    """
    # Try Google News RSS first (most reliable)
    content = await _fetch_via_google_news_rss(company_name, domain)

    # Fall back to DuckDuckGo
    if content is None:
        logger.info("Google News RSS failed, trying DuckDuckGo...")
        content = await _fetch_via_duckduckgo(company_name, domain)

    # Fall back to NewsAPI
    if content is None:
        logger.info("DDG also failed, trying NewsAPI fallback...")
        content = await _fetch_via_newsapi(company_name)

    if content:
        # Truncate to avoid blowing up LLM context
        if len(content) > 5000:
            content = content[:5000] + "\n\n[News results truncated]"

        return SourceResult(
            source_type=SourceType.NEWS,
            url=f"https://news.google.com/search?q={company_name}",
            content=content,
            fetched=True,
        )
    else:
        return SourceResult(
            source_type=SourceType.NEWS,
            url="",
            content="",
            fetched=False,
            error="No recent news found for this company",
        )
