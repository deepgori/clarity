"""
Clarity API - News Source Fetcher

Fetches recent news about a company.
Primary: DuckDuckGo search (free, no API key)
Fallback: NewsAPI.org (free tier, 100 req/day)
"""

import os
import logging
import httpx
from duckduckgo_search import DDGS
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)


async def _fetch_via_duckduckgo(company_name: str, domain: str) -> str | None:
    """Primary: Search DuckDuckGo for recent news about the company."""
    try:
        ddgs = DDGS()
        # Search for news specifically
        news_results = ddgs.news(
            f"{company_name} {domain}",
            max_results=5,
        )

        if not news_results:
            # Try a broader text search as backup
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
    """Fallback: Use NewsAPI.org free tier."""
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
    Tries DuckDuckGo first, falls back to NewsAPI.org.

    Gracefully handles "no news found" so the API still works without news data.
    """
    # Try DuckDuckGo first
    content = await _fetch_via_duckduckgo(company_name, domain)

    # Fall back to NewsAPI if DDG fails
    if content is None:
        logger.info("DDG failed, trying NewsAPI fallback...")
        content = await _fetch_via_newsapi(company_name)

    if content:
        # Truncate to avoid blowing up LLM context
        if len(content) > 5000:
            content = content[:5000] + "\n\n[News results truncated]"

        return SourceResult(
            source_type=SourceType.NEWS,
            url=f"https://duckduckgo.com/?q={company_name}+news",
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
