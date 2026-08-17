"""
Clarity API - Customer Reviews Source (via SerpAPI)

Fetches customer review data by searching Google for "{company} reviews"
via SerpAPI. Google's search results contain structured rich snippets
from G2, Capterra, Gartner Peer Insights, Trustpilot, and TrustRadius,
all platforms that block direct access.

This gives us verified ratings, review counts, and sentiment snippets
from locked-down review platforms without scraping any of them directly.

One SerpAPI call per company (~$0.01 cost).
"""

import os
import asyncio
import httpx
import logging
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

SERPAPI_BASE = "https://serpapi.com/search"
REQUEST_TIMEOUT = 15.0

# Platforms we care about for B2B review data
REVIEW_PLATFORMS = {
    "g2.com": "G2",
    "capterra.com": "Capterra",
    "gartner.com": "Gartner Peer Insights",
    "trustradius.com": "TrustRadius",
    "trustpilot.com": "Trustpilot",
    "reddit.com": "Reddit",
    "glassdoor.com": "Glassdoor",
    "indeed.com": "Indeed",
}


def _extract_company_name(domain: str) -> str:
    """Extract a clean company name from domain for search."""
    base = domain.replace("www.", "").split(".")[0]
    return base.capitalize()


def _identify_platform(url: str) -> str | None:
    """Check if a URL belongs to a known review platform."""
    url_lower = url.lower()
    for domain, name in REVIEW_PLATFORMS.items():
        if domain in url_lower:
            return name
    return None


def _parse_serp_results(data: dict, company_name: str) -> str:
    """Parse SerpAPI response into structured review text for synthesis."""
    parts = [f"Source: Customer Reviews for {company_name} (via Google Search)"]
    
    ratings_found = []
    reddit_threads = []
    other_mentions = []

    # Extract People Also Ask (reveals buyer concerns)
    related_questions = data.get("related_questions", [])
    if related_questions:
        parts.append("")
        parts.append("BUYER QUESTIONS (People Also Ask):")
        for q in related_questions[:5]:
            question = q.get("question", "")
            if question:
                parts.append(f"  - {question}")

    # Extract organic results
    for result in data.get("organic_results", []):
        url = result.get("link", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        source = result.get("source", "")
        platform = _identify_platform(url)

        # Extract rich snippet ratings
        rich_snippet = result.get("rich_snippet", {})
        if rich_snippet:
            top = rich_snippet.get("top", {})
            extensions = top.get("detected_extensions", {})
            rating = extensions.get("rating")
            review_count = extensions.get("reviews")

            if rating and platform:
                # Normalize TrustRadius (rates out of 10, not 5)
                if platform == "TrustRadius" and rating > 5:
                    display_rating = f"{rating}/10"
                else:
                    display_rating = f"{rating}/5"

                count_str = f" ({review_count:,} reviews)" if review_count else ""
                ratings_found.append(
                    f"  {platform}: {display_rating}{count_str}"
                )

        # Capture Reddit threads
        if platform == "Reddit":
            reddit_threads.append(f"  [{source}] {title}")
            if snippet:
                reddit_threads.append(f"    \"{snippet[:200]}\"")

        # Capture other review mentions without rich snippets
        elif platform and not rich_snippet:
            if snippet:
                other_mentions.append(f"  [{platform}] {snippet[:200]}")

        # Capture blog/comparison reviews with ratings
        elif rich_snippet and not platform:
            top = rich_snippet.get("top", {})
            extensions = top.get("detected_extensions", {})
            rating = extensions.get("rating")
            if rating:
                ratings_found.append(
                    f"  {source}: {rating}/5 (independent review)"
                )

    # Format ratings section
    if ratings_found:
        parts.append("")
        parts.append("CUSTOMER RATINGS (from review platforms):")
        for r in ratings_found:
            parts.append(r)

    # Format Reddit section
    if reddit_threads:
        parts.append("")
        parts.append("REDDIT DISCUSSIONS:")
        for t in reddit_threads:
            parts.append(t)

    # Format other mentions
    if other_mentions:
        parts.append("")
        parts.append("OTHER REVIEW MENTIONS:")
        for m in other_mentions:
            parts.append(m)

    # If we found basically nothing
    if not ratings_found and not reddit_threads and not related_questions:
        return ""

    return "\n".join(parts)


async def fetch_reviews(domain: str) -> SourceResult:
    """
    Fetch customer review data via SerpAPI Google search.

    Searches for "{company} reviews" and extracts:
    - Rich snippet ratings from G2, Capterra, Gartner, TrustRadius, Trustpilot
    - Reddit thread titles and snippets
    - People Also Ask questions (reveals buyer concerns)
    - Blog review ratings and snippets
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url="",
            content="Review search unavailable (no SerpAPI key configured).",
            fetched=False,
            error="SERPAPI_KEY not configured",
        )

    company_name = _extract_company_name(domain)
    base_name = domain.replace("www.", "").split(".")[0]

    logger.info(f"Searching for customer reviews: {company_name}")

    try:
        async with httpx.AsyncClient() as client:
            response = await asyncio.wait_for(
                client.get(
                    SERPAPI_BASE,
                    params={
                        "q": f"{company_name} reviews",
                        "api_key": api_key,
                        "engine": "google",
                        "num": 10,
                        "gl": "us",
                        "hl": "en",
                    },
                    timeout=REQUEST_TIMEOUT,
                ),
                timeout=REQUEST_TIMEOUT + 2,
            )

        if response.status_code != 200:
            logger.warning(f"SerpAPI returned {response.status_code} for {domain}")
            return SourceResult(
                source_type=SourceType.REVIEWS,
                url=f"https://www.google.com/search?q={base_name}+reviews",
                content="Review search failed.",
                fetched=False,
                error=f"SerpAPI error: {response.status_code}",
            )

        data = response.json()
        formatted = _parse_serp_results(data, company_name)

        if not formatted:
            return SourceResult(
                source_type=SourceType.REVIEWS,
                url=f"https://www.google.com/search?q={base_name}+reviews",
                content=(
                    f"No customer review data found for {company_name}. "
                    f"This company may not have reviews on major platforms."
                ),
                fetched=True,
            )

        logger.info(f"Review data extracted for {company_name} ({len(formatted)} chars)")
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url=f"https://www.google.com/search?q={base_name}+reviews",
            content=formatted,
            fetched=True,
        )

    except asyncio.TimeoutError:
        logger.warning(f"Review search timed out for {domain}")
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url=f"https://www.google.com/search?q={base_name}+reviews",
            content="Review search timed out.",
            fetched=False,
            error="SerpAPI request timed out",
        )
    except Exception as e:
        logger.warning(f"Review search failed for {domain}: {e}")
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url=f"https://www.google.com/search?q={base_name}+reviews",
            content="Review search encountered an error.",
            fetched=False,
            error=str(e),
        )
