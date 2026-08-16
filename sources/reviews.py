"""
Clarity API - Customer Reviews Source (G2, Capterra, Trustpilot)

Fetches customer review data for a company's product(s).
G2 and Capterra are the primary B2B software review platforms.
Trustpilot serves as a fallback for broader company reputation.

These sites block direct scraping (DataDome, Cloudflare), so we use a
two-step approach:
1. DuckDuckGo search for "{company} site:g2.com reviews" to find the
   right product page and extract rating/review data from search snippets.
2. Jina Reader to fetch the actual page content if the URL is found.

This source enables a new contradiction type: company claims vs customer
experience. "Claims easy onboarding but G2 reviews cite 3-month setup"
is a high-credibility sales angle.
"""

import re
import asyncio
import httpx
import logging
from duckduckgo_search import DDGS
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

JINA_BASE_URL = "https://r.jina.ai"
REQUEST_TIMEOUT = 10.0
MAX_REVIEW_CHARS = 4000


def _extract_company_name(domain: str) -> str:
    """Extract a clean company name from domain for search."""
    base = domain.replace("www.", "").split(".")[0]
    return base.capitalize()


async def _search_reviews(company_name: str, domain: str, site: str) -> list[dict]:
    """
    Search for review pages on a specific site using DuckDuckGo.
    Returns search results with titles, snippets, and URLs.
    """
    query = f'"{company_name}" site:{site} reviews'
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=5)),
        )
        return results or []
    except Exception as e:
        logger.debug(f"Review search failed for {company_name} on {site}: {e}")
        return []


def _extract_rating_from_text(text: str) -> dict | None:
    """
    Extract star rating and review count from search snippet text.
    G2 snippets often contain: "4.5 out of 5 stars" or "Rating: 4.3/5"
    or "★ 4.5 (1,234 reviews)"
    """
    rating = None
    review_count = None

    # Pattern: "X.X out of 5" or "X out of 5"
    match = re.search(r'(\d+\.?\d*)\s*(?:out of|\/)\s*5(?:\s*star)?', text, re.IGNORECASE)
    if match:
        rating = float(match.group(1))

    # Pattern: "Rating: X.X" or "rated X.X"
    if not rating:
        match = re.search(r'(?:rating|rated)[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            rating = float(match.group(1))

    # Pattern: just "4.5 stars" or "4.5 ★"
    if not rating:
        match = re.search(r'(\d+\.\d+)\s*(?:stars?|★)', text, re.IGNORECASE)
        if match:
            rating = float(match.group(1))

    # Review count: "(1,234 reviews)" or "1234 reviews"
    count_match = re.search(r'([\d,]+)\s*reviews?', text, re.IGNORECASE)
    if count_match:
        review_count = int(count_match.group(1).replace(",", ""))

    if rating and 0 < rating <= 5:
        return {"rating": rating, "review_count": review_count}
    return None


async def _fetch_review_page(url: str) -> str | None:
    """Try to fetch a review page via Jina Reader for richer content."""
    try:
        jina_url = f"{JINA_BASE_URL}/{url}"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                jina_url,
                headers={"Accept": "text/markdown"},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200 and len(response.text.strip()) > 200:
                return response.text[:MAX_REVIEW_CHARS]
    except Exception as e:
        logger.debug(f"Jina fetch failed for review page {url}: {e}")
    return None


def _extract_review_themes(snippets: list[str]) -> list[str]:
    """
    Extract common themes from review snippets.
    Looks for pros/cons patterns and sentiment keywords.
    """
    themes = []
    all_text = " ".join(snippets).lower()

    # Negative indicators
    negative_terms = {
        "slow": "Performance complaints",
        "expensive": "Pricing concerns",
        "complex": "Complexity/learning curve complaints",
        "difficult": "Usability concerns",
        "buggy": "Reliability/bug complaints",
        "support": "Customer support mentioned",
        "outage": "Reliability/outage complaints",
        "missing": "Missing features noted",
        "clunky": "UX/interface complaints",
        "steep learning curve": "Steep learning curve",
        "onboarding": "Onboarding friction mentioned",
        "integration": "Integration challenges noted",
        "downtime": "Downtime/reliability issues",
        "poor documentation": "Documentation gaps",
    }

    # Positive indicators
    positive_terms = {
        "easy to use": "Praised for ease of use",
        "intuitive": "Praised for intuitive UX",
        "fast": "Praised for speed/performance",
        "reliable": "Praised for reliability",
        "great support": "Praised for customer support",
        "love": "Strong positive sentiment",
        "recommend": "Users recommend to others",
        "best": "Considered best-in-class",
        "affordable": "Praised for value/pricing",
    }

    for term, theme in negative_terms.items():
        if term in all_text:
            themes.append(f"NEGATIVE: {theme}")

    for term, theme in positive_terms.items():
        if term in all_text:
            themes.append(f"POSITIVE: {theme}")

    return themes


def _format_review_data(
    g2_results: list[dict],
    capterra_results: list[dict],
    trustpilot_results: list[dict],
    company_name: str,
) -> str:
    """Format all review data into structured text for synthesis."""
    parts = ["Source: Customer Review Platforms (G2, Capterra, Trustpilot)"]
    found_any = False
    all_snippets = []

    # Process G2
    if g2_results:
        parts.append("")
        parts.append("=== G2 REVIEWS ===")
        for r in g2_results[:3]:
            title = r.get("title", "")
            snippet = r.get("body", "")
            url = r.get("href", "")

            rating_info = _extract_rating_from_text(f"{title} {snippet}")
            if rating_info:
                found_any = True
                rating_str = f"Rating: {rating_info['rating']}/5"
                if rating_info.get("review_count"):
                    rating_str += f" ({rating_info['review_count']} reviews)"
                parts.append(f"  {rating_str}")

            if snippet:
                all_snippets.append(snippet)
                parts.append(f"  Snippet: {snippet[:300]}")
            if url:
                parts.append(f"  URL: {url}")
            parts.append("")

    # Process Capterra
    if capterra_results:
        parts.append("=== CAPTERRA REVIEWS ===")
        for r in capterra_results[:3]:
            title = r.get("title", "")
            snippet = r.get("body", "")
            url = r.get("href", "")

            rating_info = _extract_rating_from_text(f"{title} {snippet}")
            if rating_info:
                found_any = True
                rating_str = f"Rating: {rating_info['rating']}/5"
                if rating_info.get("review_count"):
                    rating_str += f" ({rating_info['review_count']} reviews)"
                parts.append(f"  {rating_str}")

            if snippet:
                all_snippets.append(snippet)
                parts.append(f"  Snippet: {snippet[:300]}")
            if url:
                parts.append(f"  URL: {url}")
            parts.append("")

    # Process Trustpilot
    if trustpilot_results:
        parts.append("=== TRUSTPILOT REVIEWS ===")
        for r in trustpilot_results[:3]:
            title = r.get("title", "")
            snippet = r.get("body", "")
            url = r.get("href", "")

            rating_info = _extract_rating_from_text(f"{title} {snippet}")
            if rating_info:
                found_any = True
                rating_str = f"Rating: {rating_info['rating']}/5"
                if rating_info.get("review_count"):
                    rating_str += f" ({rating_info['review_count']} reviews)"
                parts.append(f"  {rating_str}")

            if snippet:
                all_snippets.append(snippet)
                parts.append(f"  Snippet: {snippet[:300]}")
            if url:
                parts.append(f"  URL: {url}")
            parts.append("")

    # Extract themes from all snippets
    if all_snippets:
        themes = _extract_review_themes(all_snippets)
        if themes:
            parts.append("KEY REVIEW THEMES (use for contradiction detection):")
            for theme in themes:
                parts.append(f"  - {theme}")

    if not found_any:
        return ""

    return "\n".join(parts)


async def fetch_reviews(domain: str) -> SourceResult:
    """
    Fetch customer review data from G2, Capterra, and Trustpilot.

    Uses DuckDuckGo search to find review pages and extract ratings
    and sentiment from search snippets, since these platforms block
    direct scraping.
    """
    company_name = _extract_company_name(domain)
    base_name = domain.replace("www.", "").split(".")[0]

    logger.info(f"Searching for customer reviews: {company_name}")

    # Search all three platforms in parallel
    g2_task = _search_reviews(base_name, domain, "g2.com")
    capterra_task = _search_reviews(base_name, domain, "capterra.com")
    trustpilot_task = _search_reviews(base_name, domain, "trustpilot.com")

    try:
        g2_results, capterra_results, trustpilot_results = await asyncio.wait_for(
            asyncio.gather(g2_task, capterra_task, trustpilot_task),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Review search timed out for {domain}")
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url=f"https://www.g2.com/search?query={base_name}",
            content="Review search timed out.",
            fetched=False,
            error="Timeout searching review platforms",
        )

    total_results = len(g2_results) + len(capterra_results) + len(trustpilot_results)
    logger.info(
        f"Review search results: G2={len(g2_results)}, "
        f"Capterra={len(capterra_results)}, Trustpilot={len(trustpilot_results)}"
    )

    if total_results == 0:
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url=f"https://www.g2.com/search?query={base_name}",
            content=(
                f"No customer reviews found for {company_name} on G2, Capterra, or Trustpilot.\n"
                f"This company may not have a listed product on major review platforms,\n"
                f"or may be too early-stage for reviews."
            ),
            fetched=True,
        )

    formatted = _format_review_data(
        g2_results, capterra_results, trustpilot_results, company_name
    )

    if not formatted:
        return SourceResult(
            source_type=SourceType.REVIEWS,
            url=f"https://www.g2.com/search?query={base_name}",
            content=f"Review pages found for {company_name} but no structured rating data could be extracted.",
            fetched=True,
        )

    return SourceResult(
        source_type=SourceType.REVIEWS,
        url=f"https://www.g2.com/search?query={base_name}",
        content=formatted,
        fetched=True,
    )
