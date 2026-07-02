"""
Clarity API - Website Source Fetcher

Fetches and extracts clean text content from a company's website.
Primary: Jina Reader API (handles SPAs + static sites, returns markdown)
Fallback: httpx + trafilatura (static HTML only, no external API dependency)
"""

import asyncio
import httpx
import trafilatura
import logging
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

# Pages to fetch from any company website
TARGET_PAGES = ["", "/about", "/pricing", "/careers", "/products"]

# Pages with strong claims worth capturing in full
CLAIMS_DENSE_PAGES = {"/about", "/pricing"}

JINA_BASE_URL = "https://r.jina.ai"
REQUEST_TIMEOUT = 15.0
PAGE_TIMEOUT = 15.0  # max seconds per page (prevents one slow page from blocking)
MAX_CHARS_DEFAULT = 2500
MAX_CHARS_CLAIMS = 4000  # more context for claims-dense pages
MAX_TOTAL_CHARS = 12000


async def _fetch_via_jina(url: str, client: httpx.AsyncClient) -> str | None:
    """Fetch clean markdown via Jina Reader API. Handles SPAs and JS-rendered sites."""
    try:
        jina_url = f"{JINA_BASE_URL}/{url}"
        response = await client.get(
            jina_url,
            headers={"Accept": "text/markdown"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200 and len(response.text.strip()) > 100:
            return response.text
        return None
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")
        return None


async def _fetch_via_trafilatura(url: str, client: httpx.AsyncClient) -> str | None:
    """Fallback: fetch raw HTML and extract content with trafilatura."""
    try:
        response = await client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        if response.status_code == 200:
            extracted = trafilatura.extract(response.text)
            if extracted and len(extracted.strip()) > 50:
                return extracted
        return None
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning(f"Trafilatura fallback failed for {url}: {e}")
        return None


async def _fetch_single_page(
    base_url: str, page_path: str, client: httpx.AsyncClient
) -> tuple[str, str | None]:
    """Fetch a single page with a hard timeout. Returns (label, content)."""
    url = f"{base_url}{page_path}"
    section_label = page_path.strip("/").upper() or "HOMEPAGE"

    try:
        # Hard timeout per page so one slow page can't block everything
        content = await asyncio.wait_for(
            _fetch_page_content(url, client),
            timeout=PAGE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Page timeout after {PAGE_TIMEOUT}s: {url}")
        return section_label, None

    if content:
        logger.info(f"Fetched {url} ({len(content)} chars)")
        max_chars = MAX_CHARS_CLAIMS if page_path in CLAIMS_DENSE_PAGES else MAX_CHARS_DEFAULT
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[Page truncated]"
        return section_label, content
    else:
        logger.info(f"No content from {url}")
        return section_label, None


async def _fetch_page_content(url: str, client: httpx.AsyncClient) -> str | None:
    """Try Jina first, then trafilatura."""
    content = await _fetch_via_jina(url, client)
    if content is None:
        content = await _fetch_via_trafilatura(url, client)
    return content


async def fetch_website(domain: str) -> SourceResult:
    """
    Fetch content from a company's website across multiple pages in parallel.
    Tries Jina Reader first (handles SPAs), falls back to trafilatura.

    Returns a SourceResult with concatenated content from all available pages.
    """
    base_url = f"https://{domain}"

    async with httpx.AsyncClient(
        headers={"User-Agent": "ClarityBot/1.0 (company-research)"},
        follow_redirects=True,
    ) as client:
        # Fetch all pages in parallel for speed
        results = await asyncio.gather(
            *[_fetch_single_page(base_url, page, client) for page in TARGET_PAGES]
        )

    # Collect successful fetches
    all_content = []
    for label, content in results:
        if content:
            all_content.append(f"=== {label} ===\n{content}")

    if all_content:
        combined = "\n\n".join(all_content)
        # Final safety truncation
        if len(combined) > MAX_TOTAL_CHARS:
            combined = combined[:MAX_TOTAL_CHARS] + "\n\n[Content truncated for processing]"

        return SourceResult(
            source_type=SourceType.WEBSITE,
            url=base_url,
            content=combined,
            fetched=True,
        )
    else:
        return SourceResult(
            source_type=SourceType.WEBSITE,
            url=base_url,
            content="",
            fetched=False,
            error=f"Could not extract content from {domain}",
        )
