"""
Clarity API - Website Source Fetcher

Fetches and extracts clean text content from a company's website.
Primary: Jina Reader API (handles SPAs + static sites, returns markdown)
Fallback: httpx + trafilatura (static HTML only, no external API dependency)
"""

import asyncio
import re
import httpx
import trafilatura
import logging
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

# Pages to fetch from any company website
TARGET_PAGES = ["", "/about", "/pricing", "/careers", "/products", "/blog", "/engineering"]

# Pages with strong claims worth capturing in full
CLAIMS_DENSE_PAGES = {"/about", "/pricing"}

# Pages with high-signal content (pain points, priorities, technical decisions)
HIGH_SIGNAL_PAGES = {"/blog", "/engineering"}

JINA_BASE_URL = "https://r.jina.ai"
REQUEST_TIMEOUT = 15.0
PAGE_TIMEOUT = 15.0  # max seconds per page (prevents one slow page from blocking)
MAX_CHARS_DEFAULT = 2500
MAX_CHARS_CLAIMS = 4000  # more context for claims-dense pages
MAX_CHARS_BLOG = 3000   # blog posts reveal pain points and priorities
MAX_TOTAL_CHARS = 16000


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
        if page_path in CLAIMS_DENSE_PAGES:
            max_chars = MAX_CHARS_CLAIMS
        elif page_path in HIGH_SIGNAL_PAGES:
            max_chars = MAX_CHARS_BLOG
        else:
            max_chars = MAX_CHARS_DEFAULT
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
    Also detects tech stack from HTTP response headers.

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

        # Also detect tech stack from HTTP headers (lightweight HEAD request)
        tech_fingerprint = await _detect_tech_stack(base_url, client)

    # Collect successful fetches
    all_content = []
    for label, content in results:
        if content:
            all_content.append(f"=== {label} ===\n{content}")

    # Append tech fingerprint if detected
    if tech_fingerprint:
        all_content.append(f"=== DETECTED TECH STACK (from HTTP headers) ===\n{tech_fingerprint}")

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


async def _detect_tech_stack(base_url: str, client: httpx.AsyncClient) -> str | None:
    """
    Detect tech stack from HTTP response headers and common fingerprints.
    Uses a lightweight HEAD request + one GET for script/meta tag analysis.
    """
    detected = []

    try:
        # HEAD request for server headers
        response = await client.head(base_url, timeout=5.0, follow_redirects=True)
        headers = response.headers

        # Server header
        server = headers.get("server", "")
        if server:
            detected.append(f"Server: {server}")

        # X-Powered-By
        powered_by = headers.get("x-powered-by", "")
        if powered_by:
            detected.append(f"X-Powered-By: {powered_by}")

        # Common CDN/platform headers
        if "x-vercel-id" in headers or "x-vercel-cache" in headers:
            detected.append("Platform: Vercel")
        if "x-amz-cf-id" in headers or "x-amz-cf-pop" in headers:
            detected.append("CDN: AWS CloudFront")
        if "cf-ray" in headers or "cf-cache-status" in headers:
            detected.append("CDN: Cloudflare")
        if "x-netlify" in headers:
            detected.append("Platform: Netlify")
        if "x-github-request-id" in headers:
            detected.append("Platform: GitHub Pages")
        if "x-shopify-stage" in headers:
            detected.append("Platform: Shopify")
        if "x-wix-request-id" in headers:
            detected.append("Platform: Wix")

        # Now do a GET to check for script/meta fingerprints in the HTML
        page_response = await client.get(base_url, timeout=5.0, follow_redirects=True)
        if page_response.status_code == 200:
            html = page_response.text[:50000]  # Only scan first 50KB

            # Framework detection from script tags and meta
            framework_patterns = [
                (r'__NEXT_DATA__', 'Framework: Next.js'),
                (r'__NUXT__', 'Framework: Nuxt.js'),
                (r'data-reactroot|_reactRoot|__react', 'Framework: React'),
                (r'ng-version|ng-app', 'Framework: Angular'),
                (r'data-svelte|__svelte', 'Framework: Svelte'),
                (r'__VUE__', 'Framework: Vue.js'),
                (r'wp-content|wordpress', 'CMS: WordPress'),
                (r'hubspot', 'Marketing: HubSpot'),
                (r'Shopify\.theme', 'Platform: Shopify'),
                (r'segment\.com/analytics|analytics\.js', 'Analytics: Segment'),
                (r'googletagmanager\.com', 'Analytics: Google Tag Manager'),
                (r'hotjar\.com', 'Analytics: Hotjar'),
                (r'intercom', 'Support: Intercom'),
                (r'zendesk', 'Support: Zendesk'),
                (r'sentry\.io|sentry-cdn', 'Monitoring: Sentry'),
                (r'datadog', 'Monitoring: Datadog'),
                (r'launchdarkly', 'Feature Flags: LaunchDarkly'),
                (r'stripe\.com/v3|Stripe\.js', 'Payments: Stripe'),
            ]

            for pattern, label in framework_patterns:
                if re.search(pattern, html, re.IGNORECASE):
                    detected.append(label)

    except Exception as e:
        logger.debug(f"Tech stack detection failed: {e}")

    if detected:
        # Deduplicate
        seen = set()
        unique = []
        for d in detected:
            if d not in seen:
                seen.add(d)
                unique.append(d)
        return "\n".join(unique)
    return None
