"""
Clarity API - Community Sentiment Source (Hacker News)

Fetches public discussion threads from Hacker News via the Algolia API.
HN threads are brutally honest technical discussions about companies and
products, written by the exact audience (developers, engineers, technical
buyers) whose opinions matter most for B2B sales.

This source enables a new contradiction type: company claims vs public
community sentiment. "Claims enterprise-ready but HN threads describe
scaling problems" is a sales angle that converts.

API: https://hn.algolia.com/api/v1/search
- Free, no auth, no API key required
- Returns structured JSON with title, points, comments, date
- Stable (powered by Algolia infrastructure)
"""

import asyncio
import httpx
import logging
from datetime import datetime, timezone
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
REQUEST_TIMEOUT = 10.0
MAX_STORIES = 10  # Top stories to analyze
MAX_COMMENTS_PER_STORY = 5  # Top comments to fetch per story


def _extract_search_terms(domain: str) -> list[str]:
    """
    Generate search terms from a domain.
    stripe.com -> ["stripe", "stripe.com"]
    notion.so -> ["notion", "notion.so"]
    """
    base = domain.replace("www.", "").split(".")[0].lower()
    terms = [base]
    # Add full domain for more specific results
    if domain != base:
        terms.append(domain.replace("www.", ""))
    return terms


async def _fetch_hn_stories(
    query: str, client: httpx.AsyncClient
) -> list[dict] | None:
    """
    Fetch top HN stories mentioning the query.
    Returns stories sorted by relevance with points and comment counts.
    """
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": MAX_STORIES,
    }
    try:
        response = await client.get(
            HN_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            if hits:
                return hits
        return None
    except Exception as e:
        logger.debug(f"HN story search failed for '{query}': {e}")
        return None


async def _fetch_hn_comments(
    story_id: str, client: httpx.AsyncClient
) -> list[dict] | None:
    """
    Fetch top comments for a specific HN story.
    Comments contain the actual sentiment and opinions.
    """
    params = {
        "tags": f"comment,story_{story_id}",
        "hitsPerPage": MAX_COMMENTS_PER_STORY,
    }
    try:
        response = await client.get(
            HN_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("hits", [])
        return None
    except Exception as e:
        logger.debug(f"HN comment fetch failed for story {story_id}: {e}")
        return None


def _clean_comment_text(text: str | None) -> str:
    """Clean HTML from HN comment text."""
    if not text:
        return ""
    import re
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    # Truncate
    if len(clean) > 300:
        clean = clean[:300] + "..."
    return clean


def _time_ago(date_str: str) -> str:
    """Convert ISO date to relative time string."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt
        days = diff.days
        if days < 30:
            return f"{days} days ago"
        elif days < 365:
            months = days // 30
            return f"{months} months ago"
        else:
            years = days // 365
            return f"{years} years ago"
    except Exception:
        return "unknown date"


def _analyze_hn_data(
    stories: list[dict],
    comments_by_story: dict[str, list[dict]],
    company_name: str,
) -> str:
    """
    Analyze HN stories and comments to produce structured sentiment text.
    Extracts: discussion themes, sentiment signals, specific criticisms/praise.
    """
    if not stories:
        return ""

    parts = [f"Source: Hacker News (Algolia API)"]
    parts.append(f"Total discussions found: {len(stories)}")

    # Separate recent vs old stories
    recent_stories = []
    older_stories = []
    for s in stories:
        created = s.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            days_old = (datetime.now(timezone.utc) - dt).days
            if days_old <= 365:
                recent_stories.append(s)
            else:
                older_stories.append(s)
        except Exception:
            older_stories.append(s)

    parts.append(f"Recent discussions (last 12 months): {len(recent_stories)}")
    parts.append(f"Older discussions: {len(older_stories)}")

    # High-engagement threads (signals community interest)
    high_engagement = [s for s in stories if s.get("points", 0) > 100]
    if high_engagement:
        parts.append(f"High-engagement threads (100+ points): {len(high_engagement)}")

    # List top stories with metadata
    parts.append("")
    parts.append("Top discussions:")
    for s in stories[:8]:
        title = s.get("title", "Untitled")
        points = s.get("points", 0)
        num_comments = s.get("num_comments", 0)
        created = s.get("created_at", "")
        age = _time_ago(created)
        parts.append(f"  - \"{title}\" ({points} points, {num_comments} comments, {age})")

    # Generate explicit contradiction evidence from titles
    # This section helps the LLM understand what HN reveals about the company
    parts.append("")
    parts.append("KEY FINDINGS FROM HN DISCUSSIONS (use for contradiction detection):")
    
    title_text = " ".join(s.get("title", "").lower() for s in stories)
    
    # Check for bankruptcy/financial issues
    financial_terms = ["bankrupt", "bankruptcy", "losses", "layoff", "fired", 
                       "shut down", "collapse", "fraud", "scam", "ponzi"]
    financial_found = [t for t in financial_terms if t in title_text]
    if financial_found:
        parts.append(f"  - FINANCIAL ISSUES in HN titles: {', '.join(financial_found)}")
        parts.append(f"    If the company's website claims growth or stability, these HN")
        parts.append(f"    discussions are direct contradiction evidence.")
    
    # Check for product/quality issues
    quality_terms = ["broken", "down", "outage", "problem", "terrible", "worst",
                     "slow", "unreliable", "security breach", "data loss"]
    quality_found = [t for t in quality_terms if t in title_text]
    if quality_found:
        parts.append(f"  - QUALITY/RELIABILITY ISSUES in HN titles: {', '.join(quality_found)}")

    # Check for leadership issues
    leadership_terms = ["steps down", "ceo", "fired", "resign", "leaves", "controversy"]
    leadership_found = [t for t in leadership_terms if t in title_text]
    if leadership_found:
        parts.append(f"  - LEADERSHIP CHANGES in HN titles: {', '.join(leadership_found)}")
    
    # Check for positive signals
    positive_terms = ["launch", "raises", "funding", "acquired", "ipo", "growth",
                      "open source", "release"]
    positive_found = [t for t in positive_terms if t in title_text]
    if positive_found:
        parts.append(f"  - POSITIVE SIGNALS in HN titles: {', '.join(positive_found)}")

    return "\n".join(parts)


async def fetch_community(domain: str) -> SourceResult:
    """
    Fetch community discussions about a company from Hacker News.

    Returns structured analysis of public discussions including
    sentiment signals, engagement levels, and specific criticisms/praise.
    This data enables contradiction detection against company claims.
    """
    search_terms = _extract_search_terms(domain)

    async with httpx.AsyncClient(
        headers={"User-Agent": "ClarityBot/1.0 (company-research)"},
    ) as client:
        # Search HN for stories with all search terms
        all_stories = []
        seen_ids = set()

        story_tasks = [_fetch_hn_stories(term, client) for term in search_terms]
        results = await asyncio.gather(*story_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            for story in result:
                sid = story.get("objectID", "")
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    all_stories.append(story)

        if not all_stories:
            logger.info(f"No HN discussions found for {domain}")
            return SourceResult(
                source_type=SourceType.COMMUNITY,
                url=f"https://news.ycombinator.com/",
                content=(
                    f"No Hacker News discussions found for {domain}.\n"
                    f"Search terms tried: {', '.join(search_terms)}\n"
                    f"This company may not have significant presence in technical communities."
                ),
                fetched=True,
            )

        # Sort by points (engagement) and take top stories
        all_stories.sort(key=lambda x: x.get("points", 0), reverse=True)
        top_stories = all_stories[:MAX_STORIES]

        logger.info(f"HN: Found {len(all_stories)} stories for {domain}, analyzing top {len(top_stories)}")

        # Analyze and format (titles-only, no comment fetching needed)
        company_name = domain.split(".")[0]
        analysis = _analyze_hn_data(top_stories, {}, company_name)

        return SourceResult(
            source_type=SourceType.COMMUNITY,
            url=f"https://news.ycombinator.com/",
            content=analysis,
            fetched=True,
        )
