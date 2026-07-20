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

    # Extract and summarize top comments (actual sentiment)
    all_comments = []
    for story_id, comments in comments_by_story.items():
        if comments:
            for c in comments:
                text = _clean_comment_text(c.get("comment_text", ""))
                if text and len(text) > 50:
                    all_comments.append({
                        "text": text,
                        "story_id": story_id,
                        "points": c.get("points", 0),
                    })

    if all_comments:
        # Sort by engagement
        all_comments.sort(key=lambda x: x.get("points", 0), reverse=True)
        parts.append("")
        parts.append(f"Sample community comments ({min(len(all_comments), 8)} shown):")
        for c in all_comments[:8]:
            parts.append(f"  - \"{c['text']}\"")

    # Detect sentiment patterns
    patterns = []
    titles_lower = " ".join(s.get("title", "").lower() for s in stories)
    comments_lower = " ".join(c.get("text", "").lower() for c in all_comments)
    combined = titles_lower + " " + comments_lower

    # Negative sentiment keywords
    negative_terms = ["bankrupt", "layoff", "lawsuit", "scam", "fraud", "crash",
                      "broken", "terrible", "worst", "problem", "issue", "bug",
                      "slow", "expensive", "overpriced", "downtime", "outage",
                      "security breach", "data loss", "unreliable"]
    negative_found = [t for t in negative_terms if t in combined]
    if negative_found:
        patterns.append(f"Negative sentiment detected in discussions: {', '.join(negative_found)}")

    # Positive sentiment keywords
    positive_terms = ["love", "great", "excellent", "best", "amazing", "recommend",
                      "fast", "reliable", "solid", "impressive"]
    positive_found = [t for t in positive_terms if t in combined]
    if positive_found:
        patterns.append(f"Positive sentiment detected: {', '.join(positive_found)}")

    # Competition mentions
    competitor_terms = ["alternative", "switch", "migrate", "moved from", "replaced",
                       "competitor", "vs", "versus", "compared to"]
    competitor_found = [t for t in competitor_terms if t in combined]
    if competitor_found:
        patterns.append(f"Competitive discussion themes: {', '.join(competitor_found)}")

    if patterns:
        parts.append("")
        parts.append("Sentiment patterns:")
        for p in patterns:
            parts.append(f"  - {p}")

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

        # Fetch comments for top 3 highest-engagement stories
        comments_by_story = {}
        comment_stories = top_stories[:3]
        comment_tasks = [
            _fetch_hn_comments(s.get("objectID", ""), client)
            for s in comment_stories
            if s.get("objectID")
        ]
        comment_results = await asyncio.gather(*comment_tasks, return_exceptions=True)

        for i, result in enumerate(comment_results):
            if isinstance(result, Exception) or result is None:
                continue
            story_id = comment_stories[i].get("objectID", "")
            comments_by_story[story_id] = result

        # Analyze and format
        company_name = domain.split(".")[0]
        analysis = _analyze_hn_data(top_stories, comments_by_story, company_name)

        return SourceResult(
            source_type=SourceType.COMMUNITY,
            url=f"https://news.ycombinator.com/",
            content=analysis,
            fetched=True,
        )
