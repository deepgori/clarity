"""
Clarity API - Candidate Profile Reader

Candidate mode can take a short bio and/or a GitHub profile URL. If a GitHub
URL is present, read the person's recently pushed public repos so the intro
note can point at real work instead of generic claims.

Only api.github.com is ever called, with a validated username, so the profile
text cannot be used to make the server fetch arbitrary URLs.
"""

import re
import logging
import httpx

from sources.github import GITHUB_API_BASE, _github_get

logger = logging.getLogger(__name__)

MAX_PROFILE_CHARS = 1500

# GitHub usernames: alphanumeric or single hyphens, max 39 chars, no leading hyphen
_GITHUB_URL = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(?![A-Za-z0-9-])",
    re.IGNORECASE,
)

# Paths on github.com that are not user profiles
_RESERVED = {
    "orgs", "settings", "features", "pricing", "about", "topics", "explore",
    "marketplace", "sponsors", "login", "join", "enterprise", "collections",
}


def extract_github_username(profile: str | None) -> str | None:
    """Return the first GitHub username linked in the profile text, if any."""
    if not profile:
        return None
    for match in _GITHUB_URL.finditer(profile):
        name = match.group(1)
        if name.lower() not in _RESERVED:
            return name
    return None


async def fetch_candidate_github(username: str) -> str | None:
    """Summarize a user's most recently pushed public repos (non-fork)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await _github_get(
                f"{GITHUB_API_BASE}/users/{username}/repos", client,
                params={"sort": "pushed", "direction": "desc", "per_page": 15, "type": "owner"},
            )
        if response.status_code != 200:
            logger.info(f"Candidate GitHub lookup returned {response.status_code} for {username}")
            return None

        repos = [r for r in response.json() if not r.get("fork")][:6]
        if not repos:
            return None

        lines = [f"GitHub: github.com/{username} (recently pushed public repos)"]
        for r in repos:
            desc = (r.get("description") or "").strip()
            lang = r.get("language") or "unknown language"
            pushed = (r.get("pushed_at") or "")[:10]
            line = f"- {r.get('name')} ({lang}, last push {pushed})"
            if desc:
                line += f": {desc[:160]}"
            lines.append(line)
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Candidate GitHub fetch failed for {username}: {e}")
        return None


async def build_candidate_context(profile: str | None) -> str | None:
    """Combine the candidate's own text with their GitHub repos, if linked."""
    if not profile or not profile.strip():
        return None
    text = profile.strip()[:MAX_PROFILE_CHARS]

    username = extract_github_username(text)
    if username:
        gh = await fetch_candidate_github(username)
        if gh:
            text = f"{text}\n\n{gh}"
    return text
