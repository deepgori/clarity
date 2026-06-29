"""
Clarity API - GitHub Source Fetcher

Fetches technical footprint from a company's GitHub org.
Uses the GitHub REST API (free: 60 req/hr unauthenticated, 5000 with token).
"""

import os
import logging
import httpx
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
REQUEST_TIMEOUT = 10.0


def _get_headers() -> dict:
    """Build GitHub API headers, with auth token if available."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ClarityBot/1.0",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _find_org(domain: str, client: httpx.AsyncClient) -> str | None:
    """Try to find the GitHub org name from a domain."""
    # Common patterns: stripe.com -> stripe, vercel.com -> vercel
    org_guess = domain.split(".")[0]

    # Check if the org exists
    response = await client.get(
        f"{GITHUB_API_BASE}/orgs/{org_guess}",
        headers=_get_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 200:
        return org_guess

    # Also try as a user (some companies use user accounts)
    response = await client.get(
        f"{GITHUB_API_BASE}/users/{org_guess}",
        headers=_get_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 200:
        return org_guess

    return None


async def _get_top_repos(org: str, client: httpx.AsyncClient) -> list[dict]:
    """Get top repos by stars for an org."""
    response = await client.get(
        f"{GITHUB_API_BASE}/orgs/{org}/repos",
        headers=_get_headers(),
        params={"sort": "stars", "direction": "desc", "per_page": 5, "type": "public"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 200:
        return response.json()

    # Try as user repos if org endpoint fails
    response = await client.get(
        f"{GITHUB_API_BASE}/users/{org}/repos",
        headers=_get_headers(),
        params={"sort": "stars", "direction": "desc", "per_page": 5, "type": "public"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 200:
        return response.json()

    return []


async def _get_languages(repo_full_name: str, client: httpx.AsyncClient) -> dict:
    """Get language breakdown for a repo."""
    response = await client.get(
        f"{GITHUB_API_BASE}/repos/{repo_full_name}/languages",
        headers=_get_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 200:
        return response.json()
    return {}


async def fetch_github(domain: str) -> SourceResult:
    """
    Fetch technical footprint from a company's GitHub presence.
    Returns structured content about repos, languages, and activity.

    Gracefully handles companies with no GitHub presence.
    """
    async with httpx.AsyncClient() as client:
        # Find the org
        org = await _find_org(domain, client)

        if not org:
            return SourceResult(
                source_type=SourceType.GITHUB,
                url=f"https://github.com/{domain.split('.')[0]}",
                content="",
                fetched=False,
                error=f"No GitHub organization found for {domain}",
            )

        # Get top repos
        repos = await _get_top_repos(org, client)

        if not repos:
            return SourceResult(
                source_type=SourceType.GITHUB,
                url=f"https://github.com/{org}",
                content="No public repositories found.",
                fetched=True,
            )

        # Build detailed content
        content_parts = [f"GitHub Organization: {org}\nPublic Repositories:\n"]

        all_languages = {}

        for repo in repos[:5]:
            name = repo.get("name", "")
            description = repo.get("description", "No description")
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            language = repo.get("language", "Unknown")
            updated = repo.get("updated_at", "")
            archived = repo.get("archived", False)

            content_parts.append(
                f"\n--- REPO: {name} ---\n"
                f"Description: {description}\n"
                f"Stars: {stars} | Forks: {forks}\n"
                f"Primary Language: {language}\n"
                f"Last Updated: {updated}\n"
                f"Archived: {archived}\n"
            )

            # Get language breakdown for top repos
            if stars > 0 or repos.index(repo) < 3:
                languages = await _get_languages(f"{org}/{name}", client)
                for lang, bytes_count in languages.items():
                    all_languages[lang] = all_languages.get(lang, 0) + bytes_count

        # Summarize overall tech stack
        if all_languages:
            sorted_langs = sorted(all_languages.items(), key=lambda x: x[1], reverse=True)
            lang_summary = ", ".join(
                f"{lang} ({bytes // 1024}KB)" for lang, bytes in sorted_langs[:10]
            )
            content_parts.append(f"\n=== OVERALL TECH STACK ===\n{lang_summary}")

        combined = "\n".join(content_parts)

        return SourceResult(
            source_type=SourceType.GITHUB,
            url=f"https://github.com/{org}",
            content=combined,
            fetched=True,
        )
