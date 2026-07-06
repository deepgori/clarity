"""
Clarity API - GitHub Source Fetcher

Fetches technical footprint from a company's GitHub org.
Uses the GitHub REST API (free: 60 req/hr unauthenticated, 5000 with token).
"""

import os
import re
import asyncio
import logging
from datetime import datetime, timezone
import httpx
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
REQUEST_TIMEOUT = 10.0


def _get_headers(use_auth: bool = True) -> dict:
    """Build GitHub API headers, with auth token if available."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ClarityBot/1.0",
    }
    if use_auth:
        # Use CLARITY_GITHUB_TOKEN to avoid picking up invalid system tokens
        token = os.getenv("CLARITY_GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


async def _github_get(url: str, client: httpx.AsyncClient, **kwargs) -> httpx.Response:
    """Make a GitHub API request. Retries without auth if we get a 401."""
    response = await client.get(url, headers=_get_headers(), timeout=REQUEST_TIMEOUT, **kwargs)
    if response.status_code == 401:
        # Token is invalid, retry without auth (60 req/hr unauthenticated)
        logger.info("GitHub token invalid, retrying without auth")
        response = await client.get(url, headers=_get_headers(use_auth=False), timeout=REQUEST_TIMEOUT, **kwargs)
    return response


async def _find_org(domain: str, client: httpx.AsyncClient) -> str | None:
    """Try to find the GitHub org name from a domain.

    Many companies use variations: scale.ai -> scaleapi, linear.app -> linearapp.
    We try multiple candidates and pick the one with the most public repos
    to avoid mapping to an unrelated org with the same short name.
    """
    base = domain.split(".")[0]
    tld = domain.split(".")[-1] if "." in domain else ""

    # Generate candidates - keep it small to avoid GitHub rate limits
    candidates = [base]
    # Most common patterns only: {name}api, {name}-ai, {name}hq
    for suffix in ["api", "-ai", "hq"]:
        candidates.append(base + suffix)
    # If TLD is unusual (not .com), try {base}{tld} too (linear.app -> linearapp)
    if tld and tld not in ("com", "org", "net", "io"):
        candidates.append(base + tld)

    # Check all candidates in parallel (with timeout to avoid rate limit stalls)
    async def _check_candidate(name):
        try:
            response = await _github_get(f"{GITHUB_API_BASE}/orgs/{name}", client)
            if response.status_code == 200:
                data = response.json()
                return name, data.get("public_repos", 0)
        except Exception:
            pass
        return name, -1

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_check_candidate(c) for c in candidates]),
            timeout=8.0,
        )
    except asyncio.TimeoutError:
        logger.warning(f"GitHub org lookup timed out for {domain}, falling back to '{base}'")
        return base

    best_org = None
    best_repos = -1
    for name, repos in results:
        if repos > best_repos:
            best_repos = repos
            best_org = name

    if best_org:
        logger.info(f"GitHub org resolved: {domain} -> {best_org} ({best_repos} public repos)")
        return best_org

    # Fallback: try as a user account
    response = await _github_get(f"{GITHUB_API_BASE}/users/{base}", client)
    if response.status_code == 200:
        return base

    return None


def _extract_github_org_from_content(website_content: str) -> str | None:
    """Extract GitHub org from website content (footer links, about page, etc.).

    Most companies link to their GitHub from their homepage or footer.
    This is the most reliable signal for org name, especially for companies
    where the org name doesn't match the domain (e.g. notion.so -> makenotion).
    """
    if not website_content:
        return None

    # Match github.com/{org} patterns
    matches = re.findall(r'github\.com/(?:orgs/)?([a-zA-Z0-9][a-zA-Z0-9_-]*)', website_content)

    # Filter out common non-org paths and repo-specific paths
    IGNORE = {
        'about', 'pricing', 'features', 'login', 'signup', 'settings',
        'explore', 'topics', 'trending', 'collections', 'events', 'sponsors',
        'readme', 'issues', 'pulls', 'actions', 'projects', 'wiki',
        'security', 'pulse', 'community', 'marketplace', 'apps',
    }
    candidates = [m.lower() for m in matches if m.lower() not in IGNORE and len(m) > 1]

    if not candidates:
        return None

    # Count occurrences - the actual org name typically appears most often
    from collections import Counter
    counts = Counter(candidates)
    best = counts.most_common(1)[0]
    logger.info(f"Extracted GitHub org from website content: '{best[0]}' (appeared {best[1]} times)")
    return best[0]


async def _get_top_repos(org: str, client: httpx.AsyncClient) -> list[dict]:
    """Get top repos by stars AND recently pushed repos for a balanced view.

    Fetching only by stars causes the model to see famous-but-abandoned repos
    and miss active development. We fetch both and deduplicate.
    """
    starred = []
    recent = []

    # Top by stars (the famous repos)
    response = await _github_get(
        f"{GITHUB_API_BASE}/orgs/{org}/repos", client,
        params={"sort": "stars", "direction": "desc", "per_page": 5, "type": "public"},
    )
    if response.status_code == 200:
        starred = response.json()
    else:
        # Try as user repos if org endpoint fails
        response = await _github_get(
            f"{GITHUB_API_BASE}/users/{org}/repos", client,
            params={"sort": "stars", "direction": "desc", "per_page": 5, "type": "public"},
        )
        if response.status_code == 200:
            starred = response.json()

    # Recently pushed (the active repos)
    response = await _github_get(
        f"{GITHUB_API_BASE}/orgs/{org}/repos", client,
        params={"sort": "pushed", "direction": "desc", "per_page": 5, "type": "public"},
    )
    if response.status_code == 200:
        recent = response.json()
    else:
        response = await _github_get(
            f"{GITHUB_API_BASE}/users/{org}/repos", client,
            params={"sort": "pushed", "direction": "desc", "per_page": 5, "type": "public"},
        )
        if response.status_code == 200:
            recent = response.json()

    # Deduplicate: starred first, then any recent repos not already included
    seen_ids = {r.get("id") for r in starred}
    combined = list(starred)
    for r in recent:
        if r.get("id") not in seen_ids:
            combined.append(r)
            seen_ids.add(r.get("id"))

    return combined[:8]  # Cap at 8 repos total


async def _get_languages(repo_full_name: str, client: httpx.AsyncClient) -> dict:
    """Get language breakdown for a repo."""
    response = await _github_get(
        f"{GITHUB_API_BASE}/repos/{repo_full_name}/languages", client,
    )
    if response.status_code == 200:
        return response.json()
    return {}


async def _get_last_commit_date(repo_full_name: str, client: httpx.AsyncClient) -> tuple[str, int]:
    """Get the date of the most recent commit and days since then."""
    try:
        response = await _github_get(
            f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits",
            client,
            params={"per_page": 1},
        )
        if response.status_code == 200:
            commits = response.json()
            if commits:
                date_str = commits[0].get("commit", {}).get("committer", {}).get("date", "")
                if date_str:
                    commit_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    days_ago = (datetime.now(timezone.utc) - commit_date).days
                    return date_str[:10], days_ago
    except Exception as e:
        logger.warning(f"Failed to get commit date for {repo_full_name}: {e}")
    return "unknown", -1


def _activity_label(days_ago: int) -> str:
    """Label a repo's activity level based on days since last commit."""
    if days_ago < 0:
        return "unknown"
    if days_ago <= 30:
        return "ACTIVE (committed within 30 days)"
    if days_ago <= 180:
        return f"STALE (no commits in {days_ago} days)"
    return f"ABANDONED (no commits in {days_ago} days)"


async def _build_github_content(org: str, client: httpx.AsyncClient) -> SourceResult:
    """Build GitHub content for a resolved org. Shared by fetch_github and refetch."""
    repos = await _get_top_repos(org, client)

    if not repos:
        return SourceResult(
            source_type=SourceType.GITHUB,
            url=f"https://github.com/{org}",
            content="No public repositories found.",
            fetched=True,
        )

    content_parts = [f"GitHub Organization: {org}\nPublic Repositories:\n"]
    all_languages = {}
    activity_counts = {"active": 0, "stale": 0, "abandoned": 0, "unknown": 0}

    for idx, repo in enumerate(repos[:8]):
        name = repo.get("name", "")
        description = repo.get("description", "No description")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        language = repo.get("language", "Unknown")
        updated = repo.get("updated_at", "")
        archived = repo.get("archived", False)

        last_commit_date, days_ago = await _get_last_commit_date(f"{org}/{name}", client)
        activity_status = _activity_label(days_ago)

        if days_ago < 0:
            activity_counts["unknown"] += 1
        elif days_ago <= 30:
            activity_counts["active"] += 1
        elif days_ago <= 180:
            activity_counts["stale"] += 1
        else:
            activity_counts["abandoned"] += 1

        content_parts.append(
            f"\n--- REPO: {name} ---\n"
            f"Description: {description}\n"
            f"Stars: {stars} | Forks: {forks}\n"
            f"Primary Language: {language}\n"
            f"Last Updated: {updated}\n"
            f"Last Commit: {last_commit_date} ({activity_status})\n"
            f"Archived: {archived}\n"
        )

        if stars > 0 or idx < 3:
            languages = await _get_languages(f"{org}/{name}", client)
            for lang, bytes_count in languages.items():
                all_languages[lang] = all_languages.get(lang, 0) + bytes_count

    total_checked = sum(activity_counts.values())
    summary = (
        f"\n=== ORG ACTIVITY SUMMARY ===\n"
        f"Repos checked: {total_checked}\n"
        f"Active (committed within 30 days): {activity_counts['active']}\n"
        f"Stale (30-180 days): {activity_counts['stale']}\n"
        f"Abandoned (180+ days): {activity_counts['abandoned']}\n"
    )
    content_parts.insert(1, summary)

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


async def fetch_github(domain: str) -> SourceResult:
    """
    Fetch technical footprint from a company's GitHub presence.
    Returns structured content about repos, languages, and activity.

    Gracefully handles companies with no GitHub presence.
    """
    async with httpx.AsyncClient() as client:
        org = await _find_org(domain, client)

        if not org:
            return SourceResult(
                source_type=SourceType.GITHUB,
                url=f"https://github.com/{domain.split('.')[0]}",
                content="",
                fetched=False,
                error=f"No GitHub organization found for {domain}",
            )

        return await _build_github_content(org, client)


async def refetch_github_with_website_hint(domain: str, website_content: str) -> SourceResult | None:
    """Re-fetch GitHub data using a better org name.

    Tries two strategies:
    1. Extract GitHub link from the website content
    2. Search GitHub API for orgs whose blog URL matches the domain

    Called by main.py when the initial GitHub fetch looks suspicious (all repos
    abandoned, very few repos, etc.). Returns None if no better org is found.
    """
    async with httpx.AsyncClient() as client:
        # Strategy 1: Extract from website content
        extracted_org = _extract_github_org_from_content(website_content)
        if extracted_org:
            result = await _verify_and_build(extracted_org, domain, "website link", client)
            if result:
                return result

        # Strategy 2: Search GitHub for orgs matching the company name
        # whose blog URL matches the domain
        base = domain.split(".")[0]
        try:
            response = await _github_get(
                f"{GITHUB_API_BASE}/search/users", client,
                params={"q": f"{base} type:org", "per_page": 5},
            )
            if response.status_code == 200:
                items = response.json().get("items", [])
                for item in items:
                    login = item.get("login", "")
                    # Get full org details (has blog URL)
                    org_resp = await _github_get(
                        f"{GITHUB_API_BASE}/orgs/{login}", client,
                    )
                    if org_resp.status_code == 200:
                        org_data = org_resp.json()
                        blog = (org_data.get("blog") or "").lower().strip("/")
                        description = (org_data.get("description") or "").lower()
                        pub_repos = org_data.get("public_repos", 0)

                        # Match if blog URL contains the domain
                        domain_base = domain.lower().replace("www.", "")
                        if pub_repos > 0 and (domain_base in blog or domain_base in description):
                            logger.info(
                                f"GitHub search found org '{login}' with blog '{blog}' "
                                f"matching domain '{domain}' ({pub_repos} repos)"
                            )
                            return await _build_github_content(login, client)
        except Exception as e:
            logger.warning(f"GitHub search fallback failed for {domain}: {e}")

        return None


async def _verify_and_build(org: str, domain: str, source: str, client: httpx.AsyncClient) -> SourceResult | None:
    """Verify an org exists and has repos, then build content."""
    try:
        response = await _github_get(f"{GITHUB_API_BASE}/orgs/{org}", client)
        if response.status_code != 200:
            logger.info(f"Org '{org}' from {source} not found on GitHub")
            return None

        pub_repos = response.json().get("public_repos", 0)
        if pub_repos == 0:
            logger.info(f"Org '{org}' from {source} has 0 public repos")
            return None

        logger.info(f"Re-fetching GitHub for {domain} using {source} org: {org} ({pub_repos} repos)")
        return await _build_github_content(org, client)

    except Exception as e:
        logger.warning(f"Failed to re-fetch GitHub with {source} org '{org}': {e}")
        return None
