"""
Clarity API - External Job Board Fetcher

Fetches structured job posting data from external ATS platforms
(Greenhouse, Lever, Ashby) that companies use to post their open roles.

This source fills the critical gap when GitHub repos are private:
job descriptions reveal what a company is ACTUALLY building right now,
what tech stack they use, and what their priorities are.
"""

import asyncio
import httpx
import json
import logging
import re
from urllib.parse import urlparse
from models.schemas import SourceResult, SourceType

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10.0
MAX_JOBS_TO_ANALYZE = 25  # Cap to avoid massive payloads


def _extract_company_slug(domain: str) -> str:
    """
    Extract a likely company slug from a domain.
    stripe.com -> stripe, databricks.com -> databricks
    """
    parts = domain.replace("www.", "").split(".")
    slug = parts[0]
    # Handle common patterns
    slug = slug.replace("-", "").replace("_", "")
    return slug.lower()


def _extract_company_slug_variants(domain: str) -> list[str]:
    """
    Generate multiple slug variants to try against ATS APIs.
    notion.so -> [notion, notionhq, notion-hq]
    scale.ai -> [scale, scaleai, scale-ai]
    """
    base = domain.replace("www.", "").split(".")[0].lower()
    tld = domain.split(".")[-1] if "." in domain else ""

    variants = [base]

    # Add with-TLD variant for unusual TLDs
    if tld not in ("com", "org", "net", "io", "co"):
        variants.append(f"{base}{tld}")
        variants.append(f"{base}-{tld}")

    # Add common suffixes
    variants.append(f"{base}hq")
    variants.append(f"{base}-hq")
    variants.append(f"{base}inc")
    variants.append(f"{base}io")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


async def _fetch_greenhouse_jobs(
    slug: str, client: httpx.AsyncClient
) -> list[dict] | None:
    """
    Fetch jobs from Greenhouse's public board API.
    Greenhouse is used by: Stripe, Notion, Figma, Ramp, etc.
    API: https://boards-api.greenhouse.io/v1/boards/{company}/jobs
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    try:
        response = await client.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("jobs", [])
            if jobs:
                logger.info(f"Greenhouse: Found {len(jobs)} jobs for '{slug}'")
                return [
                    {
                        "title": j.get("title", ""),
                        "location": j.get("location", {}).get("name", "Unknown"),
                        "department": (
                            j.get("departments", [{}])[0].get("name", "General")
                            if j.get("departments") else "General"
                        ),
                        "content": _clean_html(j.get("content", "")),
                        "updated_at": j.get("updated_at", ""),
                    }
                    for j in jobs[:MAX_JOBS_TO_ANALYZE]
                ]
        return None
    except Exception as e:
        logger.debug(f"Greenhouse fetch failed for '{slug}': {e}")
        return None


async def _fetch_lever_jobs(
    slug: str, client: httpx.AsyncClient
) -> list[dict] | None:
    """
    Fetch jobs from Lever's public posting API.
    Lever is used by: Netflix, Cloudflare, Reddit, etc.
    API: https://api.lever.co/v0/postings/{company}
    """
    url = f"https://api.lever.co/v0/postings/{slug}"
    try:
        response = await client.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            jobs = response.json()
            if isinstance(jobs, list) and len(jobs) > 0:
                logger.info(f"Lever: Found {len(jobs)} jobs for '{slug}'")
                return [
                    {
                        "title": j.get("text", ""),
                        "location": j.get("categories", {}).get("location", "Unknown"),
                        "department": j.get("categories", {}).get("team", "General"),
                        "content": _clean_html(j.get("descriptionPlain", j.get("description", ""))),
                        "updated_at": "",
                    }
                    for j in jobs[:MAX_JOBS_TO_ANALYZE]
                ]
        return None
    except Exception as e:
        logger.debug(f"Lever fetch failed for '{slug}': {e}")
        return None


async def _fetch_ashby_jobs(
    slug: str, client: httpx.AsyncClient
) -> list[dict] | None:
    """
    Fetch jobs from Ashby's public job board API.
    Ashby is used by: Ramp, Linear, Vercel, etc.
    API: https://api.ashbyhq.com/posting-api/job-board/{company}
    """
    url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"
    try:
        response = await client.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("jobs", [])
            if jobs:
                logger.info(f"Ashby: Found {len(jobs)} jobs for '{slug}'")
                return [
                    {
                        "title": j.get("title", ""),
                        "location": j.get("location", "Unknown"),
                        "department": j.get("departmentName", j.get("team", "General")),
                        "content": j.get("descriptionPlain", "")[:500] if j.get("descriptionPlain") else "",
                        "updated_at": j.get("publishedAt", ""),
                    }
                    for j in jobs[:MAX_JOBS_TO_ANALYZE]
                ]
        return None
    except Exception as e:
        logger.debug(f"Ashby fetch failed for '{slug}': {e}")
        return None


def _clean_html(text: str) -> str:
    """Remove HTML tags from job description content."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    # Truncate long descriptions
    if len(clean) > 500:
        clean = clean[:500] + "..."
    return clean


def _analyze_jobs(jobs: list[dict], source_platform: str) -> str:
    """
    Analyze job listings and produce structured text for synthesis.
    Extracts: tech stack, department breakdown, seniority, locations, patterns.
    """
    if not jobs:
        return ""

    total = len(jobs)

    # Department breakdown
    departments: dict[str, int] = {}
    locations: set[str] = set()
    tech_mentioned: set[str] = set()
    seniority_counts = {"junior": 0, "mid": 0, "senior": 0, "lead": 0, "director+": 0}

    # Common tech keywords to detect in job descriptions
    tech_keywords = {
        "python", "javascript", "typescript", "java", "go", "golang", "rust",
        "react", "vue", "angular", "next.js", "node.js", "django", "flask",
        "fastapi", "kubernetes", "docker", "aws", "gcp", "azure",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "rabbitmq", "terraform", "pytorch", "tensorflow",
        "machine learning", "deep learning", "llm", "nlp", "computer vision",
        "spark", "airflow", "dbt", "snowflake", "bigquery", "databricks",
        "graphql", "grpc", "microservices", "ci/cd", "github actions",
        "datadog", "sentry", "grafana", "prometheus",
    }

    for job in jobs:
        # Count departments
        dept = job.get("department", "General")
        departments[dept] = departments.get(dept, 0) + 1

        # Collect locations
        loc = job.get("location", "")
        if loc and loc != "Unknown":
            locations.add(loc)

        # Detect seniority from title
        title_lower = job.get("title", "").lower()
        if any(w in title_lower for w in ["intern", "junior", "entry", "associate", "new grad"]):
            seniority_counts["junior"] += 1
        elif any(w in title_lower for w in ["senior", "sr.", "sr ", "staff", "principal"]):
            seniority_counts["senior"] += 1
        elif any(w in title_lower for w in ["lead", "team lead", "tech lead"]):
            seniority_counts["lead"] += 1
        elif any(w in title_lower for w in ["director", "vp", "head of", "chief", "cto", "ceo"]):
            seniority_counts["director+"] += 1
        else:
            seniority_counts["mid"] += 1

        # Extract tech from description content
        content_lower = (job.get("content", "") + " " + job.get("title", "")).lower()
        for tech in tech_keywords:
            if tech in content_lower:
                tech_mentioned.add(tech)

    # Build output text
    parts = [f"Source: {source_platform} job board"]
    parts.append(f"Total open positions: {total}")

    # Department breakdown
    dept_sorted = sorted(departments.items(), key=lambda x: x[1], reverse=True)
    dept_str = ", ".join(f"{k}: {v}" for k, v in dept_sorted)
    parts.append(f"Department breakdown: {dept_str}")

    # Seniority distribution
    seniority_str = ", ".join(
        f"{k}: {v}" for k, v in seniority_counts.items() if v > 0
    )
    parts.append(f"Seniority distribution: {seniority_str}")

    # Locations
    if locations:
        parts.append(f"Hiring locations: {', '.join(sorted(locations))}")

    # Tech mentioned in job descriptions
    if tech_mentioned:
        parts.append(f"Technologies in job descriptions: {', '.join(sorted(tech_mentioned))}")

    # Notable patterns (high-signal observations)
    patterns = []

    # Engineering vs non-engineering ratio
    eng_depts = {"Engineering", "Product", "Design", "R&D", "Technology", "Platform"}
    sales_depts = {"Sales", "Business Development", "Revenue", "Growth", "GTM", "Marketing"}
    eng_count = sum(v for k, v in departments.items() if any(e.lower() in k.lower() for e in eng_depts))
    sales_count = sum(v for k, v in departments.items() if any(s.lower() in k.lower() for s in sales_depts))

    if eng_count > 0 and sales_count > 0:
        if sales_count > eng_count * 1.5:
            patterns.append(f"Sales-heavy hiring ({sales_count} sales vs {eng_count} engineering) suggests GTM expansion phase")
        elif eng_count > sales_count * 2:
            patterns.append(f"Engineering-heavy hiring ({eng_count} eng vs {sales_count} sales) suggests product-building phase")

    # AI/ML signal
    ai_tech = tech_mentioned & {"pytorch", "tensorflow", "machine learning", "deep learning", "llm", "nlp", "computer vision"}
    if ai_tech:
        patterns.append(f"Active AI/ML hiring: job descriptions mention {', '.join(sorted(ai_tech))}")
    elif total > 5:
        patterns.append("No AI/ML technologies mentioned across any job descriptions")

    # Remote vs on-site
    remote_count = sum(1 for j in jobs if "remote" in j.get("location", "").lower())
    if remote_count > total * 0.5:
        patterns.append(f"{remote_count}/{total} roles are remote, indicating distributed team")
    elif remote_count == 0 and total > 3:
        patterns.append("All roles appear to be on-site/in-office")

    if patterns:
        parts.append("Notable patterns:")
        for p in patterns:
            parts.append(f"  - {p}")

    # Sample roles (top 10)
    parts.append(f"Sample roles ({min(total, 10)} shown):")
    for j in jobs[:10]:
        parts.append(f"  - {j['title']} [{j.get('department', 'General')}] ({j.get('location', 'Unknown')})")

    return "\n".join(parts)


async def fetch_jobs(domain: str) -> SourceResult:
    """
    Fetch job postings from external ATS platforms.

    Tries Greenhouse, Lever, and Ashby in parallel with multiple
    slug variants. Returns structured analysis of job postings.

    This source fills the critical gap when GitHub repos are private:
    job descriptions reveal actual tech stack, priorities, and team structure.
    """
    slugs = _extract_company_slug_variants(domain)

    async with httpx.AsyncClient(
        headers={"User-Agent": "ClarityBot/1.0 (company-research)"},
        follow_redirects=True,
    ) as client:
        # Try all ATS platforms with all slug variants in parallel
        tasks = []
        task_labels = []

        for slug in slugs[:4]:  # Limit variants to avoid too many requests
            tasks.append(_fetch_greenhouse_jobs(slug, client))
            task_labels.append(("Greenhouse", slug))
            tasks.append(_fetch_lever_jobs(slug, client))
            task_labels.append(("Lever", slug))
            tasks.append(_fetch_ashby_jobs(slug, client))
            task_labels.append(("Ashby", slug))

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Job board fetch timed out for {domain}")
            return SourceResult(
                source_type=SourceType.JOBS,
                url=f"https://{domain}/careers",
                content="",
                fetched=False,
                error="Job board fetch timed out",
            )

    # Find the first successful result
    for i, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            continue
        platform, slug = task_labels[i]
        logger.info(f"Job postings found via {platform} (slug: {slug}) for {domain}")
        analysis = _analyze_jobs(result, platform)
        if analysis:
            return SourceResult(
                source_type=SourceType.JOBS,
                url=f"https://{domain}/careers",
                content=analysis,
                fetched=True,
            )

    logger.info(f"No external job board found for {domain}")
    return SourceResult(
        source_type=SourceType.JOBS,
        url=f"https://{domain}/careers",
        content="",
        fetched=False,
        error="No external job board found (checked Greenhouse, Lever, Ashby)",
    )
