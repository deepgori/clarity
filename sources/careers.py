"""
Clarity API - Structured Careers Page Analyzer

Takes raw careers page text (already fetched by website.py) and uses
gpt-4o-mini to extract structured hiring data: roles, departments,
locations, tech mentioned, and seniority patterns.

This structured data enables sharp contradiction detection by
cross-referencing hiring behavior against marketing claims.
"""

import os
import json
import logging
from openai import AsyncOpenAI
from costs import cost_tracker

logger = logging.getLogger(__name__)

CAREERS_EXTRACTION_PROMPT = """Extract structured hiring data from this careers page.

Analyze the page and extract:
- total_roles_found: How many open positions are listed
- roles: Array of individual roles with title, department, location, seniority
- departments_summary: A comma-separated breakdown like "Engineering: 5, Sales: 3, Marketing: 2"
- locations: Array of hiring locations mentioned
- tech_mentioned: Technologies, languages, frameworks mentioned in job descriptions
- notable_patterns: Non-obvious insights about hiring priorities, examples:
  "First CISO hire (security maturity signal)"
  "Heavy enterprise sales hiring (7 AE roles focused on Fortune 500)"
  "All engineering roles require Go, no Python/ML mentioned"

RULES:
- Only extract what's ACTUALLY on the page. Don't invent roles.
- "notable_patterns" should highlight things that reveal company priorities
  or contradict their public positioning. Be specific with numbers.
- If the page has very few roles or is mostly marketing copy, say so.
- Keep roles list to max 20 entries (summarize if more).
- For seniority, use: "junior", "mid", "senior", "lead", "director", "vp", "c-level"

CAREERS PAGE CONTENT:
{careers_text}"""

CAREERS_SCHEMA = {
    "type": "object",
    "properties": {
        "total_roles_found": {"type": "integer"},
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "department": {"type": "string"},
                    "location": {"type": "string"},
                    "seniority": {"type": "string"},
                },
                "required": ["title", "department", "location", "seniority"],
                "additionalProperties": False,
            },
        },
        "departments_summary": {
            "type": "string",
            "description": "Comma-separated summary like 'Engineering: 5, Sales: 3, Marketing: 2'",
        },
        "locations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "tech_mentioned": {
            "type": "array",
            "items": {"type": "string"},
        },
        "notable_patterns": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "total_roles_found", "roles", "departments_summary",
        "locations", "tech_mentioned", "notable_patterns",
    ],
    "additionalProperties": False,
}


async def extract_careers_data(careers_text: str) -> dict | None:
    """
    Extract structured hiring data from a careers page.
    Uses gpt-4o-mini for speed and cost (~$0.001 per call).

    Returns a dict with structured hiring data, or None if extraction fails.
    """
    if not careers_text or len(careers_text.strip()) < 100:
        logger.info("Careers text too short for extraction, skipping")
        return None

    # Truncate to avoid blowing up context
    if len(careers_text) > 6000:
        careers_text = careers_text[:6000] + "\n[Truncated]"

    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": CAREERS_EXTRACTION_PROMPT.format(
                    careers_text=careers_text,
                )},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "careers_data",
                    "strict": True,
                    "schema": CAREERS_SCHEMA,
                },
            },
            temperature=0.1,
            max_tokens=1500,
        )

        usage = response.usage
        if usage:
            cost_tracker.record(
                "gpt-4o-mini", usage.prompt_tokens, usage.completion_tokens,
                caller="careers_extraction",
            )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        logger.info(
            f"Careers extraction: {data.get('total_roles_found', 0)} roles, "
            f"{len(data.get('notable_patterns', []))} patterns found"
        )

        return data

    except Exception as e:
        logger.warning(f"Careers extraction failed: {e}")
        return None


def format_careers_for_synthesis(careers_data: dict) -> str:
    """Format structured careers data into text for the synthesis prompt."""
    if not careers_data:
        return "No structured careers data available."

    parts = []
    parts.append(f"Total open roles: {careers_data.get('total_roles_found', 'unknown')}")

    # Department breakdown
    dept_summary = careers_data.get("departments_summary", "")
    if dept_summary:
        parts.append(f"Department breakdown: {dept_summary}")

    # Locations
    locs = careers_data.get("locations", [])
    if locs:
        parts.append(f"Hiring locations: {', '.join(locs)}")

    # Tech mentioned in job descriptions
    tech = careers_data.get("tech_mentioned", [])
    if tech:
        parts.append(f"Tech in job descriptions: {', '.join(tech)}")

    # Notable patterns (most important for contradiction detection)
    patterns = careers_data.get("notable_patterns", [])
    if patterns:
        parts.append("Notable hiring patterns:")
        for p in patterns:
            parts.append(f"  - {p}")

    # Sample roles
    roles = careers_data.get("roles", [])
    if roles:
        parts.append(f"Sample roles ({min(len(roles), 10)} shown):")
        for r in roles[:10]:
            parts.append(f"  - {r['title']} [{r['department']}] ({r['location']}) - {r['seniority']}")

    return "\n".join(parts)
