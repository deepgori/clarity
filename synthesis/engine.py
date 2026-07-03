"""
Clarity API - Synthesis Engine

Takes raw data from all sources and synthesizes structured intelligence
using OpenAI's structured output (JSON mode).

Key feature: contradiction detection across sources. This requires a single
reasoning pass across all data, which parallel-column enrichment architectures
can't do since they process each source independently.
"""

import os
import json
import logging
from openai import AsyncOpenAI
from models.schemas import SourceResult, CompanyIntelligence
from costs import cost_tracker

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are Clarity, an intelligence synthesis engine for AI sales agents.

You receive raw data from multiple sources about a company (website, news, GitHub, structured careers data).
Your job: produce intelligence that a human researcher would NOT find in 5 minutes of Googling.

REASONING PROCESS (follow this in order):

STEP 1 - EXTRACT CLAIMS FROM THE WEBSITE:
Read the website content and list the company's explicit positioning claims.
Look for identity statements they lead with. Examples of claims:
  - "We're developer-first" / "AI-powered" / "enterprise-ready" / "global platform"
  - "Open source" / "community-driven" / "fastest-growing"
  - Any specific metric they highlight (uptime, scale, coverage)

STEP 2 - CROSS-REFERENCE EACH CLAIM AGAINST EVIDENCE:
For EACH claim you extracted, check whether the other sources SUPPORT or UNDERMINE it:

  Website claim: "developer-first"
    CHECK careers: Are they hiring DevRel, DevEx, or developer-facing roles?
    CHECK GitHub: Are public repos active? Are docs/SDKs recently updated?
    CHECK news: Any developer community mentions or conference presence?

  Website claim: "AI-powered"
    CHECK careers: Do engineering roles mention ML, AI, or data science?
    CHECK GitHub: Are there ML frameworks, model repos, or AI tooling?
    CHECK careers tech: Is Python/PyTorch/TensorFlow mentioned, or just React/Java?

  Website claim: "global platform"
    CHECK careers: Are jobs posted in multiple countries, or just one city?
    CHECK website pricing: Multi-currency support? Localized pricing?

STEP 3 - REPORT ONLY GENUINE CONTRADICTIONS:
A contradiction exists ONLY when observable evidence ACTIVELY UNDERMINES a claim.
Not "limited evidence" but "evidence pointing the opposite direction."

  GENUINE: "Website says 'developer-first' but their careers page has zero DevRel
  or developer experience roles, their top 3 GitHub repos haven't been committed to
  in 8+ months, and all current engineering hires are backend infrastructure."
  WHY IT'S GENUINE: Multiple evidence sources all point away from the claim.

  NOT A CONTRADICTION: "Website says 'enterprise-ready' but GitHub activity is low."
  WHY: Most enterprise companies have proprietary codebases. Low GitHub activity
  is expected, not contradictory.

  NOT A CONTRADICTION: "Claims global but careers are in one country."
  WHY (sometimes): Many global SaaS companies centralize engineering. This is only
  a contradiction if the PRICING page also lacks multi-currency or the ABOUT page
  claims offices in multiple countries that don't match careers data.

  RULES:
  - If no genuine contradiction exists, return an EMPTY array. This is fine.
  - One sharp, well-evidenced contradiction is worth infinitely more than three weak ones.
  - The sales_implication must change HOW you'd pitch, not just be an observation.

SIGNALS (things that imply CURRENT MOTION, not static facts):
For each potential signal, ask: "When did this become true?"
  - If the answer is "always" or "years ago," it's a FACT, not a signal. Skip it.
  - If the answer is "recently" or "this is new," it's a SIGNAL. Report it.

  GOOD: "3 new repos created in the last 30 days focused on observability tooling"
    (implies a current engineering priority shift)
  GOOD: "Careers page shows 5 senior hires in compliance/security, none existed 90 days ago"
    (implies regulatory pressure or enterprise push)
  GOOD: "Engineering blog post from last month specifically mentioned latency issues"
    (implies an active pain point you can address)
  GOOD: "GitHub shows migration from JavaScript to TypeScript across 4 repos"
    (implies modernization effort with specific tooling implications)

  BAD: "$1.9T payment volume" (static fact, on their Wikipedia page)
  BAD: "Supports 135 currencies" (marketing copy from their homepage)
  BAD: "Active GitHub presence" (vague, not actionable)

RELEVANCE SCORING (show your math):
  Score 0-1 by mapping SPECIFIC seller capabilities to SPECIFIC target needs:
  - "Seller offers X, target needs Y because [evidence], so relevance = Z"
  - If weak fit, say WHY: "Seller does A but target already has B internally"

SALES STRATEGY:
  - recommended_angle: Reference a SPECIFIC finding from your cross-referencing
  - conversation_starter: Must reference something that would SURPRISE the prospect.
    Never open with their most famous metric or a compliment.
    BANNED PHRASES in conversation_starter (never use ANY form of these):
    "I noticed", "noticed", "I came across", "I hope this finds you well",
    "congratulations", "congrats", "impressive", "remarkable", "leverage", "synergies".
    Instead, start with a direct question or observation that implies insider knowledge.
  - avoid_topics: Explain WHY each topic would backfire
  - timing_assessment: Based on observable signals only, not speculation

WRITING STYLE (applies to ALL text fields in the output):
  - NEVER use "synergy", "synergies", "leverage", "impressive", "remarkable"
  - NEVER use em dashes
  - Sound like a sharp analyst, not a salesperson

SOURCE ATTRIBUTION (CRITICAL):
  Every specific claim in your output MUST be traceable to the provided source data.
  - If a signal mentions a product name, funding amount, partnership, or specific metric,
    it MUST appear verbatim in the WEBSITE CONTENT, NEWS, GITHUB, or CAREERS sections above.
  - NEVER invent, interpolate, or "fill in" specific details. If the source data says
    "new model announced" but doesn't name it, say "new model announced" not a made-up name.
  - If you are uncertain whether a specific claim is in the source data, DO NOT include it.
  - For each signal, mentally tag which source it came from (website, news, GitHub, careers).
    If you can't identify the source, drop the signal.

CONFIDENCE: Rate 0-1 based on source richness. Multiple sources with fresh data = high.
Thin data with gaps = low. Be honest.

Return a JSON object matching the CompanyIntelligence schema exactly."""

SYNTHESIS_USER_PROMPT = """Analyze this company and generate structured intelligence.

COMPANY DOMAIN: {domain}
{selling_context}

=== SOURCE DATA ===

--- WEBSITE CONTENT ---
{website_content}

--- RECENT NEWS ---
{news_content}

--- GITHUB PRESENCE ---
{github_content}

--- STRUCTURED CAREERS DATA ---
{careers_content}

=== END SOURCE DATA ===

Generate the CompanyIntelligence JSON object. Pay special attention to:
1. Cross-reference website CLAIMS against the structured careers data. Look for tensions
   between what the company says it does and what they're actually hiring for.
   Example: if the website says "AI-first" but no engineering roles mention ML/AI.
2. Cross-reference website claims against GitHub commit activity. Are repos marked
   as "STALE" or "ABANDONED" while the website claims active open-source?
3. Use hiring patterns (departments, locations, seniority) as signals of company priorities.
4. Specific, actionable sales strategy based on observable evidence.
5. Honest confidence scoring based on data quality."""


# JSON schema for structured output, matches CompanyIntelligence Pydantic model
INTELLIGENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "domain": {"type": "string"},
        "what_they_do": {"type": "string"},
        "industry": {"type": "string"},
        "stage": {"type": "string"},
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "signal": {"type": "string"},
                    "implication": {"type": "string"},
                    "source_url": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["signal", "implication", "source_url", "confidence"],
                "additionalProperties": False,
            },
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_a": {"type": "string"},
                    "source_a": {"type": "string"},
                    "claim_b": {"type": "string"},
                    "source_b": {"type": "string"},
                    "resolution": {"type": "string"},
                    "sales_implication": {"type": "string"},
                },
                "required": ["claim_a", "source_a", "claim_b", "source_b", "resolution", "sales_implication"],
                "additionalProperties": False,
            },
        },
        "sales_strategy": {
            "type": "object",
            "properties": {
                "relevance_score": {"type": "number"},
                "relevance_reasoning": {"type": "string"},
                "recommended_angle": {"type": "string"},
                "conversation_starter": {"type": "string"},
                "avoid_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "timing_assessment": {"type": "string"},
                "decision_maker_profile": {"type": "string"},
            },
            "required": ["relevance_score", "relevance_reasoning", "recommended_angle", "conversation_starter", "avoid_topics", "timing_assessment", "decision_maker_profile"],
            "additionalProperties": False,
        },
        "tech_stack": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hiring_signals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "sources_used": {
            "type": "array",
            "items": {"type": "string"},
        },
        "overall_confidence": {"type": "number"},
        "data_freshness": {"type": "string"},
    },
    "required": [
        "company_name", "domain", "what_they_do", "industry", "stage",
        "signals", "contradictions", "sales_strategy", "tech_stack",
        "hiring_signals", "sources_used", "overall_confidence", "data_freshness",
    ],
    "additionalProperties": False,
}


async def synthesize_intelligence(
    domain: str,
    website_result: SourceResult,
    news_result: SourceResult,
    github_result: SourceResult,
    seller_content: str | None = None,
    context: str | None = None,
    careers_data: str | None = None,
) -> CompanyIntelligence:
    """
    Synthesize raw source data into structured company intelligence.
    Uses OpenAI GPT-4o with structured output for reliable JSON.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build the user prompt with available data
    website_content = website_result.content if website_result.fetched else "No website data available."
    news_content = news_result.content if news_result.fetched else "No recent news found."
    github_content = github_result.content if github_result.fetched else "No GitHub presence found."
    careers_content = careers_data or "No structured careers data available."

    selling_context = ""
    if seller_content:
        # We have actual data about the seller's product from their website
        selling_context = (
            f"\n=== SELLER'S COMPANY (auto-extracted from their website) ===\n"
            f"{seller_content}\n"
            f"=== END SELLER DATA ===\n\n"
            f"INSTRUCTIONS FOR RELEVANCE AND STRATEGY:\n"
            f"1. First, understand what the seller actually does from their website.\n"
            f"2. Evaluate relevance in BOTH directions:\n"
            f"   - Is the seller's product useful to the target company?\n"
            f"   - Is the target company the right type of customer for the seller?\n"
            f"3. Match SPECIFIC seller features against SPECIFIC target needs.\n"
            f"4. If the seller's product has nothing to do with the target, score relevance low.\n"
        )
        if context:
            selling_context += f"\nAdditional context from the seller: {context}\n"
    elif context:
        selling_context = f"\nCONTEXT: {context}\n"

    user_prompt = SYNTHESIS_USER_PROMPT.format(
        domain=domain,
        selling_context=selling_context,
        website_content=website_content,
        news_content=news_content,
        github_content=github_content,
        careers_content=careers_content,
    )

    logger.info(f"Sending synthesis request to OpenAI ({len(user_prompt)} chars)...")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "company_intelligence",
                "strict": True,
                "schema": INTELLIGENCE_SCHEMA,
            },
        },
        temperature=0.3,  # Lower temp for more factual output
    )

    raw_json = response.choices[0].message.content

    # Track cost
    usage = response.usage
    if usage:
        cost_tracker.record("gpt-4o", usage.prompt_tokens, usage.completion_tokens, caller="synthesis")

    parsed = json.loads(raw_json)

    # Post-process: scrub banned phrases from text fields
    _scrub_synthesis_output(parsed)

    logger.info(
        f"Synthesis complete: {parsed.get('company_name', 'Unknown')} | "
        f"Confidence: {parsed.get('overall_confidence', 0)} | "
        f"Contradictions found: {len(parsed.get('contradictions', []))}"
    )

    return CompanyIntelligence(**parsed)


def _scrub_synthesis_output(parsed: dict) -> None:
    """Post-process synthesis output to remove banned phrases the LLM ignores."""
    import re

    banned_patterns = [
        (r'\bsynergies\b', 'alignment'),
        (r'\bsynergy\b', 'alignment'),
        (r'\bleverage\b', 'use'),
        (r'\bimpressive\b', 'notable'),
        (r'\bremarkable\b', 'notable'),
        (r'\bI noticed\b', 'Your'),
        (r'\bI came across\b', 'Your'),
        (r'\bCongratulations\b', 'Your recent'),
        (r'\bcongratulations\b', 'your recent'),
        (r'\bcongrats\b', 'your recent'),
    ]

    def scrub(text: str) -> str:
        for pattern, replacement in banned_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = text.replace('\u2014', ',').replace('\u2013', '-')
        return text

    # Scrub key text fields in sales_strategy
    strat = parsed.get("sales_strategy", {})
    for field in ["conversation_starter", "recommended_angle", "relevance_reasoning", "timing_assessment"]:
        if field in strat and isinstance(strat[field], str):
            strat[field] = scrub(strat[field])
