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

You receive raw data from multiple sources about a company (website content, news articles, GitHub repos).
Your job is to synthesize this into structured intelligence that a human researcher would NOT find
in 5 minutes of casual Googling. Surface non-obvious patterns and tensions.

CRITICAL INSTRUCTIONS:

1. CONTRADICTION DETECTION (This is your most important job):
   A contradiction is a NON-OBVIOUS tension between what a company CLAIMS and what their
   OBSERVABLE BEHAVIOR shows. It must be something a researcher would genuinely miss.

   GOOD contradictions (surface these):
   - "Website leads with 'developer-first' but their careers page has zero open developer
     relations or DevEx roles, and their docs haven't been updated in 6 months"
   - "Blog from March says they're doubling down on SMB, but every engineering hire in
     the last 60 days is for enterprise infrastructure (SSO, audit logs, compliance)"
   - "Homepage claims 'AI-powered' but GitHub repos show no ML frameworks, and their
     only AI-related hire was posted 2 weeks ago (suggesting they're just starting)"
   - "Website says 'global platform' but careers are only posted in one country,
     and pricing page has no multi-currency support"

   BAD contradictions (never report these):
   - "Company is enterprise-ready but has limited GitHub activity" (most enterprise
     companies are proprietary, this is not a contradiction)
   - "Claims rapid growth but stock price is flat" (irrelevant for private companies)
   - Any observation that is already common knowledge about the company

   RULES for contradictions:
   - Each contradiction must cite SPECIFIC evidence from the sources
   - If no genuine contradiction exists, return an empty array. Do NOT fabricate one.
   - Quality over quantity. One sharp contradiction beats three weak ones.
   - The sales_implication must explain HOW to use this in a conversation

2. SIGNALS (Actionable intelligence, NOT facts):
   A signal is timely, actionable, and non-obvious. It implies something about the
   company's current priorities, pain points, or buying intent.

   GOOD signals:
   - "Posted 3 infrastructure engineering roles in Brazil in the last 30 days" (expansion signal)
   - "Migrating primary language from Ruby to Go based on recent GitHub activity" (tech shift)
   - "CEO mentioned 'developer tooling gaps' at a conference last week" (buying intent)
   - "Removed 'free tier' from pricing page compared to archived version" (monetization shift)
   - "Hiring their first CISO" (security maturity signal)

   BAD signals (never report these):
   - "$1.9T payment volume" (this is a well-known fact, not a signal)
   - "99.999% uptime" (this is marketing copy, not intelligence)
   - Any fact that appears on the company's Wikipedia page or is widely reported

   Each signal must have a specific, actionable sales implication.

3. RELEVANCE SCORING (show your math):
   Score 0-1 and EXPLAIN the reasoning by mapping SPECIFIC seller capabilities to
   SPECIFIC target needs:
   - "Seller offers X, target needs Y because [evidence from sources], so relevance = Z"
   - If the fit is weak, say WHY. "Seller does A but target already has B internally"
   - Be blunt. A bad fit scored high is worse than no score at all.

4. SALES STRATEGY:
   - recommended_angle: Reference a SPECIFIC finding, not a generic approach
   - conversation_starter: Must reference something that would surprise the prospect,
     proving you did real research. Never open with their most famous metric.
   - avoid_topics: Explain WHY each topic should be avoided
   - timing_assessment: Based on observable signals (hiring, product changes), not speculation

5. CONFIDENCE SCORING:
   Rate 0-1 based on data quality:
   - 0.9+: Multiple rich sources, fresh data, clear patterns
   - 0.7-0.9: Good data but some gaps
   - 0.5-0.7: Limited sources, several assumptions needed
   - <0.5: Very thin data, low confidence

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

    logger.info(
        f"Synthesis complete: {parsed.get('company_name', 'Unknown')} | "
        f"Confidence: {parsed.get('overall_confidence', 0)} | "
        f"Contradictions found: {len(parsed.get('contradictions', []))}"
    )

    return CompanyIntelligence(**parsed)
