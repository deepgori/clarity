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

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are Clarity, an intelligence synthesis engine for AI sales agents.

You receive raw data from multiple sources about a company (website content, news articles, GitHub repos).
Your job is to synthesize this into structured intelligence that an autonomous AI sales agent can 
consume and act on programmatically.

CRITICAL INSTRUCTIONS:

1. CONTRADICTION DETECTION:
   Actively look for contradictions between sources. Examples:
   - Website says "API-first" but GitHub repos are all gRPC-based
   - Website says "enterprise-ready" but no SOC2/security page exists
   - News says "rapid growth" but GitHub shows decreasing commit frequency
   - Website says "AI-powered" but tech stack shows no ML/AI frameworks
   
   When you find contradictions, explain what they likely mean and how they 
   affect the sales approach.

2. EVIDENCE-BACKED CLAIMS ONLY:
   Every signal must cite a specific source. Never fabricate information.
   If data is insufficient, say so. Don't guess.

3. RELEVANCE SCORING (Critical):
   If the user specifies what they're selling, honestly evaluate whether it's a
   good fit for this company. Score 0-1:
   - 0.8+: Strong natural fit, clear use case
   - 0.5-0.8: Some relevance, would require creative positioning
   - 0.2-0.5: Weak fit, unlikely to close
   - <0.2: Completely irrelevant, don't waste their time
   Be blunt. If someone is trying to sell a shopping website to a cloud monitoring
   company, say so. Don't invent fake reasons for relevance.

4. SALES STRATEGY MUST BE ACTIONABLE:
   Don't say generic things like "build a relationship." Say specific things like 
   "Their CTO blogged about microservices migration, reference this in your opening."

5. CONFIDENCE SCORING:
   Rate overall confidence 0-1 based on data quality:
   - 0.9+: Multiple sources, consistent data, rich content
   - 0.7-0.9: Good data but some gaps
   - 0.5-0.7: Limited data, several assumptions
   - <0.5: Very thin data, low confidence in conclusions

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

=== END SOURCE DATA ===

Generate the CompanyIntelligence JSON object. Pay special attention to:
1. Any contradictions between what the website claims and what GitHub/News shows
2. Specific, actionable sales strategy (not generic advice)
3. Honest confidence scoring based on data quality
4. Hiring signals and what they imply about the company's priorities"""


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
    parsed = json.loads(raw_json)

    logger.info(
        f"Synthesis complete: {parsed.get('company_name', 'Unknown')} | "
        f"Confidence: {parsed.get('overall_confidence', 0)} | "
        f"Contradictions found: {len(parsed.get('contradictions', []))}"
    )

    return CompanyIntelligence(**parsed)
