"""
Clarity API - AI SDR Agent

Demonstrates the value of Clarity by generating two outreach emails:
1. A generic cold email (without Clarity intelligence)
2. A personalized email powered by Clarity's structured intelligence

The side-by-side comparison is the core demo moment.
"""

import os
import json
import logging
from openai import AsyncOpenAI
from models.schemas import CompanyIntelligence

logger = logging.getLogger(__name__)


GENERIC_EMAIL_PROMPT = """Write a short cold outreach email to {company_name}.

You are selling: {selling}

You only know the company name and domain ({domain}). You have no other information.
Write the kind of generic email that most AI SDR tools send today.
Keep it under 100 words. No subject line, just the body."""


CLARITY_EMAIL_PROMPT = """Write a short cold outreach email to {company_name}.

You are selling: {selling}

You have access to detailed intelligence about this company from Clarity's API:

Company: {company_name} ({domain})
What they do: {what_they_do}
Industry: {industry}
Stage: {stage}

Key signals:
{signals_text}

Contradictions detected:
{contradictions_text}

Recommended pitch angle: {recommended_angle}
Conversation starter: {conversation_starter}
Topics to avoid: {avoid_topics}
Timing: {timing}
Decision maker: {decision_maker}

Hiring signals: {hiring_signals}

Write a highly personalized email that references specific details from the intelligence.
Make it clear this was NOT a mass email. Keep it under 120 words. No subject line, just the body.
Do not use em dashes."""


async def generate_generic_email(
    company_name: str, domain: str, selling: str
) -> str:
    """Generate a generic cold email without any intelligence."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": GENERIC_EMAIL_PROMPT.format(
                company_name=company_name,
                domain=domain,
                selling=selling,
            )},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()


async def generate_clarity_email(
    intelligence: CompanyIntelligence, selling: str
) -> str:
    """Generate a personalized email powered by Clarity intelligence."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Format signals for the prompt
    signals_text = "\n".join(
        f"- {s.signal} (implication: {s.implication})"
        for s in intelligence.signals
    ) or "None detected"

    contradictions_text = "\n".join(
        f"- {c.claim_a} vs {c.claim_b} -> {c.resolution}"
        for c in intelligence.contradictions
    ) or "None detected"

    avoid_topics = ", ".join(intelligence.sales_strategy.avoid_topics) or "None"
    hiring_signals = ", ".join(intelligence.hiring_signals) or "None detected"

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": CLARITY_EMAIL_PROMPT.format(
                company_name=intelligence.company_name,
                domain=intelligence.domain,
                what_they_do=intelligence.what_they_do,
                industry=intelligence.industry,
                stage=intelligence.stage,
                selling=selling,
                signals_text=signals_text,
                contradictions_text=contradictions_text,
                recommended_angle=intelligence.sales_strategy.recommended_angle,
                conversation_starter=intelligence.sales_strategy.conversation_starter,
                avoid_topics=avoid_topics,
                timing=intelligence.sales_strategy.timing_assessment,
                decision_maker=intelligence.sales_strategy.decision_maker_profile,
                hiring_signals=hiring_signals,
            )},
        ],
        temperature=0.7,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()
