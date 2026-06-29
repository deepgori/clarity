"""
Clarity API - Outreach Email Generator

Generates two types of outreach emails:
1. Generic cold email (baseline, for comparison in pitch docs)
2. Intelligence-driven email that references specific research findings
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


CLARITY_EMAIL_PROMPT = """You are a top-performing SDR who just spent 30 minutes researching a company.
Write a cold outreach email to {company_name} that could ONLY have been written by someone
who did real research. This is NOT a template with the company name swapped in.

You are selling: {selling}

Here is everything you found about {company_name}:

Company: {company_name} ({domain})
What they do: {what_they_do}
Industry: {industry} | Stage: {stage}

SIGNALS (recent moves that create urgency):
{signals_text}

CONTRADICTIONS (gaps between what they say and what they do):
{contradictions_text}

TECH STACK: {tech_stack}

HIRING PATTERNS: {hiring_signals}

RECOMMENDED ANGLE: {recommended_angle}
TIMING: {timing}
TARGET PERSONA: {decision_maker}

RULES:
1. Open with a SPECIFIC observation. Not "I noticed your company does X" but reference an actual signal,
   contradiction, or hiring pattern. Show you did homework.
2. Connect the observation to a real problem they likely face. Be specific about the pain.
3. Reference at least ONE concrete detail from the intelligence (a signal, a contradiction,
   a hiring pattern, or a tech stack choice) that most people would not know.
4. Keep it under 100 words. Busy people skim.
5. End with a low-friction ask (not "can I get 30 minutes of your time").
6. Do NOT use the phrases: "I noticed", "I came across", "I hope this finds you well",
   "I wanted to reach out", "leverage", "synergy".
7. Do NOT use em dashes.
8. Sound like a human, not a bot. No corporate buzzwords.
9. No subject line, just the body."""


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

    # Format signals with full context
    signals_text = "\n".join(
        f"- {s.signal} -> Sales implication: {s.implication}"
        for s in intelligence.signals
    ) or "None detected"

    contradictions_text = "\n".join(
        f"- They claim: '{c.claim_a}' BUT evidence shows: '{c.claim_b}' -> Opportunity: {c.sales_implication}"
        for c in intelligence.contradictions
    ) or "None detected"

    avoid_topics = ", ".join(intelligence.sales_strategy.avoid_topics) or "None"
    hiring_signals = ", ".join(intelligence.hiring_signals) or "None detected"
    tech_stack = ", ".join(intelligence.tech_stack) or "Unknown"

    response = await client.chat.completions.create(
        model="gpt-4o",
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
                tech_stack=tech_stack,
                recommended_angle=intelligence.sales_strategy.recommended_angle,
                timing=intelligence.sales_strategy.timing_assessment,
                decision_maker=intelligence.sales_strategy.decision_maker_profile,
                hiring_signals=hiring_signals,
            )},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()
