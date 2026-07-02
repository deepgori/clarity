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
from costs import cost_tracker

logger = logging.getLogger(__name__)


GENERIC_EMAIL_PROMPT = """Write a short cold outreach email to {company_name}.

You are selling: {selling}

You only know the company name and domain ({domain}). You have no other information.
Write the kind of generic email that most AI SDR tools send today.
Keep it under 100 words. No subject line, just the body."""


CLARITY_EMAIL_PROMPT = """You are a top-performing SDR writing a cold email that PROVES you did real research.
The email should make the reader think "how did they know that?" not "this is a template."

You are selling: {selling}

TARGET COMPANY INTELLIGENCE:
Company: {company_name} ({domain})
What they do: {what_they_do}
Industry: {industry} | Stage: {stage}

SIGNALS (recent moves, non-obvious patterns):
{signals_text}

CONTRADICTIONS (tensions between claims and behavior):
{contradictions_text}

TECH STACK: {tech_stack}

HIRING PATTERNS: {hiring_signals}

RECOMMENDED ANGLE: {recommended_angle}
TIMING: {timing}
TARGET PERSONA: {decision_maker}

STRICT RULES:
1. OPENING LINE: Start with a specific, non-obvious observation. NOT their revenue, scale,
   or most famous product metric. Reference a signal, a contradiction, a hiring pattern,
   or a recent move that most people wouldn't notice. The reader should think "they actually
   looked into us" within the first sentence.
2. NEVER open with a compliment about their size, revenue, volume, or uptime. Every email
   they get starts this way. Do the opposite.
3. Connect your observation to a specific problem they likely face right now.
4. Show exactly how your product solves THAT specific problem (not generically).
5. Keep it under 80 words. Shorter is better. Busy people skim.
6. End with a low-friction ask. Not "can I get 30 minutes" but something like
   "worth a 5-min look?" or a specific question they'd want to answer.
7. BANNED PHRASES: "I noticed", "I came across", "I hope this finds you well",
   "I wanted to reach out", "leverage", "synergy", "impressive", "remarkable"
8. Do NOT use em dashes.
9. Sound like a sharp peer, not a salesperson. No corporate buzzwords.
10. No subject line, just the body."""


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

    usage = response.usage
    if usage:
        cost_tracker.record("gpt-4o-mini", usage.prompt_tokens, usage.completion_tokens, caller="generic_email")

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

    usage = response.usage
    if usage:
        cost_tracker.record("gpt-4o", usage.prompt_tokens, usage.completion_tokens, caller="clarity_email")

    return response.choices[0].message.content.strip()
