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
7. BANNED PHRASES (do NOT use any form of these, including plurals or without pronouns):
   "noticed", "I noticed", "I came across", "I hope this finds you well",
   "I wanted to reach out", "leverage", "synergy", "synergies", "impressive",
   "remarkable", "congratulations", "congrats", "kudos", "game-changing",
   "game changer", "revolutionize", "transform". Do NOT start with any variant of "noticed".
8. Do NOT use em dashes.
9. Sound like a sharp peer, not a salesperson. No corporate buzzwords.
10. No subject line, just the body.
11. NEVER use placeholder brackets like [Name], [CTO's Name], [Your Name], [Company].
    Write the email as a ready-to-send body with no blanks to fill in.
    If you don't know someone's name, don't address anyone by name."""


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

    raw_email = response.choices[0].message.content.strip()
    return _scrub_banned_phrases(raw_email)


# Banned phrases that the LLM consistently ignores in prompt instructions.
# Post-processing is the only reliable enforcement mechanism.
import re

_BANNED_PATTERNS = [
    (r'\bsynergies\b', 'alignment'),
    (r'\bsynergy\b', 'alignment'),
    (r'\bleverage\b', 'use'),
    (r'\bLeverage\b', 'Use'),
    (r'\bimpressive\b', 'notable'),
    (r'\bremarkable\b', 'notable'),
    (r'\bgame-changing\b', 'significant'),
    (r'\bgame changer\b', 'significant shift'),
    (r'\brevolutionize\b', 'improve'),
    (r'\btransform\b', 'improve'),
    (r'\bI noticed\b', 'Your'),
    (r'\bI came across\b', 'Your'),
    (r'\bCongratulations\b', 'Your recent'),
    (r'\bcongratulations\b', 'your recent'),
    (r'\bCongrats\b', 'Your recent'),
    (r'\bcongrats\b', 'your recent'),
]

# Patterns to remove entirely (with surrounding context cleanup)
_BANNED_REMOVALS = [
    r'I hope this finds you well\.?\s*',
    r'I wanted to reach out\s*(to you\s*)?',
]


def _scrub_banned_phrases(text: str) -> str:
    """Post-process generated text to remove banned phrases the LLM ignores."""
    for pattern, replacement in _BANNED_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    for pattern in _BANNED_REMOVALS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean up em dashes
    text = text.replace('\u2014', ',')  # em dash to comma
    text = text.replace('\u2013', '-')  # en dash to hyphen

    return text.strip()
