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

CUSTOMER REVIEW DATA:
{review_signals}

RECOMMENDED ANGLE: {recommended_angle}
CONVERSATION STARTER (use this or improve it): {conversation_starter}
WHY THIS ANGLE (the evidence): {relevance_reasoning}
TIMING: {timing}
TARGET PERSONA: {decision_maker}
TOPICS TO AVOID: {avoid_topics}

EMAIL STRUCTURE (follow this exactly):
1. OPENING LINE: Use the conversation_starter above or write something better. The opening
   must reference a SPECIFIC data point — a review rating, a contradiction, a recent move,
   or a hiring pattern. Not a compliment. Not a generic observation. A fact that shows
   you did research they didn't expect.
   
   GOOD: "Your Gartner rating is 4.5 but Trustpilot is 1.8 — that gap usually means
          enterprise buyers are happy but self-serve users aren't."
   GOOD: "Consolidating under 'Chewy Made' while expanding Vet Care to Cedar Park suggests
          you're betting on brand loyalty. Your Trustpilot 3.3 says it's not there yet."
   BAD:  "Your company is doing great things in the monitoring space."
   BAD:  "As you continue to grow..."

2. BRIDGE: In ONE sentence, connect the observation to a specific problem they likely face.
   This must be a logical inference from the data, not a generic claim.

3. PITCH: In ONE sentence, show how your product solves THAT specific problem.
   Not "we help companies like yours" — specifically how YOUR tool addresses THEIR issue.

4. ASK: Low-friction close. "Worth a 5-min look?" or a question they'd want to answer.

HARD CONSTRAINTS:
- Under 80 words total. Shorter is better. Busy people skim.
- BANNED PHRASES (never use any form of these):
  "noticed", "I noticed", "I came across", "I hope this finds you well",
  "I wanted to reach out", "leverage", "synergy", "synergies", "impressive",
  "remarkable", "congratulations", "congrats", "kudos", "game-changing",
  "game changer", "revolutionize", "transform". Do NOT start with any variant of "noticed".
- No em dashes.
- Sound like a sharp peer, not a salesperson. No corporate buzzwords.
- No subject line, just the body.
- NEVER use placeholder brackets like [Name], [CTO's Name], [Your Name], [Company].
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

    # Extract review-specific signals for the email
    review_signals = "\n".join(
        f"- {s.signal}"
        for s in intelligence.signals
        if any(kw in s.signal.lower() for kw in [
            "trustpilot", "g2", "gartner", "capterra", "review", "rating",
            "reddit", "glassdoor", "trustradius",
        ])
    ) or "No customer review data available"

    contradictions_text = "\n".join(
        f"- They claim: '{c.claim_a}' BUT evidence shows: '{c.claim_b}' -> Opportunity: {c.sales_implication}"
        for c in intelligence.contradictions
    ) or "None detected"

    avoid_topics = ", ".join(intelligence.sales_strategy.avoid_topics) or "None"

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
                review_signals=review_signals,
                recommended_angle=intelligence.sales_strategy.recommended_angle,
                conversation_starter=intelligence.sales_strategy.conversation_starter,
                relevance_reasoning=intelligence.sales_strategy.relevance_reasoning,
                timing=intelligence.sales_strategy.timing_assessment,
                decision_maker=intelligence.sales_strategy.decision_maker_profile,
                avoid_topics=avoid_topics,
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
