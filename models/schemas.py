"""
Clarity API - Data Models

Structured output schema for company intelligence.
These models define the JSON response that agents consume via the API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# --- Source Data Models ---

class SourceType(str, Enum):
    WEBSITE = "website"
    NEWS = "news"
    GITHUB = "github"
    JOBS = "jobs"
    COMMUNITY = "community"
    REVIEWS = "reviews"


class SourceResult(BaseModel):
    """Raw data from a single source."""
    source_type: SourceType
    url: str
    content: str
    fetched: bool = True
    error: Optional[str] = None


# --- Intelligence Output Models ---

class Signal(BaseModel):
    """A specific, evidence-backed signal about the company."""
    signal: str = Field(description="What was observed")
    implication: str = Field(description="What this means for a sales approach")
    source_url: str = Field(description="URL where this was found")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")


class Contradiction(BaseModel):
    """
    A detected contradiction between sources.
    
    This is the main differentiator for Clarity. Parallel enrichment tools
    process each source independently and can't cross-reference claims.
    Clarity runs a single reasoning pass across all sources to catch conflicts.
    """
    claim_a: str = Field(description="What one source says")
    source_a: str = Field(description="URL of first source")
    claim_b: str = Field(description="What another source says that conflicts")
    source_b: str = Field(description="URL of second source")
    resolution: str = Field(description="Most likely explanation for the contradiction")
    sales_implication: str = Field(description="How this affects the sales approach")


class SalesStrategy(BaseModel):
    """Typed decision object an AI agent can branch on programmatically."""
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="How relevant is what we're selling to this company? 0 = completely irrelevant, 1 = perfect fit"
    )
    relevance_reasoning: str = Field(
        description="Why this product is or isn't a good fit for this company. Be honest when it's a bad match."
    )
    recommended_angle: str = Field(description="The best approach to pitch this company")
    conversation_starter: str = Field(description="A specific, relevant opening line")
    avoid_topics: list[str] = Field(description="Topics or approaches that would backfire")
    timing_assessment: str = Field(description="Is now a good time to reach out? Why?")
    decision_maker_profile: str = Field(description="Who likely makes buying decisions and their likely priorities")


class CompanyIntelligence(BaseModel):
    """
    Full structured intelligence response.
    Designed for programmatic consumption by AI agents, not for human dashboards.
    """
    # Company basics
    company_name: str
    domain: str
    what_they_do: str = Field(description="One-sentence description of the company")
    industry: str
    stage: str = Field(description="e.g., 'Seed-stage startup', 'Series B', 'Public company'")

    # Structured signals
    signals: list[Signal] = Field(description="Evidence-backed observations")
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Detected contradictions between sources"
    )

    # Agent-native decision object
    sales_strategy: SalesStrategy

    # Tech footprint (when available)
    tech_stack: list[str] = Field(default_factory=list)
    hiring_signals: list[str] = Field(default_factory=list)

    # Metadata
    sources_used: list[str] = Field(description="URLs of all sources consulted")
    overall_confidence: float = Field(ge=0.0, le=1.0)
    data_freshness: str = Field(description="How recent the underlying data is")


# --- API Request/Response Models ---

class ClarityRequest(BaseModel):
    """API request body."""
    domain: str = Field(description="Target company domain to analyze, e.g., 'stripe.com'")
    seller_domain: Optional[str] = Field(
        default=None,
        description="Your company's domain. Clarity will fetch your website to understand what you sell."
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional extra context (e.g., 'focus on their enterprise tier', 'we just launched a new feature')"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language for the suggested email as an ISO 639-1 code: en, fr, de, es, it, pt, nl. Defaults to English."
    )


class ClarityResponse(BaseModel):
    """API response wrapper."""
    success: bool
    intelligence: Optional[CompanyIntelligence] = None
    suggested_email: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: int = Field(description="Total processing time in milliseconds")


# --- Candidate Mode Models ---
#
# Same sources and pipeline as sales mode, pointed at a job seeker instead of
# a sales agent: someone deciding whether to apply to a company, or preparing
# for a call with its founder.

class CandidateSignal(BaseModel):
    """An evidence-backed observation that matters to someone considering a job there."""
    signal: str = Field(description="What was observed")
    why_it_matters: str = Field(description="What this means for a candidate")
    source_url: str = Field(description="URL where this was found")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")


class CandidateContradiction(BaseModel):
    """A tension between sources, turned into a question worth asking the founder."""
    claim_a: str = Field(description="What one source says")
    source_a: str = Field(description="URL of first source")
    claim_b: str = Field(description="What another source says that conflicts")
    source_b: str = Field(description="URL of second source")
    resolution: str = Field(description="Most likely innocent explanation")
    question_to_ask: str = Field(description="A neutral question a candidate could ask the founder about it")


class FounderQuestion(BaseModel):
    """A question for the founder call, tied to a specific finding."""
    question: str
    based_on: str = Field(description="The specific finding that prompts this question")
    source_url: str = Field(default="", description="Where that finding came from")


class ClaimCheck(BaseModel):
    """One of the company's own claims, checked against independent sources."""
    claim: str = Field(description="What the company says about itself")
    claim_url: str = Field(description="Where the company says it")
    verdict: str = Field(description="verified, partly, contradicted or unverified")
    evidence: str = Field(description="The detail that decided the verdict")
    evidence_url: str = Field(default="", description="Where that evidence came from")


class CandidateIntelligence(BaseModel):
    """Company research shaped for a job seeker."""
    company_name: str
    domain: str
    what_they_do: str = Field(description="One-sentence description of the company")
    who_they_sell_to: str = Field(description="Their customers, as stated in the sources")
    industry: str
    stage: str = Field(description="Stage if the sources state it, otherwise says it was not found")

    bottom_line: str = Field(default="", description="Two sentences: what's solid, what's unknown")
    claims_checked: list[ClaimCheck] = Field(default_factory=list)
    signals: list[CandidateSignal] = Field(default_factory=list)
    contradictions: list[CandidateContradiction] = Field(default_factory=list)
    questions_for_founder: list[FounderQuestion] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Things a candidate would want to know that the sources did not answer",
    )
    unrelated_sources: list[str] = Field(
        default_factory=list,
        description="Sources whose results were about something else with the same name",
    )

    tech_stack: list[str] = Field(default_factory=list)
    hiring_signals: list[str] = Field(default_factory=list)

    sources_used: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    data_freshness: str


class CandidateRequest(BaseModel):
    """Candidate mode request body."""
    domain: str = Field(description="Company domain to research, e.g. 'linear.app'")
    candidate_profile: Optional[str] = Field(
        default=None,
        max_length=1500,
        description=(
            "Optional: a short bio, and/or a GitHub profile URL. Used only to "
            "tailor the intro note. Public GitHub repos are read if a URL is given."
        ),
    )


class CandidateResponse(BaseModel):
    """Candidate mode response wrapper."""
    success: bool
    intelligence: Optional[CandidateIntelligence] = None
    intro_note: Optional[str] = None
    sources_checked: dict[str, bool] = Field(
        default_factory=dict,
        description="Which sources returned data, so gaps are visible",
    )
    error: Optional[str] = None
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
