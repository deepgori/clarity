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
    domain: str = Field(description="Company domain to analyze, e.g., 'stripe.com'")
    selling: Optional[str] = Field(
        default=None,
        description="What you're selling (short description, used if seller_domain not provided)"
    )
    seller_domain: Optional[str] = Field(
        default=None,
        description="Your company's domain. Clarity will research your product too for better pitch matching."
    )


class ClarityResponse(BaseModel):
    """API response wrapper."""
    success: bool
    intelligence: Optional[CompanyIntelligence] = None
    error: Optional[str] = None
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
