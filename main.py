"""
Clarity API - Main Application

Company intelligence API designed for AI sales agents.
One API call, parallel source fetching, structured JSON output.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.schemas import ClarityRequest, ClarityResponse, CompanyIntelligence
from sources.website import fetch_website
from sources.news import fetch_news
from sources.github import fetch_github
from synthesis.engine import synthesize_intelligence
from agent import generate_generic_email, generate_clarity_email

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("clarity")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Clarity API starting up")
    yield
    logger.info("Clarity API shutting down")


app = FastAPI(
    title="Clarity API",
    description=(
        "Company intelligence API for AI sales agents. "
        "One API call returns structured intelligence that agents "
        "can consume and act on programmatically."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the demo frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "clarity-api"}


@app.post("/api/company", response_model=ClarityResponse)
async def analyze_company(request: ClarityRequest):
    """
    Analyze a company and return structured intelligence.

    Fetches data from website, news, and GitHub in parallel,
    then synthesizes everything through OpenAI with contradiction detection.

    Args:
        request: ClarityRequest with domain and optional selling context

    Returns:
        ClarityResponse with structured intelligence or error
    """
    start_time = time.time()
    domain = request.domain.strip().lower()

    # Remove protocol prefix if provided
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

    logger.info(f"Analyzing: {domain}")

    try:
        # Phase 1: Parallel source fetching
        logger.info("Phase 1: Parallel source fetching")

        website_result, news_result, github_result = await asyncio.gather(
            fetch_website(domain),
            fetch_news(domain.split(".")[0], domain),
            fetch_github(domain),
        )

        sources_status = (
            f"Website: {'ok' if website_result.fetched else 'miss'} | "
            f"News: {'ok' if news_result.fetched else 'miss'} | "
            f"GitHub: {'ok' if github_result.fetched else 'miss'}"
        )
        logger.info(f"Source results: {sources_status}")

        # We need at minimum the website to produce useful intelligence
        if not website_result.fetched:
            elapsed = int((time.time() - start_time) * 1000)
            return ClarityResponse(
                success=False,
                error=f"Could not fetch website for {domain}. The website may be unreachable.",
                processing_time_ms=elapsed,
            )

        # Phase 2: AI synthesis
        logger.info("Phase 2: Synthesis with contradiction detection")

        intelligence = await synthesize_intelligence(
            domain=domain,
            website_result=website_result,
            news_result=news_result,
            github_result=github_result,
            selling=request.selling,
        )

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Complete: {domain} in {elapsed}ms")

        return ClarityResponse(
            success=True,
            intelligence=intelligence,
            processing_time_ms=elapsed,
        )

    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error(f"Error analyzing {domain}: {e}", exc_info=True)
        return ClarityResponse(
            success=False,
            error=f"Analysis failed: {str(e)}",
            processing_time_ms=elapsed,
        )


class CompareRequest(BaseModel):
    """Request for side-by-side email comparison."""
    domain: str = Field(description="Company domain to analyze")
    selling: str = Field(description="What you're selling")


class CompareResponse(BaseModel):
    """Side-by-side email comparison response."""
    success: bool
    company_name: Optional[str] = None
    generic_email: Optional[str] = None
    clarity_email: Optional[str] = None
    intelligence: Optional[CompanyIntelligence] = None
    processing_time_ms: int = 0
    error: Optional[str] = None


@app.post("/api/compare", response_model=CompareResponse)
async def compare_emails(request: CompareRequest):
    """
    Run the full pipeline: analyze a company, then generate two emails.

    Returns a generic cold email (no intelligence) side-by-side with
    a Clarity-powered personalized email. This is the core demo endpoint.
    """
    start_time = time.time()
    domain = request.domain.strip().lower()
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

    logger.info(f"Compare request: {domain}")

    try:
        # Step 1: Get intelligence
        website_result, news_result, github_result = await asyncio.gather(
            fetch_website(domain),
            fetch_news(domain.split(".")[0], domain),
            fetch_github(domain),
        )

        if not website_result.fetched:
            elapsed = int((time.time() - start_time) * 1000)
            return CompareResponse(
                success=False,
                error=f"Could not fetch website for {domain}",
                processing_time_ms=elapsed,
            )

        intelligence = await synthesize_intelligence(
            domain=domain,
            website_result=website_result,
            news_result=news_result,
            github_result=github_result,
            selling=request.selling,
        )

        # Step 2: Generate both emails in parallel
        generic_email, clarity_email = await asyncio.gather(
            generate_generic_email(
                company_name=intelligence.company_name,
                domain=domain,
                selling=request.selling,
            ),
            generate_clarity_email(
                intelligence=intelligence,
                selling=request.selling,
            ),
        )

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Compare complete: {domain} in {elapsed}ms")

        return CompareResponse(
            success=True,
            company_name=intelligence.company_name,
            generic_email=generic_email,
            clarity_email=clarity_email,
            intelligence=intelligence,
            processing_time_ms=elapsed,
        )

    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error(f"Compare error for {domain}: {e}", exc_info=True)
        return CompareResponse(
            success=False,
            error=f"Comparison failed: {str(e)}",
            processing_time_ms=elapsed,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
