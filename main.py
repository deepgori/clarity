"""
Clarity API - Main Application

Company intelligence API designed for AI sales agents.
One API call, parallel source fetching, structured JSON output.
"""

import asyncio
import os
import time
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from cache import get_cached, set_cached

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("clarity")


def normalize_domain(raw: str) -> str:
    """Clean up user input into a proper domain."""
    domain = raw.strip().lower()
    domain = domain.replace("https://", "").replace("http://", "")
    domain = domain.rstrip("/")
    # Remove www. prefix
    if domain.startswith("www."):
        domain = domain[4:]
    # If no TLD (no dot), assume .com
    if "." not in domain:
        domain = domain + ".com"
    return domain


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

# API key auth (optional, only enforced if CLARITY_API_KEY is set)
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str | None:
    """Check API key if one is configured. Skip auth if no key is set."""
    expected_key = os.getenv("CLARITY_API_KEY", "").strip()
    if not expected_key:
        return None  # No key configured, allow all requests

    if not credentials or credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

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
async def analyze_company(
    request: ClarityRequest,
    _key: str | None = Depends(verify_api_key),
):
    """
    Analyze a company and return structured intelligence.

    Fetches data from website, news, and GitHub in parallel,
    then synthesizes everything through OpenAI with contradiction detection.
    """
    start_time = time.time()
    domain = normalize_domain(request.domain)
    seller_domain_str = normalize_domain(request.seller_domain) if request.seller_domain else None

    # Self-targeting check
    if seller_domain_str and seller_domain_str == domain:
        return ClarityResponse(
            success=False,
            error="Target and seller are the same company. Use a different target domain.",
            processing_time_ms=0,
        )

    logger.info(f"Analyzing: {domain}")

    # Check cache first
    cached = get_cached(domain, seller_domain_str, request.context)
    if cached:
        elapsed = int((time.time() - start_time) * 1000)
        cached["processing_time_ms"] = elapsed
        return ClarityResponse(**cached)

    try:
        # Phase 1: Parallel source fetching with overall timeout
        logger.info("Phase 1: Parallel source fetching")

        fetch_tasks = [
            fetch_website(domain),
            fetch_news(domain.split(".")[0], domain),
            fetch_github(domain),
        ]

        seller_content = None
        if request.seller_domain:
            seller_domain = normalize_domain(request.seller_domain)
            logger.info(f"Also fetching seller website: {seller_domain}")
            fetch_tasks.append(fetch_website(seller_domain))

        # Hard 45s timeout on all fetches combined
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*fetch_tasks),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            elapsed = int((time.time() - start_time) * 1000)
            logger.warning(f"Source fetching timed out for {domain} after 45s")
            return ClarityResponse(
                success=False,
                error=f"Fetching data for {domain} took too long. Try again or use a more specific domain.",
                processing_time_ms=elapsed,
            )

        website_result = results[0]
        news_result = results[1]
        github_result = results[2]

        if request.seller_domain and len(results) > 3:
            seller_result = results[3]
            if seller_result.fetched:
                seller_content = seller_result.content
                logger.info(f"Seller website fetched ({len(seller_content)} chars)")

        sources_status = (
            f"Website: {'ok' if website_result.fetched else 'miss'} | "
            f"News: {'ok' if news_result.fetched else 'miss'} | "
            f"GitHub: {'ok' if github_result.fetched else 'miss'}"
        )
        logger.info(f"Source results: {sources_status}")

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
            seller_content=seller_content,
            context=request.context,
        )

        # Phase 3: Generate suggested outreach email
        selling_desc = "our product"
        if seller_content:
            selling_desc = seller_content[:300].strip()
        if request.context:
            selling_desc = request.context

        suggested_email = await generate_clarity_email(
            intelligence=intelligence,
            selling=selling_desc,
        )

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Complete: {domain} in {elapsed}ms")

        response_data = ClarityResponse(
            success=True,
            intelligence=intelligence,
            suggested_email=suggested_email,
            processing_time_ms=elapsed,
        )

        # Cache the successful response
        set_cached(domain, response_data.model_dump(), seller_domain_str, request.context)

        return response_data

    except asyncio.TimeoutError:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error(f"Timeout analyzing {domain} after {elapsed}ms")
        return ClarityResponse(
            success=False,
            error=f"Analysis timed out for {domain}. The website may be too large.",
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
    domain: str = Field(description="Target company domain")
    seller_domain: str = Field(description="Your company's domain")
    context: Optional[str] = Field(
        default=None,
        description="Optional extra context"
    )


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
    domain = normalize_domain(request.domain)

    logger.info(f"Compare request: {domain}")

    try:
        # Step 1: Fetch target company data + optional seller data
        fetch_tasks = [
            fetch_website(domain),
            fetch_news(domain.split(".")[0], domain),
            fetch_github(domain),
        ]

        seller_content = None
        if request.seller_domain:
            seller_domain = normalize_domain(request.seller_domain)
            logger.info(f"Also fetching seller website: {seller_domain}")
            fetch_tasks.append(fetch_website(seller_domain))

        results = await asyncio.gather(*fetch_tasks)

        website_result = results[0]
        news_result = results[1]
        github_result = results[2]

        if request.seller_domain and len(results) > 3:
            seller_result = results[3]
            if seller_result.fetched:
                seller_content = seller_result.content

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
            seller_content=seller_content,
            context=request.context,
        )

        # Build a selling description from seller content for email generation
        selling_desc = "our product"
        if seller_content:
            # Use the first 200 chars of seller content as a brief description
            selling_desc = seller_content[:200].strip()
        if request.context:
            selling_desc = request.context

        # Step 2: Generate both emails in parallel
        generic_email, clarity_email = await asyncio.gather(
            generate_generic_email(
                company_name=intelligence.company_name,
                domain=domain,
                selling=selling_desc,
            ),
            generate_clarity_email(
                intelligence=intelligence,
                selling=selling_desc,
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
