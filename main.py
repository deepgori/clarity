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
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.schemas import ClarityRequest, ClarityResponse, CompanyIntelligence
from sources.website import fetch_website
from sources.news import fetch_news
from sources.github import fetch_github, refetch_github_with_website_hint
from sources.careers import extract_careers_data, format_careers_for_synthesis
from sources.jobs import fetch_jobs
from sources.community import fetch_community
from sources.reviews import fetch_reviews
from synthesis.engine import synthesize_intelligence
from agent import generate_generic_email, generate_clarity_email
from cache import get_cached, set_cached
from security import rate_limiter, validate_domain
from analytics import log_request, log_feedback, get_analytics_summary

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
    # Remove any path components
    if "/" in domain:
        domain = domain.split("/")[0]
    # Strip spaces (handles company names like "amazon prime video")
    domain = domain.replace(" ", "")
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
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str | None:
    """Check API key if one is configured. Skip auth for the built-in frontend."""
    expected_key = os.getenv("CLARITY_API_KEY", "").strip()
    if not expected_key:
        return None  # No key configured, allow all requests

    # Allow requests from the built-in frontend (same-origin)
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    if host and (host in referer or host in origin):
        return None

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


@app.post("/admin/clear-cache")
async def admin_clear_cache(_key: str | None = Depends(verify_api_key)):
    """Clear the response cache. Requires API key."""
    from cache import clear_cache
    clear_cache()
    return {"status": "cache_cleared"}

@app.get("/costs")
async def get_costs(_key: str | None = Depends(verify_api_key)):
    """Get cumulative OpenAI spend stats. Requires API key."""
    from costs import cost_tracker
    return cost_tracker.get_summary()


@app.get("/stats")
async def get_stats():
    """Get public usage and feedback stats."""
    return get_analytics_summary()


class FeedbackRequest(BaseModel):
    """User feedback on analysis quality."""
    domain: str = Field(description="Domain that was analyzed")
    rating: str = Field(description="'up' or 'down'")


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest, http_request: Request):
    """Record thumbs up/down feedback on analysis quality."""
    if feedback.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")
    client_ip = http_request.headers.get("x-forwarded-for", http_request.client.host).split(",")[0].strip()
    log_feedback(feedback.domain, feedback.rating, client_ip)
    return {"status": "recorded"}


class AnalysisResult:
    """Internal result from the shared analysis pipeline."""
    def __init__(self, intelligence=None, seller_content=None, error=None):
        self.intelligence = intelligence
        self.seller_content = seller_content
        self.error = error
        self.success = intelligence is not None


async def _run_analysis(
    domain: str,
    seller_domain_str: str | None = None,
    context: str | None = None,
) -> AnalysisResult:
    """
    Shared analysis pipeline: fetch all sources, correct GitHub org,
    extract careers, and synthesize intelligence.

    Both /api/company and /api/compare call this.
    """
    # Phase 1: Parallel source fetching with overall timeout
    logger.info("Phase 1: Parallel source fetching")

    fetch_tasks = [
        fetch_website(domain),
        fetch_news(domain.split(".")[0], domain),
        fetch_github(domain),
        fetch_jobs(domain),
        fetch_community(domain),
        fetch_reviews(domain),
    ]

    if seller_domain_str:
        logger.info(f"Also fetching seller website: {seller_domain_str}")
        fetch_tasks.append(fetch_website(seller_domain_str))

    # Hard 45s timeout on all fetches combined
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*fetch_tasks, return_exceptions=True),
            timeout=45.0,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Source fetching timed out for {domain} after 45s")
        return AnalysisResult(error=f"Fetching data for {domain} took too long. Try again or use a more specific domain.")

    website_result = results[0]
    news_result = results[1]
    github_result = results[2]
    jobs_result = results[3]
    community_result = results[4]
    reviews_result = results[5]

    # Extract seller content if fetched
    seller_content = None
    if seller_domain_str and len(results) > 6:
        seller_result = results[6]
        if not isinstance(seller_result, Exception) and seller_result.fetched:
            seller_content = seller_result.content
            logger.info(f"Seller website fetched ({len(seller_content)} chars)")

    # Log source status (handle exceptions from gather)
    def _source_ok(r):
        return not isinstance(r, Exception) and r.fetched

    sources_status = (
        f"Website: {'ok' if _source_ok(website_result) else 'miss'} | "
        f"News: {'ok' if _source_ok(news_result) else 'miss'} | "
        f"GitHub: {'ok' if _source_ok(github_result) else 'miss'} | "
        f"Jobs: {'ok' if _source_ok(jobs_result) else 'miss'} | "
        f"Community: {'ok' if _source_ok(community_result) else 'miss'} | "
        f"Reviews: {'ok' if _source_ok(reviews_result) else 'miss'}"
    )
    logger.info(f"Source results: {sources_status}")

    # Website is required
    if isinstance(website_result, Exception) or not website_result.fetched:
        return AnalysisResult(error=f"Could not fetch website for {domain}. The website may be unreachable.")

    # Phase 1.1: Check if GitHub org mapping looks wrong
    # If GitHub data has all abandoned/unknown repos, try extracting the
    # real org from the website content (e.g. notion.so -> makenotion)
    if _source_ok(github_result) and website_result.fetched:
        gh_content = github_result.content
        has_active = "ACTIVE (committed within 30 days)" in gh_content
        all_abandoned = (
            "Active (committed within 30 days): 0" in gh_content
            and ("Abandoned" in gh_content or "No public repositories" in gh_content)
        )
        if not has_active and all_abandoned:
            logger.info("GitHub data looks suspicious (0 active repos), trying website hint...")
            refetched = await refetch_github_with_website_hint(
                domain, website_result.content
            )
            if refetched and refetched.fetched:
                github_result = refetched
                logger.info("GitHub org corrected via website hint")

    # Phase 1.5: Extract structured careers data
    careers_text = None
    if website_result.content:
        content = website_result.content
        careers_start = content.find("=== CAREERS ===")
        if careers_start != -1:
            careers_end = content.find("===", careers_start + 15)
            careers_text = content[careers_start:careers_end] if careers_end != -1 else content[careers_start:]

    careers_formatted = None
    if careers_text and len(careers_text.strip()) > 100:
        logger.info("Phase 1.5: Extracting structured careers data")
        try:
            careers_data = await asyncio.wait_for(
                extract_careers_data(careers_text),
                timeout=15.0,
            )
            if careers_data:
                careers_formatted = format_careers_for_synthesis(careers_data)
                logger.info(f"Careers extraction: {careers_data.get('total_roles_found', 0)} roles parsed")
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Careers extraction failed: {e}")

    # Phase 2: AI synthesis (30s timeout)
    logger.info("Phase 2: Synthesis with contradiction detection")

    # Safely pass source results (may be exceptions from gather)
    safe_news = news_result if not isinstance(news_result, Exception) else None
    safe_github = github_result if not isinstance(github_result, Exception) else None
    safe_jobs = jobs_result if not isinstance(jobs_result, Exception) else None
    safe_community = community_result if not isinstance(community_result, Exception) else None
    safe_reviews = reviews_result if not isinstance(reviews_result, Exception) else None

    try:
        intelligence = await asyncio.wait_for(
            synthesize_intelligence(
                domain=domain,
                website_result=website_result,
                news_result=safe_news or website_result,  # fallback to avoid None
                github_result=safe_github or website_result,
                jobs_result=safe_jobs,
                community_result=safe_community,
                reviews_result=safe_reviews,
                seller_content=seller_content,
                context=context,
                careers_data=careers_formatted,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        return AnalysisResult(error="AI synthesis timed out. OpenAI may be experiencing delays.")

    return AnalysisResult(intelligence=intelligence, seller_content=seller_content)


@app.post("/api/company", response_model=ClarityResponse)
async def analyze_company(
    request: ClarityRequest,
    http_request: Request,
    _key: str | None = Depends(verify_api_key),
):
    """
    Analyze a company and return structured intelligence.

    Fetches data from website, news, GitHub, jobs, and community in parallel,
    then synthesizes everything through OpenAI with contradiction detection.
    """
    # Rate limiting
    client_ip = http_request.headers.get("x-forwarded-for", http_request.client.host).split(",")[0].strip()
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {rate_limiter.max_requests} requests per minute.",
        )

    start_time = time.time()
    domain = normalize_domain(request.domain)
    seller_domain_str = normalize_domain(request.seller_domain) if request.seller_domain else None

    # Domain validation
    domain_error = validate_domain(domain)
    if domain_error:
        return ClarityResponse(success=False, error=domain_error, processing_time_ms=0)
    if seller_domain_str:
        seller_error = validate_domain(seller_domain_str)
        if seller_error:
            return ClarityResponse(success=False, error=f"Seller domain: {seller_error}", processing_time_ms=0)

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
        # Run the shared analysis pipeline
        result = await _run_analysis(domain, seller_domain_str, request.context)

        if not result.success:
            elapsed = int((time.time() - start_time) * 1000)
            log_request(domain, False, elapsed, client_ip, seller_domain_str)
            return ClarityResponse(success=False, error=result.error, processing_time_ms=elapsed)

        # Generate suggested outreach email (30s timeout)
        selling_desc = "our product"
        if result.seller_content:
            selling_desc = result.seller_content[:300].strip()
        if request.context:
            selling_desc = request.context

        try:
            suggested_email = await asyncio.wait_for(
                generate_clarity_email(
                    intelligence=result.intelligence,
                    selling=selling_desc,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Email generation timed out, returning without email")
            suggested_email = None

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Complete: {domain} in {elapsed}ms")

        response_data = ClarityResponse(
            success=True,
            intelligence=result.intelligence,
            suggested_email=suggested_email,
            processing_time_ms=elapsed,
        )

        # Cache the successful response
        set_cached(domain, response_data.model_dump(), seller_domain_str, request.context)

        # Log analytics
        log_request(domain, True, elapsed, client_ip, seller_domain_str, has_email=bool(suggested_email))

        return response_data

    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error(f"Error analyzing {domain}: {e}", exc_info=True)
        log_request(domain, False, elapsed, client_ip, seller_domain_str)
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
async def compare_emails(
    request: CompareRequest,
    http_request: Request,
    _key: str | None = Depends(verify_api_key),
):
    """
    Run the full pipeline: analyze a company, then generate two emails.

    Returns a generic cold email (no intelligence) side-by-side with
    a Clarity-powered personalized email. This is the core demo endpoint.
    """
    # Rate limiting
    client_ip = http_request.headers.get("x-forwarded-for", http_request.client.host).split(",")[0].strip()
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {rate_limiter.max_requests} requests per minute.",
        )

    start_time = time.time()
    domain = normalize_domain(request.domain)

    # Domain validation
    domain_error = validate_domain(domain)
    if domain_error:
        return CompareResponse(success=False, error=domain_error, processing_time_ms=0)

    seller_domain_str = None
    if request.seller_domain:
        seller_domain_str = normalize_domain(request.seller_domain)
        seller_error = validate_domain(seller_domain_str)
        if seller_error:
            return CompareResponse(success=False, error=f"Seller domain: {seller_error}", processing_time_ms=0)

    logger.info(f"Compare request: {domain}")

    try:
        # Run the shared analysis pipeline
        result = await _run_analysis(domain, seller_domain_str, request.context)

        if not result.success:
            elapsed = int((time.time() - start_time) * 1000)
            log_request(domain, False, elapsed, client_ip, seller_domain_str)
            return CompareResponse(success=False, error=result.error, processing_time_ms=elapsed)

        # Build selling description
        selling_desc = "our product"
        if result.seller_content:
            selling_desc = result.seller_content[:200].strip()
        if request.context:
            selling_desc = request.context

        # Generate both emails in parallel (30s timeout)
        try:
            generic_email, clarity_email = await asyncio.wait_for(
                asyncio.gather(
                    generate_generic_email(
                        company_name=result.intelligence.company_name,
                        domain=domain,
                        selling=selling_desc,
                    ),
                    generate_clarity_email(
                        intelligence=result.intelligence,
                        selling=selling_desc,
                    ),
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            elapsed = int((time.time() - start_time) * 1000)
            logger.warning("Email generation timed out in compare")
            generic_email = None
            clarity_email = None

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Compare complete: {domain} in {elapsed}ms")

        # Log analytics
        log_request(domain, True, elapsed, client_ip, seller_domain_str, has_email=bool(clarity_email))

        return CompareResponse(
            success=True,
            company_name=result.intelligence.company_name,
            generic_email=generic_email,
            clarity_email=clarity_email,
            intelligence=result.intelligence,
            processing_time_ms=elapsed,
        )

    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error(f"Compare error for {domain}: {e}", exc_info=True)
        log_request(domain, False, elapsed, client_ip, seller_domain_str)
        return CompareResponse(
            success=False,
            error=f"Comparison failed: {str(e)}",
            processing_time_ms=elapsed,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
