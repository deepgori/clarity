# Clarity

Company intelligence API for AI sales agents. One API call returns structured intelligence about any company: signals, contradictions, customer review sentiment, and evidence-backed outreach angles.

**Live at [clarityapi.co](https://clarityapi.co)**

## What it does

Clarity takes a target company domain, researches it across 7 data sources in parallel, and returns:

- **Company profile**: industry, stage, and description
- **Sales signals** with implications (e.g., "Trustpilot 1.8/5 vs Gartner 4.5/5" → "enterprise buyers happy, consumers aren't")
- **Contradiction detection**: cross-references website claims against GitHub, news, jobs, community, and reviews
- **Customer review sentiment**: ratings from G2, Gartner, Trustpilot, Capterra, and Reddit via SerpAPI
- **Tech stack** extracted from GitHub repos
- **Hiring patterns** from external ATS platforms (Greenhouse, Lever, Ashby)
- **Evidence-based relevance scoring**: honest fit assessment that says "no angle identified" when the evidence doesn't support one
- **Suggested outreach email** that references specific findings, not generic bridges

## Quick start

```bash
# Clone and set up
git clone https://github.com/deepgori/clarity.git
cd clarity
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your API keys to .env

# Run
python main.py
# Open http://localhost:8000
```

## API

### POST /api/company

Analyze a company and return structured intelligence.

```bash
curl -X POST https://clarityapi.co/api/company \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "datadog.com",
    "seller_domain": "grafana.com"
  }'
```

**Request fields:**
| Field | Required | Description |
|-------|----------|-------------|
| `domain` | Yes | Target company domain |
| `seller_domain` | No | Your company domain (for relevance scoring and outreach) |
| `context` | No | Extra context for the analysis |

**Response:**
```json
{
  "success": true,
  "intelligence": {
    "company_name": "Datadog",
    "domain": "datadog.com",
    "what_they_do": "Cloud monitoring and security platform...",
    "industry": "Cloud Infrastructure / DevOps",
    "stage": "Public",
    "signals": [
      {
        "signal": "Customer reviews show Gartner 4.5/5 but Trustpilot 1.8/5",
        "implication": "Enterprise buyers satisfied, consumer experience is poor",
        "confidence": 0.7
      }
    ],
    "contradictions": [],
    "tech_stack": ["Go", "Rust", "TypeScript"],
    "hiring_signals": ["Hiring data unavailable - company likely uses enterprise ATS platforms"],
    "sales_strategy": {
      "recommended_angle": "...",
      "conversation_starter": "...",
      "relevance_score": 0.8,
      "relevance_reasoning": "..."
    },
    "overall_confidence": 0.85
  },
  "suggested_email": "...",
  "processing_time_ms": 23500
}
```

### POST /api/compare

Same as `/api/company` but also generates a generic email for side-by-side comparison.

## Architecture

```
Request → Parallel fetch (7 sources) → AI synthesis → Post-processing gates → Response
```

**Data sources (all fetched in parallel):**
| Source | Method | What it provides |
|--------|--------|-----------------|
| Website | Jina Reader (trafilatura fallback) | Company claims, positioning, product info |
| News | Google News RSS (NewsAPI fallback) | Recent announcements, funding, partnerships |
| GitHub | GitHub API | Repo activity, tech stack, commit freshness |
| Careers | Direct scraping | Internal job listings, team structure |
| External Jobs | Greenhouse, Lever, Ashby APIs | Department breakdown, tech requirements |
| Community | Hacker News Algolia API | Developer sentiment, public criticism |
| Reviews | SerpAPI (Google search) | G2, Gartner, Trustpilot, Capterra ratings; Reddit threads |

**AI layer:**
- GPT-4o with structured JSON output for intelligence synthesis
- GPT-4o for evidence-driven email generation
- Cross-source contradiction detection
- Evidence-based relevance scoring (honest "no angle" when fit is weak)

**Post-processing gates (deterministic, code-level):**
- Enterprise ATS filter: suppresses false contradictions from missing job data for public companies
- Hiring inference guard: prevents fabricated claims about job postings when no data exists
- Banned phrase scrubber: removes AI-isms from output text

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `SERPAPI_KEY` | Yes | SerpAPI key for customer review data |
| `CLARITY_API_KEY` | No | API key for authentication (disabled if not set) |
| `CLARITY_GITHUB_TOKEN` | No | GitHub token for higher rate limits |
| `NEWS_API_KEY` | No | NewsAPI key for news fallback |

## Cost per query

| Component | Cost |
|-----------|------|
| SerpAPI (reviews) | ~$0.01 |
| OpenAI GPT-4o (synthesis + email) | ~$0.03-0.05 |
| All other sources | Free |
| **Total** | **~$0.04-0.06** |

## License

MIT
