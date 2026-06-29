# Clarity

Company intelligence API for sales teams and AI agents. One API call returns structured intelligence about any company, including signals, contradictions, tech stack, hiring patterns, and a personalized outreach email.

## What it does

Clarity takes a target company domain and your company domain, researches both in parallel, and returns:

- **Company profile** with industry, stage, and description
- **Sales signals** with implications (e.g., "just raised Series C" -> "budget available for new tooling")
- **Contradiction detection** between what the company says and what evidence shows
- **Tech stack** extracted from GitHub repos
- **Hiring patterns** and what they imply about priorities
- **Bidirectional relevance scoring** (is your product relevant to them? are they the right customer for you?)
- **Suggested outreach email** that references specific findings

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
# Add your OpenAI API key to .env

# Run
python main.py
# Open http://localhost:8000
```

## API

### POST /api/company

Analyze a company and return structured intelligence.

```bash
curl -X POST http://localhost:8000/api/company \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "domain": "datadog.com",
    "seller_domain": "sentry.io",
    "context": "focus on their enterprise monitoring gaps"
  }'
```

**Request fields:**
| Field | Required | Description |
|-------|----------|-------------|
| `domain` | Yes | Target company domain |
| `seller_domain` | No | Your company domain (Clarity auto-extracts what you sell) |
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
    "stage": "Public (NASDAQ: DDOG)",
    "signals": [...],
    "contradictions": [...],
    "tech_stack": ["Go", "Python", "TypeScript", ...],
    "hiring_signals": [...],
    "sales_strategy": {
      "recommended_angle": "...",
      "relevance_score": 0.85,
      "relevance_reasoning": "..."
    },
    "overall_confidence": 0.9
  },
  "suggested_email": "...",
  "processing_time_ms": 12000
}
```

### POST /api/compare

Same as `/api/company` but also generates a generic email for side-by-side comparison.

## Architecture

```
Request -> Parallel fetch (website + news + GitHub + seller) -> AI synthesis -> Response
```

**Data sources:**
- Website content via Jina Reader (with trafilatura fallback)
- News via DuckDuckGo (with NewsAPI fallback)
- GitHub repos, languages, and org data via GitHub API
- Seller website for bidirectional matching

**AI layer:**
- GPT-4o with structured JSON output for intelligence synthesis
- GPT-4o for personalized email generation
- Contradiction detection across data sources
- Bidirectional relevance scoring

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `CLARITY_API_KEY` | No | API key for authentication (if not set, auth is disabled) |
| `CLARITY_GITHUB_TOKEN` | No | GitHub token for higher rate limits |
| `NEWS_API_KEY` | No | NewsAPI key for news fallback |

## License

MIT
