# Clarity

Company intelligence API for AI sales agents.

Clarity takes a company domain, pulls data from their website, news, and GitHub in parallel, then synthesizes structured intelligence through an LLM. The output is a JSON object designed for programmatic consumption by autonomous agents, not for human dashboards.

The key feature is **contradiction detection**: Clarity runs a single reasoning pass across all sources to catch conflicts that parallel enrichment tools miss (e.g., a company's website claims "API-first" but their GitHub repos are all gRPC-based).

## Quick Start

```bash
# Clone and set up
git clone https://github.com/yourusername/clarity.git
cd clarity
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the server
python main.py
```

## Usage

```bash
curl -X POST http://localhost:8000/api/company \
  -H "Content-Type: application/json" \
  -d '{"domain": "stripe.com", "selling": "a real-time analytics platform"}'
```

The `selling` field is optional. When provided, the sales strategy in the response will be tailored to pitching that specific product.

## API Response

```json
{
  "success": true,
  "intelligence": {
    "company_name": "Stripe",
    "what_they_do": "Financial infrastructure for businesses",
    "industry": "Financial Technology",
    "stage": "Growth",
    "signals": [
      {
        "signal": "Processes $1.9 trillion in payments volume",
        "implication": "Large-scale data needs, good fit for analytics tooling",
        "source_url": "https://stripe.com/",
        "confidence": 0.9
      }
    ],
    "contradictions": [
      {
        "claim_a": "Website claims enterprise-ready security",
        "source_a": "https://stripe.com/",
        "claim_b": "Recent news reports a third-party security breach",
        "source_b": "https://news.example.com/...",
        "resolution": "Breach was via a third-party tool, not core infrastructure",
        "sales_implication": "Avoid leading with security claims in outreach"
      }
    ],
    "sales_strategy": {
      "recommended_angle": "...",
      "conversation_starter": "...",
      "avoid_topics": ["..."],
      "timing_assessment": "...",
      "decision_maker_profile": "..."
    },
    "tech_stack": ["Ruby", "Go", "React"],
    "overall_confidence": 0.85
  },
  "processing_time_ms": 11500
}
```

## Architecture

```
Input: domain
    |
    v
+---------------------------+
|   FastAPI Orchestrator     |
|   (asyncio.gather)        |
+--+--------+--------+------+
   |        |        |
   v        v        v
Website   News    GitHub
(Jina)   (DDG)    (API)
   |        |        |
   +--------+--------+
            |
            v
+---------------------------+
|  OpenAI Structured Output |
|  (contradiction detection)|
+---------------------------+
            |
            v
    Structured JSON Response
```

**Data sources:**
- **Website**: Jina Reader (primary, handles SPAs) with trafilatura fallback
- **News**: DuckDuckGo search (primary) with NewsAPI.org fallback
- **GitHub**: REST API for org repos and language analysis

All sources are fetched in parallel. Each has a coded fallback path so a single source failure doesn't break the response.

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o synthesis |
| `GITHUB_TOKEN` | No | GitHub PAT for higher rate limits (5000/hr vs 60/hr) |
| `NEWS_API_KEY` | No | NewsAPI.org key for news fallback |

## Tech Stack

- Python 3.12+
- FastAPI + Uvicorn
- httpx (async HTTP)
- OpenAI (structured output)
- Jina Reader + trafilatura (web content extraction)
- DuckDuckGo search + NewsAPI (news)
- GitHub REST API

## License

MIT
