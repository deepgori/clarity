"""Candidate mode tests. No network or OpenAI calls: sources and models are stubbed."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from models.schemas import (
    CandidateRequest, CandidateIntelligence, SourceResult, SourceType,
)
from synthesis.candidate import (
    HIRING_UNAVAILABLE, apply_candidate_gates, scrub_text, sources_checked_map,
)
from synthesis.engine import format_source_sections
from sources.candidate_profile import extract_github_username


def src(kind, url, content="data", fetched=True):
    return SourceResult(source_type=kind, url=url, content=content, fetched=fetched)


def base_parsed(**overrides):
    parsed = {
        "company_name": "Acme",
        "domain": "acme.dev",
        "what_they_do": "Builds deploy tooling.",
        "who_they_sell_to": "Platform teams",
        "industry": "Developer tools",
        "stage": "Not stated in sources",
        "signals": [],
        "contradictions": [],
        "questions_for_founder": [],
        "evidence_gaps": [],
        "tech_stack": [],
        "hiring_signals": [],
        "sources_used": [],
        "overall_confidence": 0.9,
        "data_freshness": "Data is current as of 2026-11-06",
    }
    parsed.update(overrides)
    return parsed


ALL_SOURCES = {"website": True, "news": True, "github": True, "jobs": True, "community": True, "reviews": True}
URLS = {
    "website": "https://acme.dev",
    "github": "https://github.com/acme",
    "news": "https://news.google.com/search?q=acme",
    "jobs": "https://acme.dev/careers",
}


def gates(parsed, **kw):
    args = dict(
        domain="acme.dev", corpus="", source_urls=URLS,
        careers_available=False, jobs_fetched=True, sources_checked=ALL_SOURCES,
    )
    args.update(kw)
    return apply_candidate_gates(parsed, **args)


# --- Request validation --------------------------------------------------------

def test_request_requires_domain():
    with pytest.raises(ValidationError):
        CandidateRequest()


def test_request_profile_is_optional_and_length_capped():
    assert CandidateRequest(domain="acme.dev").candidate_profile is None
    with pytest.raises(ValidationError):
        CandidateRequest(domain="acme.dev", candidate_profile="x" * 1501)


def test_endpoint_rejects_private_domain_without_fetching(monkeypatch):
    called = []
    monkeypatch.setattr(main, "run_candidate_pipeline", lambda *a, **k: called.append(1))
    client = TestClient(main.app)
    r = client.post("/api/candidate", json={"domain": "localhost"})
    assert r.status_code == 200
    assert r.json()["success"] is False
    assert not called


def test_speedrun_page_is_served():
    r = TestClient(main.app).get("/speedrun")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# --- Evidence-gap path ------------------------------------------------------------

def test_no_hiring_data_strips_hiring_claims_and_keeps_honest_question():
    parsed = base_parsed(
        signals=[
            {"signal": "No open roles on job boards", "why_it_matters": "Hiring freeze",
             "source_url": "https://acme.dev", "confidence": 0.7},
            {"signal": "Main repo pushed 3 days ago", "why_it_matters": "Shipping weekly",
             "source_url": "https://github.com/acme/core", "confidence": 0.9},
        ],
        contradictions=[{
            "claim_a": "Scaling fast", "source_a": "website",
            "claim_b": "Zero job postings on Greenhouse", "source_b": "External job postings",
            "resolution": "May hire via referrals", "question_to_ask": "How are you hiring?",
        }],
        questions_for_founder=[
            {"question": "Why no openings?", "based_on": "No job postings found", "source_url": ""},
            {"question": "What roles come next?", "based_on": "Main repo is active", "source_url": ""},
        ],
        hiring_signals=["Not hiring"],
    )
    out = gates(parsed, jobs_fetched=False,
                sources_checked={**ALL_SOURCES, "jobs": False})

    assert out["hiring_signals"] == [HIRING_UNAVAILABLE]
    assert [s["signal"] for s in out["signals"]] == ["Main repo pushed 3 days ago"]
    assert out["contradictions"] == []
    assert [q["question"] for q in out["questions_for_founder"]] == ["What roles come next?"]
    assert "No listings found on Greenhouse, Lever or Ashby" in out["evidence_gaps"]
    CandidateIntelligence(**out)  # still a valid response


def test_missing_sources_become_gaps_and_cap_confidence():
    thin = {"website": True, "news": False, "github": False, "jobs": False, "community": False, "reviews": False}
    out = gates(base_parsed(evidence_gaps=["Funding not disclosed"]), sources_checked=thin)
    assert out["overall_confidence"] <= 0.4
    assert out["evidence_gaps"][-1] == "Funding not disclosed"
    assert any("GitHub" in g for g in out["evidence_gaps"])
    assert any("news" in g for g in out["evidence_gaps"])


def test_hiring_data_present_keeps_hiring_signals():
    out = gates(base_parsed(hiring_signals=["4 backend roles open on Ashby"]))
    assert out["hiring_signals"] == ["4 backend roles open on Ashby"]


# --- Source attribution ------------------------------------------------------------

def test_untraceable_signal_urls_are_dropped_and_labels_mapped():
    corpus = "Series A announced https://techcrunch.com/2026/10/acme-raises ..."
    parsed = base_parsed(signals=[
        {"signal": "Raised Series A", "why_it_matters": "Runway", "source_url": "https://techcrunch.com/2026/10/acme-raises/", "confidence": 0.8},
        {"signal": "Invented launch", "why_it_matters": "x", "source_url": "https://madeup.example/acme", "confidence": 0.8},
        {"signal": "Repo active", "why_it_matters": "x", "source_url": "GitHub", "confidence": 0.8},
        {"signal": "Pricing page", "why_it_matters": "x", "source_url": "https://www.acme.dev/pricing-unfetched", "confidence": 0.8},
    ])
    out = gates(parsed, corpus=corpus)
    by_signal = {s["signal"]: s["source_url"] for s in out["signals"]}
    assert "Invented launch" not in by_signal
    assert by_signal["Raised Series A"].startswith("https://techcrunch.com/")
    assert by_signal["Repo active"] == "https://github.com/acme"
    assert by_signal["Pricing page"] == "https://acme.dev"


def test_github_repo_under_verified_org_is_kept_but_other_org_is_not():
    parsed = base_parsed(signals=[
        {"signal": "a", "why_it_matters": "x", "source_url": "https://github.com/acme/cli", "confidence": 0.5},
        {"signal": "b", "why_it_matters": "x", "source_url": "https://github.com/acmecorp/cli", "confidence": 0.5},
    ])
    out = gates(parsed)
    assert [s["signal"] for s in out["signals"]] == ["a"]


# --- Writing style -------------------------------------------------------------------

def test_scrub_removes_em_dashes_and_filler():
    assert scrub_text("Fast — and impressive") == "Fast, and notable"
    out = gates(base_parsed(what_they_do="Deploys—fast"))
    assert "—" not in out["what_they_do"]


# --- Candidate profile -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I build things. github.com/deepgori", "deepgori"),
    ("https://www.github.com/some-user/repo", "some-user"),
    ("see https://github.com/orgs/acme", None),
    ("no links here", None),
    (None, None),
    ("github.com/-bad", None),
])
def test_extract_github_username(text, expected):
    assert extract_github_username(text) == expected


# --- Pipeline wiring -----------------------------------------------------------------

def _fetched(**kw):
    return main.FetchedSources(
        website=src(SourceType.WEBSITE, "https://acme.dev", "Acme builds deploy tooling."),
        news=src(SourceType.NEWS, "", "", fetched=False),
        github=src(SourceType.GITHUB, "https://github.com/acme"),
        jobs=None, community=None, reviews=None,
        **kw,
    )


def test_candidate_pipeline_reports_fetch_error(monkeypatch):
    async def fail(*a, **k):
        return main.FetchedSources(error="Could not fetch website")
    monkeypatch.setattr(main, "_fetch_sources", fail)
    resp = asyncio.run(main.run_candidate_pipeline("acme.dev", None))
    assert resp.success is False and "Could not fetch" in resp.error


def test_candidate_pipeline_returns_sources_checked(monkeypatch):
    async def fetch(*a, **k):
        return _fetched()

    async def synth(**k):
        return CandidateIntelligence(**base_parsed())

    async def note(intel, ctx):
        return "Hi"

    monkeypatch.setattr(main, "_fetch_sources", fetch)
    monkeypatch.setattr(main, "synthesize_candidate_intelligence", synth)
    monkeypatch.setattr(main, "generate_intro_note", note)
    resp = asyncio.run(main.run_candidate_pipeline("acme.dev", None))
    assert resp.success and resp.intro_note == "Hi"
    assert resp.sources_checked == {
        "website": True, "news": False, "github": True,
        "jobs": False, "community": False, "reviews": False,
    }


def test_sales_pipeline_still_gets_website_fallbacks(monkeypatch):
    """Regression for the _fetch_sources split: sales synthesis inputs are unchanged."""
    seen = {}

    async def fetch(*a, **k):
        f = _fetched(seller_content="We sell monitoring")
        f.news = None
        f.github = None
        return f

    async def synth(**k):
        seen.update(k)
        return "intel"

    monkeypatch.setattr(main, "_fetch_sources", fetch)
    monkeypatch.setattr(main, "synthesize_intelligence", synth)
    result = asyncio.run(main._run_analysis("acme.dev", "seller.io", "ctx"))
    assert result.success and result.seller_content == "We sell monitoring"
    assert seen["news_result"] is seen["website_result"]
    assert seen["github_result"] is seen["website_result"]
    assert seen["context"] == "ctx"


def test_format_source_sections_missing_messages_unchanged():
    w = src(SourceType.WEBSITE, "https://acme.dev", "site")
    miss = src(SourceType.NEWS, "", "", fetched=False)
    s = format_source_sections(w, miss, miss)
    assert s["news_content"] == "No recent news found."
    assert s["github_content"] == "No GitHub presence found."
    assert s["jobs_content"].startswith("No external job board data found")
    assert s["careers_content"] == "No structured careers data available."


def test_sources_checked_map_handles_exceptions():
    m = sources_checked_map(RuntimeError("x"), None, None, None, None)
    assert m["news"] is False and m["website"] is True


def test_unrelated_sources_become_gaps_and_dont_count_as_evidence():
    checked = {"website": True, "news": False, "github": False, "jobs": False, "community": True, "reviews": True}
    out = gates(base_parsed(unrelated_sources=["community", "reviews", "community", "news", "bogus"]),
                sources_checked=checked)
    assert out["unrelated_sources"] == ["community", "reviews"]  # deduped; unfetched/unknown dropped
    assert any("Hacker News results were about something else" in g for g in out["evidence_gaps"])
    assert out["overall_confidence"] <= 0.4  # only the website really counted


def test_unrelated_job_board_counts_as_no_hiring_data():
    parsed = base_parsed(
        unrelated_sources=["jobs"],
        hiring_signals=["11 roles across Bangalore, Manila and Toronto"],
        signals=[{"signal": "11 open roles on Ashby", "why_it_matters": "Scaling", "source_url": "https://acme.dev/careers", "confidence": 0.8},
                 {"signal": "Shipped v2 CLI", "why_it_matters": "Pace", "source_url": "https://acme.dev", "confidence": 0.8}],
    )
    out = gates(parsed)  # jobs fetched, but judged to be a different company
    assert out["hiring_signals"] == [HIRING_UNAVAILABLE]
    assert [s["signal"] for s in out["signals"]] == ["Shipped v2 CLI"]
    assert any("different company" in g for g in out["evidence_gaps"])


def test_signals_citing_unrelated_news_are_dropped():
    corpus = "https://news.google.com/rss/articles/abc"
    parsed = base_parsed(unrelated_sources=["news"], signals=[
        {"signal": "Hotel chain files for bankruptcy", "why_it_matters": "x", "source_url": "https://news.google.com/rss/articles/abc", "confidence": 0.9},
    ])
    out = gates(parsed, corpus=corpus)
    assert out["signals"] == []


def test_claims_need_traceable_independent_evidence():
    corpus = "https://acme.dev/pricing https://github.com/acme/core"
    parsed = base_parsed(claims_checked=[
        {"claim": "Open source", "claim_url": "https://acme.dev", "verdict": "verified",
         "evidence": "Core repo pushed this week", "evidence_url": "https://github.com/acme/core"},
        {"claim": "Used by 10,000 teams", "claim_url": "https://acme.dev", "verdict": "verified",
         "evidence": "Homepage says so", "evidence_url": "https://acme.dev"},
        {"claim": "Available on iOS", "claim_url": "https://acme.dev", "verdict": "partly",
         "evidence": "TechCrunch says beta", "evidence_url": "https://techcrunch.com/made-up"},
        {"claim": "SOC 2", "claim_url": "https://acme.dev/security", "verdict": "unverified",
         "evidence": "A report would settle it", "evidence_url": "https://acme.dev/x"},
        {"claim": "Bogus", "claim_url": "", "verdict": "maybe", "evidence": "", "evidence_url": ""},
    ])
    out = gates(parsed, corpus=corpus)
    by = {c["claim"]: c for c in out["claims_checked"]}
    assert "Bogus" not in by
    assert by["Open source"]["verdict"] == "verified"
    assert by["Used by 10,000 teams"]["verdict"] == "unverified"  # only its own page
    assert "own page" in by["Used by 10,000 teams"]["evidence"]
    assert by["Available on iOS"]["verdict"] == "unverified"       # untraceable evidence
    assert by["SOC 2"]["evidence_url"] == ""
    assert by["SOC 2"]["claim_url"] == "https://acme.dev"          # unfetched page falls back to homepage
    CandidateIntelligence(**out)


def test_zero_job_listings_show_as_not_found():
    zero = src(SourceType.JOBS, "https://acme.dev/careers", "ZERO job listings found across all checked ATS platforms.")
    real = src(SourceType.JOBS, "https://jobs.ashbyhq.com/acme", "Source: Ashby job board\nTotal open positions: 4")
    assert sources_checked_map(None, None, zero, None, None)["jobs"] is False
    assert sources_checked_map(None, None, real, None, None)["jobs"] is True
