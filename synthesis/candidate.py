"""
Clarity API - Candidate Mode Synthesis

Same sources as sales mode, read for a job seeker instead of a sales agent.
A candidate looking at an early-stage company usually has a landing page, a
LinkedIn and a founder call coming up. This turns the same source data into:
what the company does, signals that matter to a candidate, contradictions
reframed as questions to ask, explicit evidence gaps, and a short intro note.

Honesty rules are enforced in code after the model runs, not only in the prompt:
- hiring claims are stripped when no hiring data was found
- the sales-mode enterprise ATS filter is reused
- signals whose source URL does not appear in the fetched data are dropped
- confidence is capped when most sources came back empty
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from openai import AsyncOpenAI

from models.schemas import SourceResult, CandidateIntelligence
from synthesis.engine import (
    format_source_sections,
    _filter_enterprise_ats_contradictions,
    _fix_enterprise_hiring_inference,
    _usable,
)
from costs import cost_tracker

logger = logging.getLogger(__name__)

HIRING_UNAVAILABLE = (
    "Hiring data unavailable: no roles found on the careers page or on Greenhouse, Lever or Ashby"
)

CANDIDATE_SYSTEM_PROMPT = """You are Clarity, a research engine for job seekers.

The reader is a candidate deciding whether to apply to this company, or preparing for a
call with its founder. They want what a sharp friend who spent an hour researching would
tell them: what the company really does, whether it is moving, and what to ask.

ABSOLUTE RULE - NO DATA MEANS NO INFERENCE:
If hiring data is marked unavailable ("No structured careers data available" AND "No
external job board data found"), job postings do not exist for your reasoning. No field
may claim the company is or is not hiring, or reason from the absence of postings. Put
nothing about hiring in signals or contradictions. You may still ask the founder what
roles they are hiring for.

FIELDS:

what_they_do: one plain sentence. No marketing language.

who_they_sell_to: their customers as the sources describe them (e.g. "mid-market finance
teams", "developers at AI startups"). If unclear, say "Not clear from the sources".

stage: under 10 words, ONLY from what a source states (funding round, "Series A",
"YC W25", public listing, amount raised, a reported valuation), with the date if given.
Examples: "Seed, $4M raised in March 2026", "Backed by a16z speedrun, round not
disclosed", "Not stated". Never guess.

bottom_line: two short sentences for a candidate deciding whether to take the call.
  Sentence 1: what is solidly backed by evidence. Sentence 2: the biggest unknown or risk.
  Concrete, no hype. Example: "Shipping weekly and hiring two founding engineers at listed
  pay. Funding and team size aren't public, so ask about runway first."

claims_checked: 3 to 5 of the company's OWN claims about itself (from its website, blog or
  pricing page), each checked against the other sources. Pick claims a candidate would care
  about: traction ("used by 10,000 teams"), activity ("open source", "shipping fast"),
  product reach ("available on iOS"), maturity ("enterprise-ready"), hiring ("we're growing
  the team").
  claim: what they say, briefly, close to their wording.
  claim_url: the company page where they say it (copy the exact URL from the data).
  verdict: one of
    "verified"     - independent evidence (GitHub, job boards, news, another page) backs it
    "partly"       - some evidence backs it, but part of the claim doesn't hold or is narrower
    "contradicted" - evidence points the other way
    "unverified"   - nothing in the other sources can confirm or deny it
  evidence: one sentence with the concrete detail that decided the verdict. For
    "unverified", say what would settle it.
  evidence_url: exact URL of that evidence, or "" for "unverified".
  A claim checked only against the same page that makes it is "unverified". Be fair: use
  "contradicted" only when the evidence clearly points the other way.
  CHOOSING CLAIMS:
  - Prefer claims the other sources can actually test (activity, platforms, open source,
    hiring, numbers that also appear in GitHub, news or job data).
  - Use "unverified" for at most 2 claims, and only for claims a candidate would worry
    about (traction numbers, compliance or security badges, "enterprise-ready").
  - Skip investor or accelerator badges ("Backed by X"); they belong in stage. Skip
    trivial lines (setup takes "one line of code", a count of open roles, nav labels).
  - Skip pricing and plan terms, and the mere existence of a link, signup or docs page:
    the company controls those, so checking them tells a candidate nothing.

READ THE SOURCES SKEPTICALLY:
  - NAME COLLISIONS: search-based sources (Hacker News, reviews, news) often return
    results about something else with the same name (a common word, another company,
    an unrelated product). Ignore those results completely, and put that source's key
    in unrelated_sources. Keys: "news", "github", "jobs", "community", "reviews".
    Only flag a source when its results are mostly or entirely about something else.
  - JOB BOARDS ARE MATCHED BY NAME, so the listings can belong to a different company
    with a similar name. Compare them with what the website says. If the roles, teams,
    locations or scale clearly don't fit (property managers at an AI developer tool,
    11 roles across five countries at a company that describes two founders), put
    "jobs" in unrelated_sources and say nothing about those roles anywhere.
  - The same goes for GitHub: if the org's repos have nothing to do with the product,
    put "github" in unrelated_sources.
  - "Notable patterns" lines in the job data are automated guesses. Repeat one only if
    the listed roles support it. Example: roles listed by region ("Europe", "North
    America") rather than by city suggest remote hiring, whatever the guess says.
  - The company's own careers page often shows nothing while Ashby/Greenhouse/Lever
    list real roles. Trust the external job board.

SPECIFICITY RULE (the most important rule for signals and questions):
  Every signal and every question must contain a concrete detail from the data: a
  number, a date, a role title, a repo name, a product name, a headline, a location.
  Test: if you could paste it into a report about a different company and it would
  still read fine, it is too vague. Rewrite it or drop it.
  BAD: "The company has active GitHub repositories with recent commits."
  GOOD: "5 of 7 checked public repos had commits in the last 30 days, mostly SDKs
        and a VS Code extension."
  BAD: "The company is hiring across multiple departments."
  GOOD: "18 open roles on Greenhouse: 9 engineering, 3 in a new 'Payments' team,
        all in New York or Toronto."
  BAD: "How will you use the new funding?"
  GOOD: "A September TechCrunch headline says you're moving upmarket. Which of the
        open engineering roles is closest to that work?"

signals: things that imply CURRENT MOTION and matter to a candidate. For each, ask "when
did this become true?" If the answer is "always", it is a fact, not a signal. Skip it.
Customer counts, famous customer logos and taglines are static facts unless a source
says they are new.
Look hardest at: what leadership says the company is becoming (news headlines, blog,
changelog), the exact shape of hiring (counts, titles, seniority, locations, new teams),
and what the active repos say about what is being built right now.
  Good candidate signals:
  - shipping velocity: repos marked ACTIVE, recent releases, changelog entries
  - hiring shape: which teams, seniority, locations, remote policy (only from real data)
  - money and momentum: a funding round, a launch, a big customer, in the last ~12 months
  - risk: layoffs, leadership exits, pivots, outages, in recent news or discussions
  - what users say: review ratings and recurring praise or complaints
  why_it_matters: one sentence on what this means for someone joining (scope, pace,
  stability, what the work would be). Not a sales angle.
  source_url: copy the exact URL from the source data where you saw it.
  Aim for 4 to 7 signals. Fewer is fine when the data is thin.

contradictions: tensions where evidence ACTIVELY UNDERMINES a claim, not "limited evidence".
  If the tension is about one of the company's own claims, put it in claims_checked
  (as "partly" or "contradicted") and NOT here, so it isn't shown twice. Use this array
  only for tensions between two outside sources, which is rare.
  Same standard as a careful analyst:
  - A company's own sparse careers page proves nothing. Never use it.
  - Low GitHub activity at a closed-source company is expected, not a contradiction.
  - GITHUB: use the ORG ACTIVITY SUMMARY. If ANY repo is ACTIVE, the org is active. Never
    generalize from one abandoned repo.
  - Public or large companies use enterprise ATS platforms. Zero Greenhouse/Lever/Ashby
    listings means nothing for them.
  - Old, publicly acknowledged history (a past pivot or layoff) is context, not a
    contradiction of present claims.
  resolution: the most likely innocent explanation. Be fair to the company.
  question_to_ask: a neutral, curious question the candidate could ask the founder about
  it. Never accusatory. Example: "The site leads with the open-source SDK, but the main
  repo has been quiet since spring. Is most of the work happening in a private repo now?"
  An empty array is the right answer most of the time.

questions_for_founder: 3 to 5 questions a strong candidate would ask on a first call.
  Each must come from a specific finding (based_on) with its source_url when there is one,
  and must name the detail it is about. Ask about scope, priorities, what is being built
  next, how the team works. A good question cannot be answered with a press-release line.
  At least one question must target the most important evidence gap (often runway, team
  size or pay), phrased so the founder can answer it in a sentence. Include questions for
  any "partly" or "contradicted" claims.
  BAD: "What is the culture like?" (generic, no finding)
  BAD: "What challenges do you face keeping up the pace?" (works for any company)
  GOOD: "Three of the five open roles are in solutions engineering. Is the next year
  more about enterprise deployments than new product?" (specific, grounded)

evidence_gaps: what a candidate would want to know that the sources did NOT answer, such
  as funding or runway, team size, compensation, remote policy, who the customers are,
  product traction. List only real gaps: never list something your own output already
  answers (if news reports a valuation, funding is not a gap; if roles are listed by
  location, location is not a gap). Do not list missing data sources; those are added
  separately. Write each gap as a short phrase, e.g. "Funding and runway", "Team size and
  founders' backgrounds", "Pay for the open roles". Never write "in the provided sources".

unrelated_sources: keys of sources whose results were about something else (see NAME
  COLLISIONS). Usually empty.

tech_stack: only languages, frameworks and infrastructure that appear in the source data.
  Not product features (SSO, audit logs) and not vendors' export targets.
hiring_signals: short factual lines from careers or job board data, or empty.

SOURCE ATTRIBUTION (CRITICAL):
  Every specific claim must be traceable to the source data. Never invent product names,
  numbers, funding amounts, customers or people. If unsure, leave it out. If the data is
  thin, say so through evidence_gaps and a low overall_confidence. An honest "not enough
  evidence" is more useful than a confident guess.

CONFIDENCE: 0-1, based on how much fresh evidence the sources contained.

WRITING STYLE (all text fields):
  - Plain, specific, short. Sound like a thoughtful engineer, not a recruiter.
  - Never use em dashes.
  - Never use: "synergy", "leverage", "impressive", "remarkable", "exciting", "cutting-edge",
    "innovative", "passionate", "rockstar", "game-changing", "revolutionize".

Return a JSON object matching the schema exactly."""

CANDIDATE_USER_PROMPT = """Research this company for a job seeker.

DATA FETCHED ON: {fetch_date}
COMPANY DOMAIN: {domain}

=== SOURCE DATA ===

--- WEBSITE CONTENT ---
{website_content}

--- RECENT NEWS ---
{news_content}

--- GITHUB PRESENCE ---
{github_content}

--- STRUCTURED CAREERS DATA ---
{careers_content}

--- EXTERNAL JOB POSTINGS (from ATS platforms) ---
{jobs_content}

--- COMMUNITY DISCUSSIONS (Hacker News) ---
{community_content}

--- CUSTOMER REVIEWS (G2, Capterra, Gartner, Reddit) ---
{reviews_content}

=== END SOURCE DATA ===

For data_freshness write exactly "Data is current as of {fetch_date}"."""


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}


def _obj(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


CANDIDATE_SCHEMA = _obj({
    "company_name": _STR,
    "domain": _STR,
    "what_they_do": _STR,
    "who_they_sell_to": _STR,
    "industry": _STR,
    "stage": _STR,
    "signals": {"type": "array", "items": _obj({
        "signal": _STR,
        "why_it_matters": _STR,
        "source_url": _STR,
        "confidence": {"type": "number"},
    })},
    "contradictions": {"type": "array", "items": _obj({
        "claim_a": _STR,
        "source_a": _STR,
        "claim_b": _STR,
        "source_b": _STR,
        "resolution": _STR,
        "question_to_ask": _STR,
    })},
    "questions_for_founder": {"type": "array", "items": _obj({
        "question": _STR,
        "based_on": _STR,
        "source_url": _STR,
    })},
    "evidence_gaps": _STR_LIST,
    "bottom_line": _STR,
    "claims_checked": {"type": "array", "items": _obj({
        "claim": _STR,
        "claim_url": _STR,
        "verdict": {"type": "string", "enum": ["verified", "partly", "contradicted", "unverified"]},
        "evidence": _STR,
        "evidence_url": _STR,
    })},
    "unrelated_sources": {"type": "array", "items": {
        "type": "string", "enum": ["news", "github", "jobs", "community", "reviews"],
    }},
    "tech_stack": _STR_LIST,
    "hiring_signals": _STR_LIST,
    "sources_used": _STR_LIST,
    "overall_confidence": {"type": "number"},
    "data_freshness": _STR,
})


# --- Model selection --------------------------------------------------------------

# Fast enough for the live form; scripts/prerun_speedrun.py uses a larger model offline.
DEFAULT_CANDIDATE_MODEL = "gpt-5.4-mini"


def candidate_model() -> str:
    """Candidate mode's model. Sales mode is unaffected and stays on gpt-4o."""
    return os.getenv("CANDIDATE_MODEL", "").strip() or DEFAULT_CANDIDATE_MODEL


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def model_kwargs(model: str, temperature: float, max_tokens: int | None = None) -> dict:
    """Sampling params differ by model family.

    Reasoning models (gpt-5.x, o-series) reject temperature, count hidden reasoning
    toward the token limit, and take a reasoning effort instead.
    """
    if _is_reasoning_model(model):
        kw = {"reasoning_effort": os.getenv("CANDIDATE_REASONING_EFFORT", "low")}
        if max_tokens:
            kw["max_completion_tokens"] = max_tokens + 4000
        return kw
    kw = {"temperature": temperature}
    if max_tokens:
        kw["max_tokens"] = max_tokens
    return kw


# --- Deterministic post-processing -------------------------------------------

_JOB_DATA_KEYWORDS = [
    "job listing", "job posting", "job description", "job board", "open role",
    "open position", "openings", "ats", "greenhouse", "lever", "ashby",
    "hiring", "careers page", "no roles",
]

_BANNED = [
    (r"\bsynergies\b", "overlap"),
    (r"\bsynergy\b", "overlap"),
    (r"\bleveraging\b", "using"),
    (r"\bleveraged\b", "used"),
    (r"\bleverages\b", "uses"),
    (r"\bleverage\b", "use"),
    (r"\bimpressive\b", "notable"),
    (r"\bremarkable\b", "notable"),
    (r"\bcutting-edge\b", "new"),
    (r"\bgame-changing\b", "significant"),
    (r"\brevolutionize\b", "change"),
]

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


def _mentions_job_data(text: str) -> bool:
    t = text.lower()
    # Whole words (plurals allowed) so "leverage" isn't read as "lever"
    return any(re.search(rf"\b{re.escape(kw)}s?\b", t) for kw in _JOB_DATA_KEYWORDS)


def scrub_text(text: str) -> str:
    """Remove em dashes and phrases that read as filler."""
    text = re.sub(r"\s*—\s*", ", ", text)
    text = text.replace("–", "-")
    for pattern, replacement in _BANNED:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _scrub_all(value):
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [_scrub_all(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub_all(v) for k, v in value.items()}
    return value


def _norm_url(url: str) -> str:
    return url.strip().rstrip("/.,;").lower().replace("://www.", "://")


def _resolve_source_url(cited: str, known_urls: set[str], label_map: dict[str, str], domain: str) -> str | None:
    """Return a verifiable URL for a citation, or None if it can't be traced.

    - exact (normalized) match against URLs present in the fetched data: keep
    - a repo under the verified GitHub org (github.com/<org>/...): keep
    - a bare label like "GitHub" or "website": map to that source's URL
    - a path on the company's own domain we didn't fetch: fall back to the homepage
    - anything else (an invented article URL, a different site): untraceable
    """
    cited = (cited or "").strip()
    if not cited:
        return None
    if _norm_url(cited) in known_urls:
        return cited
    gh = label_map.get("github")
    if gh and _norm_url(cited).startswith(_norm_url(gh) + "/"):
        return cited

    label = cited.lower()
    for key, url in label_map.items():
        if key in label and not label.startswith("http"):
            return url

    host = urlparse(cited if "://" in cited else f"https://{cited}").hostname or ""
    if host == domain or host.endswith("." + domain):
        return label_map.get("website")
    return None


_COLLISION_TEXT = {
    "news": "News results were about something else with the same name, so they were ignored",
    "github": "The GitHub organization found looks unrelated, so it was ignored",
    "jobs": "Job board listings appear to belong to a different company with a similar name, so they were ignored",
    "community": "Hacker News results were about something else with the same name, so they were ignored",
    "reviews": "Review results were about something else with the same name, so they were ignored",
}

_VERDICTS = {"verified", "partly", "contradicted", "unverified"}

# Hosts that citations from each search-based source point at
_SOURCE_HOSTS = {
    "news": "news.google.com",
    "github": "github.com",
    "community": "news.ycombinator.com",
    "reviews": "google.com",
}


def apply_candidate_gates(
    parsed: dict,
    *,
    domain: str,
    corpus: str,
    source_urls: dict[str, str],
    careers_available: bool,
    jobs_fetched: bool,
    sources_checked: dict[str, bool],
) -> dict:
    """Code-level honesty gates. Runs after the model, so prompts can't override it."""
    # 0. Sources the model judged to be about a different company (name collisions)
    unrelated = [
        k for k in dict.fromkeys(parsed.get("unrelated_sources") or [])
        if k in _COLLISION_TEXT and sources_checked.get(k, False)
    ]
    parsed["unrelated_sources"] = unrelated
    if "jobs" in unrelated:
        jobs_fetched = False
    if "github" in unrelated:
        source_urls = {k: v for k, v in source_urls.items() if k != "github"}
    hiring_data_available = careers_available or jobs_fetched

    # Drop signals that cite an unrelated search source
    unrelated_hosts = {_SOURCE_HOSTS[k] for k in unrelated if k in _SOURCE_HOSTS}
    if unrelated_hosts:
        parsed["signals"] = [
            s for s in parsed.get("signals", [])
            if (urlparse(s.get("source_url", "")).hostname or "").removeprefix("www.") not in unrelated_hosts
        ]

    # 1. No hiring data: no hiring claims anywhere except a question about it
    if not hiring_data_available:
        parsed["hiring_signals"] = [HIRING_UNAVAILABLE]
        before = len(parsed.get("signals", []))
        parsed["signals"] = [
            s for s in parsed.get("signals", [])
            if not _mentions_job_data(f"{s.get('signal', '')} {s.get('why_it_matters', '')}")
        ]
        if len(parsed["signals"]) < before:
            logger.info(f"Candidate gate: dropped {before - len(parsed['signals'])} hiring signal(s) with no hiring data")
        parsed["questions_for_founder"] = [
            q for q in parsed.get("questions_for_founder", [])
            if not _mentions_job_data(q.get("based_on", ""))
        ]

    # 2. Sales-mode enterprise ATS gates. The contradiction filter also drops
    #    job-based contradictions whenever hiring_signals says "unavailable".
    _filter_enterprise_ats_contradictions(parsed)
    if not jobs_fetched:
        # Only overwrite hiring signals when the ATS data really was empty
        _fix_enterprise_hiring_inference(parsed)

    # 3. Source attribution: every signal must point to something we fetched
    known = {_norm_url(u) for u in _URL_RE.findall(corpus)}
    known |= {_norm_url(u) for u in source_urls.values() if u}
    label_map = {
        "hacker news": source_urls.get("community", ""),
        "community": source_urls.get("community", ""),
        "github": source_urls.get("github", ""),
        "news": source_urls.get("news", ""),
        "job": source_urls.get("jobs", ""),
        "careers": source_urls.get("website", ""),
        "review": source_urls.get("reviews", ""),
        "website": source_urls.get("website", ""),
    }
    label_map = {k: v for k, v in label_map.items() if v}

    kept = []
    for s in parsed.get("signals", []):
        url = _resolve_source_url(s.get("source_url", ""), known, label_map, domain)
        if url is None:
            logger.info(f"Candidate gate: dropped untraceable signal '{s.get('signal', '')[:60]}' ({s.get('source_url')})")
            continue
        s["source_url"] = url
        kept.append(s)
    parsed["signals"] = kept

    for q in parsed.get("questions_for_founder", []):
        url = _resolve_source_url(q.get("source_url", ""), known, label_map, domain)
        q["source_url"] = url or ""

    # Claims: a verdict other than "unverified" needs evidence we can trace.
    # Untraceable evidence downgrades the verdict instead of hiding the claim.
    checked_claims = []
    for c in parsed.get("claims_checked", []):
        if c.get("verdict") not in _VERDICTS:
            continue
        c["claim_url"] = _resolve_source_url(c.get("claim_url", ""), known, label_map, domain) or source_urls.get("website", "")
        if c["verdict"] == "unverified":
            c["evidence_url"] = ""
        else:
            ev = _resolve_source_url(c.get("evidence_url", ""), known, label_map, domain)
            ev_host = (urlparse(ev or "").hostname or "").removeprefix("www.")
            same_page = ev is not None and _norm_url(ev) == _norm_url(c["claim_url"] or "")
            if ev is None or ev_host in unrelated_hosts or (same_page and c["verdict"] == "verified"):
                logger.info(f"Candidate gate: downgraded claim '{c.get('claim', '')[:50]}' (untraceable evidence)")
                c["verdict"] = "unverified"
                c["evidence"] = (
                    "Only the company's own page says this, so it isn't independently confirmed."
                    if same_page else "The evidence for this couldn't be traced to a fetched source."
                )
                c["evidence_url"] = ""
            else:
                c["evidence_url"] = ev
        checked_claims.append(c)
    parsed["claims_checked"] = checked_claims

    # 4. Missing sources become explicit gaps
    gap_text = {
        "news": "No recent news coverage found",
        "github": "No public GitHub organization found, so shipping activity could not be checked",
        "jobs": "No listings found on Greenhouse, Lever or Ashby",
        "community": "No Hacker News discussions found",
        "reviews": "No customer reviews found on G2, Capterra, Gartner or Reddit",
    }
    source_gaps = [msg for key, msg in gap_text.items() if not sources_checked.get(key, False)]
    source_gaps += [_COLLISION_TEXT[k] for k in unrelated]
    model_gaps = [g for g in parsed.get("evidence_gaps", []) if g and g.strip()]
    parsed["evidence_gaps"] = source_gaps + model_gaps

    # 5. Thin evidence caps confidence (unrelated sources don't count as evidence)
    fetched_count = sum(1 for k, v in sources_checked.items() if v and k not in unrelated)
    try:
        conf = float(parsed.get("overall_confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    if fetched_count <= 2:
        conf = min(conf, 0.4)
    elif fetched_count <= 3:
        conf = min(conf, 0.6)
    parsed["overall_confidence"] = max(0.0, min(1.0, conf))
    for s in parsed["signals"]:
        try:
            s["confidence"] = max(0.0, min(1.0, float(s.get("confidence", 0))))
        except (TypeError, ValueError):
            s["confidence"] = 0.0

    # 6. Writing style, everywhere
    return _scrub_all(parsed)


# --- Synthesis ----------------------------------------------------------------

async def synthesize_candidate_intelligence(
    domain: str,
    website_result: SourceResult,
    news_result: SourceResult | None,
    github_result: SourceResult | None,
    jobs_result: SourceResult | None = None,
    community_result: SourceResult | None = None,
    reviews_result: SourceResult | None = None,
    careers_data: str | None = None,
) -> CandidateIntelligence:
    """Synthesize fetched sources into candidate-facing intelligence."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = candidate_model()

    sections = format_source_sections(
        website_result,
        news_result or _missing(),
        github_result or _missing(),
        jobs_result, community_result, reviews_result, careers_data,
    )
    fetch_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_prompt = CANDIDATE_USER_PROMPT.format(domain=domain, fetch_date=fetch_date, **sections)

    logger.info(f"Sending candidate synthesis request ({len(user_prompt)} chars)...")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "candidate_intelligence", "strict": True, "schema": CANDIDATE_SCHEMA},
        },
        **model_kwargs(model, temperature=0),
    )
    usage = response.usage
    if usage:
        cost_tracker.record(model, usage.prompt_tokens, usage.completion_tokens, caller="candidate_synthesis")

    parsed = json.loads(response.choices[0].message.content)

    careers_in_site = "=== CAREERS ===" in (website_result.content or "")

    parsed = apply_candidate_gates(
        parsed,
        domain=domain,
        corpus=user_prompt,
        source_urls=source_url_map(website_result, news_result, github_result, jobs_result, community_result, reviews_result),
        careers_available=bool(careers_data) or careers_in_site,
        jobs_fetched=_usable(jobs_result),
        sources_checked=sources_checked_map(news_result, github_result, jobs_result, community_result, reviews_result),
    )
    parsed["domain"] = domain

    logger.info(
        f"Candidate synthesis complete: {parsed.get('company_name')} | "
        f"signals={len(parsed['signals'])} contradictions={len(parsed.get('contradictions', []))} "
        f"gaps={len(parsed['evidence_gaps'])} confidence={parsed['overall_confidence']}"
    )
    return CandidateIntelligence(**parsed)


def _missing() -> SourceResult:
    from models.schemas import SourceType
    return SourceResult(source_type=SourceType.WEBSITE, url="", content="", fetched=False)


def source_url_map(website, news, github, jobs, community, reviews) -> dict[str, str]:
    pairs = {"website": website, "news": news, "github": github, "jobs": jobs,
             "community": community, "reviews": reviews}
    return {k: v.url for k, v in pairs.items() if _usable(v) and v.url}


def sources_checked_map(news, github, jobs, community, reviews) -> dict[str, bool]:
    return {
        "website": True,  # required; the pipeline stops earlier without it
        "news": _usable(news),
        "github": _usable(github),
        # "ZERO job listings" is reported as fetched (absence is data for sales
        # mode), but for the coverage chips it means nothing was found
        "jobs": _usable(jobs) and not jobs.content.lstrip().startswith("ZERO job listings"),
        "community": _usable(community),
        "reviews": _usable(reviews),
    }


# --- Intro note ------------------------------------------------------------------

INTRO_NOTE_PROMPT = """Write a short note from a job candidate to the founder of {company_name}.

What the research found about the company:
What they do: {what_they_do}
Who they sell to: {who_they_sell_to}
Signals:
{signals_text}
Open questions:
{questions_text}

About the candidate:
{candidate_text}

SHAPE (three short paragraphs, 60 to 110 words total):
1. Open with a specific finding and what it suggests, in the candidate's own words. Two
   findings at most. The first sentence must contain a concrete detail (a number, a
   name, a date, a role title). Never open with "I", with interest, or with praise.
2. One or two sentences in the first person that CONNECT one specific thing the candidate
   has done to one specific finding about the company ("I've built X, which is close to
   the Y work your repos show"). Do not restate the bio as a list. Use ONLY facts given
   about the candidate. Never invent experience, employers, projects or skills, and
   never claim the candidate has used the company's product.
{placeholder_rule}
3. One real question taken from the open questions, then ask for a 20-minute call. The
   question must add something new, not restate paragraph 1.

EXAMPLE OF THE TONE (different company, do not copy its content):
"Four of your six open engineering roles mention the new on-prem deployment, and the
installer repo has had commits every week since August. That reads like the next year
is about getting into regulated customers.

[one line on what you have built]

Is the on-prem work its own team, or does every engineer rotate through it? Happy to do
a 20-minute call if it's useful."

RULES:
- No subject line. No greeting by name (the founder's name is unknown). No sign-off name.
- Mention only findings listed above, accurately.
- Never use: "I am interested", "I'm interested", "aligns with", "I'd love", "passionate",
  "excited", "thrilled", "impressive", "remarkable", "I came across", "I noticed",
  "I hope this finds you well", "leverage", "synergy", "reach out".
- No em dashes. Plain sentences. Sound like an engineer writing to another engineer."""


async def generate_intro_note(intel: CandidateIntelligence, candidate_context: str | None) -> str:
    """Draft a short note to the founder that cites specific findings."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    signals_text = "\n".join(f"- {s.signal}" for s in intel.signals[:6]) or "- None found"
    questions_text = "\n".join(f"- {q.question}" for q in intel.questions_for_founder[:4]) or "- None"
    if candidate_context:
        candidate_text = candidate_context
        placeholder_rule = "   Do not use placeholders."
    else:
        candidate_text = "Nothing provided."
        placeholder_rule = (
            "   Nothing is known about the candidate, so paragraph 2 is exactly this placeholder on "
            "its own line: [one line on what you have built]. No other placeholders anywhere."
        )

    model = candidate_model()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": INTRO_NOTE_PROMPT.format(
            company_name=intel.company_name,
            what_they_do=intel.what_they_do,
            who_they_sell_to=intel.who_they_sell_to,
            signals_text=signals_text,
            questions_text=questions_text,
            candidate_text=candidate_text,
            placeholder_rule=placeholder_rule,
        )}],
        **model_kwargs(model, temperature=0.4, max_tokens=300),
    )
    usage = response.usage
    if usage:
        cost_tracker.record(model, usage.prompt_tokens, usage.completion_tokens, caller="intro_note")

    note = response.choices[0].message.content.strip()
    for phrase in [r"I hope this finds you well\.?\s*", r"\bI came across\b\s*", r"\bI noticed\b\s*"]:
        note = re.sub(phrase, "", note, flags=re.IGNORECASE)
    note = re.sub(r"\b(passionate|thrilled)\b", "interested", note, flags=re.IGNORECASE)
    return scrub_text(note).strip()
