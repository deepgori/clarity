"""
Pre-run candidate mode on a few companies and save the results as static JSON
for the /speedrun page, so reviewers see real output instantly at no cost per view.

Usage (from the repo root):
    python scripts/prerun_speedrun.py company1.com company2.ai ...

Writes static/speedrun/examples/<domain>.json and rebuilds index.json from
every example file present. Failed runs are reported and not written, so the
page never shows an error as an example.
"""

import sys
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

# Examples are generated once and viewed many times, so use the stronger model
# and give it time. Override with CANDIDATE_MODEL=... if needed.
os.environ.setdefault("CANDIDATE_MODEL", "gpt-5.5")
os.environ.setdefault("CANDIDATE_SYNTH_TIMEOUT", "120")

import main  # noqa: E402
from security import validate_domain  # noqa: E402

OUT_DIR = ROOT / "static" / "speedrun" / "examples"


def rebuild_index() -> list[dict]:
    """List every example file. Keeps the existing index order (the first one
    loads by default, so put the strongest first) and appends new files."""
    index_path = OUT_DIR / "index.json"
    previous = []
    if index_path.exists():
        previous = [e["file"] for e in json.loads(index_path.read_text())]
    files = {f.name: f for f in OUT_DIR.glob("*.json") if f.name != "index.json"}
    ordered = [files[n] for n in previous if n in files] + sorted(f for n, f in files.items() if n not in previous)

    entries = []
    for f in ordered:
        data = json.loads(f.read_text())
        intel = data.get("intelligence") or {}
        entries.append({"domain": intel.get("domain", f.stem), "company_name": intel.get("company_name", f.stem), "file": f.name})
    (OUT_DIR / "index.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


async def run(domains: list[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for raw in domains:
        domain = main.normalize_domain(raw)
        err = validate_domain(domain)
        if err:
            print(f"SKIP {raw}: {err}")
            failures += 1
            continue
        print(f"Running {domain} ...", flush=True)
        try:
            resp = await main.run_candidate_pipeline(domain, None)
        except Exception as e:
            print(f"FAIL {domain}: {e}")
            failures += 1
            continue
        if not resp.success:
            print(f"FAIL {domain}: {resp.error}")
            failures += 1
            continue
        path = OUT_DIR / f"{domain}.json"
        path.write_text(resp.model_dump_json(indent=2) + "\n")
        i = resp.intelligence
        print(f"  ok in {resp.processing_time_ms / 1000:.1f}s: {len(i.signals)} signals, "
              f"{len(i.contradictions)} contradictions, {len(i.evidence_gaps)} gaps, "
              f"confidence {i.overall_confidence:.2f} -> {path.relative_to(ROOT)}")
    entries = rebuild_index()
    print(f"index.json lists {len(entries)} example(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(asyncio.run(run(sys.argv[1:])))
