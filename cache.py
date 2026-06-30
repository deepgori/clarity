"""
Simple SQLite cache for Clarity API responses.

Caches intelligence results by (target_domain, seller_domain) pair.
Default TTL is 6 hours since company data does not change frequently.
"""

import os
import json
import time
import sqlite3
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data"
CACHE_DB = CACHE_DIR / "cache.db"
DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", 6 * 60 * 60))  # 6 hours


def _get_db() -> sqlite3.Connection:
    """Get a database connection, creating the table if needed."""
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_key(domain: str, seller_domain: str | None, context: str | None) -> str:
    """Create a cache key from the request parameters."""
    # Context changes the analysis, so it is part of the key
    raw = f"{domain}|{seller_domain or ''}|{context or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def get_cached(domain: str, seller_domain: str | None = None, context: str | None = None) -> dict | None:
    """Look up a cached response. Returns None on miss or expiry."""
    try:
        conn = _get_db()
        key = _make_key(domain, seller_domain, context)
        row = conn.execute(
            "SELECT response_json, created_at FROM cache WHERE cache_key = ?",
            (key,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        response_json, created_at = row
        age = time.time() - created_at

        if age > DEFAULT_TTL:
            logger.info(f"Cache expired for {domain} (age: {age:.0f}s)")
            return None

        logger.info(f"Cache hit for {domain} (age: {age:.0f}s)")
        return json.loads(response_json)

    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def set_cached(domain: str, response: dict, seller_domain: str | None = None, context: str | None = None):
    """Store a response in the cache."""
    try:
        conn = _get_db()
        key = _make_key(domain, seller_domain, context)
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, response_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(response), time.time())
        )
        conn.commit()
        conn.close()
        logger.info(f"Cached response for {domain}")

    except Exception as e:
        logger.warning(f"Cache write error: {e}")


def clear_cache():
    """Clear all cached entries."""
    try:
        conn = _get_db()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        logger.info("Cache cleared")
    except Exception as e:
        logger.warning(f"Cache clear error: {e}")
