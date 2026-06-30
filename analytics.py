"""
Usage analytics and feedback tracking for Clarity API.

Stores every API request and user feedback in SQLite.
Provides summary stats for the pitch dashboard.
"""

import sqlite3
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ANALYTICS_DB = DATA_DIR / "analytics.db"


def _get_db() -> sqlite3.Connection:
    """Get a database connection with WAL mode."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(ANALYTICS_DB), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            domain TEXT NOT NULL,
            seller_domain TEXT,
            success INTEGER NOT NULL,
            processing_time_ms INTEGER,
            client_ip TEXT,
            has_email INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            domain TEXT NOT NULL,
            rating TEXT NOT NULL,
            client_ip TEXT
        );
    """)
    conn.commit()
    return conn


def log_request(
    domain: str,
    success: bool,
    processing_time_ms: int,
    client_ip: str = "",
    seller_domain: str | None = None,
    has_email: bool = False,
):
    """Log an API request."""
    try:
        conn = _get_db()
        conn.execute(
            """INSERT INTO requests
               (timestamp, domain, seller_domain, success, processing_time_ms, client_ip, has_email)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), domain, seller_domain, int(success), processing_time_ms, client_ip, int(has_email)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to log request: {e}")


def log_feedback(domain: str, rating: str, client_ip: str = ""):
    """Log a thumbs up/down feedback."""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO feedback (timestamp, domain, rating, client_ip) VALUES (?, ?, ?, ?)",
            (time.time(), domain, rating, client_ip),
        )
        conn.commit()
        conn.close()
        logger.info(f"Feedback recorded: {rating} for {domain}")
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")


def get_analytics_summary() -> dict:
    """Get a summary of usage and feedback stats."""
    try:
        conn = _get_db()

        # Total requests
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        successful = conn.execute("SELECT COUNT(*) FROM requests WHERE success = 1").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM requests WHERE success = 0").fetchone()[0]

        # Unique domains analyzed
        unique_domains = conn.execute(
            "SELECT COUNT(DISTINCT domain) FROM requests WHERE success = 1"
        ).fetchone()[0]

        # Unique visitors (by IP)
        unique_visitors = conn.execute(
            "SELECT COUNT(DISTINCT client_ip) FROM requests WHERE client_ip != ''"
        ).fetchone()[0]

        # Average processing time
        avg_time = conn.execute(
            "SELECT AVG(processing_time_ms) FROM requests WHERE success = 1"
        ).fetchone()[0] or 0

        # Top queried domains
        top_domains = conn.execute(
            """SELECT domain, COUNT(*) as cnt
               FROM requests WHERE success = 1
               GROUP BY domain ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()

        # Feedback stats
        total_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        thumbs_up = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating = 'up'"
        ).fetchone()[0]
        thumbs_down = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating = 'down'"
        ).fetchone()[0]

        # Requests in last 24h
        day_ago = time.time() - 86400
        last_24h = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE timestamp > ?", (day_ago,)
        ).fetchone()[0]

        conn.close()

        approval_rate = round(thumbs_up / max(total_feedback, 1) * 100, 1)

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": round(successful / max(total, 1) * 100, 1),
            "unique_domains_analyzed": unique_domains,
            "unique_visitors": unique_visitors,
            "avg_processing_time_ms": round(avg_time),
            "requests_last_24h": last_24h,
            "top_domains": [{"domain": r["domain"], "count": r["cnt"]} for r in top_domains],
            "feedback": {
                "total": total_feedback,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "approval_rate": approval_rate,
            },
        }
    except Exception as e:
        logger.warning(f"Failed to get analytics: {e}")
        return {"error": str(e)}
