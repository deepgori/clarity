"""
Rate limiter and input validation for Clarity API.
"""

import re
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Rate limiting: max requests per IP per window
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 5  # max requests per window

# Domain validation
MAX_DOMAIN_LENGTH = 100
DOMAIN_PATTERN = re.compile(
    r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?'
    r'(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*'
    r'\.[a-zA-Z]{2,}$'
)

# Blocked TLDs and patterns (internal/private)
BLOCKED_PATTERNS = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "192.168.",
    "169.254.",
    ".internal",
    ".local",
    ".localhost",
]


class RateLimiter:
    """Simple in-memory per-IP rate limiter."""

    def __init__(self, window: int = RATE_LIMIT_WINDOW, max_requests: int = RATE_LIMIT_MAX):
        self.window = window
        self.max_requests = max_requests
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from this IP is allowed."""
        now = time.time()
        cutoff = now - self.window

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > cutoff
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def remaining(self, client_ip: str) -> int:
        """How many requests remain for this IP in the current window."""
        now = time.time()
        cutoff = now - self.window
        recent = [t for t in self._requests[client_ip] if t > cutoff]
        return max(0, self.max_requests - len(recent))


def validate_domain(domain: str) -> str | None:
    """
    Validate a normalized domain. Returns an error message if invalid, None if ok.
    """
    if not domain:
        return "Domain cannot be empty"

    if len(domain) > MAX_DOMAIN_LENGTH:
        return f"Domain too long (max {MAX_DOMAIN_LENGTH} characters)"

    # Check against blocked patterns (SSRF protection)
    for pattern in BLOCKED_PATTERNS:
        if pattern in domain:
            return f"Domain '{domain}' is not allowed"

    # Basic format check
    if not DOMAIN_PATTERN.match(domain):
        return f"Invalid domain format: '{domain}'"

    return None


# Global rate limiter instance
rate_limiter = RateLimiter()
