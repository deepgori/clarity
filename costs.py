"""
OpenAI cost tracking for Clarity API.

Tracks token usage and estimated cost per request and cumulatively.
Logs warnings when spend thresholds are crossed.
"""

import time
import logging
import threading

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of mid-2025, update if pricing changes)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Spend alert thresholds in USD
ALERT_THRESHOLDS = [1.0, 5.0, 10.0, 25.0, 50.0]


class CostTracker:
    """Thread-safe cumulative OpenAI cost tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
        self._cost_by_model: dict[str, float] = {}
        self._start_time = time.time()
        self._alerts_fired: set[float] = set()

    def record(self, model: str, input_tokens: int, output_tokens: int, caller: str = ""):
        """Record token usage from an OpenAI response and log the cost."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        request_cost = input_cost + output_cost

        with self._lock:
            self._total_cost += request_cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_requests += 1
            self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + request_cost
            cumulative = self._total_cost

        label = f"[{caller}] " if caller else ""
        logger.info(
            f"{label}OpenAI cost: ${request_cost:.4f} "
            f"({input_tokens} in / {output_tokens} out) "
            f"| Cumulative: ${cumulative:.4f} ({self._total_requests} calls)"
        )

        # Check alert thresholds
        for threshold in ALERT_THRESHOLDS:
            if cumulative >= threshold and threshold not in self._alerts_fired:
                self._alerts_fired.add(threshold)
                logger.warning(
                    f"SPEND ALERT: Cumulative OpenAI cost has crossed ${threshold:.2f} "
                    f"(now ${cumulative:.4f} across {self._total_requests} API calls)"
                )

        return request_cost

    def get_summary(self) -> dict:
        """Get a snapshot of current spend stats."""
        uptime = time.time() - self._start_time
        hours = uptime / 3600

        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 4),
                "total_requests": self._total_requests,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "cost_by_model": {k: round(v, 4) for k, v in self._cost_by_model.items()},
                "uptime_hours": round(hours, 2),
                "avg_cost_per_request": round(
                    self._total_cost / max(self._total_requests, 1), 4
                ),
                "projected_daily_cost": round(
                    (self._total_cost / max(hours, 0.01)) * 24, 2
                ),
            }


# Global instance
cost_tracker = CostTracker()
