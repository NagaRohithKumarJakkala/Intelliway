"""Per-upstream circuit breakers with a short rolling health score."""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field

from .telemetry import CIRCUIT_STATE, UPSTREAM_HEALTH, observe_upstream


class CircuitOpenError(Exception):
    def __init__(self, name: str, retry_after: int):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit for {name} is open; retry in {retry_after}s")


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = field(default_factory=lambda: int(os.getenv("GATEWAY_CIRCUIT_FAILURES", "3")))
    reset_seconds: int = field(default_factory=lambda: int(os.getenv("GATEWAY_CIRCUIT_RESET_SECONDS", "30")))
    outcomes: deque[tuple[float, bool]] = field(default_factory=deque)
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def before_call(self) -> None:
        async with self.lock:
            if self.opened_at is None:
                return
            elapsed = time.monotonic() - self.opened_at
            if elapsed < self.reset_seconds:
                raise CircuitOpenError(self.name, max(1, int(self.reset_seconds - elapsed)))
            if self.probe_in_flight:
                raise CircuitOpenError(self.name, 1)
            self.probe_in_flight = True

    async def success(self) -> None:
        await self._record(True)

    async def failure(self) -> None:
        await self._record(False)

    async def _record(self, succeeded: bool) -> None:
        now = time.monotonic()
        async with self.lock:
            self.outcomes.append((now, succeeded))
            while self.outcomes and self.outcomes[0][0] < now - 60:
                self.outcomes.popleft()
            self.probe_in_flight = False
            if succeeded:
                self.failures = 0
                self.opened_at = None
                observe_upstream(self.name, "success")
            else:
                self.failures += 1
                observe_upstream(self.name, "failure")
                if self.failures >= self.failure_threshold or self.opened_at is not None:
                    self.opened_at = now
            score = 100.0 * sum(ok for _, ok in self.outcomes) / len(self.outcomes) if self.outcomes else 100.0
            CIRCUIT_STATE.labels(upstream=self.name).set(1 if self.opened_at is not None else 0)
            UPSTREAM_HEALTH.labels(upstream=self.name).set(score)

    async def snapshot(self) -> dict:
        async with self.lock:
            now = time.monotonic()
            while self.outcomes and self.outcomes[0][0] < now - 60:
                self.outcomes.popleft()
            score = 100.0 * sum(ok for _, ok in self.outcomes) / len(self.outcomes) if self.outcomes else 100.0
            remaining = max(0, int(self.reset_seconds - (now - self.opened_at))) if self.opened_at else 0
            return {
                "state": "open" if self.opened_at and remaining else "half_open" if self.opened_at else "closed",
                "health_score": round(score, 1),
                "failures": self.failures,
                "retry_after_seconds": remaining,
            }


BREAKERS: dict[str, CircuitBreaker] = {}


def breaker_for(name: str) -> CircuitBreaker:
    return BREAKERS.setdefault(name, CircuitBreaker(name))
