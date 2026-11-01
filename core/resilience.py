"""
RegLingua — Agent-call resilience (Day 10, M4)
==================================================
"Add retries, timeouts, fallbacks and structured output validation."
Architecture doc Section 15: every agent call should have a timeout, retry
limit, structured-output validation, confidence/status, fallback and trace
event:

    Agent fails -> retry once -> still fails? -> deterministic fallback
                                              -> still fails? NEEDS_REVIEW

`call_with_resilience` implements exactly that ladder, independent of which
agent is being called, so the Supervisor never needs agent-specific
try/except blocks and no agent (deterministic or AI-backed) can take a case
down.

Notes
  - Timeouts run the call on a worker thread and stop *waiting* for it.
    Python cannot kill a running thread, so a call that is genuinely stuck
    keeps running in the background until it returns; for this project's
    CPU-bound, in-process agents that is acceptable, and the result is
    simply discarded.
  - `validate` implements structured-output validation: it receives the
    agent's return value and must raise (any exception) if the value is not
    usable. A value that fails validation is treated exactly like an
    exception.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Optional


class StructuredOutputError(ValueError):
    """Raised by validators when an agent's output breaks its contract."""


@dataclass
class CallOutcome:
    value: Any = None
    status: str = "success"          # success | retried | fallback | failed
    attempts: int = 0
    used_fallback: bool = False
    error: Optional[str] = None      # last error seen, even if later recovered
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def _run_once(fn: Callable[[], Any], timeout_s: float, validate: Optional[Callable[[Any], None]]) -> Any:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        try:
            value = future.result(timeout=timeout_s)
        except FutureTimeout as exc:
            raise TimeoutError(f"agent call exceeded {timeout_s:.2f}s") from exc
    finally:
        pool.shutdown(wait=False)
    if validate is not None:
        validate(value)
    return value


def call_with_resilience(
    primary: Callable[[], Any],
    *,
    fallback: Optional[Callable[[], Any]] = None,
    validate: Optional[Callable[[Any], None]] = None,
    timeout_s: float = 5.0,
    max_retries: int = 1,
) -> CallOutcome:
    started = time.perf_counter()
    outcome = CallOutcome()

    for attempt in range(1, max_retries + 2):
        outcome.attempts = attempt
        try:
            outcome.value = _run_once(primary, timeout_s, validate)
            outcome.status = "success" if attempt == 1 else "retried"
            outcome.latency_ms = (time.perf_counter() - started) * 1000
            return outcome
        except Exception as exc:  # noqa: BLE001 - every failure mode is handled identically
            outcome.error = f"{type(exc).__name__}: {exc}"

    if fallback is not None:
        try:
            outcome.value = _run_once(fallback, timeout_s, validate)
            outcome.status = "fallback"
            outcome.used_fallback = True
            outcome.latency_ms = (time.perf_counter() - started) * 1000
            return outcome
        except Exception as exc:  # noqa: BLE001
            outcome.error = f"{outcome.error} | fallback failed: {type(exc).__name__}: {exc}"

    outcome.status = "failed"
    outcome.value = None
    outcome.latency_ms = (time.perf_counter() - started) * 1000
    return outcome
