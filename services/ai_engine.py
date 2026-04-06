"""
ai_engine.py
~~~~~~~~~~~~
Unified Gemini AI wrapper for HydroVision.

Provides a single entry point — ``generate(prompt)`` — that handles:
  - API key lookup from the environment.
  - Disk-based response caching (via analysis_service._gemini_generate).
  - Automatic model rotation on 429 RESOURCE_EXHAUSTED quota errors.
  - Circuit breaker: after ``failure_threshold`` consecutive failures the
    breaker opens and all calls return ``None`` immediately for ``cooldown``
    seconds, protecting quota and latency. It resets automatically.

All services should call ``generate()`` instead of instantiating
``google.genai.Client`` directly or importing ``_gemini_generate``.

Usage::

    from .ai_engine import generate, is_available

    text = generate(prompt)          # returns str | None — never raises
    if text:
        result['analysis'] = markdown.markdown(text)
    else:
        result['analysis'] = _local_fallback(data)
"""
from __future__ import annotations

import os
import time
import threading
from typing import Optional

# Re-use cache + model rotation already in analysis_service.
from .analysis_service import _gemini_generate


# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Three-state circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

    CLOSED    Normal operation; all calls go through.
    OPEN      Too many recent failures; calls are blocked for ``cooldown`` s.
    HALF_OPEN Cooldown elapsed; next call is a trial.
              Success  → CLOSED (reset failure count).
              Failure  → OPEN   (restart cooldown).
    """

    CLOSED    = 'closed'
    OPEN      = 'open'
    HALF_OPEN = 'half_open'

    def __init__(self, failure_threshold: int = 5, cooldown: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._failures = 0
        self._state = self.CLOSED
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - (self._opened_at or 0) >= self.cooldown:
                    self._state = self.HALF_OPEN
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Manually reset to CLOSED (useful for testing or admin endpoints)."""
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED
            self._opened_at = None

    def __repr__(self) -> str:
        return (
            f'CircuitBreaker(state={self._state!r}, '
            f'failures={self._failures}/{self.failure_threshold})'
        )


# Module-level singleton — shared across all services in the process.
_breaker = CircuitBreaker(failure_threshold=5, cooldown=60.0)


# ── Public API ────────────────────────────────────────────────────────────────

def generate(prompt: str) -> Optional[str]:
    """
    Generate AI text for the given prompt.

    Returns the generated text string (stripped), or ``None`` when:
      - ``GEMINI_API_KEY`` is not set in the environment.
      - The circuit breaker is open (too many recent consecutive failures).
      - Gemini raises any exception.

    This function **never raises**. Callers must always provide a local
    fallback for the ``None`` case.
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return None

    if _breaker.is_open:
        return None

    try:
        text = _gemini_generate(api_key, prompt)
        _breaker.record_success()
        return text or None
    except Exception:
        _breaker.record_failure()
        return None


def is_available() -> bool:
    """
    Return True if Gemini is configured and the circuit breaker is closed.

    Services can use this to decide whether to even build a prompt, avoiding
    the overhead of prompt construction when AI is known to be unavailable.
    """
    return bool(os.getenv('GEMINI_API_KEY')) and not _breaker.is_open


def breaker_state() -> str:
    """Return the current circuit breaker state string (for health endpoints)."""
    return _breaker.state
