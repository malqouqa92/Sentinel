"""Phase 15b -- write-origin provenance via ``contextvars``.

Every persistent write to the KB or memory tables now carries a
``created_by_origin`` column saying WHO triggered it. The writers
read the value from a process-wide ``ContextVar`` so we never had
to thread an extra argument through the existing call-site graph.

The pattern is borrowed from Hermes' write-attribution work and is
asyncio-safe: each task gets its own copy of the var when it
forks, so a background extraction running concurrently with a
foreground /code never cross-contaminates.

Valid values (callers should use the constants, not raw strings):
  FOREGROUND              -- user-driven request (default)
  BACKGROUND              -- internal sweep / scheduled job /
                             post-pipeline hook (e.g. adaptive
                             title-filter learning after /jobsearch)
  BACKGROUND_EXTRACTION   -- specifically the Phase 10 auto-extract
                             of durable facts from working memory.
                             Distinct from generic background so we
                             can curate it separately.
"""
from __future__ import annotations

import contextvars

FOREGROUND = "foreground"
BACKGROUND = "background"
BACKGROUND_EXTRACTION = "background_extraction"

VALID_ORIGINS = frozenset({FOREGROUND, BACKGROUND, BACKGROUND_EXTRACTION})

# Default is FOREGROUND -- if no caller has explicitly entered a
# background scope, we attribute the write to the user request that
# kicked off the chain. This keeps the column meaningful even on
# direct (non-wrapped) writes.
_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_write_origin", default=FOREGROUND,
)


def get_current_write_origin() -> str:
    """Return the active origin label. Cheap (single ContextVar
    read). Always returns one of the VALID_ORIGINS values; an
    earlier ``set_current_write_origin`` with an unknown value gets
    coerced to FOREGROUND on read so callers never see junk."""
    val = _write_origin.get()
    if val not in VALID_ORIGINS:
        return FOREGROUND
    return val


def set_current_write_origin(
    origin: str,
) -> contextvars.Token[str]:
    """Set the active origin and return a token. Caller MUST pass
    the token back to ``reset_current_write_origin`` (typically in a
    ``try/finally``) to restore the previous value -- otherwise the
    setting leaks across tasks/calls.

    Unknown origin values are accepted (no exception) but will be
    normalised to FOREGROUND when read back; raising would defeat
    the "best-effort" character of the provenance system.
    """
    return _write_origin.set(origin)


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior origin. Idempotent in practice -- calling
    twice with the same token raises ``ValueError`` from contextvars,
    so callers should keep their try/finally tight."""
    _write_origin.reset(token)


def is_background() -> bool:
    """Convenience predicate -- True for any non-foreground origin."""
    return get_current_write_origin() != FOREGROUND
