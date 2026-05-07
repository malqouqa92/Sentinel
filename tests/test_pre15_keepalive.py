"""Pre-Phase-15b: verify the worker no longer aggressively unloads
models after GPU tasks. Trust OLLAMA_KEEP_ALIVE (set externally to
2m) to manage VRAM eviction.

Why this matters: aggressive unload was the root cause of Phase 14a
pattern #52's 15-minute graduation hang -- worker model was unloaded
between /code completion and graduation start, forcing a cold-load.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_worker_no_unload_call_in_source():
    """Verify the explicit unload_ollama_model() call has been removed
    from worker.py's GPU-task cleanup path. Source-level check avoids
    needing pytest-asyncio just for this assertion.

    Pattern #52's 15-min hang was caused by aggressive unloading
    between /code completion and Phase 14b graduation. This test
    locks in the fix."""
    import inspect
    from core import worker
    src = inspect.getsource(worker.execute_task)
    # The unload call SHOULD NOT appear in _run_task anymore.
    assert "unload_ollama_model" not in src, (
        "worker.execute_task() should no longer call unload_ollama_model "
        "after GPU tasks. Trust OLLAMA_KEEP_ALIVE to manage VRAM. "
        "See pre-Phase-15b for rationale."
    )


def test_worker_still_releases_gpu_lock_in_source():
    """Defensive: lock release MUST stay -- removing unload doesn't
    affect lock discipline. The lock was always the real serializer;
    unload was only an eviction hint."""
    import inspect
    from core import worker
    src = inspect.getsource(worker.execute_task)
    assert "release_lock" in src, (
        "worker.execute_task() must still release the GPU lock"
    )


def test_inference_client_unload_method_still_exists():
    """The unload helper isn't removed -- /code teaching loops or
    future explicit calls may still want it. Just NOT called auto-
    matically after every GPU task."""
    from core.llm import InferenceClient
    assert hasattr(InferenceClient, "unload_ollama_model")
