"""Phase 11 -- health endpoint + startup spreading + /dashboard.

Tests K-P. All mocked: no real aiohttp port bind, no nvidia-smi
required (handled via tri-state probe), no real Telegram API.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from core.health import HealthMonitor, render_dashboard
from core.scheduler import Scheduler


def _add_overdue(name: str, seconds_ago: int = 60) -> int:
    past = (datetime.now(timezone.utc)
            - timedelta(seconds=seconds_ago)).isoformat()
    return database.add_job(
        name=name, schedule_type="interval", schedule_value="1h",
        command="/ping", next_run_at=past,
    )


# ---------------------------------------------------------------------
# Test K -- 3 overdue jobs spread across 300s
# ---------------------------------------------------------------------

def test_k_three_overdue_jobs_get_spread():
    ids = [_add_overdue(f"old-{i}") for i in range(3)]
    # Patch the spread budget so the math is deterministic.
    with patch.object(config, "SCHEDULER_STARTUP_SPREAD_SECONDS", 300):
        sched = Scheduler(
            router_fn=MagicMock(),
            wait_for_task_fn=AsyncMock(),
            shutdown_event=asyncio.Event(),
        )
        n = asyncio.run(sched.spread_overdue_jobs())
    assert n == 3
    times = [
        datetime.fromisoformat(database.get_job(j)["next_run_at"])
        for j in ids
    ]
    times.sort()
    # Spaced by ~100s each (300/3); allow generous slack for clock jitter.
    diffs = [(times[i + 1] - times[i]).total_seconds()
             for i in range(len(times) - 1)]
    for d in diffs:
        assert 80 <= d <= 120, f"unexpected gap {d}s between spread jobs"


# ---------------------------------------------------------------------
# Test L -- 0 overdue, no DB writes, no errors
# ---------------------------------------------------------------------

def test_l_zero_overdue_is_noop():
    sched = Scheduler(
        router_fn=MagicMock(),
        wait_for_task_fn=AsyncMock(),
        shutdown_event=asyncio.Event(),
    )
    n = asyncio.run(sched.spread_overdue_jobs())
    assert n == 0


# ---------------------------------------------------------------------
# Test M -- /health snapshot has all required keys + valid types
# ---------------------------------------------------------------------

def test_m_snapshot_has_required_keys():
    mon = HealthMonitor(
        start_time=datetime.now(timezone.utc) - timedelta(minutes=5)
    )
    snap = mon.snapshot()
    required = {
        "status", "ts", "uptime_seconds", "queue", "gpu", "models",
        "scheduler", "memory", "kb", "disk", "logs", "telegram",
    }
    assert set(snap.keys()) >= required, f"missing keys: {required - set(snap)}"
    assert snap["status"] == "healthy"
    assert isinstance(snap["uptime_seconds"], int)
    assert snap["uptime_seconds"] >= 0
    # queue subkeys
    for k in ("pending", "processing", "failed", "completed_total"):
        assert k in snap["queue"]
    # gpu subkeys
    for k in ("lock_holder", "lock_age_seconds", "vram_used_mb",
              "vram_limit_mb"):
        assert k in snap["gpu"]


# ---------------------------------------------------------------------
# Test N -- GPU lock observability shows holder + age
# ---------------------------------------------------------------------

def test_n_gpu_lock_holder_and_age_visible():
    # Pre-populate a lock with a known timestamp 90s ago.
    locked_at = (datetime.now(timezone.utc)
                 - timedelta(seconds=90)).isoformat()
    database._test_only_force_lock("gpu", "task-XYZ", locked_at)

    mon = HealthMonitor(start_time=datetime.now(timezone.utc))
    g = mon.snapshot()["gpu"]
    assert g["lock_holder"] == "task-XYZ"
    assert g["lock_age_seconds"] is not None
    assert 80 <= g["lock_age_seconds"] <= 120, \
        f"expected ~90s, got {g['lock_age_seconds']}s"


def test_n2_gpu_lock_unheld_returns_none():
    mon = HealthMonitor(start_time=datetime.now(timezone.utc))
    g = mon.snapshot()["gpu"]
    assert g["lock_holder"] is None
    assert g["lock_age_seconds"] is None


# ---------------------------------------------------------------------
# Test O -- model availability is read from cache, not probed
# ---------------------------------------------------------------------

def test_o_model_availability_uses_cache_no_network():
    mon = HealthMonitor(start_time=datetime.now(timezone.utc))
    mon.set_model_availability({
        "qwen-coder": True,
        "claude-cli": False,
        "qwen3-brain": True,
    })
    # Patch the registry to blow up if anyone tries to probe network.
    with patch("core.model_registry.MODEL_REGISTRY") as fake_reg:
        fake_reg.check_availability.side_effect = AssertionError(
            "no network probe expected"
        )
        snap = mon.snapshot()
    assert snap["models"] == {
        "qwen-coder": True,
        "claude-cli": False,
        "qwen3-brain": True,
    }


# ---------------------------------------------------------------------
# Test P -- /dashboard renders a non-empty body
# ---------------------------------------------------------------------

def test_p_dashboard_renders_non_empty_body():
    mon = HealthMonitor(
        start_time=datetime.now(timezone.utc) - timedelta(minutes=15),
    )
    body = render_dashboard(mon.snapshot())
    assert body and len(body) > 50
    # Must contain the major section headers.
    assert "Sentinel status" in body
    assert "Queue" in body
    assert "GPU" in body
    assert "Scheduler" in body
    assert "Memory" in body
    assert "Telegram polling" in body


def test_p2_dashboard_handles_held_gpu_lock():
    """When a lock IS held, the rendered line has a holder + age, not '(free)'."""
    locked_at = (datetime.now(timezone.utc)
                 - timedelta(seconds=42)).isoformat()
    database._test_only_force_lock("gpu", "agent.code_assistant", locked_at)
    mon = HealthMonitor(start_time=datetime.now(timezone.utc))
    body = render_dashboard(mon.snapshot())
    assert "agent.code_assistant" in body
    assert "(free)" not in body


# ---------------------------------------------------------------------
# HTTP endpoint smoke test (uses ephemeral port to avoid clashes)
# ---------------------------------------------------------------------

def test_health_endpoint_serves_json():
    """Bind on an ephemeral port, hit /health, parse the body."""
    import json
    import socket
    from aiohttp import web

    mon = HealthMonitor(start_time=datetime.now(timezone.utc))

    async def _run():
        app = web.Application()
        app.router.add_get("/health", mon.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        # Pick a free port.
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"http://127.0.0.1:{port}/health", timeout=2,
                ) as r:
                    assert r.status == 200
                    body = await r.text()
                    return json.loads(body)
        finally:
            await runner.cleanup()

    snap = asyncio.run(_run())
    assert snap["status"] == "healthy"
    assert "queue" in snap and "gpu" in snap
