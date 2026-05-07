"""Phase 11 -- localhost health endpoint + system probes.

Bound to 127.0.0.1 only. Never exposes externally. Cheap to compute:
no per-request network calls. Model availability is read from the
registry's last cached check (refreshed on startup); GPU lock holder +
age come from the locks table; queue/memory counts are SQLite COUNTs.

The /dashboard Telegram command renders the same data as a chat-
readable summary so the user can check status from their phone
without exposing the HTTP port.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import Any

from aiohttp import web

from core import config, database
from core.logger import log_event


# nvidia-smi probe -- one-time WARN if missing, then skip.
_NVSMI_AVAILABLE: bool | None = None  # tri-state: None=unprobed


def _nvsmi_path() -> str | None:
    """Return path to nvidia-smi, or None if not on PATH. Logs a one-
    time WARNING the first time it's missing."""
    global _NVSMI_AVAILABLE
    if _NVSMI_AVAILABLE is False:
        return None
    p = shutil.which("nvidia-smi")
    if p is None:
        if _NVSMI_AVAILABLE is None:
            log_event("SEN-system", "WARNING", "health",
                      "nvidia-smi not found on PATH; VRAM probe will "
                      "report None (one-time warning)")
            _NVSMI_AVAILABLE = False
        return None
    _NVSMI_AVAILABLE = True
    return p


def _vram_used_mb() -> int | None:
    """Total VRAM in use across all GPUs, MB. None if nvidia-smi absent
    or query fails.

    Phase 16 C-safety: pass CREATE_NO_WINDOW on Windows so the bot
    running detached (via Phase 11 Task Scheduler supervisor) doesn't
    flash a console window every time the scheduler's Resource probe
    fires (~every 10 min). On non-Windows hosts, the flag is 0 and
    has no effect."""
    p = _nvsmi_path()
    if p is None:
        return None
    import subprocess
    import sys
    creationflags = (
        subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )
    try:
        out = subprocess.run(
            [p, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=creationflags,
        )
        if out.returncode != 0:
            return None
        total = sum(int(x.strip()) for x in out.stdout.splitlines()
                    if x.strip())
        return total
    except Exception:
        return None


class HealthMonitor:
    """Snapshot of system + scheduler + memory + telegram state.

    `bot` and `scheduler` are wired in by main.py after construction.
    Both may be None during very early startup; JSON shape stays the
    same with null values.
    """

    def __init__(
        self,
        start_time: datetime,
        scheduler: Any | None = None,
        bot: Any | None = None,
    ) -> None:
        self.start_time = start_time
        self.scheduler = scheduler
        self.bot = bot
        self._model_avail: dict[str, bool] | None = None

    def set_model_availability(self, avail: dict[str, bool]) -> None:
        self._model_avail = dict(avail)

    def _uptime_seconds(self) -> int:
        return int((datetime.now(timezone.utc) - self.start_time)
                   .total_seconds())

    def _queue(self) -> dict:
        return {
            "pending": database.count_tasks_by_status("pending"),
            "processing": database.count_tasks_by_status("processing"),
            "failed": database.count_tasks_by_status("failed"),
            "completed_total": database.count_tasks_by_status("completed"),
        }

    def _gpu(self) -> dict:
        lock = database.get_lock("gpu")
        lock_age_s = None
        if lock and lock.get("locked_at"):
            try:
                t = datetime.fromisoformat(lock["locked_at"])
                lock_age_s = int(
                    (datetime.now(timezone.utc) - t).total_seconds()
                )
            except Exception:
                lock_age_s = None
        return {
            "lock_holder": lock["locked_by"] if lock else None,
            "lock_age_seconds": lock_age_s,
            "vram_used_mb": _vram_used_mb(),
            "vram_limit_mb": config.VRAM_LIMIT_MB,
        }

    def _scheduler_block(self) -> dict:
        if self.scheduler is None:
            return {"enabled_jobs": 0, "running_jobs": 0, "next_run_at": None}
        jobs = database.list_jobs(enabled_only=True)
        running = sum(1 for j in jobs if database.has_running_run(j["id"]))
        next_iso: str | None = None
        for j in sorted(jobs, key=lambda x: x["next_run_at"] or "9999"):
            if j.get("next_run_at"):
                next_iso = j["next_run_at"]
                break
        return {
            "enabled_jobs": len(jobs),
            "running_jobs": running,
            "next_run_at": next_iso,
        }

    def _memory_block(self) -> dict:
        try:
            from core.memory import get_memory
            stats = get_memory().stats()
            return {
                "episodes": int(stats.get("total_episodes", 0)),
                "facts": int(stats.get("total_facts", 0)),
            }
        except Exception:
            return {"episodes": 0, "facts": 0}

    def _kb_block(self) -> dict:
        try:
            from core.knowledge_base import KnowledgeBase
            return {"patterns": int(
                KnowledgeBase().stats().get("total_entries", 0)
            )}
        except Exception:
            return {"patterns": 0}

    def _disk_block(self) -> dict:
        try:
            d = shutil.disk_usage(str(config.PROJECT_ROOT))
            return {
                "free_bytes": int(d.free),
                "free_gb": round(d.free / (1024 ** 3), 1),
            }
        except Exception:
            return {"free_bytes": None, "free_gb": None}

    def _logs_block(self) -> dict:
        try:
            log_file = config.LOG_DIR / config.LOG_FILE
            size = log_file.stat().st_size if log_file.exists() else 0
            rotated = sum(
                f.stat().st_size for f in config.LOG_DIR.glob(
                    f"{config.LOG_FILE}.*"
                )
                if f.is_file()
            )
            return {
                "current_bytes": int(size),
                "total_bytes": int(size + rotated),
            }
        except Exception:
            return {"current_bytes": None, "total_bytes": None}

    def _telegram_block(self) -> dict:
        if self.bot is None:
            return {"polling_alive": False}
        try:
            updater = getattr(self.bot.app, "updater", None)
            alive = bool(updater and getattr(updater, "running", False))
        except Exception:
            alive = False
        return {"polling_alive": alive}

    def _models_block(self) -> dict:
        if self._model_avail is not None:
            return dict(self._model_avail)
        try:
            from core.model_registry import MODEL_REGISTRY
            return {m["name"]: None for m in MODEL_REGISTRY.list_models()}
        except Exception:
            return {}

    def snapshot(self) -> dict:
        return {
            "status": "healthy",
            "ts": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": self._uptime_seconds(),
            "queue": self._queue(),
            "gpu": self._gpu(),
            "models": self._models_block(),
            "scheduler": self._scheduler_block(),
            "memory": self._memory_block(),
            "kb": self._kb_block(),
            "disk": self._disk_block(),
            "logs": self._logs_block(),
            "telegram": self._telegram_block(),
        }

    async def handle(self, request: web.Request) -> web.Response:
        return web.json_response(self.snapshot())


async def start_health_server(monitor: HealthMonitor) -> web.AppRunner:
    """Bind aiohttp on 127.0.0.1:HEALTH_PORT. Returns the runner so the
    caller can `await runner.cleanup()` on shutdown."""
    app = web.Application()
    app.router.add_get("/health", monitor.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(
        runner,
        host=getattr(config, "HEALTH_BIND_HOST", "127.0.0.1"),
        port=getattr(config, "HEALTH_PORT", 18700),
    )
    await site.start()
    log_event("SEN-system", "INFO", "health",
              f"health endpoint listening on http://127.0.0.1:"
              f"{getattr(config, 'HEALTH_PORT', 18700)}/health")
    return runner


def render_dashboard(snap: dict) -> str:
    """Compact chat-readable summary of a HealthMonitor snapshot."""
    lines: list[str] = []
    up = snap.get("uptime_seconds", 0)
    h, rem = divmod(up, 3600)
    m, s = divmod(rem, 60)
    lines.append(f"📊 Sentinel status (up {h}h{m:02d}m{s:02d}s)")
    q = snap["queue"]
    lines.append(
        f"Queue: pending={q['pending']} processing={q['processing']} "
        f"failed={q['failed']}"
    )
    g = snap["gpu"]
    lock = g["lock_holder"] or "(free)"
    age = (f"{g['lock_age_seconds']}s"
           if g["lock_age_seconds"] is not None else "-")
    vram = (f"{g['vram_used_mb']}MB"
            if g["vram_used_mb"] is not None else "?")
    lines.append(
        f"GPU: holder={lock} age={age} vram={vram}/{g['vram_limit_mb']}MB"
    )
    s = snap["scheduler"]
    lines.append(
        f"Scheduler: {s['enabled_jobs']} enabled / {s['running_jobs']} "
        f"running" + (f" -- next {s['next_run_at']}"
                      if s["next_run_at"] else "")
    )
    mem = snap["memory"]
    lines.append(
        f"Memory: {mem['episodes']} episodes / {mem['facts']} facts / "
        f"{snap['kb']['patterns']} KB patterns"
    )
    d = snap["disk"]
    if d["free_gb"] is not None:
        lines.append(f"Disk free: {d['free_gb']} GB")
    lg = snap["logs"]
    if lg["current_bytes"] is not None:
        lines.append(
            f"Logs: {lg['current_bytes']//1024} KB current / "
            f"{lg['total_bytes']//1024} KB total"
        )
    tg = snap["telegram"]
    lines.append("Telegram polling: " + (
        "alive ✅" if tg["polling_alive"] else "dead ❌"
    ))
    models = snap.get("models") or {}
    if models:
        on = [k for k, v in models.items() if v]
        off = [k for k, v in models.items() if v is False]
        if on or off:
            mline = []
            if on:
                mline.append("on=" + ",".join(on))
            if off:
                mline.append("off=" + ",".join(off))
            lines.append("Models: " + " / ".join(mline))
    return "\n".join(lines)
