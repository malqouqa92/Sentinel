import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from core import config

_LOG_PATH = config.LOG_DIR / config.LOG_FILE


class _WindowsTolerantRotatingHandler(RotatingFileHandler):
    """RotatingFileHandler that doesn't crash if rollover fails because
    another process holds the file open (Windows-only concern: rename on
    a held file raises PermissionError/WinError 32). The current write
    is appended to the existing file; rotation will be retried on the
    next overflow once the other holder releases. On Linux/macOS the
    underlying behavior is identical (rename of an open file just works)
    so this is a no-op there."""

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except (OSError, PermissionError) as e:
            # Re-open the stream so subsequent writes still land on disk
            # even though we couldn't roll. Don't raise -- losing rotation
            # is acceptable; losing logs is not.
            try:
                if self.stream:
                    self.stream.close()
                self.stream = self._open()
            except Exception:
                pass
            # Best-effort note about the deferral.
            try:
                _LOGGER.warning(
                    "log rotation deferred (another process holds the file): %s",
                    type(e).__name__,
                    extra={"trace_id": "SEN-system", "component": "logger"},
                )
            except Exception:
                pass


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": getattr(record, "trace_id", ""),
            "level": record.levelname,
            "component": getattr(record, "component", ""),
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("sentinel")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "baseFilename", None) == str(_LOG_PATH)
        for h in logger.handlers
    ):
        # Phase 11: rotate at LOG_MAX_BYTES, keep LOG_BACKUP_COUNT siblings.
        # Windows-tolerant subclass: defers rotation if another process
        # holds the file open (PermissionError/WinError 32 on rename).
        handler = _WindowsTolerantRotatingHandler(
            _LOG_PATH, mode="a", encoding="utf-8",
            maxBytes=getattr(config, "LOG_MAX_BYTES", 10 * 1024 * 1024),
            backupCount=getattr(config, "LOG_BACKUP_COUNT", 5),
        )
        handler.setFormatter(_JsonlFormatter())
        logger.addHandler(handler)
    return logger


_LOGGER = _build_logger()


def log_event(trace_id: str, level: str, component: str, message: str) -> None:
    _LOGGER.log(
        getattr(logging, level.upper()),
        message,
        extra={"trace_id": trace_id, "component": component},
    )
