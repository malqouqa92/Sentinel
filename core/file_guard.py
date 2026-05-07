"""Hash-based diff-watch for protected persona files.

PROTECTED_FILES are SHA-256'd at startup. Any subsequent change that
doesn't go through ``authorize_update()`` is logged CRITICAL and triggers
a Telegram alert callback. The hash baseline is in-memory only -- we
re-snapshot on startup, so this catches *runtime* tampering. Offline
tampering between sessions is by design out of scope (we'd need a
trusted store for that, which we don't have on a single-user PC).

Telegram alert callback is injected by ``main.py`` when the bot is up;
tests inject a list-append spy. Both share the same Callable[[str], None]
shape.
"""
import hashlib
from pathlib import Path
from typing import Callable

from core import config
from core.logger import log_event


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


class FileGuard:
    def __init__(
        self,
        directory: Path | None = None,
        protected: set[str] | None = None,
        alert_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.directory = directory or config.PERSONA_DIR
        self.protected = set(protected) if protected else set(
            config.PROTECTED_FILES,
        )
        self._alert = alert_callback
        self._hashes: dict[str, str | None] = {}
        self.snapshot()

    def snapshot(self) -> dict[str, str | None]:
        """Recompute baseline from disk for all protected files."""
        for name in self.protected:
            self._hashes[name] = _sha256(self.directory / name)
        return dict(self._hashes)

    def check_integrity(self) -> list[str]:
        """Return list of filenames whose current hash differs from
        the stored baseline. Logs CRITICAL and fires the alert
        callback for each tampered file. Does NOT update the baseline
        -- only ``authorize_update()`` can do that, and only for the
        specific file being authorized.
        """
        tampered: list[str] = []
        for name in self.protected:
            current = _sha256(self.directory / name)
            if current != self._hashes.get(name):
                tampered.append(name)
                expected = self._hashes.get(name)
                log_event(
                    "SEN-system", "CRITICAL", "file_guard",
                    f"protected file modified outside authorize_update: "
                    f"{name} expected={expected} actual={current}",
                )
                if self._alert is not None:
                    try:
                        self._alert(
                            f"⚠️ Protected file modified: {name}\n"
                            f"This was NOT done via the curation flow.",
                        )
                    except Exception as e:
                        log_event(
                            "SEN-system", "ERROR", "file_guard",
                            f"alert callback failed: "
                            f"{type(e).__name__}: {e}",
                        )
        return tampered

    def authorize_update(self, filename: str, new_content: str) -> None:
        """Write ``new_content`` to a protected file and refresh its
        hash baseline. Raises ValueError if the filename isn't
        protected -- prevents the curation flow from accidentally
        clobbering arbitrary files.
        """
        if filename not in self.protected:
            raise ValueError(
                f"file_guard.authorize_update only handles protected "
                f"files (one of {sorted(self.protected)}); "
                f"got {filename!r}",
            )
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")
        self._hashes[filename] = _sha256(path)
        prefix = (self._hashes[filename] or "")[:8]
        log_event(
            "SEN-system", "INFO", "file_guard",
            f"authorized update to {filename} (new hash {prefix}...)",
        )

    def hashes(self) -> dict[str, str | None]:
        return dict(self._hashes)


# Process-wide singleton; ``main.py`` constructs it with a real Telegram
# alert callback. Stays None at import time so test fixtures can
# instantiate isolated FileGuards over tmp_path persona dirs without
# fighting a global.
FILE_GUARD: FileGuard | None = None


def get_file_guard() -> FileGuard | None:
    return FILE_GUARD


def install_file_guard(guard: FileGuard) -> None:
    global FILE_GUARD
    FILE_GUARD = guard
