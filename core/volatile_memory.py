"""Volatile (in-process) key-value memory.

Resets every time Sentinel restarts.  No SQLite, no disk, no lifecycle.
Useful for ephemeral state that should not survive a crash or reboot.

Usage::

    from core.volatile_memory import get_volatile_memory
    vm = get_volatile_memory()
    vm.set("last_query", "RSM Chicago")
    vm.get("last_query")          # "RSM Chicago"
    vm.get("missing", "default")  # "default"
    vm.delete("last_query")
    vm.all()                      # {}
    vm.clear()                    # wipes everything
"""
from __future__ import annotations

from threading import Lock
from typing import Any


class VolatileMemory:
    """Thread-safe in-process key/value store.

    Values may be any Python object.  All data is lost when the process
    exits -- intentional, that is the contract.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, overwriting any previous entry."""
        with self._lock:
            self._store[key] = value

    def delete(self, key: str) -> bool:
        """Remove *key*.  Returns True if the key existed, False otherwise."""
        with self._lock:
            return self._store.pop(key, _MISSING) is not _MISSING

    def clear(self) -> None:
        """Wipe the entire store."""
        with self._lock:
            self._store.clear()

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not present."""
        with self._lock:
            return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Return True if *key* is present."""
        with self._lock:
            return key in self._store

    def all(self) -> dict[str, Any]:
        """Return a shallow copy of the entire store."""
        with self._lock:
            return dict(self._store)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:  # pragma: no cover
        return f"VolatileMemory({len(self)} keys)"


# sentinel for delete
_MISSING = object()

# ------------------------------------------------------------------
# module-level singleton
# ------------------------------------------------------------------

_instance: VolatileMemory | None = None


def get_volatile_memory() -> VolatileMemory:
    """Return the process-global VolatileMemory singleton.

    Safe to call from multiple threads; the instance itself is also
    thread-safe.  Reset (clear) is intentionally NOT done here --
    callers that want a clean slate should call ``.clear()``.
    """
    global _instance
    if _instance is None:
        _instance = VolatileMemory()
    return _instance
