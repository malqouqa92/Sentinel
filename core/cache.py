import threading
import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Thread-safe LRU cache with TTL expiry and atomic eviction.

    * Lock is held for every public method, so concurrent get/set/delete
      calls are safe across threads.
    * A background daemon thread ('janitor') runs every ttl/2 seconds and
      bulk-removes expired entries so memory is reclaimed even for keys
      that are never accessed again.
    * On-access expiry check means callers never observe a stale value even
      if the janitor hasn't run yet.
    * LRU eviction: when the cache is full the *least-recently-used* entry
      is evicted atomically inside the same lock acquisition as the insert.
    """

    def __init__(self, maxsize: int = 128, ttl: float = 60.0) -> None:
        if maxsize < 1:
            raise ValueError('maxsize must be >= 1')
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._janitor = threading.Thread(target=self._evict_loop, daemon=True)
        self._janitor.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_loop(self) -> None:
        """Background thread: bulk-remove expired entries periodically."""
        interval = max(self._ttl / 2, 0.05)
        while not self._stop.wait(timeout=interval):
            now = time.monotonic()
            with self._lock:
                expired = [k for k, (_, e) in self._cache.items() if e <= now]
                for k in expired:
                    del self._cache[k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if missing / expired.

        Moves an unexpired hit to the MRU end of the order so it is the
        last candidate for LRU eviction.
        """
        with self._lock:
            if key not in self._cache:
                return default
            value, expiry = self._cache[key]
            if time.monotonic() > expiry:
                del self._cache[key]
                return default
            self._cache.move_to_end(key)
            return value

    def set(self, key: Any, value: Any) -> None:
        """Insert or update *key* with *value*.

        If *key* already exists its expiry is refreshed and it is moved to
        the MRU end.  If the cache is at capacity, the LRU entry is evicted
        atomically before the new entry is inserted.
        """
        with self._lock:
            expiry = time.monotonic() + self._ttl
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (value, expiry)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)  # evict LRU atomically
                self._cache[key] = (value, expiry)

    def delete(self, key: Any) -> bool:
        """Remove *key* from the cache.  Returns True if it was present."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return the number of entries currently held in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        """Check if *key* is present in the cache."""
        with self._lock:
            return key in self._cache