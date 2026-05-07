import asyncio
import functools
import time


def retry(attempts: int = 3, base_delay: float = 1.0, backoff: float = 2.0):
    """Decorator factory: retry a function on exception with exponential backoff.

    Works for both synchronous and async (coroutine) functions.

    Args:
        attempts:   Total number of attempts (default 3).
        base_delay: Seconds to wait before the 2nd attempt (default 1.0).
        backoff:    Multiplier applied to delay after each failure (default 2.0).
                    Delays: base_delay, base_delay*backoff, base_delay*backoff**2 …

    Raises:
        The last exception raised if all attempts are exhausted.
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = base_delay
                last_exc: BaseException | None = None
                for attempt in range(1, attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        if attempt < attempts:
                            await asyncio.sleep(delay)
                            delay *= backoff
                raise last_exc  # type: ignore[misc]
            async_wrapper.retry_attempts = attempts
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                delay = base_delay
                last_exc: BaseException | None = None
                for attempt in range(1, attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        if attempt < attempts:
                            time.sleep(delay)
                            delay *= backoff
                raise last_exc  # type: ignore[misc]
            sync_wrapper.retry_attempts = attempts
            return sync_wrapper
    return decorator
