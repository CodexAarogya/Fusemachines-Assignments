"""
Two complementary reliability primitives:

1. RateLimiter - a per-client sliding-window limiter (requests/minute)
   applied at the API edge so a single caller can't exhaust the LLM
   provider's quota or starve other users.

2. ConcurrencyLimiter - a global asyncio.Semaphore bounding how many
   LLM calls are in flight at once, which is what actually controls
   throughput/latency under load and protects the process from being
   overwhelmed by a burst of concurrent requests.

Both are process-local (in-memory). For multi-instance deployments,
swap RateLimiter's backing store for Redis (see comment below).
"""
import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.limit = requests_per_minute
        self.window_seconds = 60
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, client_key: str) -> None:
        async with self._lock:
            now = time.monotonic()
            q = self._hits[client_key]
            while q and now - q[0] > self.window_seconds:
                q.popleft()
            if len(q) >= self.limit:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {self.limit} requests/{self.window_seconds}s"
                )
            q.append(now)

    # --- For horizontal scaling, replace the deque above with Redis, e.g.: ---
    # INCR client_key with EXPIRE window_seconds, compare to self.limit.


class ConcurrencyLimiter:
    """Bounds simultaneous in-flight LLM calls to control latency/throughput."""

    def __init__(self, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()
