"""
RateLimiter (sliding-window per-client) + ConcurrencyLimiter (semaphore).
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


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()
