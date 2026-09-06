"""
Prompt/response cache. Avoids paying LLM latency+cost twice for an
identical (message history + params) request. Backed by cachetools'
TTLCache which handles both max-size eviction (LRU-ish) and
time-based expiry in one structure.
"""
import hashlib
import json
from typing import Any, Optional

from cachetools import TTLCache


class ResponseCache:
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 600):
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    @staticmethod
    def make_key(**kwargs) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def stats(self) -> dict:
        return {"size": len(self._cache), "maxsize": self._cache.maxsize, "ttl": self._cache.ttl}
