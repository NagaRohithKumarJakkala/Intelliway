"""Sliding-window rate limiting with shared Redis counters."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import defaultdict, deque

from redis.asyncio import Redis


_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cutoff = now - tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  redis.call('PEXPIRE', key, ARGV[2])
  return {0, oldest[2]}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, ARGV[2])
return {1, 0}
"""


class RateLimiter:
    def __init__(self) -> None:
        self._redis: Redis | None = None
        host = os.getenv("GATEWAY_REDIS_HOST")
        if host:
            self._redis = Redis(
                host=host,
                port=int(os.getenv("GATEWAY_REDIS_PORT", "6379")),
                password=os.getenv("GATEWAY_REDIS_PASSWORD") or os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        self._memory: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    @property
    def distributed(self) -> bool:
        return self._redis is not None

    async def check(self, identity: str, *, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        if self._redis is not None:
            result = await self._redis.eval(
                _SLIDING_WINDOW_LUA,
                1,
                f"intelliway:gateway:rpm:{identity}",
                now_ms,
                window_ms,
                limit,
                f"{now_ms}:{uuid.uuid4().hex}",
            )
            if int(result[0]) == 1:
                return True, 0
            oldest_ms = int(float(result[1]))
            retry_after = max(1, (oldest_ms + window_ms - now_ms + 999) // 1000)
            return False, retry_after

        now = time.monotonic()
        cutoff = now - window_seconds
        async with self._lock:
            bucket = self._memory[identity]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False, max(1, int(bucket[0] + window_seconds - now + 0.999))
            bucket.append(now)
            if len(self._memory) > 10000:
                for old_identity, old_bucket in list(self._memory.items()):
                    if not old_bucket or old_bucket[-1] <= cutoff:
                        self._memory.pop(old_identity, None)
            return True, 0

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def ping(self) -> bool:
        if self._redis is None:
            return False
        return bool(await self._redis.ping())
