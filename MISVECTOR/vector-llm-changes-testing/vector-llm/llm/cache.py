"""
llm/cache.py
============
Redis-backed response cache for GENERAL-intent /query requests (P4).

WHY:
  During demo-heavy periods the same questions get asked repeatedly.
  Each one currently runs a full Ollama inference. A short-lived,
  per-user cache lets identical questions return instantly without
  burning credits or LLM time.

WHAT'S CACHED:
  GENERAL intent only, non-streaming. DASHBOARD and DECISION results
  depend on the user's uploaded data (which can change between
  requests), so they are never cached.

KEY DESIGN:
  key = "qcache:" + sha256(user_id + ":" + normalised_query)
  - Scoped per user_id, so User A's cached answer can never leak to User B.
  - Query is lowercased/stripped before hashing so trivial whitespace/case
    differences still hit the cache.

TTL:
  Default 1 hour. Override with CACHE_TTL_SECONDS env var.

LIBRARY:
  Uses redis.asyncio (redis-py >= 4.2).
  Install: pip install redis
  Set env: REDIS_URL=redis://localhost:6379/0

FAIL-OPEN:
  If Redis is unreachable, get()/set() raise CacheServiceError, which
  callers should catch and treat as a cache miss / no-op — a Redis
  outage must never break /query.

USAGE (in main.py):
  from llm.cache import cache, CacheServiceError

  try:
      cached = await cache.get(user_id, req.query)
  except CacheServiceError:
      cached = None

  if cached is not None:
      return QueryResponse(answer=cached, intent="general",
                            confidence=route.confidence, cache_hit=True)

  ... run inference, get `answer` ...

  try:
      await cache.set(user_id, req.query, answer)
  except CacheServiceError:
      pass
"""

import os
import hashlib
from typing import Optional

DEFAULT_TTL_SECONDS = 3600  # 1 hour


class CacheServiceError(Exception):
    """Raised when Redis is unreachable or the redis package isn't installed."""
    pass


class QueryCache:
    """
    Thin async wrapper around redis.asyncio for caching GENERAL-intent
    query responses.

    Usage pattern:
        cached = await cache.get(user_id, query)   # -> str | None
        if cached is None:
            answer = await ollama_client.complete(prompt)
            await cache.set(user_id, query, answer)
    """

    def __init__(self):
        self._redis = None  # redis.asyncio.Redis — lazy init
        self._ttl = int(os.getenv("CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def _get_client(self):
        """Lazy-init the redis.asyncio client."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
            except ImportError:
                raise CacheServiceError(
                    "redis package is not installed. Run: pip install redis"
                )

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis = redis.from_url(redis_url, decode_responses=True)

        return self._redis

    async def close(self):
        """Close the Redis connection. Wire to app shutdown in main.py."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    def make_key(self, user_id: str, query: str) -> str:
        """
        Build the cache key: qcache:sha256(user_id + ':' + normalised_query)

        Normalises whitespace and casing so "What is revenue?" and
        "what is revenue? " hit the same cache entry.
        """
        normalised = " ".join(query.strip().lower().split())
        raw = f"{user_id}:{normalised}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"qcache:{digest}"

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    async def get(self, user_id: str, query: str) -> Optional[str]:
        """
        Return the cached answer for this user+query, or None on a miss.
        Raises CacheServiceError if Redis is unreachable — callers should
        treat that the same as a miss (fail open).
        """
        try:
            client = await self._get_client()
            return await client.get(self.make_key(user_id, query))
        except CacheServiceError:
            raise
        except Exception as e:
            raise CacheServiceError(f"Redis GET failed: {e}")

    async def set(
        self,
        user_id: str,
        query: str,
        answer: str,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store the answer for this user+query with a TTL (default 1 hour).
        Raises CacheServiceError if Redis is unreachable.
        """
        try:
            client = await self._get_client()
            await client.set(self.make_key(user_id, query), answer, ex=ttl or self._ttl)
        except CacheServiceError:
            raise
        except Exception as e:
            raise CacheServiceError(f"Redis SET failed: {e}")


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.cache import cache, CacheServiceError
# ------------------------------------------------------------------
cache = QueryCache()
