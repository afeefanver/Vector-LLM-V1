"""
tests/test_cache.py
====================
Unit tests for llm/cache.py (P4 — Redis response cache for GENERAL queries).

No live Redis required:
  - Key-construction tests run against the real QueryCache instance.
  - get()/set() roundtrip tests inject a fake `redis.asyncio` module into
    sys.modules so _get_client() picks it up without needing redis-py installed.
  - "service unavailable" tests rely on redis-py genuinely not being
    installed in the test environment, mirroring the asyncpg-not-installed
    pattern in test_credits.py.
"""

import sys
import types
import importlib
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_cache_module():
    """Reload llm.cache so each test gets a clean singleton (no cached _redis)."""
    sys.modules.pop("llm.cache", None)
    return importlib.import_module("llm.cache")


def _install_fake_redis(mock_client):
    """
    Inject fake `redis` / `redis.asyncio` modules into sys.modules so that
    `import redis.asyncio as redis` inside cache.py resolves to our mock,
    without requiring redis-py to actually be installed.
    """
    fake_redis_asyncio = types.ModuleType("redis.asyncio")
    fake_redis_asyncio.from_url = MagicMock(return_value=mock_client)

    fake_redis = types.ModuleType("redis")
    fake_redis.asyncio = fake_redis_asyncio

    sys.modules["redis"] = fake_redis
    sys.modules["redis.asyncio"] = fake_redis_asyncio


def _remove_fake_redis():
    sys.modules.pop("redis", None)
    sys.modules.pop("redis.asyncio", None)


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------

class TestMakeKey:

    def setup_method(self):
        self.cache_mod = _fresh_cache_module()

    def test_key_is_deterministic(self):
        cache = self.cache_mod.QueryCache()
        k1 = cache.make_key("user-1", "what is revenue?")
        k2 = cache.make_key("user-1", "what is revenue?")
        assert k1 == k2

    def test_key_differs_by_user(self):
        cache = self.cache_mod.QueryCache()
        k1 = cache.make_key("user-1", "what is revenue?")
        k2 = cache.make_key("user-2", "what is revenue?")
        assert k1 != k2

    def test_key_differs_by_query(self):
        cache = self.cache_mod.QueryCache()
        k1 = cache.make_key("user-1", "what is revenue?")
        k2 = cache.make_key("user-1", "what is profit?")
        assert k1 != k2

    def test_key_normalises_whitespace_and_case(self):
        cache = self.cache_mod.QueryCache()
        k1 = cache.make_key("user-1", "What is   Revenue?")
        k2 = cache.make_key("user-1", "what is revenue?")
        assert k1 == k2

    def test_key_has_qcache_prefix(self):
        cache = self.cache_mod.QueryCache()
        key = cache.make_key("user-1", "what is revenue?")
        assert key.startswith("qcache:")


# ---------------------------------------------------------------------------
# Fail-open: redis-py not installed
# ---------------------------------------------------------------------------

class TestRedisUnavailable:

    def setup_method(self):
        _remove_fake_redis()
        self.cache_mod = _fresh_cache_module()

    @pytest.mark.asyncio
    async def test_get_raises_cache_service_error_without_redis(self):
        cache = self.cache_mod.QueryCache()
        with pytest.raises(self.cache_mod.CacheServiceError):
            await cache.get("user-1", "what is revenue?")

    @pytest.mark.asyncio
    async def test_set_raises_cache_service_error_without_redis(self):
        cache = self.cache_mod.QueryCache()
        with pytest.raises(self.cache_mod.CacheServiceError):
            await cache.set("user-1", "what is revenue?", "Revenue is up 15%.")


# ---------------------------------------------------------------------------
# get() / set() roundtrip with a mocked Redis client
# ---------------------------------------------------------------------------

class TestGetSetRoundtrip:

    def setup_method(self):
        self.mock_client = MagicMock()
        self.mock_client.get = AsyncMock(return_value=None)
        self.mock_client.set = AsyncMock(return_value=True)
        _install_fake_redis(self.mock_client)
        self.cache_mod = _fresh_cache_module()

    def teardown_method(self):
        _remove_fake_redis()

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self):
        cache = self.cache_mod.QueryCache()
        result = await cache.get("user-1", "what is revenue?")
        assert result is None
        self.mock_client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_returns_cached_value_on_hit(self):
        self.mock_client.get = AsyncMock(return_value="Revenue is up 15%.")
        cache = self.cache_mod.QueryCache()
        result = await cache.get("user-1", "what is revenue?")
        assert result == "Revenue is up 15%."

    @pytest.mark.asyncio
    async def test_set_calls_redis_with_ttl(self):
        cache = self.cache_mod.QueryCache()
        await cache.set("user-1", "what is revenue?", "Revenue is up 15%.")

        self.mock_client.set.assert_awaited_once()
        args, kwargs = self.mock_client.set.call_args
        key, value = args[0], args[1]
        assert key.startswith("qcache:")
        assert value == "Revenue is up 15%."
        assert kwargs["ex"] == self.cache_mod.DEFAULT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_set_uses_custom_ttl_when_given(self):
        cache = self.cache_mod.QueryCache()
        await cache.set("user-1", "what is revenue?", "Revenue is up 15%.", ttl=120)

        _, kwargs = self.mock_client.set.call_args
        assert kwargs["ex"] == 120

    @pytest.mark.asyncio
    async def test_different_users_get_different_keys(self):
        cache = self.cache_mod.QueryCache()
        await cache.set("user-1", "what is revenue?", "Answer A")
        await cache.set("user-2", "what is revenue?", "Answer B")

        key_a = self.mock_client.set.call_args_list[0][0][0]
        key_b = self.mock_client.set.call_args_list[1][0][0]
        assert key_a != key_b
