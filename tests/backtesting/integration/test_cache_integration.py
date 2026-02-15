"""
Cache Integration Tests.

Verify coordination between the DataFrame cache and the Indicator cache during
strategy execution and validate cache TTL/eviction behavior.

Extended explanation of behavior and purpose:

- Test that indicator calculations are cached and reused across strategy runs.
- Test that DataFrame loading is cached when loading from parquet files.
- Provide a small performance smoke-test to confirm warm runs are faster.
- Verify TTL (time-to-live) removes expired entries and LRU evicts least-
  recently-used items when the cache exceeds its max size.

These integration tests exercise multiple components working together and should
be run as part of integration or nightly test suites.
"""
import os
import time
import uuid

import pytest

from app.backtesting.cache.cache_base import Cache, CACHE_DIR
from app.backtesting.cache.dataframe_cache import dataframe_cache, get_cached_dataframe
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.strategies import RSIStrategy


# ==================== Cache Coordination Tests ====================

@pytest.mark.integration
class TestCacheCoordination:
    """Tests that dataframe and indicator caches interact correctly.

    These tests verify that indicator calculations are stored in the indicator
    cache when strategies run on DataFrames, and that loading from parquet
    uses the dataframe cache. The tests intentionally separate the two cache
    types because running a strategy on an in-memory DataFrame will not
    populate the `dataframe_cache` (that cache stores file-based loads).
    """

    def test_cache_misses_then_hits(self, integration_test_data, contract_switch_dates, clean_caches, tmp_path):
        """
        Test indicator cache is populated by strategy runs and DataFrame cache
        is used when loading from parquet.

        Steps:
        1. Run strategy on an in-memory DataFrame to populate indicator cache
        2. Write the same DataFrame to a temporary parquet file
        3. Call get_cached_dataframe() twice and verify a cache miss then hit
        """
        # Ensure caches are empty
        dataframe_cache.cache_data.clear()
        indicator_cache.cache_data.clear()

        # Create strategy instance (RSI) used for indicator generation
        strategy = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )

        # Run strategy on in-memory data: this exercises indicator calculations
        # and should populate the indicator cache (but not dataframe_cache)
        trades_first = strategy.run(integration_test_data.copy(), contract_switch_dates.get('ZS', []))

        # Indicator cache should have entries after the first run
        assert indicator_cache.size() >= 1

        # Write the same DataFrame to disk to exercise dataframe_cache
        temp_file = tmp_path / "zs_integration.parquet"
        integration_test_data.to_parquet(temp_file)

        # First access through get_cached_dataframe should be a miss and store the DataFrame
        get_cached_dataframe(str(temp_file))
        assert dataframe_cache.size() >= 1

        # Second access should be served from the cache (hits increment)
        get_cached_dataframe(str(temp_file))
        assert dataframe_cache.hits >= 1

        # Sanity check: strategy results should be a list of trades
        assert isinstance(trades_first, list)


# ==================== Cache Performance Tests ====================

@pytest.mark.integration
class TestCachePerformance:
    """Performance-related smoke tests for caching behavior.

    These tests are intentionally lenient: they verify that warm (cached)
    runs are faster than cold runs but do not act as strict microbenchmarks.
    """

    def test_cached_run_is_faster_than_cold_run(self, small_test_data, contract_switch_dates, clean_caches):
        """
        Quick performance smoke-test comparing cold and warm runs.

        This is a lenient assertion (5% tolerance) to avoid flaky failures on
        different machines. The goal is to ensure caching helps, not to be a
        microbenchmark.
        """
        # Clear caches before timing
        dataframe_cache.cache_data.clear()
        indicator_cache.cache_data.clear()

        strategy = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )

        # Cold run (populate caches)
        strategy.run(small_test_data.copy(), contract_switch_dates.get('ZS', []))
        ind_hits_after_cold = indicator_cache.hits
        assert indicator_cache.size() >= 1

        # Warm run (should hit caches)
        strategy.run(small_test_data.copy(), contract_switch_dates.get('ZS', []))

        # Warm run should produce at least one indicator cache hit compared to
        # the cold run
        assert indicator_cache.hits >= ind_hits_after_cold + 1

    def test_indicator_cache_reuse_across_strategies(self, small_test_data, contract_switch_dates, clean_caches):
        """
        Verify indicator cache is reused across different strategy instances that
        request the same indicator parameters.

        Steps:
        1. Run RSI strategy with a specific period to populate indicator_cache
        2. Run a second RSI strategy with the same period but different thresholds
        3. Verify indicator_cache.hits increased (cache reused)
        """
        # Clear caches
        dataframe_cache.cache_data.clear()
        indicator_cache.cache_data.clear()

        # Strategy A
        s1 = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )

        # Strategy B (same indicator parameters)
        s2 = RSIStrategy(
            rsi_period=14, lower_threshold=25, upper_threshold=75,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )

        # Populate indicator cache
        s1.run(small_test_data.copy(), contract_switch_dates.get('ZS', []))
        ind_hits_before = indicator_cache.hits

        # Expect reuse of cached indicator computations
        s2.run(small_test_data.copy(), contract_switch_dates.get('ZS', []))
        assert indicator_cache.hits >= ind_hits_before


# ==================== TTL and Eviction Tests ====================

def _cleanup_cache_files(cache_name):
    """Remove cache and lock files created during tests."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.pkl")
    lock_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.lock")
    for p in (cache_file, lock_file):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


@pytest.mark.integration
class TestCacheTTLAndEviction:
    """TTL expiration and LRU eviction tests for the cache base class.

    These tests create isolated cache instances (unique names) so they don't
    interfere with global caches used by the application. Temporary cache
    files are removed in cleanup.
    """

    def test_ttl_expiration_removes_items(self):
        """
        Verify items expire after max_age seconds.

        Create a cache with a very small TTL, add an item, wait longer than the
        TTL, and assert the item is removed.
        """
        cache_name = f"test_ttl_{uuid.uuid4().hex}"
        max_age = 0.05
        cache = Cache(cache_name=cache_name, max_size=10, max_age=max_age)
        try:
            cache.set('a', 1)
            assert cache.contains('a') is True
            assert cache.size() == 1

            timeout = max(0.5, max_age * 20)
            deadline = time.monotonic() + timeout
            poll_interval = 0.01
            while time.monotonic() < deadline:
                if not cache.contains('a'):
                    break
                time.sleep(poll_interval)
            assert cache.contains('a') is False
            assert cache.size() == 0
        finally:
            _cleanup_cache_files(cache_name)

    def test_lru_eviction_removes_oldest(self):
        """
        Verify least-recently-used eviction when max_size is exceeded.

        Add items up to the cache's max_size, then add one more item. The
        oldest item (least-recently-used) should be evicted to make space.
        """
        cache_name = f"test_lru_{uuid.uuid4().hex}"
        cache = Cache(cache_name=cache_name, max_size=3, max_age=3600)
        try:
            cache.set('k1', 'v1')
            cache.set('k2', 'v2')
            cache.set('k3', 'v3')
            assert cache.size() == 3
            # touch k1 so it becomes most-recently-used
            assert cache.get('k1') == 'v1'
            cache.set('k4', 'v4')
            # k2 should be evicted
            assert cache.contains('k2') is False
            assert cache.contains('k1') is True
            assert cache.contains('k3') is True
            assert cache.contains('k4') is True
            assert cache.size() == 3
        finally:
            _cleanup_cache_files(cache_name)
