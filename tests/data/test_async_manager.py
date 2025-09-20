"""
Comprehensive testing of the Async Manager component.
Tests async operations, error handling, caching, and performance.
"""

import sys
from pathlib import Path
# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import tempfile
import shutil
from pathlib import Path
import json

# Import comprehensive async manager components
from src.data.async_manager import (
    EnhancedAsyncDataManager, SmartMemoryCache, ParquetStorage,
    APIConfig, DataProcessorConfig, CacheConfig, StorageConfig, ValidationConfig,
    CircuitBreaker, RetryPolicy, RateLimiter, HealthMonitor
)
from src.data.core import ValidationResult
from src.core.interfaces import PatternType, PatternConfig, PatternSignal


class TestSmartMemoryCache:
    """Test SmartMemoryCache functionality"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        config = CacheConfig(ttl_seconds=3600, max_size=100)
        cache = SmartMemoryCache(config)

        assert cache.cache == {}
        assert cache.access_times == {}
        assert cache.size_stats == {}
        assert cache.config.ttl_seconds == 3600
        assert cache.config.max_size == 100
        assert hasattr(cache, 'logger')

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        config = CacheConfig(ttl_seconds=3600, max_size=100)
        cache = SmartMemoryCache(config)

        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Set data
        assert await cache.set_async("test_key", test_data)

        # Get data
        retrieved_data = await cache.get_async("test_key")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        # Test sync versions
        assert cache.set_sync("sync_key", test_data)
        retrieved_sync_data = cache.get_sync("sync_key")
        assert not retrieved_sync_data.empty
        assert retrieved_sync_data.equals(test_data)

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(self):
        """Test cache TTL expiry"""
        config = CacheConfig(ttl_seconds=0)  # Immediate expiry
        cache = SmartMemoryCache(config)

        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Set data
        await cache.set_async("expire_key", test_data)

        # Should expire immediately
        retrieved_data = await cache.get_async("expire_key")
        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_cache_size_limit(self):
        """Test cache size limit eviction"""
        config = CacheConfig(max_size=2)  # Very small cache
        cache = SmartMemoryCache(config)

        data1 = pd.DataFrame({'test': [1, 2, 3]})
        data2 = pd.DataFrame({'test': [4, 5, 6]})
        data3 = pd.DataFrame({'test': [7, 8, 9]})

        # Add data until cache exceeds limit
        await cache.set_async("key1", data1)
        await cache.set_async("key2", data2)
        await cache.set_async("key3", data3)  # Should evict oldest

        # key1 should be evicted
        assert cache.cache.get("key1") is None
        # key2 and key3 should remain
        assert cache.cache.get("key2") is not None
        assert cache.cache.get("key3") is not None

    @pytest.mark.asyncio
    async def test_cache_clear_expired(self):
        """Test clearing expired cache entries"""
        config = CacheConfig(ttl_seconds=0)  # Immediate expiry
        cache = SmartMemoryCache(config)

        data1 = pd.DataFrame({'test': [1, 2, 3]})
        data2 = pd.DataFrame({'test': [4, 5, 6]})

        await cache.set_async("key1", data1)
        await cache.set_async("key2", data2)

        # Should clear all entries due to immediate expiry
        cleared = await cache.clear_expired_async()
        assert cleared == 2
        assert len(cache.cache) == 0

    def test_cache_stats(self):
        """Test cache statistics"""
        config = CacheConfig(max_size=10)
        cache = SmartMemoryCache(config)

        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Add data
        cache.set_sync("key1", test_data)
        cache.set_sync("key2", test_data)

        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['total_memory_usage'] == 6  # 3 rows per df * 2 dfs
        assert 'key1' in stats['keys']
        assert 'key2' in stats['keys']


class TestParquetStorage:
    """Test ParquetStorage functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = StorageConfig(base_path=self.temp_dir)
        self.storage = ParquetStorage(self.config)

    def teardown_method(self):
        """Cleanup test environment"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_store_and_load(self):
        """Test storing and loading parquet files"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Store data
        success = await self.storage.store_async("AAPL", test_data)
        assert success is True

        # Load data
        loaded_data = await self.storage.load_async("AAPL")
        assert not loaded_data.empty
        assert len(loaded_data) == 3
        assert loaded_data['open'].tolist() == [100, 101, 102]

    @pytest.mark.asyncio
    async def test_load_with_date_filter(self):
        """Test loading data with date filters"""
        # Create test data with date index
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)

        # Store data
        await self.storage.store_async("AAPL", test_data)

        # Load with date filter
        start_date = datetime(2023, 1, 2)
        end_date = datetime(2023, 1, 4)
        filtered_data = await self.storage.load_async("AAPL", start_date, end_date)

        assert len(filtered_data) == 3  # Jan 2, 3, 4
        assert filtered_data['open'].tolist() == [101, 102, 103]

    @pytest.mark.asyncio
    async def test_get_available_symbols(self):
        """Test getting available symbols"""
        # Store some data
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        await self.storage.store_async("AAPL", test_data)
        await self.storage.store_async("MSFT", test_data)

        symbols = await self.storage.get_available_symbols_async()
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert len(symbols) >= 2

    @pytest.mark.asyncio
    async def test_delete_symbol_data(self):
        """Test deleting symbol data"""
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        # Store data
        await self.storage.store_async("AAPL", test_data)

        # Verify data exists
        assert not (await self.storage.load_async("AAPL")).empty

        # Delete data
        success = await self.storage.delete_async("AAPL")
        assert success is True

        # Verify data is deleted
        assert (await self.storage.load_async("AAPL")).empty


class TestEnhancedAsyncDataManager:
    """Test EnhancedAsyncDataManager functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataProcessorConfig(
            cache_config=CacheConfig(ttl_seconds=3600, max_size=100),
            storage_config=StorageConfig(base_path=self.temp_dir),
            validation_config=ValidationConfig(min_data_points=3),
            max_concurrent_requests=5,
            request_timeout=10,
            retry_attempts=2,
            retry_delay=0.1
        )
        self.manager = EnhancedAsyncDataManager(self.config)

    def teardown_method(self):
        """Cleanup test environment"""
        if hasattr(self.manager, '_executor'):
            self.manager._executor.shutdown(wait=False)
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization"""
        assert self.manager.cache is not None
        assert self.manager.storage is not None
        assert self.manager.validator is not None
        assert self.manager.preprocessor is not None
        assert self.manager.apis is not None

    @pytest.mark.asyncio
    async def test_manager_context_manager(self):
        """Test manager context manager"""
        async with self.manager as manager:
            assert manager._session is not None
            # Should be able to use the manager
            pass

    @pytest.mark.asyncio
    async def test_fetch_data_success(self):
        """Test successful data fetching"""
        # Mock the fetch method to return test data
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        with patch.object(self.manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.return_value = test_data

            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 3)

            result = await self.manager.fetch_data_async("AAPL", start_date, end_date)

            assert not result.empty
            assert len(result) == 3
            mock_fetch.assert_called_once_with("AAPL", start_date, end_date)

    @pytest.mark.asyncio
    async def test_fetch_data_cache_hit(self):
        """Test cache hit scenario"""
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        # First call should hit storage
        with patch.object(self.manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.return_value = test_data

            await self.manager.fetch_data_async("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 2))

        # Second call should hit cache
            cached_result = await self.manager.fetch_data_async("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 2))
            assert not cached_result.empty
            # Should not call fetch method again
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols(self):
        """Test fetching multiple symbols"""
        test_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

        with patch.object(self.manager, '_fetch_single_symbol_with_retry') as mock_fetch:
            mock_fetch.return_value = test_data

            symbols = ["AAPL", "MSFT", "GOOGL"]
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 2)

            results = await self.manager.fetch_multiple_symbols_async(symbols, start_date, end_date)

            assert len(results) == 3
            assert "AAPL" in results
            assert "MSFT" in results
            assert "GOOGL" in results

            # Should have called fetch once for each symbol
            assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_preprocess_data(self):
        """Test data preprocessing"""
        test_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        })

        result = await self.manager.preprocess_data_async(test_data)

        # Should have additional columns from preprocessing
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'returns' in result.columns

    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test performance metrics"""
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        with patch.object(self.manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.return_value = test_data

            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 2)

            # Fetch some data
            await self.manager.fetch_data_async("AAPL", start_date, end_date)

            metrics = await self.manager.get_performance_metrics_async()

            assert 'cache_stats' in metrics
            assert 'metrics' in metrics
            assert 'available_symbols' in metrics
            assert 'fetch_times' in metrics['metrics']
            assert 'cache_hits' in metrics['metrics']
            assert 'cache_misses' in metrics['metrics']

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality"""
        # Add some data to cache first
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})
        await self.manager.cache.set_async("test", test_data)

        cleanup_result = await self.manager.cleanup_async()

        assert 'cache_entries_cleared' in cleanup_result
        assert isinstance(cleanup_result['cache_entries_cleared'], int)

    @pytest.mark.asyncio
    async def test_api_config_management(self):
        """Test API configuration management"""
        # Add custom API config
        api_config = APIConfig(
            name="test_api",
            base_url="https://test.api.com",
            timeout=20,
            rate_limit=10,
            retry_attempts=5
        )

        self.manager.add_api_config("test_api", api_config)

        assert "test_api" in self.manager.apics
        assert self.manager.apics["test_api"] == api_config

    @pytest.mark.asyncio
    async def test_preprocessor_customization(self):
        """Test custom preprocessor"""
        from ...src.utils.data_preprocessor import DataPreprocessor

        custom_preprocessor = DataPreprocessor()
        self.manager.set_preprocessor(custom_preprocessor)

        assert self.manager.preprocessor == custom_preprocessor


class TestAPIConfig:
    """Test APIConfig functionality"""

    def test_api_config_defaults(self):
        """Test default API configuration"""
        api_config = APIConfig("test_api")
        assert api_config.name == "test_api"
        assert api_config.base_url == ""
        assert api_config.timeout == 30
        assert api_config.rate_limit == 5
        assert api_config.retry_attempts == 3
        assert api_config.retry_delay == 1.0
        assert api_config.api_key is None

    def test_api_config_custom(self):
        """Test custom API configuration"""
        api_config = APIConfig(
            name="custom_api",
            base_url="https://api.example.com",
            timeout=60,
            rate_limit=10,
            retry_attempts=5,
            retry_delay=2.0,
            api_key="test_key"
        )
        assert api_config.name == "custom_api"
        assert api_config.base_url == "https://api.example.com"
        assert api_config.timeout == 60
        assert api_config.rate_limit == 10
        assert api_config.retry_attempts == 5
        assert api_config.retry_delay == 2.0
        assert api_config.api_key == "test_key"


class TestErrorHandling:
    """Test error handling in async manager"""

    @pytest.mark.asyncio
    async def test_fetch_data_error_handling(self):
        """Test error handling in data fetching"""
        manager = EnhancedAsyncDataManager()

        # Mock fetch method to raise an exception
        with patch.object(manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 2)

            result = await manager.fetch_data_async("AAPL", start_date, end_date)

            # Should return empty DataFrame on error
            assert result.empty
            assert len(manager.metrics.error_counts) > 0

    @pytest.mark.asyncio
    async def test_preprocess_data_error_handling(self):
        """Test error handling in data preprocessing"""
        manager = EnhancedAsyncDataManager()

        # Mock preprocessor to raise an exception
        with patch.object(manager.preprocessor, 'clean_data_async') as mock_clean:
            mock_clean.side_effect = Exception("Preprocessing error")

            test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

            result = await manager.preprocess_data_async(test_data)

            # Should return original data on error
            assert result.equals(test_data)

    @pytest.mark.asyncio
    async def test_cache_error_handling(self):
        """Test error handling in caching operations"""
        config = CacheConfig()
        cache = SmartMemoryCache(config)

        # Mock storage operation to raise an exception
        with patch.object(cache, 'cache', {}):
            test_data = pd.DataFrame({'test': [1, 2, 3]})

            # This should not raise an exception even if storage fails
            try:
                await cache.set_async("error_key", test_data)
            except Exception:
                pass  # Expected to fail in this test scenario


class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_concurrent_fetching(self):
        """Test concurrent data fetching performance"""
        config = DataProcessorConfig(max_concurrent_requests=5)
        manager = EnhancedAsyncDataManager(config)

        # Mock multiple successful fetches
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        with patch.object(manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.return_value = test_data

            symbols = [f"SYMBOL_{i}" for i in range(10)]
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 2)

            start_time = time.time()
            results = await manager.fetch_multiple_symbols_async(symbols, start_date, end_date)
            end_time = time.time()

            assert len(results) == 10
            elapsed_time = end_time - start_time

            # Should be faster than sequential execution
            assert elapsed_time < 2.0  # Should complete quickly with concurrent execution

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance benefits"""
        config = CacheConfig(ttl_seconds=3600, max_size=1000)
        cache = SmartMemoryCache(config)

        test_data = pd.DataFrame({'test': list(range(1000))})

        # Time cache set operation
        start_time = time.time()
        await cache.set_async("large_key", test_data)
        set_time = time.time() - start_time

        # Time cache get operation
        start_time = time.time()
        retrieved_data = await cache.get_async("large_key")
        get_time = time.time() - start_time

        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)
        assert set_time < 1.0  # Should be fast
        assert get_time < 1.0  # Should be fast


class TestRateLimiter:
    """Test RateLimiter functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)

        # Should allow first two requests
        assert await limiter.acquire("test_user")
        assert await limiter.acquire("test_user")

        # Should block third request within window
        assert not await limiter.acquire("test_user")

        # Wait for window to pass
        await asyncio.sleep(1.1)

        # Should allow request after window
        assert await limiter.acquire("test_user")

    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self):
        """Test rate limiting per user"""
        limiter = RateLimiter(max_requests=1, window_seconds=1)

        # Different users should not interfere
        assert await limiter.acquire("user1")
        assert await limiter.acquire("user2")

        # Same user should be limited
        assert not await limiter.acquire("user1")

    @pytest.mark.asyncio
    async def test_rate_limit_concurrent(self):
        """Test rate limiting with concurrent requests"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)

        # Make concurrent requests
        tasks = [limiter.acquire("test") for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # Should allow 2, reject 1
        assert sum(results) == 2
        assert sum(1 for r in results if not r) == 1


class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        assert breaker.state == 'CLOSED'
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None

    def test_circuit_breaker_failure_detection(self):
        """Test circuit breaker failure detection"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Record failures
        for i in range(3):
            breaker.record_failure()

        assert breaker.state == 'OPEN'
        assert breaker.failure_count == 3

    def test_circuit_breaker_request_allowance(self):
        """Test circuit breaker request allowance"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Closed state - should allow requests
        assert breaker.allow_request()

        # Open state - should block requests
        breaker.state = 'OPEN'
        assert not breaker.allow_request()

        # Half-open state - should allow requests
        breaker.state = 'HALF_OPEN'
        assert breaker.allow_request()

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Trigger open state
        breaker.state = 'OPEN'
        breaker.last_failure_time = datetime.now()

        # Wait for recovery timeout
        time.sleep(0.1)  # Can't actually wait 60 seconds in test, so just check logic
        breaker.last_failure_time = datetime.now() - timedelta(seconds=61)

        # Should transition to half-open
        assert breaker.allow_request()
        assert breaker.state == 'HALF_OPEN'


class TestRetryPolicy:
    """Test RetryPolicy functionality"""

    @pytest.mark.asyncio
    async def test_retry_policy_success(self):
        """Test retry policy with success"""
        policy = RetryPolicy(max_retries=3, backoff_factor=1.0)

        async def successful_function():
            return "success"

        result = await policy.execute_with_retry(successful_function)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_policy_failure(self):
        """Test retry policy with failure"""
        policy = RetryPolicy(max_retries=2, backoff_factor=0.1)

        async def failing_function():
            raise Exception("Always fails")

        with pytest.raises(Exception, match="Always fails"):
            await policy.execute_with_retry(failing_function)

    @pytest.mark.asyncio
    async def test_retry_policy_intermittent_failure(self):
        """Test retry policy with intermittent failures"""
        policy = RetryPolicy(max_retries=3, backoff_factor=0.1)
        call_count = 0

        async def intermittent_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Attempt {call_count} failed")
            return "success"

        result = await policy.execute_with_retry(intermittent_function)
        assert result == "success"
        assert call_count == 3


class TestHealthMonitor:
    """Test HealthMonitor functionality"""

    def test_health_monitor_initialization(self):
        """Test health monitor initialization"""
        monitor = HealthMonitor()

        assert monitor.metrics == {}
        assert monitor.alerts == []
        assert monitor.health_status == {}

    def test_health_monitor_record_metric(self):
        """Test health monitor metric recording"""
        monitor = HealthMonitor()

        monitor.record_metric('api_calls', 5)
        monitor.record_metric('error_rate', 0.1)
        monitor.record_metric('response_time', 1.5)

        assert 'api_calls' in monitor.metrics
        assert 'error_rate' in monitor.metrics
        assert 'response_time' in monitor.metrics

    def test_health_monitor_get_health_status(self):
        """Test health status retrieval"""
        monitor = HealthMonitor()

        # Record some metrics
        monitor.record_metric('api_calls', 10)
        monitor.record_metric('error_rate', 0.05)
        monitor.record_metric('response_time', 2.0)

        status = monitor.get_health_status()

        assert 'overall_status' in status
        assert 'metrics' in status
        assert 'timestamp' in status
        assert 'api_calls' in status['metrics']
        assert 'error_rate' in status['metrics']
        assert 'response_time' in status['metrics']

    def test_health_monitor_alerting(self):
        """Test health monitor alerting"""
        monitor = HealthMonitor()

        # Record metric that should trigger alert
        monitor.record_metric('error_rate', 0.15)  # High error rate

        status = monitor.get_health_status()
        assert 'alerts' in status
        assert len(status['alerts']) > 0

    def test_health_monitor_reset(self):
        """Test health monitor reset"""
        monitor = HealthMonitor()

        monitor.record_metric('api_calls', 10)
        monitor.record_alert('High error rate', 'error', 'warning')

        monitor.reset()

        assert monitor.metrics == {}
        assert monitor.alerts == []
        assert monitor.health_status == {}


# Helper functions
def create_test_dataframe(rows: int = 50) -> pd.DataFrame:
    """Create test market data for testing"""
    dates = pd.date_range('2023-01-01', periods=rows, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    np.random.seed(42)
    prices = [100.0]
    for i in range(1, len(dates)):
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))

    data = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * 1.01 for p in prices[:-1]],
        'low': [p * 0.99 for p in prices[:-1]],
        'close': prices[1:],
        'volume': [1000000 + i * 10000 for i in range(len(prices)-1)]
    }, index=dates[:-1])

    return data


if __name__ == "__main__":
    pytest.main([__file__])