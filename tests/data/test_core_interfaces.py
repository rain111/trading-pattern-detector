"""
Comprehensive testing of core interfaces and abstract base classes.
Tests the foundation of the enhanced data management system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List

# Import core interfaces from multiple locations to cover the enhanced system
from src.data.core import (
    CacheConfig, StorageConfig, ValidationConfig, DataProcessorConfig,
    CacheType, StorageType, DataFormat, ValidationResult,
    DataProcessor, DataCache, DataStorage, DataValidator,
    DataProcessorResult, DataMetrics
)
from src.core.interfaces import (
    PatternType, PatternConfig, PatternSignal,
    EnhancedPatternDetector, BaseDetector, PatternEngine,
    DataValidator as CoreDataValidator
)
from src.data.async_manager import (
    APIConfig, SmartMemoryCache, ParquetStorage, EnhancedAsyncDataManager
)
from src.utils.data_preprocessor import DataPreprocessor


class TestCacheConfig:
    """Test CacheConfig functionality"""

    def test_cache_config_defaults(self):
        """Test default cache configuration"""
        config = CacheConfig()
        assert config.cache_type == CacheType.MEMORY
        assert config.ttl_seconds == 3600
        assert config.max_size == 10000
        assert config.compression is True

    def test_cache_config_custom(self):
        """Test custom cache configuration"""
        config = CacheConfig(
            cache_type=CacheType.DISK,
            ttl_seconds=7200,
            max_size=5000,
            compression=False,
            parquet_path="/tmp/cache"
        )
        assert config.cache_type == CacheType.DISK
        assert config.ttl_seconds == 7200
        assert config.max_size == 5000
        assert config.compression is False
        assert config.parquet_path == "/tmp/cache"


class TestStorageConfig:
    """Test StorageConfig functionality"""

    def test_storage_config_defaults(self):
        """Test default storage configuration"""
        config = StorageConfig()
        assert config.storage_type == StorageType.PARQUET
        assert config.base_path == "data"
        assert config.compression == "snappy"
        assert config.partition_size == 100000
        assert config.format == DataFormat.OHLCV

    def test_storage_config_custom(self):
        """Test custom storage configuration"""
        config = StorageConfig(
            storage_type=StorageType.CSV,
            base_path="/tmp/storage",
            compression="gzip",
            partition_size=50000,
            format=DataFormat.TICK
        )
        assert config.storage_type == StorageType.CSV
        assert config.base_path == "/tmp/storage"
        assert config.compression == "gzip"
        assert config.partition_size == 50000
        assert config.format == DataFormat.TICK


class TestValidationConfig:
    """Test ValidationConfig functionality"""

    def test_validation_config_defaults(self):
        """Test default validation configuration"""
        config = ValidationConfig()
        assert config.min_data_points == 20
        assert config.max_missing_pct == 5.0
        assert config.price_consistency_check is True
        assert config.volume_consistency_check is True
        assert config.timezone_check is True
        assert config.timezone == "UTC"

    def test_validation_config_custom(self):
        """Test custom validation configuration"""
        config = ValidationConfig(
            min_data_points=50,
            max_missing_pct=10.0,
            price_consistency_check=False,
            volume_consistency_check=False,
            timezone_check=False,
            timezone="US/Eastern"
        )
        assert config.min_data_points == 50
        assert config.max_missing_pct == 10.0
        assert config.price_consistency_check is False
        assert config.volume_consistency_check is False
        assert config.timezone_check is False
        assert config.timezone == "US/Eastern"


class TestDataProcessorConfig:
    """Test DataProcessorConfig functionality"""

    def test_data_processor_config_defaults(self):
        """Test default data processor configuration"""
        config = DataProcessorConfig()
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0

    def test_data_processor_config_with_components(self):
        """Test data processor configuration with component configs"""
        cache_config = CacheConfig()
        storage_config = StorageConfig()
        validation_config = ValidationConfig()

        config = DataProcessorConfig(
            cache_config=cache_config,
            storage_config=storage_config,
            validation_config=validation_config,
            max_concurrent_requests=20,
            request_timeout=60,
            retry_attempts=5,
            retry_delay=2.0
        )
        assert config.cache_config == cache_config
        assert config.storage_config == storage_config
        assert config.validation_config == validation_config
        assert config.max_concurrent_requests == 20
        assert config.request_timeout == 60
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0


class TestDataProcessorResult:
    """Test DataProcessorResult functionality"""

    def test_data_processor_result_defaults(self):
        """Test default data processor result"""
        result = DataProcessorResult()
        assert result.data is None
        assert result.success is False
        assert result.error_message is None
        assert result.metadata == {}
        assert result.processing_time is None

    def test_data_processor_result_with_data(self):
        """Test data processor result with data"""
        data = pd.DataFrame({'test': [1, 2, 3]})
        result = DataProcessorResult(
            data=data,
            success=True,
            error_message="No error",
            metadata={'symbol': 'AAPL'},
            processing_time=1.5
        )
        assert result.data is not None
        assert result.success is True
        assert result.error_message == "No error"
        assert result.metadata == {'symbol': 'AAPL'}
        assert result.processing_time == 1.5

    def test_data_processor_result_to_dict(self):
        """Test data processor result to dictionary conversion"""
        data = pd.DataFrame({'test': [1, 2, 3]})
        result = DataProcessorResult(
            data=data,
            success=True,
            error_message="No error",
            metadata={'symbol': 'AAPL'},
            processing_time=1.5
        )

        result_dict = result.to_dict()
        assert 'data' in result_dict
        assert 'success' in result_dict
        assert 'error_message' in result_dict
        assert 'metadata' in result_dict
        assert 'processing_time' in result_dict
        assert result_dict['success'] is True
        assert result_dict['error_message'] == "No error"


class TestDataMetrics:
    """Test DataMetrics functionality"""

    def test_data_metrics_initialization(self):
        """Test data metrics initialization"""
        metrics = DataMetrics()
        assert metrics.fetch_times == []
        assert metrics.process_times == []
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.validation_results == []
        assert metrics.error_counts == {}

    def test_record_fetch_time(self):
        """Test recording fetch time"""
        metrics = DataMetrics()
        metrics.record_fetch_time(1.5)
        metrics.record_fetch_time(2.0)
        assert len(metrics.fetch_times) == 2
        assert metrics.fetch_times[0] == 1.5
        assert metrics.fetch_times[1] == 2.0

    def test_record_process_time(self):
        """Test recording process time"""
        metrics = DataMetrics()
        metrics.record_process_time(0.5)
        metrics.record_process_time(1.0)
        assert len(metrics.process_times) == 2
        assert metrics.process_times[0] == 0.5
        assert metrics.process_times[1] == 1.0

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation"""
        metrics = DataMetrics()
        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        assert metrics.get_cache_hit_rate() == 2/3

    def test_success_rate(self):
        """Test validation success rate calculation"""
        metrics = DataMetrics()
        metrics.record_validation_result(ValidationResult.VALID)
        metrics.record_validation_result(ValidationResult.VALID)
        metrics.record_validation_result(ValidationResult.INVALID)
        assert metrics.get_success_rate() == 2/3

    def test_error_recording(self):
        """Test error recording"""
        metrics = DataMetrics()
        metrics.record_error('fetch_error')
        metrics.record_error('validation_error')
        metrics.record_error('fetch_error')
        assert metrics.error_counts == {
            'fetch_error': 2,
            'validation_error': 1
        }

    def test_metrics_to_dict(self):
        """Test metrics to dictionary conversion"""
        metrics = DataMetrics()
        metrics.record_fetch_time(1.5)
        metrics.record_process_time(0.5)
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        metrics.record_validation_result(ValidationResult.VALID)
        metrics.record_error('test_error')

        metrics_dict = metrics.to_dict()
        assert 'fetch_times' in metrics_dict
        assert 'process_times' in metrics_dict
        assert 'cache_hits' in metrics_dict
        assert 'cache_misses' in metrics_dict
        assert 'validation_results' in metrics_dict
        assert 'error_counts' in metrics_dict
        assert 'average_fetch_time' in metrics_dict
        assert 'average_process_time' in metrics_dict
        assert 'cache_hit_rate' in metrics_dict
        assert 'success_rate' in metrics_dict

        assert metrics_dict['cache_hits'] == 1
        assert metrics_dict['cache_misses'] == 1
        assert metrics_dict['cache_hit_rate'] == 1/2
        assert metrics_dict['success_rate'] == 1.0


class MockDataProcessor(DataProcessor):
    """Mock implementation of DataProcessor for testing"""

    def __init__(self, config=None):
        super().__init__(config)
        self.last_data = None

    async def fetch_data_async(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Mock async fetch"""
        self.last_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start_date, periods=3, freq='D'))
        return self.last_data

    def fetch_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Mock sync fetch"""
        return asyncio.run(self.fetch_data_async(symbol, start_date, end_date))

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock preprocessing"""
        return data

    async def preprocess_data_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock async preprocessing"""
        return data

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """Mock validation"""
        return ValidationResult.VALID


class MockDataCache(DataCache):
    """Mock implementation of DataCache for testing"""

    def __init__(self, config):
        super().__init__(config)
        self.cache_data = {}

    async def get_async(self, key: str):
        """Mock async get"""
        return self.cache_data.get(key)

    def get_sync(self, key: str):
        """Mock sync get"""
        return self.cache_data.get(key)

    async def set_async(self, key: str, data, ttl_seconds=None):
        """Mock async set"""
        self.cache_data[key] = data
        return True

    def set_sync(self, key: str, data, ttl_seconds=None):
        """Mock sync set"""
        self.cache_data[key] = data
        return True

    async def delete_async(self, key: str):
        """Mock async delete"""
        if key in self.cache_data:
            del self.cache_data[key]
            return True
        return False

    def delete_sync(self, key: str):
        """Mock sync delete"""
        if key in self.cache_data:
            del self.cache_data[key]
            return True
        return False

    async def clear_expired_async(self):
        """Mock async clear expired"""
        count = len(self.cache_data)
        self.cache_data.clear()
        return count

    def clear_expired_sync(self):
        """Mock sync clear expired"""
        count = len(self.cache_data)
        self.cache_data.clear()
        return count


class TestDataProcessor:
    """Test DataProcessor abstract base class"""

    def test_data_processor_initialization(self):
        """Test data processor initialization"""
        processor = MockDataProcessor()
        assert processor.config is None
        assert processor.logger is not None

    def test_data_processor_fetch_data(self):
        """Test data processor fetch data methods"""
        processor = MockDataProcessor()
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        # Test sync fetch
        sync_data = processor.fetch_data_sync("AAPL", start_date, end_date)
        assert not sync_data.empty
        assert len(sync_data) == 3

        # Test async fetch
        async def test_async():
            return await processor.fetch_data_async("AAPL", start_date, end_date)

        async_data = asyncio.run(test_async())
        assert not async_data.empty
        assert len(async_data) == 3

    def test_data_processor_preprocess_data(self):
        """Test data processor preprocessing methods"""
        processor = MockDataProcessor()
        data = pd.DataFrame({'test': [1, 2, 3]})

        # Test sync preprocess
        sync_result = processor.preprocess_data(data)
        assert not sync_result.empty

        # Test async preprocess
        async def test_async():
            return await processor.preprocess_data_async(data)

        async_result = asyncio.run(test_async())
        assert not async_result.empty

    def test_data_processor_validate_data(self):
        """Test data processor validation"""
        processor = MockDataProcessor()
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        result = processor.validate_data(data)
        assert result == ValidationResult.VALID


class MockDataStorage(DataStorage):
    """Mock implementation of DataStorage for testing"""

    def __init__(self, config):
        super().__init__(config)
        self.storage_data = {}

    async def store_async(self, symbol: str, data: pd.DataFrame):
        """Mock async store"""
        self.storage_data[symbol] = data
        return True

    def store_sync(self, symbol: str, data: pd.DataFrame):
        """Mock sync store"""
        self.storage_data[symbol] = data
        return True

    async def load_async(self, symbol: str, start_date=None, end_date=None):
        """Mock async load"""
        return self.storage_data.get(symbol, pd.DataFrame())

    def load_sync(self, symbol: str, start_date=None, end_date=None):
        """Mock sync load"""
        return self.storage_data.get(symbol, pd.DataFrame())

    async def delete_async(self, symbol: str):
        """Mock async delete"""
        if symbol in self.storage_data:
            del self.storage_data[symbol]
            return True
        return False

    def delete_sync(self, symbol: str):
        """Mock sync delete"""
        if symbol in self.storage_data:
            del self.storage_data[symbol]
            return True
        return False

    async def get_available_symbols_async(self):
        """Mock async get available symbols"""
        return list(self.storage_data.keys())

    def get_available_symbols_sync(self):
        """Mock sync get available symbols"""
        return list(self.storage_data.keys())


class TestAbstractImplementations:
    """Test that abstract implementations work correctly"""

    def test_mock_data_processor(self):
        """Test mock data processor"""
        processor = MockDataProcessor()
        config = DataProcessorConfig()
        processor_with_config = MockDataProcessor(config)

        # Test with configuration
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        data = processor_with_config.fetch_data_sync("AAPL", start_date, end_date)
        assert not data.empty
        assert processor_with_config.config == config

    def test_mock_data_cache(self):
        """Test mock data cache"""
        config = CacheConfig()
        cache = MockDataCache(config)

        # Test cache operations
        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Set and get
        assert cache.set_sync("test_key", test_data)
        retrieved_data = cache.get_sync("test_key")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        # Delete
        assert cache.delete_sync("test_key")
        assert cache.get_sync("test_key") is None

        # Clear expired
        assert cache.clear_expired_sync() == 0

    def test_mock_data_storage(self):
        """Test mock data storage"""
        config = StorageConfig()
        storage = MockDataStorage(config)

        # Test storage operations
        test_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        })

        # Store and load
        assert storage.store_sync("AAPL", test_data)
        retrieved_data = storage.load_sync("AAPL")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        # Delete
        assert storage.delete_sync("AAPL")
        assert storage.load_sync("AAPL").empty

        # Get available symbols
        storage.store_sync("MSFT", test_data)
        symbols = storage.get_available_symbols_sync()
        assert "AAPL" not in symbols
        assert "MSFT" in symbols

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async operations"""
        config = CacheConfig()
        cache = MockDataCache(config)

        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Test async cache operations
        assert await cache.set_async("async_key", test_data)
        retrieved_data = await cache.get_async("async_key")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        assert await cache.delete_async("async_key")
        assert await cache.get_async("async_key") is None

        # Test async storage operations
        storage_config = StorageConfig()
        storage = MockDataStorage(storage_config)

        assert await storage.store_async("async_symbol", test_data)
        retrieved_data = await storage.load_async("async_symbol")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        symbols = await storage.get_available_symbols_async()
        assert "async_symbol" in symbols


class TestIntegration:
    """Integration tests for core components"""

    def test_metrics_collection(self):
        """Test metrics collection across components"""
        config = DataProcessorConfig()
        processor = MockDataProcessor(config)
        metrics = DataMetrics()

        # Simulate some operations
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 3)

        # Fetch data
        data = processor.fetch_data_sync("AAPL", start_time, end_time)
        metrics.record_fetch_time(1.5)

        # Preprocess data
        processed_data = processor.preprocess_data(data)
        metrics.record_process_time(0.5)

        # Validate data
        result = processor.validate_data(data)
        metrics.record_validation_result(result)

        # Record cache operations
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        metrics.record_error("test_error")

        # Check metrics
        assert len(metrics.fetch_times) == 1
        assert len(metrics.process_times) == 1
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1
        assert len(metrics.validation_results) == 1
        assert "test_error" in metrics.error_counts

        # Convert to dict
        metrics_dict = metrics.to_dict()
        assert metrics_dict['cache_hit_rate'] == 1/2
        assert metrics_dict['success_rate'] == 1.0


class TestPatternConfig:
    """Test PatternConfig configuration class for enhanced system"""

    def test_pattern_config_defaults(self):
        """Test default values for PatternConfig"""
        config = PatternConfig()

        assert config.min_confidence == 0.6
        assert config.max_lookback == 100
        assert config.timeframe == "1d"
        assert config.volume_threshold == 1000000.0
        assert config.volatility_threshold == 0.001
        assert config.reward_ratio == 2.0

    def test_pattern_config_custom_values(self):
        """Test custom values for PatternConfig"""
        config = PatternConfig(
            min_confidence=0.8,
            max_lookback=200,
            timeframe="4h",
            volume_threshold=5000000.0,
            volatility_threshold=0.005,
            reward_ratio=3.0
        )

        assert config.min_confidence == 0.8
        assert config.max_lookback == 200
        assert config.timeframe == "4h"
        assert config.volume_threshold == 5000000.0
        assert config.volatility_threshold == 0.005
        assert config.reward_ratio == 3.0

    def test_pattern_config_serialization(self):
        """Test that PatternConfig can be serialized/deserialized"""
        config = PatternConfig()

        # Test to_dict conversion
        config_dict = config.__dict__.copy()
        assert 'min_confidence' in config_dict
        assert 'max_lookback' in config_dict

        # Test from dict reconstruction
        new_config = PatternConfig(**config_dict)
        assert new_config.min_confidence == config.min_confidence
        assert new_config.max_lookback == config.max_lookback


class TestPatternType:
    """Test PatternType enum for enhanced system"""

    def test_pattern_type_values(self):
        """Test all pattern type enum values"""
        assert PatternType.VCP_BREAKOUT.value == "vcp_breakout"
        assert PatternType.FLAG_PATTERN.value == "flag_pattern"
        assert PatternType.CUP_HANDLE.value == "cup_handle"
        assert PatternType.ASCENDING_TRIANGLE.value == "ascending_triangle"
        assert PatternType.DOUBLE_BOTTOM.value == "double_bottom"
        assert PatternType.WEDGE_PATTERN.value == "wedge_pattern"
        assert PatternType.HEAD_AND_SHOULDERS.value == "head_and_shoulders"
        assert PatternType.ROUNDING_BOTTOM.value == "rounding_bottom"
        assert PatternType.DESCENDING_TRIANGLE.value == "descending_triangle"
        assert PatternType.RISING_WEDGE.value == "rising_wedge"
        assert PatternType.FALLING_WEDGE.value == "falling_wedge"

    def test_pattern_type_unique_values(self):
        """Test that all pattern type values are unique"""
        values = [pt.value for pt in PatternType]
        assert len(values) == len(set(values)), "Pattern type values must be unique"


class TestPatternSignal:
    """Test PatternSignal dataclass for enhanced system"""

    def test_pattern_signal_creation(self):
        """Test creation of PatternSignal instances"""
        signal = PatternSignal(
            symbol="AAPL",
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=0.85,
            entry_price=150.0,
            stop_loss=145.0,
            target_price=170.0,
            timeframe="1d",
            timestamp=datetime.now(),
            metadata={"test": "value"}
        )

        assert signal.symbol == "AAPL"
        assert signal.pattern_type == PatternType.VCP_BREAKOUT
        assert signal.confidence == 0.85
        assert signal.entry_price == 150.0
        assert signal.stop_loss == 145.0
        assert signal.target_price == 170.0
        assert signal.timeframe == "1d"
        assert isinstance(signal.timestamp, datetime)
        assert signal.metadata == {"test": "value"}
        assert signal.signal_strength == 0.0
        assert signal.risk_level == "medium"
        assert signal.expected_duration is None
        assert signal.probability_target is None

    def test_pattern_signal_optional_fields(self):
        """Test PatternSignal with optional fields"""
        signal = PatternSignal(
            symbol="MSFT",
            pattern_type=PatternType.FLAG_PATTERN,
            confidence=0.75,
            entry_price=300.0,
            stop_loss=295.0,
            target_price=330.0,
            timeframe="1d",
            timestamp=datetime.now(),
            metadata={},
            signal_strength=0.8,
            risk_level="low",
            expected_duration="2-3 weeks",
            probability_target=0.85
        )

        assert signal.signal_strength == 0.8
        assert signal.risk_level == "low"
        assert signal.expected_duration == "2-3 weeks"
        assert signal.probability_target == 0.85


class TestDataValidator:
    """Test DataValidator validation functionality for enhanced system"""

    def create_test_data(self, rows: int = 50) -> pd.DataFrame:
        """Create test market data"""
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

    def test_valid_data_validation(self):
        """Test validation of valid market data"""
        data = self.create_test_data()

        result = DataValidator.validate_price_data_safe(data)
        assert result is True

        # Test strict validation
        try:
            DataValidator.validate_price_data(data)
            assert True  # Should not raise exception
        except ValueError:
            pytest.fail("Valid data should not raise ValueError")

    def test_invalid_data_validation(self):
        """Test validation of invalid market data"""
        # Create invalid data
        data = self.create_test_data()
        data.loc[data.index[0], 'close'] = -10.0  # Negative price

        result = DataValidator.validate_price_data_safe(data)
        assert result is False

        # Test strict validation
        with pytest.raises(ValueError, match="non-positive values"):
            DataValidator.validate_price_data(data)

    def test_missing_columns_validation(self):
        """Test validation with missing required columns"""
        data = self.create_test_data()
        data = data.drop('volume', axis=1)  # Remove volume column

        result = DataValidator.validate_price_data_safe(data)
        assert result is False

        with pytest.raises(ValueError, match="Missing required columns"):
            DataValidator.validate_price_data(data)

    def test_nan_values_validation(self):
        """Test validation with NaN values"""
        data = self.create_test_data()
        data.loc[data.index[10:15], 'close'] = np.nan  # Add NaN values

        result = DataValidator.validate_price_data_safe(data)
        assert result is False

        with pytest.raises(ValueError, match="contains NaN values"):
            DataValidator.validate_price_data(data)

    def test_ohlc_inconsistency_validation(self):
        """Test validation with OHLC inconsistencies"""
        data = self.create_test_data()
        data.loc[data.index[0], 'high'] = 50.0  # High < Low
        data.loc[data.index[1], 'low'] = 200.0  # Low > High

        result = DataValidator.validate_price_data_safe(data)
        assert result is False

        with pytest.raises(ValueError, match="High prices cannot be lower"):
            DataValidator.validate_price_data(data)

    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        # Create corrupted data
        data = self.create_test_data()
        data.loc[data.index[0], 'high'] = 50.0  # Inconsistent high
        data.loc[data.index[1], 'low'] = 200.0  # Inconsistent low

        cleaned_data = DataValidator.clean_ohlc_data(data)

        # Check that inconsistencies were fixed
        assert cleaned_data.loc[cleaned_data.index[0], 'high'] >= cleaned_data.loc[cleaned_data.index[0], 'close']
        assert cleaned_data.loc[cleaned_data.index[1], 'low'] <= cleaned_data.loc[cleaned_data.index[1], 'close']


class TestEnhancedPatternDetector:
    """Test EnhancedPatternDetector base class"""

    def create_test_detector(self):
        """Create a test implementation of EnhancedPatternDetector"""

        class TestDetector(EnhancedPatternDetector):
            def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
                # Simple test detection
                signals = []
                if len(data) > 10:
                    signals.append(PatternSignal(
                        symbol="AAPL",
                        pattern_type=PatternType.VCP_BREAKOUT,
                        confidence=min(1.0, len(data) / 100.0),
                        entry_price=data.iloc[-1]['close'],
                        stop_loss=data.iloc[-1]['close'] * 0.95,
                        target_price=data.iloc[-1]['close'] * 1.2,
                        timeframe=self.config.timeframe,
                        timestamp=data.index[-1],
                        metadata={"test": True}
                    ))
                return signals

            def get_required_columns(self) -> List[str]:
                return ["open", "high", "low", "close", "volume"]

        config = PatternConfig(min_confidence=0.5, max_lookback=20)
        return TestDetector(config)

    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = self.create_test_detector()

        assert detector.config.min_confidence == 0.5
        assert detector.config.max_lookback == 20
        assert detector.config.timeframe == "1d"
        assert hasattr(detector, 'logger')
        assert hasattr(detector, 'volatility_analyzer')
        assert hasattr(detector, 'volume_analyzer')
        assert hasattr(detector, 'trend_analyzer')

    def test_detector_data_validation(self):
        """Test detector data validation"""
        detector = self.create_test_detector()

        # Create valid test data
        valid_data = self.create_test_data()
        assert detector.validate_data(valid_data) is True

        # Test invalid data
        invalid_data = pd.DataFrame()
        assert detector.validate_data(invalid_data) is False

        # Test data with missing columns
        incomplete_data = valid_data.drop('volume', axis=1)
        assert detector.validate_data(incomplete_data) is False

    def test_detector_signal_generation(self):
        """Test detector signal generation"""
        detector = self.create_test_detector()
        data = self.create_test_data(rows=25)  # Sufficient data

        signals = detector.detect_pattern(data)

        assert len(signals) > 0
        assert all(isinstance(signal, PatternSignal) for signal in signals)
        assert all(signal.symbol == "AAPL" for signal in signals)
        assert all(signal.pattern_type == PatternType.VCP_BREAKOUT for signal in signals)
        assert all(signal.confidence >= detector.config.min_confidence for signal in signals)

    def test_detector_confidence_calculation(self):
        """Test detector confidence calculation"""
        detector = self.create_test_detector()

        # Test confidence calculation with various pattern data
        pattern_data = {
            "volume_ratio": 2.0,
            "volatility_score": 0.03,
            "trend_strength": 0.8
        }

        confidence = detector.calculate_confidence(pattern_data)

        # Confidence should be between 0 and 1
        assert 0.5 <= confidence <= 1.0  # Base confidence of 0.5 + additions

    def test_detector_required_columns(self):
        """Test detector required columns"""
        detector = self.create_test_detector()

        columns = detector.get_required_columns()

        assert isinstance(columns, list)
        assert "open" in columns
        assert "high" in columns
        assert "low" in columns
        assert "close" in columns
        assert "volume" in columns


class TestAPIConfig:
    """Test APIConfig for enhanced async manager"""

    def test_api_config_creation(self):
        """Test APIConfig creation"""
        config = APIConfig(
            name="test_api",
            base_url="https://api.test.com",
            timeout=30,
            rate_limit=5,
            retry_attempts=3,
            retry_delay=1.0,
            api_key="test_key"
        )

        assert config.name == "test_api"
        assert config.base_url == "https://api.test.com"
        assert config.timeout == 30
        assert config.rate_limit == 5
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.api_key == "test_key"

    def test_api_config_defaults(self):
        """Test APIConfig defaults"""
        config = APIConfig(name="test_api", base_url="http://test.com")

        assert config.name == "test_api"
        assert config.base_url == "http://test.com"
        assert config.timeout == 30
        assert config.rate_limit == 5
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.api_key is None


def create_test_data(rows: int = 50) -> pd.DataFrame:
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
    pytest.main([__file__, "-v"])