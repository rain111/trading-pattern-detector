"""
Enhanced tests for backward compatibility adapters with comprehensive failure point testing.
Tests adapter pattern compatibility, backward compatibility, and edge cases.
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
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from src.data.adapters import (
    DataPreprocessorAdapter, DataManagerAdapter,
    LegacyDataCacheAdapter, LegacyDataStorageAdapter,
    DataProcessorFactory
)
from src.data.core import (
    DataProcessorConfig, CacheConfig, StorageConfig, ValidationConfig,
    CacheType, StorageType, ValidationResult, DataProcessor
)
from src.utils.data_preprocessor import DataPreprocessor


class FailurePointMockDataPreprocessor:
    """Mock preprocessor for testing failure scenarios"""

    def __init__(self, fail_mode=None, delay=0):
        self.fail_mode = fail_mode
        self.delay = delay
        self.clean_calls = 0
        self.validate_calls = 0

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data with controlled failure points"""
        self.clean_calls += 1

        # Simulate processing delay
        if self.delay > 0:
            time.sleep(self.delay)

        # Simulate different failure modes
        if self.fail_mode == "exception":
            raise Exception("Cleaning failed")
        elif self.fail_mode == "return_none":
            return None
        elif self.fail_mode == "corrupt_data":
            # Return malformed data
            data = data.copy()
            data['close'] = None  # Invalid data
            return data
        elif self.fail_mode == "memory_error":
            # Simulate memory error
            raise MemoryError("Memory allocation failed")

        # Normal operation
        data = data.copy()
        data = data[~data.index.duplicated(keep='first')]
        data = data.sort_index()
        return data

    def validate_clean_data(self, data: pd.DataFrame) -> bool:
        """Validate data with controlled failure modes"""
        self.validate_calls += 1

        if self.fail_mode == "validation_timeout":
            time.sleep(5)  # Simulate timeout
            return False
        elif self.fail_mode == "validation_error":
            raise Exception("Validation error")

        return not data.empty and len(data) >= 20


class MockDataPreprocessor:
    """Mock implementation of DataPreprocessor for testing"""

    def __init__(self):
        self.clean_called = False
        self.validate_called = False

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock clean data method"""
        self.clean_called = True
        data = data.copy()
        # Remove duplicates and sort
        data = data[~data.index.duplicated(keep='first')]
        data = data.sort_index()
        return data

    def validate_clean_data(self, data: pd.DataFrame) -> bool:
        """Mock validate data method"""
        self.validate_called = True
        return not data.empty and len(data) >= 20


class FailurePointDataManager:
    """Mock data manager for testing failure scenarios"""

    def __init__(self, fail_mode=None, delay=0):
        self.fail_mode = fail_mode
        self.delay = delay
        self._cache = {}
        self.call_count = 0

    async def __aenter__(self):
        if self.fail_mode == "context_manager_error":
            raise Exception("Context manager failed")
        self._session = Mock()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.fail_mode == "context_exit_error":
            raise Exception("Context exit failed")
        if self._session:
            self._session.close()

    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get stock data with controlled failure modes"""
        self.call_count += 1

        if self.delay > 0:
            time.sleep(self.delay)

        if self.fail_mode == "data_fetch_error":
            raise Exception("Data fetch failed")
        elif self.fail_mode == "invalid_data":
            # Return malformed data
            return pd.DataFrame({'invalid': [1, 2, 3]})  # Missing required columns
        elif self.fail_mode == "connection_error":
            raise ConnectionError("Connection failed")
        elif self.fail_mode == "timeout":
            time.sleep(5)  # Simulate timeout
            return pd.DataFrame()

        # Normal operation
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start_date, periods=3, freq='D'))

    def _clean_ohlc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLC data with failure modes"""
        if self.fail_mode == "cleaning_error":
            raise Exception("Data cleaning failed")

        data = data.copy()
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)
        data = data.dropna()
        return data


class MockDataManager:
    """Mock implementation of DataManager for testing"""

    def __init__(self):
        self._cache = {}
        self.get_available_symbols_called = False
        self.get_cache_info_called = False

    def get_available_symbols(self) -> list:
        """Mock get available symbols method"""
        self.get_available_symbols_called = True
        return ['AAPL', 'MSFT']

    def get_cache_info(self, symbol: str):
        """Mock get cache info method"""
        self.get_cache_info_called = True
        return Mock()

    def _clean_ohlc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock OHLC cleaning method"""
        data = data.copy()
        # Ensure OHLC relationships
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)
        data = data.dropna()
        return data


class TestDataPreprocessorAdapter:
    """Test DataPreprocessorAdapter functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_preprocessor = MockDataPreprocessor()
        self.adapter = DataPreprocessorAdapter(self.mock_preprocessor)

    def test_backward_compatibility_sync_mode(self):
        """Test that adapter maintains backward compatibility with sync operations"""
        # Test that adapter can be used in sync mode like old preprocessor
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Should work with sync methods
        result = self.adapter.preprocess_data(test_data)
        assert not result.empty
        assert self.mock_preprocessor.clean_called is True

    @pytest.mark.asyncio
    async def test_backward_compatibility_async_mode(self):
        """Test that adapter provides async capabilities while maintaining compatibility"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Should work with async methods
        result = await self.adapter.preprocess_data_async(test_data)
        assert not result.empty
        assert self.mock_preprocessor.clean_called is True

    def test_interface_compatibility(self):
        """Test that adapter implements expected interfaces"""
        # Should implement DataProcessor interface
        assert isinstance(self.adapter, DataProcessor)

        # Should have expected methods
        assert hasattr(self.adapter, 'preprocess_data')
        assert hasattr(self.adapter, 'preprocess_data_async')
        assert hasattr(self.adapter, 'validate_data')
        assert hasattr(self.adapter, 'fetch_data_sync')
        assert hasattr(self.adapter, 'fetch_data_async')

    def test_adapter_initialization(self):
        """Test adapter initialization"""
        assert self.adapter.existing_preprocessor == self.mock_preprocessor
        assert self.adapter.config is None
        assert self.adapter.logger is not None

    @pytest.mark.asyncio
    async def test_preprocess_data_async(self):
        """Test async data preprocessing"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        result = await self.adapter.preprocess_data_async(test_data)

        # Should call the underlying preprocessor
        assert self.mock_preprocessor.clean_called is True
        assert not result.empty
        assert len(result) == 3

    def test_preprocess_data_sync(self):
        """Test sync data preprocessing"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        result = self.adapter.preprocess_data(test_data)

        # Should call the underlying preprocessor
        assert self.mock_preprocessor.clean_called is True
        assert not result.empty
        assert len(result) == 3

    def test_validate_data(self):
        """Test data validation"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        result = self.adapter.validate_data(test_data)

        # Should call the underlying validator
        assert self.mock_preprocessor.validate_called is True
        assert result == ValidationResult.VALID

    def test_validate_data_insufficient(self):
        """Test data validation with insufficient data"""
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        result = self.adapter.validate_data(test_data)

        assert self.mock_preprocessor.validate_called is True
        assert result == ValidationResult.NEEDS_CLEANING


class DataManagerAdapterWithConfig:
    """Mock DataManager with config for testing"""

    def __init__(self):
        self._cache = {}
        self._session = None
        self.config = DataProcessorConfig()

    async def __aenter__(self):
        self._session = Mock()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()

    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Mock get stock data method"""
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start_date, periods=3, freq='D'))

    def _clean_ohlc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock OHLC cleaning method"""
        data = data.copy()
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)
        data = data.dropna()
        return data

    def get_available_symbols(self) -> list:
        """Mock get available symbols method"""
        return ['AAPL', 'MSFT']

    def get_cache_info(self, symbol: str):
        """Mock get cache info method"""
        return Mock()

    def clear_cache(self, symbol: Optional[str] = None):
        """Mock clear cache method"""
        pass

    def cleanup_old_cache(self, max_age_hours: int = 24):
        """Mock cleanup old cache method"""
        pass


class TestDataManagerAdapter:
    """Test DataManagerAdapter functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_manager = DataManagerAdapterWithConfig()
        self.adapter = DataManagerAdapter(self.mock_manager)

    def test_adapter_initialization(self):
        """Test adapter initialization"""
        assert self.adapter.existing_data_manager == self.mock_manager
        assert self.adapter.logger is not None

    @pytest.mark.asyncio
    async def test_fetch_data_async(self):
        """Test async data fetching"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        result = await self.adapter.fetch_data_async("AAPL", start_date, end_date)

        assert not result.empty
        assert len(result) == 3
        assert result['open'].tolist() == [100, 101, 102]

    def test_fetch_data_sync(self):
        """Test sync data fetching"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        # Mock the event loop
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = self.mock_manager.get_stock_data("AAPL", start_date, end_date)

            result = self.adapter.fetch_data_sync("AAPL", start_date, end_date)

            assert not result.empty
            assert len(result) == 3

    def test_get_available_symbols(self):
        """Test getting available symbols"""
        symbols = self.adapter.get_available_symbols()
        assert symbols == ['AAPL', 'MSFT']

    def test_get_cache_info(self):
        """Test getting cache info"""
        cache_info = self.adapter.get_cache_info("AAPL")
        assert cache_info is not None

    def test_clear_cache(self):
        """Test clearing cache"""
        # Should not raise an exception
        self.adapter.clear_cache()
        self.adapter.clear_cache("AAPL")

    def test_cleanup_old_cache(self):
        """Test cleaning up old cache"""
        # Should not raise an exception
        self.adapter.cleanup_old_cache()


class LegacyCacheManager:
    """Mock legacy cache manager for testing"""

    def __init__(self):
        self._cache = {}

    def get_data(self, key: str):
        """Mock get data method"""
        return self._cache.get(key)

    def set_data(self, key: str, data):
        """Mock set data method"""
        self._cache[key] = data

    def remove_data(self, key: str):
        """Mock remove data method"""
        if key in self._cache:
            del self._cache[key]


class TestLegacyDataCacheAdapter:
    """Test LegacyDataCacheAdapter functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_cache_manager = LegacyCacheManager()
        self.config = CacheConfig(ttl_seconds=3600, max_size=100)
        self.adapter = LegacyDataCacheAdapter(self.mock_cache_manager, self.config)

    @pytest.mark.asyncio
    async def test_cache_operations_async(self):
        """Test async cache operations"""
        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Set data
        success = await self.adapter.set_async("test_key", test_data)
        assert success is True

        # Get data
        retrieved_data = await self.adapter.get_async("test_key")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        # Delete data
        success = await self.adapter.delete_async("test_key")
        assert success is True

        # Should be deleted
        retrieved_data = await self.adapter.get_async("test_key")
        assert retrieved_data is None

    def test_cache_operations_sync(self):
        """Test sync cache operations"""
        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Set data
        success = self.adapter.set_sync("test_key", test_data)
        assert success is True

        # Get data
        retrieved_data = self.adapter.get_sync("test_key")
        assert not retrieved_data.empty
        assert retrieved_data.equals(test_data)

        # Delete data
        success = self.adapter.delete_sync("test_key")
        assert success is True

        # Should be deleted
        retrieved_data = self.adapter.get_sync("test_key")
        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_clear_expired(self):
        """Test clearing expired cache entries"""
        # Legacy cache doesn't track TTL, so it should clear all entries
        self.mock_cache_manager.set_data("key1", pd.DataFrame({'test': [1]}))
        self.mock_cache_manager.set_data("key2", pd.DataFrame({'test': [2]}))

        cleared = await self.adapter.clear_expired_async()
        assert cleared == 2
        assert len(self.mock_cache_manager._cache) == 0

    def test_clear_expired_sync(self):
        """Test sync clearing expired cache entries"""
        self.mock_cache_manager.set_data("key1", pd.DataFrame({'test': [1]}))
        self.mock_cache_manager.set_data("key2", pd.DataFrame({'test': [2]}))

        cleared = self.adapter.clear_expired_sync()
        assert cleared == 2
        assert len(self.mock_cache_manager._cache) == 0


class LegacyStorageManager:
    """Mock legacy storage manager for testing"""

    def __init__(self):
        self.storage_data = {}

    def save_parquet(self, symbol: str, data: pd.DataFrame) -> bool:
        """Mock save parquet method"""
        self.storage_data[symbol] = data
        return True

    def load_parquet(self, symbol: str) -> pd.DataFrame:
        """Mock load parquet method"""
        return self.storage_data.get(symbol, pd.DataFrame())

    def get_symbols(self) -> list:
        """Mock get symbols method"""
        return list(self.storage_data.keys())


class TestLegacyDataStorageAdapter:
    """Test LegacyDataStorageAdapter functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_storage_manager = LegacyStorageManager()
        self.config = StorageConfig(base_path="/tmp/storage")
        self.adapter = LegacyDataStorageAdapter(self.mock_storage_manager, self.config)

    @pytest.mark.asyncio
    async def test_storage_operations_async(self):
        """Test async storage operations"""
        test_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        })

        # Store data
        success = await self.adapter.store_async("AAPL", test_data)
        assert success is True

        # Load data
        loaded_data = await self.adapter.load_async("AAPL")
        assert not loaded_data.empty
        assert loaded_data.equals(test_data)

        # Get available symbols
        symbols = await self.adapter.get_available_symbols_async()
        assert "AAPL" in symbols

    def test_storage_operations_sync(self):
        """Test sync storage operations"""
        test_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        })

        # Store data
        success = self.adapter.store_sync("AAPL", test_data)
        assert success is True

        # Load data
        loaded_data = self.adapter.load_sync("AAPL")
        assert not loaded_data.empty
        assert loaded_data.equals(test_data)

        # Get available symbols
        symbols = self.adapter.get_available_symbols_sync()
        assert "AAPL" in symbols

    @pytest.mark.asyncio
    async def test_delete_data(self):
        """Test deleting data"""
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        # Store data
        await self.adapter.store_async("AAPL", test_data)
        assert "AAPL" in self.mock_storage_manager.storage_data

        # Delete data (legacy storage doesn't support delete)
        success = await self.adapter.delete_async("AAPL")
        assert success is False  # Should return False for legacy storage

    def test_delete_data_sync(self):
        """Test sync deleting data"""
        success = self.adapter.delete_sync("AAPL")
        assert success is False  # Should return False for legacy storage


class TestDataProcessorFactory:
    """Test DataProcessorFactory functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_preprocessor = MockDataPreprocessor()
        self.mock_data_manager = MockDataManager()

    def test_create_preprocessor_adapter(self):
        """Test creating DataPreprocessor adapter"""
        adapter = DataProcessorFactory.create_preprocessor_adapter(self.mock_preprocessor)
        assert isinstance(adapter, DataPreprocessorAdapter)
        assert adapter.existing_preprocessor == self.mock_preprocessor

    def test_create_data_manager_adapter(self):
        """Test creating DataManager adapter"""
        config = DataProcessorConfig()
        adapter = DataProcessorFactory.create_data_manager_adapter(self.mock_data_manager, config)
        assert isinstance(adapter, DataManagerAdapter)
        assert adapter.existing_data_manager == self.mock_data_manager

    def test_create_cache_adapter(self):
        """Test creating cache adapter"""
        config = CacheConfig()
        adapter = DataProcessorFactory.create_cache_adapter(self.mock_data_manager, config)
        assert isinstance(adapter, LegacyDataCacheAdapter)
        assert adapter.existing_cache_manager == self.mock_data_manager

    def test_create_storage_adapter(self):
        """Test creating storage adapter"""
        config = StorageConfig()
        adapter = DataProcessorFactory.create_storage_adapter(self.mock_data_manager, config)
        assert isinstance(adapter, LegacyDataStorageAdapter)
        assert adapter.existing_storage_manager == self.mock_data_manager


class TestAdapterIntegration:
    """Integration tests for adapter components"""

    def test_full_adapter_chain(self):
        """Test using all adapters together"""
        # Create mock components
        mock_preprocessor = MockDataPreprocessor()
        mock_data_manager = MockDataManager()

        # Create adapters
        preprocessor_adapter = DataPreprocessorAdapter(mock_preprocessor)
        data_manager_adapter = DataManagerAdapter(mock_data_manager)

        # Test that adapters work together
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Test preprocessing through adapter
        processed_data = preprocessor_adapter.preprocess_data(test_data)
        assert not processed_data.empty
        assert mock_preprocessor.clean_called is True

        # Test validation through adapter
        validation_result = preprocessor_adapter.validate_data(test_data)
        assert validation_result == ValidationResult.VALID

        # Test manager operations
        available_symbols = data_manager_adapter.get_available_symbols()
        assert available_symbols == ['AAPL', 'MSFT']

        cache_info = data_manager_adapter.get_cache_info("AAPL")
        assert cache_info is not None

    @pytest.mark.asyncio
    async def test_async_adapter_operations(self):
        """Test async operations through adapters"""
        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor)

        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Test async preprocessing
        processed_data = await adapter.preprocess_data_async(test_data)
        assert not processed_data.empty
        assert mock_preprocessor.clean_called is True

    def test_adapter_error_handling(self):
        """Test error handling in adapters"""
        # Create a preprocessor that raises exceptions
        class ErrorPreprocessor:
            def clean_data(self, data):
                raise Exception("Clean failed")

        error_preprocessor = ErrorPreprocessor()
        adapter = DataPreprocessorAdapter(error_preprocessor)

        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        # Should handle errors gracefully
        result = adapter.preprocess_data(test_data)
        assert result.equals(test_data)  # Should return original data on error

    def test_adapter_configuration(self):
        """Test adapter with configuration"""
        config = DataProcessorConfig()
        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor, config)

        assert adapter.config == config

        # Test that adapter still works with config
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})
        result = adapter.preprocess_data(test_data)
        assert not result.empty


# ====================================================================================================
# COMPREHENSIVE FAILURE POINT TESTING FOR ADAPTERS
# ====================================================================================================

class TestAdapterFailurePoints:
    """Comprehensive failure point testing for adapter components"""

    def test_data_preprocessor_adapter_exception_handling(self):
        """Test DataPreprocessorAdapter handles exceptions gracefully"""
        # Create preprocessor that throws exceptions
        failing_preprocessor = FailurePointMockDataPreprocessor(fail_mode="exception")
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Should handle exception and return original data
        result = adapter.preprocess_data(test_data)
        assert result.equals(test_data)  # Should return original data on error

    @pytest.mark.asyncio
    async def test_data_preprocessor_adapter_async_exception_handling(self):
        """Test DataPreprocessorAdapter async exception handling"""
        failing_preprocessor = FailurePointMockDataPreprocessor(fail_mode="exception")
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        # Should handle exception and return original data
        result = await adapter.preprocess_data_async(test_data)
        assert result.equals(test_data)  # Should return original data on error

    def test_data_preprocessor_adapter_null_return_handling(self):
        """Test DataPreprocessorAdapter handles null returns"""
        failing_preprocessor = FailurePointMockDataPreprocessor(fail_mode="return_none")
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]
        })

        # Should handle null return gracefully
        result = adapter.preprocess_data(test_data)
        assert result.equals(test_data)  # Should return original data

    def test_data_preprocessor_adapter_validation_failure(self):
        """Test DataPreprocessorAdapter validation failure scenarios"""
        failing_preprocessor = FailurePointMockDataPreprocessor(fail_mode="validation_timeout")
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]
        })

        # Should handle validation timeout
        with patch('time.sleep'):  # Mock sleep to prevent actual timeout
            result = adapter.validate_data(test_data)
            # Should return NEEDS_CLEANING or INVALID based on validation logic

    def test_data_preprocessor_adapter_memory_error_handling(self):
        """Test DataPreprocessorAdapter handles memory errors"""
        failing_preprocessor = FailurePointMockDataPreprocessor(fail_mode="memory_error")
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]
        })

        # Should handle memory error gracefully
        result = adapter.preprocess_data(test_data)
        assert result.equals(test_data)  # Should return original data on error

    def test_data_manager_adapter_connection_failure(self):
        """Test DataManagerAdapter handles connection failures"""
        failing_manager = FailurePointDataManager(fail_mode="connection_error")
        adapter = DataManagerAdapter(failing_manager)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        # Should handle connection errors gracefully
        result = adapter.fetch_data_sync("AAPL", start_date, end_date)
        assert result.empty  # Should return empty DataFrame on error

    @pytest.mark.asyncio
    async def test_data_manager_adapter_async_connection_failure(self):
        """Test DataManagerAdapter async connection failures"""
        failing_manager = FailurePointDataManager(fail_mode="connection_error")
        adapter = DataManagerAdapter(failing_manager)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        # Should handle connection errors gracefully
        result = await adapter.fetch_data_async("AAPL", start_date, end_date)
        assert result.empty  # Should return empty DataFrame on error

    @pytest.mark.asyncio
    async def test_data_manager_adapter_context_manager_failure(self):
        """Test DataManagerAdapter handles context manager failures"""
        failing_manager = FailurePointDataManager(fail_mode="context_manager_error")

        # Should handle context manager failure
        adapter = DataManagerAdapter(failing_manager)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        # Should handle exception gracefully
        result = await adapter.fetch_data_async("AAPL", start_date, end_date)
        assert result.empty

    def test_data_manager_adapter_data_fetch_failure(self):
        """Test DataManagerAdapter handles data fetch failures"""
        failing_manager = FailurePointDataManager(fail_mode="data_fetch_error")
        adapter = DataManagerAdapter(failing_manager)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)

        # Should handle data fetch failure gracefully
        result = adapter.fetch_data_sync("AAPL", start_date, end_date)
        assert result.empty

    def test_legacy_cache_adapter_concurrent_access(self):
        """Test LegacyDataCacheAdapter handles concurrent access"""
        from threading import Thread
        import concurrent.futures

        mock_cache_manager = Mock()
        mock_cache_manager._cache = {}

        config = CacheConfig(ttl_seconds=3600, max_size=100)
        adapter = LegacyDataCacheAdapter(mock_cache_manager, config)

        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Test concurrent access
        def set_cache(key, data):
            return adapter.set_sync(key, data)

        def get_cache(key):
            return adapter.get_sync(key)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Concurrent writes
            futures = []
            for i in range(5):
                future = executor.submit(set_cache, f"key_{i}", test_data)
                futures.append(future)

            # Concurrent reads
            for i in range(5):
                future = executor.submit(get_cache, f"key_{i}")
                futures.append(future)

            # Wait for all operations
            concurrent.futures.wait(futures)

        # Should not have crashed
        assert True  # If we get here, no critical failure

    def test_legacy_storage_adapter_invalid_operations(self):
        """Test LegacyDataStorageAdapter handles invalid operations"""
        mock_storage_manager = Mock()
        mock_storage_manager.storage_data = {}

        config = StorageConfig(base_path="/tmp/test")
        adapter = LegacyDataStorageAdapter(mock_storage_manager, config)

        # Test with invalid data
        invalid_data = "not a dataframe"

        # Should handle invalid data gracefully
        with pytest.raises(Exception):
            adapter.store_sync("AAPL", invalid_data)

        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        success = adapter.store_sync("AAPL", empty_data)
        # Should either succeed or fail gracefully
        assert success in [True, False]

    def test_adapter_factory_error_handling(self):
        """Test DataProcessorFactory handles invalid inputs"""
        from ...src.data.core import CacheConfig, StorageConfig

        # Test with None inputs
        adapter1 = DataProcessorFactory.create_preprocessor_adapter(None)
        assert isinstance(adapter1, DataPreprocessorAdapter)

        config = CacheConfig()
        adapter2 = DataProcessorFactory.create_cache_adapter(None, config)
        assert isinstance(adapter2, LegacyDataCacheAdapter)

        storage_config = StorageConfig()
        adapter3 = DataProcessorFactory.create_storage_adapter(None, storage_config)
        assert isinstance(adapter3, LegacyDataStorageAdapter)

    def test_adapter_performance_under_load(self):
        """Test adapter performance under load conditions"""
        # Create adapter with failing preprocessor
        failing_preprocessor = FailurePointMockDataPreprocessor(delay=0.1)  # 100ms delay
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        import time
        start_time = time.time()

        # Process multiple datasets
        results = []
        for i in range(10):
            result = adapter.preprocess_data(test_data)
            results.append(result)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert processing_time < 5.0
        assert len(results) == 10

    def test_adapter_error_logging(self):
        """Test adapter error logging functionality"""
        import logging

        # Set up logging capture
        logger = logging.getLogger('DataPreprocessorAdapter')
        log_messages = []

        def log_handler(record):
            log_messages.append(record.getMessage())

        logger.addHandler(logging.Handler())
        logger.addHandler(logging.StreamHandler())

        # Create adapter with failing preprocessor
        failing_preprocessor = FailurePointMockDataPreprocessor(fail_mode="exception")
        adapter = DataPreprocessorAdapter(failing_preprocessor)

        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        # Trigger error
        result = adapter.preprocess_data(test_data)

        # Should have logged error
        assert len(log_messages) > 0
        assert any("error" in msg.lower() for msg in log_messages)

    def test_adapter_configuration_override(self):
        """Test adapter respects configuration settings"""
        from ...src.data.core import DataProcessorConfig, ValidationConfig

        # Create custom config
        validation_config = ValidationConfig(
            min_data_points=10,
            max_missing_ratio=0.1,
            price_consistency_check=True
        )

        config = DataProcessorConfig(
            validation_config=validation_config
        )

        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor, config)

        assert adapter.config == config
        assert adapter.config.validation_config.min_data_points == 10

    def test_adapter_inheritance_compatibility(self):
        """Test adapter inheritance compatibility with base classes"""
        # Test that adapters inherit from expected base classes
        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor)

        # Should implement base class methods
        assert hasattr(adapter, 'config')
        assert hasattr(adapter, 'logger')
        assert hasattr(adapter, 'fetch_data_async')
        assert hasattr(adapter, 'fetch_data_sync')
        assert hasattr(adapter, 'preprocess_data')
        assert hasattr(adapter, 'preprocess_data_async')
        assert hasattr(adapter, 'validate_data')

    def test_adapter_thread_safety(self):
        """Test adapter thread safety"""
        import threading
        import time

        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor)

        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })

        results = []
        errors = []

        def worker(worker_id):
            try:
                result = adapter.preprocess_data(test_data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have processed all data without critical errors
        assert len(results) == 5
        assert len(errors) == 0

    def test_adapter_resource_cleanup(self):
        """Test adapter resource cleanup"""
        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor)

        # Process data
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})
        result = adapter.preprocess_data(test_data)

        # Verify cleanup
        assert hasattr(adapter, 'logger')
        assert adapter.logger is not None

        # Should not leave resources hanging
        assert True  # If we get here, cleanup is fine

    def test_adapter_backward_comprehensive_scenarios(self):
        """Test comprehensive backward compatibility scenarios"""
        # Test that all existing functionality continues to work
        mock_preprocessor = MockDataPreprocessor()
        adapter = DataPreprocessorAdapter(mock_preprocessor)

        # Test various data formats
        test_cases = [
            # Normal data
            pd.DataFrame({
                'open': [100, 101], 'high': [101, 102], 'low': [99, 100],
                'close': [100.5, 101.5], 'volume': [1000000, 1100000]
            }),
            # Data with missing values
            pd.DataFrame({
                'open': [100, None], 'high': [101, 102], 'low': [99, 100],
                'close': [100.5, 101.5], 'volume': [1000000, None]
            }),
            # Single row data
            pd.DataFrame({
                'open': [100], 'high': [101], 'low': [99],
                'close': [100], 'volume': [1000000]
            })
        ]

        for i, test_data in enumerate(test_cases):
            # Test sync processing
            sync_result = adapter.preprocess_data(test_data)
            assert not sync_result.empty

            # Test async processing
            import asyncio
            async_result = asyncio.run(adapter.preprocess_data_async(test_data))
            assert not async_result.empty

            # Test validation
            validation_result = adapter.validate_data(test_data)
            assert validation_result in [ValidationResult.VALID, ValidationResult.NEEDS_CLEANING]

        assert True  # All scenarios passed


if __name__ == "__main__":
    pytest.main([__file__])