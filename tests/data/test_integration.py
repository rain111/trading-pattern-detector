"""
Integration tests for the complete enhanced data management architecture.
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
import time

from src.data.async_manager import EnhancedAsyncDataManager
from src.data.core import (
    DataProcessorConfig, CacheConfig, StorageConfig, ValidationConfig,
    CacheType, StorageType, ValidationResult, DataProcessorResult
)
from src.data.adapters import DataProcessorFactory, DataManagerAdapter
from src.utils.data_preprocessor import DataPreprocessor


class TestFullIntegration:
    """Test complete integration of all components"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Create comprehensive configuration
        self.config = DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=self.temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=5,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
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
    async def test_end_to_end_data_flow(self):
        """Test complete end-to-end data flow"""
        # Mock data fetching
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))

        with patch.object(self.manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.return_value = test_data

            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 5)

            # Step 1: Fetch data
            raw_data = await self.manager.fetch_data_async("AAPL", start_date, end_date)
            assert not raw_data.empty
            assert len(raw_data) == 5

            # Step 2: Preprocess data
            processed_data = await self.manager.preprocess_data_async(raw_data)
            assert not processed_data.empty
            assert 'sma_20' in processed_data.columns
            assert 'rsi' in processed_data.columns
            assert 'returns' in processed_data.columns

            # Step 3: Validate data
            validation_result = self.manager.validator.validate_sync(processed_data)
            assert validation_result == ValidationResult.VALID

            # Step 4: Store data
            storage_success = await self.manager.storage.store_async("AAPL", processed_data)
            assert storage_success is True

            # Step 5: Load data back
            loaded_data = await self.manager.storage.load_async("AAPL")
            assert not loaded_data.empty
            assert loaded_data.equals(processed_data)

    @pytest.mark.asyncio
    async def test_multiple_symbols_concurrent(self):
        """Test concurrent processing of multiple symbols"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        with patch.object(self.manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.return_value = test_data

            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 3)

            # Fetch multiple symbols concurrently
            results = await self.manager.fetch_multiple_symbols_async(symbols, start_date, end_date)

            # Verify all symbols were processed
            assert len(results) == 5
            for symbol in symbols:
                assert symbol in results
                assert not results[symbol].empty
                assert len(results[symbol]) == 3

            # Process all symbols with technical indicators
            processed_results = {}
            for symbol, data in results.items():
                processed_data = await self.manager.preprocess_data_async(data)
                processed_results[symbol] = processed_data

            # Verify all symbols have indicators
            for symbol, data in processed_results.items():
                assert 'sma_20' in data.columns
                assert 'rsi' in data.columns
                assert 'returns' in data.columns

    @pytest.mark.asyncio
    async def test_adapter_integration(self):
        """Test adapter integration with enhanced backend"""
        # Create legacy manager
        legacy_manager = Mock()
        legacy_manager.get_available_symbols.return_value = ["LEGACY_SYMBOL"]

        # Create adapter
        adapter = DataProcessorFactory.create_data_manager_adapter(legacy_manager, self.config)

        # Test that adapter uses enhanced backend
        assert adapter.existing_data_manager == legacy_manager
        assert adapter.config == self.config

        # Test adapter methods
        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        # Mock the enhanced backend call
        with patch.object(adapter, '_enhanced_manager') as mock_manager:
            mock_manager.fetch_data_async.return_value = test_data

            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 2)

            result = await adapter.fetch_data_async("AAPL", start_date, end_date)
            assert not result.empty

            # Verify enhanced backend was called
            mock_manager.fetch_data_async.assert_called_once_with("AAPL", start_date, end_date)

    @pytest.mark.asyncio
    async def test_preprocessor_pipeline_integration(self):
        """Test integration with enhanced preprocessor pipeline"""
        # Test data
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000]
        }, index=pd.date_range('2023-01-01', periods=15, freq='D'))

        # Test enhanced preprocessor pipeline
        enhanced_preprocessor = DataPreprocessor()

        # Test full pipeline
        result = await enhanced_preprocessor.preprocess_pipeline_async(
            test_data,
            operations=['clean', 'indicators', 'returns', 'normalize'],
            timeframe='5D'
        )

        assert not result.empty
        # Should have been resampled to 5D intervals
        assert len(result) == 3  # 15 days / 5 days = 3 intervals

        # Should have all processed indicators
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'returns' in result.columns
        assert 'normalized_open' in result.columns
        assert 'normalized_close' in result.columns

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache integration with data flow"""
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

            # First call - should fetch from API
            data1 = await self.manager.fetch_data_async("AAPL", start_date, end_date)
            assert mock_fetch.call_count == 1

            # Second call - should hit cache
            data2 = await self.manager.fetch_data_async("AAPL", start_date, end_date)
            assert mock_fetch.call_count == 1  # Should not call fetch again

            # Verify data is the same
            assert data1.equals(data2)

            # Check cache statistics
            cache_stats = self.manager.cache.get_stats()
            assert cache_stats['size'] > 0

    @pytest.mark.asyncio
    async def test_storage_integration(self):
        """Test storage integration with data flow"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        # Store data
        storage_success = await self.manager.storage.store_async("STOCK", test_data)
        assert storage_success is True

        # Load data back
        loaded_data = await self.manager.storage.load_async("STOCK")
        assert not loaded_data.empty
        assert loaded_data.equals(test_data)

        # Verify file was created
        stock_dir = Path(self.temp_dir) / "stock"
        assert stock_dir.exists()

        # Should have parquet file and metadata
        parquet_files = list(stock_dir.glob("stock_*.parquet"))
        metadata_files = list(stock_dir.glob("stock_metadata.json"))

        assert len(parquet_files) > 0
        assert len(metadata_files) > 0

        # Verify metadata content
        import json
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)

        assert metadata['symbol'] == 'STOCK'
        assert metadata['records_count'] == 3
        assert 'stored_at' in metadata

    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self):
        """Test performance metrics collection across components"""
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

            # Perform several operations
            await self.manager.fetch_data_async("AAPL", start_date, end_date)
            await self.manager.preprocess_data_async(test_data)
            await self.manager.preprocess_data_async(test_data)

            # Get performance metrics
            metrics = await self.manager.get_performance_metrics_async()

            # Verify metrics structure
            assert 'cache_stats' in metrics
            assert 'metrics' in metrics
            assert 'available_symbols' in metrics

            # Check specific metrics
            metrics_dict = metrics['metrics']
            assert 'fetch_times' in metrics_dict
            assert 'process_times' in metrics_dict
            assert 'cache_hits' in metrics_dict
            assert 'cache_misses' in metrics_dict
            assert 'validation_results' in metrics_dict
            assert 'error_counts' in metrics_dict

            # Should have cache hit and process time recorded
            assert metrics_dict['cache_hits'] == 1
            assert len(metrics_dict['process_times']) >= 1
            assert len(metrics_dict['fetch_times']) >= 1

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across all components"""
        # Test fetch error
        with patch.object(self.manager, '_fetch_with_fallback') as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 3)

            result = await self.manager.fetch_data_async("AAPL", start_date, end_date)
            assert result.empty  # Should return empty DataFrame on error

            # Check that error was recorded
            assert 'fetch_failed' in self.manager.metrics.error_counts

        # Test preprocessing error
        class ErrorPreprocessor:
            def add_technical_indicators(self, data):
                raise Exception("Preprocessing error")

        self.manager.set_preprocessor(ErrorPreprocessor())

        test_data = pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100], 'volume': [1000000]})

        result = await self.manager.preprocess_data_async(test_data)
        assert result.equals(test_data)  # Should return original data on error

        # Check that error was recorded
        assert 'preprocessing_failed' in self.manager.metrics.error_counts

    @pytest.mark.asyncio
    async def test_data_validation_integration(self):
        """Test data validation integration"""
        # Valid data
        valid_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105],
            'high': [101, 102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103, 104],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6, freq='D'))

        validation_result = self.manager.validator.validate_sync(valid_data)
        assert validation_result == ValidationResult.VALID

        # Invalid data
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 100],  # Inconsistent with low
            'low': [99, 98],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

        validation_result_invalid = self.manager.validator.validate_sync(invalid_data)
        assert validation_result_invalid == ValidationResult.INVALID

        # Check validation metrics
        assert len(self.manager.metrics.validation_results) >= 2

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self):
        """Test performance with concurrent operations"""
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(101, 201, 50),
            'low': np.random.uniform(99, 199, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000000, 2000000, 50)
        }, index=pd.date_range('2023-01-01', periods=50, freq='D'))

        # Create multiple concurrent tasks
        tasks = []
        symbols = [f"SYMBOL_{i}" for i in range(10)]

        for symbol in symbols:
            task = self.manager.fetch_multiple_symbols_async([symbol], datetime(2023, 1, 1), datetime(2023, 1, 50))
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # All tasks should complete without exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10

        # Should complete quickly
        elapsed_time = end_time - start_time
        assert elapsed_time < 5.0  # Should complete in under 5 seconds

        # Check performance metrics
        metrics = self.manager.metrics.to_dict()
        assert 'cache_hit_rate' in metrics
        assert 'success_rate' in metrics
        assert metrics['success_rate'] >= 0.9  # Should have high success rate

    @pytest.mark.asyncio
    async def test_cleanup_integration(self):
        """Test cleanup functionality integration"""
        # Add some data to cache
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        await self.manager.cache.set_async("test", test_data)

        # Perform cleanup
        cleanup_result = await self.manager.cleanup_async()

        # Check cleanup results
        assert 'cache_entries_cleared' in cleanup_result
        assert isinstance(cleanup_result['cache_entries_cleared'], int)

    @pytest.mark.asyncio
    async def test_full_manager_lifecycle(self):
        """Test complete manager lifecycle"""
        # Test context manager
        async with EnhancedAsyncDataManager(self.config) as manager:
            # Mock data fetching
            test_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [101, 102, 103],
                'low': [99, 100, 101],
                'close': [100.5, 101.5, 102.5],
                'volume': [1000000, 1100000, 1200000]
            }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

            with patch.object(manager, '_fetch_with_fallback') as mock_fetch:
                mock_fetch.return_value = test_data

                start_date = datetime(2023, 1, 1)
                end_date = datetime(2023, 1, 3)

                # Perform operations
                data = await manager.fetch_data_async("AAPL", start_date, end_date)
                processed_data = await manager.preprocess_data_async(data)

                # Store data
                await manager.storage.store_async("AAPL", processed_data)

                # Verify data was stored
                loaded_data = await manager.storage.load_async("AAPL")
                assert loaded_data.equals(processed_data)

        # Manager should be cleaned up after context exit
        # This test ensures no resource leaks


if __name__ == "__main__":
    pytest.main([__file__])