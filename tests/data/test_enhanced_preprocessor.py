"""
Tests for enhanced DataPreprocessor with async support.
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
import time
from concurrent.futures import ThreadPoolExecutor

from src.utils.data_preprocessor import DataPreprocessor


class TestDataPreprocessorAsync:
    """Test enhanced DataPreprocessor async functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.preprocessor = DataPreprocessor()

    def teardown_method(self):
        """Cleanup test environment"""
        if hasattr(self.preprocessor, '_executor'):
            self.preprocessor._executor.shutdown(wait=False)

    def test_initialization(self):
        """Test preprocessor initialization"""
        assert self.preprocessor.price_columns == ["open", "high", "low", "close", "volume"]
        assert self.preprocessor.logger is not None
        assert hasattr(self.preprocessor, '_executor')
        assert isinstance(self.preprocessor._executor, ThreadPoolExecutor)

    @pytest.mark.asyncio
    async def test_clean_data_async(self):
        """Test async data cleaning"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 100],
            'high': [101, 102, 103, 101],
            'low': [99, 100, 101, 99],
            'close': [100.5, 101.5, 102.5, 100.5],
            'volume': [1000000, 1100000, 1200000, 1000000]
        }, index=pd.date_range('2023-01-01', periods=4, freq='D'))

        # Add some duplicates
        test_data = pd.concat([test_data, test_data.iloc[0:1]])

        result = await self.preprocessor.clean_data_async(test_data)

        # Should remove duplicates
        assert len(result) == 4
        assert result.index.duplicated().sum() == 0
        assert result.index.is_monotonic_increasing

    @pytest.mark.asyncio
    async def test_resample_data_async(self):
        """Test async data resampling"""
        # Create minute-level data
        dates = pd.date_range('2023-01-01', periods=60, freq='min')
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 101, 60),
            'high': np.random.uniform(101, 102, 60),
            'low': np.random.uniform(99, 100, 60),
            'close': np.random.uniform(100, 101, 60),
            'volume': np.random.uniform(100000, 200000, 60)
        }, index=dates)

        result = await self.preprocessor.resample_data_async(test_data, '1H')

        # Should resample to hourly data
        assert len(result) == 1  # 60 minutes = 1 hour
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

    @pytest.mark.asyncio
    async def test_add_technical_indicators_async(self):
        """Test async adding technical indicators"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105],
            'high': [101, 102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103, 104],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6, freq='D'))

        result = await self.preprocessor.add_technical_indicators_async(test_data)

        # Should have additional columns for technical indicators
        assert 'sma_20' in result.columns
        assert 'sma_50' in result.columns
        assert 'ema_12' in result.columns
        assert 'ema_26' in result.columns
        assert 'rsi' in result.columns
        assert 'atr' in result.columns
        assert 'volume_sma' in result.columns
        assert 'volume_ratio' in result.columns

        # Check that technical indicators are calculated correctly
        # Note: with only 6 data points, many indicators will be NaN
        assert len(result) == 6

    @pytest.mark.asyncio
    async def test_normalize_data_async(self):
        """Test async data normalization"""
        test_data = pd.DataFrame({
            'open': [100, 200, 300],
            'high': [101, 202, 303],
            'low': [99, 198, 297],
            'close': [100.5, 200.5, 300.5],
            'volume': [1000000, 2000000, 3000000]
        })

        # Test minmax normalization
        result = await self.preprocessor.normalize_data_async(test_data, method="minmax")

        # Check that prices are normalized to [0, 1]
        assert result['open'].min() == 0.0
        assert result['open'].max() == 1.0
        assert result['close'].min() == 0.0
        assert result['close'].max() == 1.0

        # Test zscore normalization
        result_zscore = await self.preprocessor.normalize_data_async(test_data, method="zscore")

        # Check that prices have mean 0 and std 1
        assert abs(result_zscore['open'].mean()) < 1e-10  # Should be close to 0
        assert abs(result_zscore['open'].std() - 1.0) < 1e-10  # Should be close to 1

    @pytest.mark.asyncio
    async def test_calculate_returns_async(self):
        """Test async returns calculation"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103],
            'high': [101, 102, 103, 104],
            'low': [99, 100, 101, 102],
            'close': [100.5, 101.5, 102.5, 103.5],
            'volume': [1000000, 1100000, 1200000, 1300000]
        }, index=pd.date_range('2023-01-01', periods=4, freq='D'))

        result = await self.preprocessor.calculate_returns_async(test_data)

        # Should have additional columns for returns
        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert 'cumulative_returns' in result.columns
        assert 'volatility_20' in result.columns
        assert 'volatility_50' in result.columns

        # Check that returns are calculated correctly
        expected_returns = [None, 0.01 / 100.5, 0.01 / 101.5, 0.01 / 102.5]
        np.testing.assert_array_almost_equal(result['returns'].dropna().tolist(), expected_returns[1:], decimal=6)

    @pytest.mark.asyncio
    async def test_align_data_timestamps_async(self):
        """Test async timestamp alignment"""
        # Create data with timezone
        test_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D').tz_localize('US/Eastern'))

        result = await self.preprocessor.align_data_timestamps_async(test_data, timezone='UTC')

        # Should be converted to UTC
        assert result.tz.zone == 'UTC'

    @pytest.mark.asyncio
    async def test_validate_clean_data_async(self):
        """Test async data validation"""
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        result = await self.preprocessor.validate_clean_data_async(valid_data)
        assert result is True

    def test_validate_clean_data_sync(self):
        """Test sync data validation"""
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        result = self.preprocessor.validate_clean_data(valid_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_process_multiple_symbols_async(self):
        """Test processing multiple symbols"""
        # Create test data for multiple symbols
        data_dict = {
            'AAPL': pd.DataFrame({
                'open': [100, 101],
                'high': [101, 102],
                'low': [99, 100],
                'close': [100.5, 101.5],
                'volume': [1000000, 1100000]
            }, index=pd.date_range('2023-01-01', periods=2, freq='D')),

            'MSFT': pd.DataFrame({
                'open': [200, 201],
                'high': [201, 202],
                'low': [199, 200],
                'close': [200.5, 201.5],
                'volume': [2000000, 2100000]
            }, index=pd.date_range('2023-01-01', periods=2, freq='D'))
        }

        # Test with all operations
        results = await self.preprocessor.process_multiple_symbols_async(
            data_dict, operations=['clean', 'indicators', 'returns']
        )

        # Check that both symbols were processed
        assert 'AAPL' in results
        assert 'MSFT' in results
        assert not results['AAPL'].empty
        assert not results['MSFT'].empty

        # Check that indicators were added
        assert 'sma_20' in results['AAPL'].columns
        assert 'sma_20' in results['MSFT'].columns
        assert 'returns' in results['AAPL'].columns
        assert 'returns' in results['MSFT'].columns

    @pytest.mark.asyncio
    async def test_process_single_symbol_async(self):
        """Test processing single symbol"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        result = await self.preprocessor._process_single_symbol_async(
            'AAPL', test_data, ['clean', 'indicators']
        )

        assert not result.empty
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns

    @pytest.mark.asyncio
    async def test_preprocess_pipeline_async(self):
        """Test preprocessing pipeline"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        # Test default operations
        result = await self.preprocessor.preprocess_pipeline_async(test_data)

        assert not result.empty
        assert 'sma_20' in result.columns  # indicators
        assert 'returns' in result.columns  # returns

        # Test custom operations with timeframe
        result_custom = await self.preprocessor.preprocess_pipeline_async(
            test_data, operations=['clean', 'resample'], timeframe='2D'
        )

        assert not result_custom.empty
        assert len(result_custom) == 2  # Resampled to 2-day intervals

    @pytest.mark.asyncio
    async def test_process_with_performance_tracking_async(self):
        """Test processing with performance tracking"""
        data_dict = {
            'AAPL': pd.DataFrame({
                'open': [100, 101],
                'high': [101, 102],
                'low': [99, 100],
                'close': [100.5, 101.5],
                'volume': [1000000, 1100000]
            }),
            'MSFT': pd.DataFrame({
                'open': [200, 201],
                'high': [201, 202],
                'low': [199, 200],
                'close': [200.5, 201.5],
                'volume': [2000000, 2100000]
            })
        }

        results, performance_data = await self.preprocessor.process_with_performance_tracking_async(
            data_dict, operations=['clean', 'indicators']
        )

        # Check results
        assert 'AAPL' in results
        assert 'MSFT' in results

        # Check performance data
        assert 'AAPL' in performance_data
        assert 'MSFT' in performance_data
        assert 'total_time' in performance_data
        assert isinstance(performance_data['AAPL'], float)
        assert isinstance(performance_data['MSFT'], float)
        assert isinstance(performance_data['total_time'], float)

    @pytest.mark.asyncio
    async def test_process_large_dataset_async(self):
        """Test processing large dataset in chunks"""
        # Create a large dataset
        large_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 10000),
            'high': np.random.uniform(101, 201, 10000),
            'low': np.random.uniform(99, 199, 10000),
            'close': np.random.uniform(100, 200, 10000),
            'volume': np.random.uniform(1000000, 2000000, 10000)
        }, index=pd.date_range('2023-01-01', periods=10000, freq='D'))

        # Test with chunk size smaller than dataset
        result = await self.preprocessor.process_large_dataset_async(
            large_data, chunk_size=1000, operations=['clean', 'indicators']
        )

        assert not result.empty
        assert len(result) == 10000
        assert 'sma_20' in result.columns

        # Test with chunk size larger than dataset
        result_small = await self.preprocessor.process_large_dataset_async(
            large_data.head(100), chunk_size=1000, operations=['clean']
        )

        assert not result_small.empty
        assert len(result_small) == 100

    @pytest.mark.asyncio
    async def test_validate_async(self):
        """Test async data validation with detailed reporting"""
        # Valid data
        valid_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105],
            'high': [101, 102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103, 104],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6, freq='D'))

        report = await self.preprocessor.validate_async(valid_data)

        assert report['is_valid'] is True
        assert report['issues'] == []
        assert report['warnings'] == []
        assert report['data_quality_score'] == 100.0

        # Invalid data - missing columns
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100]
            # Missing close and volume
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

        report_invalid = await self.preprocessor.validate_async(invalid_data)

        assert report_invalid['is_valid'] is False
        assert len(report_invalid['issues']) > 0
        assert report_invalid['data_quality_score'] < 100.0

        # Data with warnings
        warning_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, np.nan]  # Some missing volume
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        report_warning = await self.preprocessor.validate_async(warning_data)

        assert report_warning['is_valid'] is True
        assert len(report_warning['warnings']) > 0

    def test_private_validation_methods(self):
        """Test private validation helper methods"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        # Test required columns check
        result = self.preprocessor._check_required_columns(test_data)
        assert result['valid'] is True
        assert result['issues'] == []

        # Test missing values check
        result_mv = self.preprocessor._check_missing_values(test_data)
        assert result_mv['valid'] is True
        assert result_mv['warnings'] == []

        # Test price consistency check
        result_pc = self.preprocessor._check_price_consistency(test_data)
        assert result_pc['valid'] is True
        assert result_pc['warnings'] == []

        # Test sufficient data check
        result_sd = self.preprocessor._check_sufficient_data(test_data)
        assert result_sd is True

    def test_close(self):
        """Test cleanup method"""
        # Should not raise an exception
        self.preprocessor.close()

    @pytest.mark.asyncio
    async def test_error_handling_async(self):
        """Test error handling in async methods"""
        # Create problematic data
        problematic_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 100],  # Inconsistent with low
            'low': [99, 98],
            'close': [100.5, 101.5],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

        # Test that methods handle errors gracefully
        result = await self.preprocessor.clean_data_async(problematic_data)
        assert not result.empty  # Should return original data on error

        result_indicators = await self.preprocessor.add_technical_indicators_async(problematic_data)
        assert not result_indicators.empty  # Should return original data on error

        result_validate = await self.preprocessor.validate_async(problematic_data)
        assert isinstance(result_validate, dict)  # Should return error report

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent processing of multiple operations"""
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = self.preprocessor.clean_data_async(test_data)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all tasks completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        assert all(len(result) == 3 for result in successful_results)


if __name__ == "__main__":
    pytest.main([__file__])