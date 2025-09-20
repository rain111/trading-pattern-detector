"""
Tests for the DataManager class
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil
from pathlib import Path

from frontend.data import DataManager
from frontend.config import settings

class TestDataManager:
    """Test cases for DataManager class"""

    @pytest.fixture
    def data_manager(self):
        """Create a DataManager instance for testing"""
        return DataManager()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 150, 100),
            'high': np.random.uniform(105, 155, 100),
            'low': np.random.uniform(95, 145, 100),
            'close': np.random.uniform(100, 150, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        return data

    @pytest.fixture
    def test_dates(self):
        """Create test date range"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 4, 1)
        return start_date, end_date

    @pytest.mark.asyncio
    async def test_data_manager_initialization(self, data_manager):
        """Test DataManager initialization"""
        await data_manager.__aenter__()
        assert data_manager._session is not None
        await data_manager.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_get_stock_data_empty_cache(self, data_manager, test_dates):
        """Test getting stock data with empty cache"""
        start_date, end_date = test_dates

        with patch('frontend.data.manager.DataManager._fetch_data_from_yfinance') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()

            result = await data_manager.get_stock_data("AAPL", start_date, end_date)

            assert isinstance(result, pd.DataFrame)
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stock_data_cache_hit(self, data_manager, test_dates):
        """Test getting stock data with cache hit"""
        start_date, end_date = test_dates
        symbol = "AAPL"

        # Cache the data first
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cached_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000000, 2000000, 3000000]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))

        data_manager._cache[cache_key] = {
            'data': cached_data,
            'timestamp': asyncio.get_event_loop().time()
        }

        # Now try to get the same data
        result = await data_manager.get_stock_data(symbol, start_date, end_date)

        assert not result.empty
        assert len(result) == 3
        assert mock_fetch.call_count == 0  # Should not call fetch function

    def test_clean_ohlc_data(self, data_manager, sample_data):
        """Test OHLC data cleaning"""
        # Introduce some invalid data
        dirty_data = sample_data.copy()
        dirty_data.loc['2023-01-01', 'high'] = 90  # High < Low
        dirty_data.loc['2023-01-02', 'low'] = 160  # Low > High
        dirty_data.loc['2023-01-03', 'close'] = np.nan  # NaN value

        cleaned_data = data_manager._clean_ohlc_data(dirty_data)

        # Check that relationships are fixed
        assert cleaned_data.loc['2023-01-01', 'high'] >= cleaned_data.loc['2023-01-01', 'low']
        assert cleaned_data.loc['2023-01-02', 'low'] <= cleaned_data.loc['2023-01-02', 'high']
        assert not cleaned_data.isnull().any().any()

    def test_get_available_symbols(self, data_manager):
        """Test getting available symbols"""
        # Create some mock parquet files
        with patch.object(settings, 'PARQUET_DIR') as mock_parquet_dir:
            mock_parquet_dir.glob.return_value = [
                Mock(name="AAPL.parquet"),
                Mock(name="MSFT.parquet"),
                Mock(name="GOOGL.parquet")
            ]

            symbols = data_manager.get_available_symbols()

            assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_get_cache_info(self, data_manager):
        """Test getting cache information"""
        # Create a mock metadata file
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '''
            {
                "symbol": "AAPL",
                "last_updated": "2023-01-01T00:00:00",
                "data_from": "yfinance",
                "records_count": 100,
                "file_path": "/path/to/AAPL.parquet"
            }
            '''

            cache_info = data_manager.get_cache_info("AAPL")

            assert cache_info.symbol == "AAPL"
            assert cache_info.records_count == 100
            assert cache_info.data_from == "yfinance"

    def test_clear_cache(self, data_manager):
        """Test clearing cache"""
        # Add some cache data
        data_manager._cache['test_key1'] = {'data': 'value1', 'timestamp': 123456}
        data_manager._cache['test_key2'] = {'data': 'value2', 'timestamp': 789012}

        data_manager.clear_cache()

        assert len(data_manager._cache) == 0

    def test_clear_symbol_cache(self, data_manager):
        """Test clearing cache for specific symbol"""
        # Add some cache data
        data_manager._cache['AAPL_key1'] = {'data': 'value1', 'timestamp': 123456}
        data_manager._cache['AAPL_key2'] = {'data': 'value2', 'timestamp': 789012}
        data_manager._cache['MSFT_key1'] = {'data': 'value3', 'timestamp': 345678}

        data_manager.clear_cache("AAPL")

        # Only AAPL keys should be removed
        assert len([k for k in data_manager._cache.keys() if k.startswith('AAPL')]) == 0
        assert 'MSFT_key1' in data_manager._cache

    @pytest.mark.asyncio
    async def test_fetch_data_from_yfinance_error_handling(self, data_manager):
        """Test error handling in yfinance data fetching"""
        with patch('frontend.data.manager.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.history.side_effect = Exception("Network error")

            result = await data_manager._fetch_data_from_yfinance("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 31))

            assert result.empty

    def test_needs_additional_data(self, data_manager):
        """Test the needs_additional_data method"""
        # Create existing data
        existing_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000000, 2000000, 3000000]
        }, index=pd.date_range(start='2023-01-15', periods=3, freq='D'))

        # Test case 1: Need data before existing range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        assert data_manager._needs_additional_data(existing_data, start_date, end_date)

        # Test case 2: Need data after existing range
        start_date = datetime(2023, 1, 20)
        end_date = datetime(2023, 2, 10)
        assert data_manager._needs_additional_data(existing_data, start_date, end_date)

        # Test case 3: No additional data needed
        start_date = datetime(2023, 1, 15)
        end_date = datetime(2023, 1, 17)
        assert not data_manager._needs_additional_data(existing_data, start_date, end_date)

    def test_combine_data(self, data_manager):
        """Test data combination logic"""
        # Create existing and new data
        existing_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000000, 2000000, 3000000]
        }, index=pd.date_range(start='2023-01-10', periods=3, freq='D'))

        new_data = pd.DataFrame({
            'open': [103, 104, 105],
            'high': [108, 109, 110],
            'low': [98, 99, 100],
            'close': [103, 104, 105],
            'volume': [4000000, 5000000, 6000000]
        }, index=pd.date_range(start='2023-01-15', periods=3, freq='D'))

        combined = data_manager._combine_data(existing_data, new_data, "AAPL")

        # Check that data is combined and sorted
        assert len(combined) == 5
        assert combined.index.is_monotonic_increasing
        assert len(combined) == len(combined.drop_duplicates())  # No duplicates

    @pytest.mark.asyncio
    async def test_save_to_parquet(self, data_manager, sample_data):
        """Test saving data to parquet"""
        with patch('frontend.data.manager.pq.write_table') as mock_write:
            with patch('frontend.data.manager.DataCache') as mock_cache:
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.write = Mock()

                    await data_manager._save_to_parquet(sample_data, "AAPL")

                    mock_write.assert_called_once()
                    mock_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_manager_context_manager(self):
        """Test DataManager context manager"""
        async with DataManager() as dm:
            assert dm._session is not None
            assert dm._cache == {}  # Should start with empty cache