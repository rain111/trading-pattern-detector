"""
Enhanced Async Data Manager - Advanced data processing with multiple API support,
smart caching, parallel processing, and technical indicators calculation.
"""

import asyncio
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    pq = None
    pa = None

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple, Union
import time
import json
from dataclasses import dataclass, asdict
import concurrent.futures

from .core import (
    DataProcessor, DataCache, DataStorage, DataValidator,
    CacheConfig, StorageConfig, ValidationConfig, DataProcessorConfig,
    ValidationResult, DataProcessorResult, DataMetrics, CacheType, StorageType
)
from .adapters import DataProcessorFactory
from ..utils.data_preprocessor import DataPreprocessor


class APIConfig:
    """Configuration for API endpoints"""

    def __init__(self, name: str, base_url: str, timeout: int = 30,
                 rate_limit: int = 5, retry_attempts: int = 3,
                 retry_delay: float = 1.0, api_key: Optional[str] = None):
        self.name = name
        self.base_url = base_url
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.api_key = api_key


class SmartMemoryCache(DataCache):
    """
    Smart in-memory cache with TTL, size limits, and performance optimization
    """

    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache = {}
        self.access_times = {}
        self.size_stats = {}

    async def get_async(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from memory cache with TTL check"""
        try:
            if key not in self.cache:
                return None

            cache_entry = self.cache[key]
            current_time = time.time()

            # Check TTL
            if current_time - cache_entry['timestamp'] > self.config.ttl_seconds:
                await self.delete_async(key)
                return None

            # Update access time
            self.access_times[key] = current_time
            return cache_entry['data']

        except Exception as e:
            self.logger.error(f"Error in async cache get for {key}: {e}")
            return None

    def get_sync(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from memory cache (sync version)"""
        try:
            if key not in self.cache:
                return None

            cache_entry = self.cache[key]
            current_time = time.time()

            # Check TTL
            if current_time - cache_entry['timestamp'] > self.config.ttl_seconds:
                self.delete_sync(key)
                return None

            # Update access time
            self.access_times[key] = current_time
            return cache_entry['data']

        except Exception as e:
            self.logger.error(f"Error in sync cache get for {key}: {e}")
            return None

    async def set_async(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """Set data in memory cache with size management"""
        try:
            # Check size limits
            current_size = len(self.cache)
            if current_size >= self.config.max_size:
                # Remove oldest entries
                await self._evict_oldest_entries()

            cache_entry = {
                'data': data,
                'timestamp': time.time(),
                'size': len(data)
            }

            self.cache[key] = cache_entry
            self.access_times[key] = time.time()
            self.size_stats[key] = len(data)

            self.logger.debug(f"Cache set for {key}, size: {len(data)}")
            return True

        except Exception as e:
            self.logger.error(f"Error in async cache set for {key}: {e}")
            return False

    def set_sync(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """Set data in memory cache (sync version)"""
        try:
            # Check size limits
            current_size = len(self.cache)
            if current_size >= self.config.max_size:
                # Remove oldest entries
                self._evict_oldest_entries_sync()

            cache_entry = {
                'data': data,
                'timestamp': time.time(),
                'size': len(data)
            }

            self.cache[key] = cache_entry
            self.access_times[key] = time.time()
            self.size_stats[key] = len(data)

            self.logger.debug(f"Cache set for {key}, size: {len(data)}")
            return True

        except Exception as e:
            self.logger.error(f"Error in sync cache set for {key}: {e}")
            return False

    async def delete_async(self, key: str) -> bool:
        """Delete data from memory cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                if key in self.size_stats:
                    del self.size_stats[key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in async cache delete for {key}: {e}")
            return False

    def delete_sync(self, key: str) -> bool:
        """Delete data from memory cache (sync version)"""
        try:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                if key in self.size_stats:
                    del self.size_stats[key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in sync cache delete for {key}: {e}")
            return False

    async def clear_expired_async(self) -> int:
        """Clear expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = []

            for key, cache_entry in self.cache.items():
                if current_time - cache_entry['timestamp'] > self.config.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                await self.delete_async(key)

            self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
            return len(expired_keys)

        except Exception as e:
            self.logger.error(f"Error in async cache cleanup: {e}")
            return 0

    def clear_expired_sync(self) -> int:
        """Clear expired cache entries (sync version)"""
        try:
            current_time = time.time()
            expired_keys = []

            for key, cache_entry in self.cache.items():
                if current_time - cache_entry['timestamp'] > self.config.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self.delete_sync(key)

            self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
            return len(expired_keys)

        except Exception as e:
            self.logger.error(f"Error in sync cache cleanup: {e}")
            return 0

    async def _evict_oldest_entries(self):
        """Evict oldest entries based on access time"""
        if len(self.cache) >= self.config.max_size:
            sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
            keys_to_remove = sorted_keys[:max(1, len(self.cache) - self.config.max_size + 1)]

            for key in keys_to_remove:
                await self.delete_async(key)

    def _evict_oldest_entries_sync(self):
        """Evict oldest entries (sync version)"""
        if len(self.cache) >= self.config.max_size:
            sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
            keys_to_remove = sorted_keys[:max(1, len(self.cache) - self.config.max_size + 1)]

            for key in keys_to_remove:
                self.delete_sync(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'total_memory_usage': sum(self.size_stats.values()),
            'hit_count': sum(1 for entry in self.cache.values() if entry.get('hit', False)),
            'keys': list(self.cache.keys())
        }


class ParquetStorage(DataStorage):
    """
    Parquet-based storage with compression and partitioning
    """

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def store_async(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store data in parquet format asynchronously"""
        try:
            if data.empty:
                return False

            # Create symbol-specific directory
            symbol_dir = self.base_path / symbol.lower()
            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with date range
            start_date = data.index.min().strftime('%Y%m%d')
            end_date = data.index.max().strftime('%Y%m%d')
            filename = f"{symbol}_{start_date}_{end_date}.parquet"

            # Save with compression
            parquet_file = symbol_dir / filename
            table = pa.Table.from_pandas(data)
            pq.write_table(table, parquet_file, compression=self.config.compression)

            # Create metadata file
            metadata = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'records_count': len(data),
                'columns': list(data.columns),
                'stored_at': datetime.now().isoformat()
            }

            metadata_file = symbol_dir / f"{symbol}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"Stored {len(data)} records for {symbol} in {parquet_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
            return False

    def store_sync(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store data in parquet format synchronously"""
        try:
            # Run async version in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.store_async(symbol, data))
        except Exception as e:
            self.logger.error(f"Error syncing data store for {symbol}: {e}")
            return False

    async def load_async(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from parquet storage asynchronously"""
        try:
            symbol_dir = self.base_path / symbol.lower()

            if not symbol_dir.exists():
                return pd.DataFrame()

            # Find all parquet files for this symbol
            parquet_files = list(symbol_dir.glob(f"{symbol}_*.parquet"))
            if not parquet_files:
                return pd.DataFrame()

            # Combine all files
            all_data = []
            for parquet_file in sorted(parquet_files):
                try:
                    table = pq.read_table(parquet_file)
                    df = table.to_pandas()
                    all_data.append(df)
                except Exception as e:
                    self.logger.warning(f"Error reading {parquet_file}: {e}")

            if not all_data:
                return pd.DataFrame()

            combined_data = pd.concat(all_data, ignore_index=True)

            # Set index if it's a datetime index
            if hasattr(combined_data, 'index') and not isinstance(combined_data.index, pd.DatetimeIndex):
                # Try to parse dates from metadata or first column
                if 'date' in combined_data.columns:
                    combined_data['date'] = pd.to_datetime(combined_data['date'])
                    combined_data.set_index('date', inplace=True)
                elif 'timestamp' in combined_data.columns:
                    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
                    combined_data.set_index('timestamp', inplace=True)

            # Apply date filters if provided
            if start_date and end_date:
                combined_data = combined_data[(combined_data.index >= start_date) & (combined_data.index <= end_date)]

            # Sort by date
            combined_data = combined_data.sort_index()

            self.logger.info(f"Loaded {len(combined_data)} records for {symbol}")
            return combined_data

        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def load_sync(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from parquet storage synchronously"""
        try:
            # Run async version in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.load_async(symbol, start_date, end_date))
        except Exception as e:
            self.logger.error(f"Error syncing data load for {symbol}: {e}")
            return pd.DataFrame()

    async def delete_async(self, symbol: str) -> bool:
        """Delete stored data asynchronously"""
        try:
            symbol_dir = self.base_path / symbol.lower()

            if symbol_dir.exists():
                import shutil
                shutil.rmtree(symbol_dir)
                self.logger.info(f"Deleted all data for {symbol}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error deleting data for {symbol}: {e}")
            return False

    def delete_sync(self, symbol: str) -> bool:
        """Delete stored data synchronously"""
        try:
            # Run async version in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.delete_async(symbol))
        except Exception as e:
            self.logger.error(f"Error syncing data delete for {symbol}: {e}")
            return False

    async def get_available_symbols_async(self) -> List[str]:
        """Get list of available symbols asynchronously"""
        try:
            symbols = []
            for symbol_dir in self.base_path.iterdir():
                if symbol_dir.is_dir():
                    symbols.append(symbol_dir.name)
            return symbols
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    def get_available_symbols_sync(self) -> List[str]:
        """Get list of available symbols synchronously"""
        try:
            # Run async version in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_available_symbols_async())
        except Exception as e:
            self.logger.error(f"Error syncing available symbols: {e}")
            return []


class EnhancedAsyncDataManager(DataProcessor):
    """
    Enhanced async data manager with multiple API support, smart caching,
    parallel processing, and technical indicators calculation.
    """

    def __init__(self, config: Optional[DataProcessorConfig] = None):
        super().__init__(config)

        # Initialize components
        self.cache = SmartMemoryCache(config.cache_config if config else CacheConfig())
        self.storage = ParquetStorage(config.storage_config if config else StorageConfig())
        self.validator = DataValidator(config.validation_config if config else ValidationConfig())
        self.preprocessor = DataPreprocessor()
        self.metrics = DataMetrics()

        # API configurations
        self.apis = {
            'yfinance': APIConfig('yfinance', '', timeout=30, rate_limit=2),
            'alpha_vantage': APIConfig('alpha_vantage', 'https://www.alphavantage.co/', timeout=30, rate_limit=5),
            'polygon': APIConfig('polygon', 'https://api.polygon.io/', timeout=30, rate_limit=5)
        }

        # Session management
        self._session = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests if config else 10)

    async def __aenter__(self):
        """Initialize async session"""
        if aiohttp:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session"""
        if self._session:
            if aiohttp:
                await self._session.close()

    async def fetch_data_async(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from multiple APIs with fallback and retry logic"""
        start_time = time.time()
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        # Check cache first
        cached_data = await self.cache.get_async(cache_key)
        if cached_data is not None:
            self.metrics.record_cache_hit()
            self.metrics.record_fetch_time(time.time() - start_time)
            self.logger.info(f"Cache hit for {symbol}")
            return cached_data

        self.metrics.record_cache_miss()

        # Check storage first
        stored_data = await self.storage.load_async(symbol, start_date, end_date)
        if not stored_data.empty:
            self.logger.info(f"Loaded {len(stored_data)} records for {symbol} from storage")
            # Validate and cache
            validated_data = await self._validate_and_cache_data(cache_key, stored_data)
            self.metrics.record_fetch_time(time.time() - start_time)
            return validated_data

        # Fetch from APIs with fallback
        data = await self._fetch_with_fallback(symbol, start_date, end_date)

        if not data.empty:
            # Validate and cache
            validated_data = await self._validate_and_cache_data(cache_key, data)
            # Store to disk
            await self.storage.store_async(symbol, data)
            self.metrics.record_fetch_time(time.time() - start_time)
            return validated_data

        self.metrics.record_fetch_time(time.time() - start_time)
        self.metrics.record_error('fetch_failed')
        return pd.DataFrame()

    def fetch_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Sync version of fetch data"""
        try:
            # Run async version in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.fetch_data_async(symbol, start_date, end_date))
        except Exception as e:
            self.logger.error(f"Error in sync fetch for {symbol}: {e}")
            return pd.DataFrame()

    async def _fetch_with_fallback(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from multiple APIs with fallback"""
        # Try Yahoo Finance first
        try:
            data = await self._fetch_from_yahoo_finance(symbol, start_date, end_date)
            if not data.empty:
                return data
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed for {symbol}: {e}")

        # Try Alpha Vantage
        try:
            data = await self._fetch_from_alpha_vantage(symbol, start_date, end_date)
            if not data.empty:
                return data
        except Exception as e:
            self.logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")

        # Try Polygon.io
        try:
            data = await self._fetch_from_polygon(symbol, start_date, end_date)
            if not data.empty:
                return data
        except Exception as e:
            self.logger.warning(f"Polygon.io fetch failed for {symbol}: {e}")

        return pd.DataFrame()

    async def _fetch_from_yahoo_finance(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf

            # Run yfinance in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            data = await loop.run_in_executor(
                None,
                ticker.history,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                '1d'
            )

            if not data.empty:
                data.attrs['symbol'] = symbol
                data.attrs['fetched_at'] = datetime.now()
                data.attrs['source'] = 'yfinance'
                return data

            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching from Yahoo Finance for {symbol}: {e}")
            return pd.DataFrame()

    async def _fetch_from_alpha_vantage(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Alpha Vantage"""
        try:
            # This would implement Alpha Vantage API calls
            # Placeholder implementation
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching from Alpha Vantage for {symbol}: {e}")
            return pd.DataFrame()

    async def _fetch_from_polygon(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Polygon.io"""
        try:
            # This would implement Polygon.io API calls
            # Placeholder implementation
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching from Polygon.io for {symbol}: {e}")
            return pd.DataFrame()

    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data using enhanced preprocessor"""
        start_time = time.time()

        try:
            # Clean data
            cleaned_data = self.preprocessor.clean_data(data)

            # Add technical indicators
            processed_data = self.preprocessor.add_technical_indicators(cleaned_data)

            # Add returns and volatility
            processed_data = self.preprocessor.calculate_returns(processed_data)

            self.metrics.record_process_time(time.time() - start_time)
            self.metrics.record_validation_result(ValidationResult.VALID)

            return processed_data
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            self.metrics.record_process_time(time.time() - start_time)
            self.metrics.record_error('preprocessing_failed')
            return data

    async def preprocess_data_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Async version of preprocess data"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.preprocess_data, data)
        except Exception as e:
            self.logger.error(f"Error in async preprocessing: {e}")
            return data

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data synchronously"""
        try:
            return self.validator.validate_sync(data)
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return ValidationResult.INVALID

    async def _validate_and_cache_data(self, cache_key: str, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and cache data"""
        validation_result = await self.validator.validate_async(data)
        self.metrics.record_validation_result(validation_result)

        if validation_result == ValidationResult.VALID:
            await self.cache.set_async(cache_key, data)
            return data
        elif validation_result == ValidationResult.NEEDS_CLEANING:
            cleaned_data = await self.validator.auto_clean_async(data)
            await self.cache.set_async(cache_key, cleaned_data)
            return cleaned_data
        else:
            self.logger.warning(f"Data validation failed for {cache_key}")
            return data

    async def fetch_multiple_symbols_async(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel"""
        tasks = []
        for symbol in symbols:
            task = self._fetch_single_symbol_with_retry(symbol, start_date, end_date)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching {symbol}: {result}")
                processed_results[symbol] = pd.DataFrame()
            else:
                processed_results[symbol] = result

        return processed_results

    async def _fetch_single_symbol_with_retry(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch single symbol with retry logic"""
        last_error = None
        for attempt in range(self.config.retry_attempts if self.config else 3):
            try:
                return await self.fetch_data_async(symbol, start_date, end_date)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < (self.config.retry_attempts if self.config else 3) - 1:
                    await asyncio.sleep(self.config.retry_delay if self.config else 1.0)

        if last_error:
            self.logger.error(f"All attempts failed for {symbol}: {last_error}")
        return pd.DataFrame()

    async def get_performance_metrics_async(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'cache_stats': self.cache.get_stats(),
            'metrics': self.metrics.to_dict(),
            'available_symbols': await self.storage.get_available_symbols_async()
        }

    async def cleanup_async(self) -> Dict[str, int]:
        """Cleanup old cache and expired entries"""
        cache_cleared = await self.cache.clear_expired_async()

        # Additional cleanup logic here
        return {
            'cache_entries_cleared': cache_cleared,
            'storage_cleanup': 0  # Would implement storage cleanup here
        }

    def get_performance_metrics_sync(self) -> Dict[str, Any]:
        """Sync version of get performance metrics"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_performance_metrics_async())
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}

    def cleanup_sync(self) -> Dict[str, int]:
        """Sync version of cleanup"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.cleanup_async())
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
            return {}

    def add_api_config(self, api_name: str, config: APIConfig):
        """Add custom API configuration"""
        self.apis[api_name] = config
        self.logger.info(f"Added API config for {api_name}")

    def set_preprocessor(self, preprocessor: DataPreprocessor):
        """Set custom preprocessor"""
        self.preprocessor = preprocessor
        self.logger.info("Updated custom preprocessor")

    def get_metrics(self) -> DataMetrics:
        """Get current metrics"""
        return self.metrics