"""
Adapters for backward compatibility with existing data management components.
Provides adapter patterns to wrap existing classes while maintaining their interfaces.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

from .core import (
    DataProcessor, DataCache, DataStorage, DataValidator,
    CacheConfig, StorageConfig, ValidationConfig, DataProcessorConfig,
    ValidationResult, DataProcessorResult
)


class DataPreprocessorAdapter(DataProcessor):
    """
    Adapter for existing DataPreprocessor class to work with new core interfaces.
    Maintains backward compatibility while providing enhanced async capabilities.
    """

    def __init__(self, existing_preprocessor, config: Optional[DataProcessorConfig] = None):
        super().__init__(config)
        self.existing_preprocessor = existing_preprocessor
        self.logger = logging.getLogger(self.__class__.__name__)

    async def fetch_data_async(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Async fetch data - delegates to sync implementation in adapter mode
        """
        try:
            # For backward compatibility, use sync method wrapped in async
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.fetch_data_sync, symbol, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error in async fetch for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Sync fetch data - note: this would need implementation or delegation
        """
        # This would typically delegate to a data fetcher, not the preprocessor
        self.logger.warning("Sync fetch_data not implemented in DataPreprocessorAdapter")
        return pd.DataFrame()

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data using existing implementation
        """
        try:
            return self.existing_preprocessor.clean_data(data)
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return data

    async def preprocess_data_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Async preprocess data using existing implementation
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.preprocess_data, data)
        except Exception as e:
            self.logger.error(f"Error in async preprocessing: {e}")
            return data

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data using existing implementation
        """
        try:
            if self.existing_preprocessor.validate_clean_data(data):
                return ValidationResult.VALID
            else:
                return ValidationResult.NEEDS_CLEANING
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return ValidationResult.INVALID


class DataManagerAdapter(DataProcessor):
    """
    Adapter for existing DataManager class to work with new core interfaces.
    Maintains full backward compatibility while enabling enhanced features.
    """

    def __init__(self, existing_data_manager, config: Optional[DataProcessorConfig] = None):
        super().__init__(config)
        self.existing_data_manager = existing_data_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    async def fetch_data_async(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Async fetch data using existing DataManager implementation
        """
        try:
            # Create adapter context manager
            async with self.existing_data_manager:
                data = await self.existing_data_manager.get_stock_data(symbol, start_date, end_date)
                return data
        except Exception as e:
            self.logger.error(f"Error in async fetch for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Sync fetch data using existing DataManager implementation
        """
        try:
            # For sync operations, run async method in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.fetch_data_async(symbol, start_date, end_date))
        except Exception as e:
            self.logger.error(f"Error in sync fetch for {symbol}: {e}")
            return pd.DataFrame()

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data using existing DataManager implementation
        """
        try:
            return self.existing_data_manager._clean_ohlc_data(data)
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return data

    async def preprocess_data_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Async preprocess data using existing DataManager implementation
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.preprocess_data, data)
        except Exception as e:
            self.logger.error(f"Error in async preprocessing: {e}")
            return data

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data using existing DataManager implementation
        """
        try:
            if data.empty:
                return ValidationResult.INVALID

            # Check required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return ValidationResult.INVALID

            # Check for NaN values
            for col in required_columns:
                if data[col].isna().any():
                    self.logger.error(f"Column '{col}' contains NaN values")
                    return ValidationResult.INVALID

            # Check price consistency
            if (data["high"] < data["low"]).any():
                self.logger.error("High prices cannot be lower than low prices")
                return ValidationResult.INVALID

            # Check for sufficient data points
            if len(data) < self.config.validation_config.min_data_points if self.config and self.config.validation_config else 20:
                self.logger.error("Insufficient data points for analysis")
                return ValidationResult.INVALID

            return ValidationResult.VALID

        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return ValidationResult.INVALID

    def get_available_symbols(self) -> list:
        """Get available symbols using existing implementation"""
        try:
            return self.existing_data_manager.get_available_symbols()
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    def get_cache_info(self, symbol: str):
        """Get cache info using existing implementation"""
        try:
            return self.existing_data_manager.get_cache_info(symbol)
        except Exception as e:
            self.logger.error(f"Error getting cache info for {symbol}: {e}")
            return None

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache using existing implementation"""
        try:
            self.existing_data_manager.clear_cache(symbol)
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def cleanup_old_cache(self, max_age_hours: int = 24):
        """Cleanup old cache using existing implementation"""
        try:
            self.existing_data_manager.cleanup_old_cache(max_age_hours)
        except Exception as e:
            self.logger.error(f"Error cleaning up old cache: {e}")


class LegacyDataCacheAdapter(DataCache):
    """
    Adapter for existing cache mechanisms to work with new Cache interface
    """

    def __init__(self, existing_cache_manager, config: CacheConfig):
        super().__init__(config)
        self.existing_cache_manager = existing_cache_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    async def get_async(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from legacy cache (async)"""
        try:
            # For async compatibility, run sync method in event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.get_sync, key)
        except Exception as e:
            self.logger.error(f"Error in async cache get for {key}: {e}")
            return None

    def get_sync(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from legacy cache (sync)"""
        try:
            # Access private cache if available
            if hasattr(self.existing_cache_manager, '_cache'):
                cache_data = self.existing_cache_manager._cache.get(key)
                if cache_data:
                    return cache_data['data']
            return None
        except Exception as e:
            self.logger.error(f"Error in sync cache get for {key}: {e}")
            return None

    async def set_async(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """Set data in legacy cache (async)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.set_sync, key, data, ttl_seconds)
        except Exception as e:
            self.logger.error(f"Error in async cache set for {key}: {e}")
            return False

    def set_sync(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """Set data in legacy cache (sync)"""
        try:
            if hasattr(self.existing_cache_manager, '_cache'):
                self.existing_cache_manager._cache[key] = {
                    'data': data,
                    'timestamp': 0  # Legacy cache doesn't track timestamps
                }
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in sync cache set for {key}: {e}")
            return False

    async def delete_async(self, key: str) -> bool:
        """Delete data from legacy cache (async)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.delete_sync, key)
        except Exception as e:
            self.logger.error(f"Error in async cache delete for {key}: {e}")
            return False

    def delete_sync(self, key: str) -> bool:
        """Delete data from legacy cache (sync)"""
        try:
            if hasattr(self.existing_cache_manager, '_cache'):
                if key in self.existing_cache_manager._cache:
                    del self.existing_cache_manager._cache[key]
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error in sync cache delete for {key}: {e}")
            return False

    async def clear_expired_async(self) -> int:
        """Clear expired cache entries (async)"""
        try:
            # Legacy cache doesn't track TTL, so we clear all
            cleared = 0
            if hasattr(self.existing_cache_manager, '_cache'):
                cleared = len(self.existing_cache_manager._cache)
                self.existing_cache_manager._cache.clear()
            return cleared
        except Exception as e:
            self.logger.error(f"Error in async cache cleanup: {e}")
            return 0

    def clear_expired_sync(self) -> int:
        """Clear expired cache entries (sync)"""
        try:
            # Legacy cache doesn't track TTL, so we clear all
            cleared = 0
            if hasattr(self.existing_cache_manager, '_cache'):
                cleared = len(self.existing_cache_manager._cache)
                self.existing_cache_manager._cache.clear()
            return cleared
        except Exception as e:
            self.logger.error(f"Error in sync cache cleanup: {e}")
            return 0


class LegacyDataStorageAdapter(DataStorage):
    """
    Adapter for existing storage mechanisms to work with new Storage interface
    """

    def __init__(self, existing_storage_manager, config: StorageConfig):
        super().__init__(config)
        self.existing_storage_manager = existing_storage_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    async def store_async(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store data using legacy storage (async)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.store_sync, symbol, data)
        except Exception as e:
            self.logger.error(f"Error in async store for {symbol}: {e}")
            return False

    def store_sync(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store data using legacy storage (sync)"""
        try:
            # Delegate to legacy manager if it has storage methods
            if hasattr(self.existing_storage_manager, '_save_to_parquet'):
                return self.existing_storage_manager._save_to_parquet(data, symbol)
            return False
        except Exception as e:
            self.logger.error(f"Error in sync store for {symbol}: {e}")
            return False

    async def load_async(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data using legacy storage (async)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.load_sync, symbol, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error in async load for {symbol}: {e}")
            return pd.DataFrame()

    def load_sync(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data using legacy storage (sync)"""
        try:
            # Delegate to legacy manager if it has load methods
            if hasattr(self.existing_storage_manager, '_load_parquet_data'):
                return self.existing_storage_manager._load_parquet_data(symbol, start_date or datetime.min, end_date or datetime.max)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error in sync load for {symbol}: {e}")
            return pd.DataFrame()

    async def delete_async(self, symbol: str) -> bool:
        """Delete data using legacy storage (async)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.delete_sync, symbol)
        except Exception as e:
            self.logger.error(f"Error in async delete for {symbol}: {e}")
            return False

    def delete_sync(self, symbol: str) -> bool:
        """Delete data using legacy storage (sync)"""
        try:
            # Legacy storage doesn't support delete, so we just return False
            return False
        except Exception as e:
            self.logger.error(f"Error in sync delete for {symbol}: {e}")
            return False

    async def get_available_symbols_async(self) -> list:
        """Get available symbols using legacy storage (async)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.get_available_symbols_sync)
        except Exception as e:
            self.logger.error(f"Error in async get available symbols: {e}")
            return []

    def get_available_symbols_sync(self) -> list:
        """Get available symbols using legacy storage (sync)"""
        try:
            if hasattr(self.existing_storage_manager, 'get_available_symbols'):
                return self.existing_storage_manager.get_available_symbols()
            return []
        except Exception as e:
            self.logger.error(f"Error in sync get available symbols: {e}")
            return []


class DataProcessorFactory:
    """
    Factory for creating data processor adapters for legacy components
    """

    @staticmethod
    def create_preprocessor_adapter(existing_preprocessor, config: Optional[DataProcessorConfig] = None) -> DataPreprocessorAdapter:
        """Create DataPreprocessor adapter"""
        return DataPreprocessorAdapter(existing_preprocessor, config)

    @staticmethod
    def create_data_manager_adapter(existing_data_manager, config: Optional[DataProcessorConfig] = None) -> DataManagerAdapter:
        """Create DataManager adapter"""
        return DataManagerAdapter(existing_data_manager, config)

    @staticmethod
    def create_cache_adapter(existing_cache_manager, config: CacheConfig) -> LegacyDataCacheAdapter:
        """Create cache adapter"""
        return LegacyDataCacheAdapter(existing_cache_manager, config)

    @staticmethod
    def create_storage_adapter(existing_storage_manager, config: StorageConfig) -> LegacyDataStorageAdapter:
        """Create storage adapter"""
        return LegacyDataStorageAdapter(existing_storage_manager, config)