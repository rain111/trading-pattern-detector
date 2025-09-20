"""
Core data management interfaces for the trading pattern detection system.
Provides unified interfaces for data processing, caching, storage, and validation.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum
import logging


class CacheType(Enum):
    """Cache storage types"""
    MEMORY = "memory"
    DISK = "disk"
    PARQUET = "parquet"
    HYBRID = "hybrid"


class StorageType(Enum):
    """Storage backend types"""
    PARQUET = "parquet"
    CSV = "csv"
    DATABASE = "database"
    LOCAL = "local"


class DataFormat(Enum):
    """Supported data formats"""
    OHLCV = "ohlcv"
    TICK = "tick"
    ORDERBOOK = "orderbook"
    CUSTOM = "custom"


class ValidationResult(Enum):
    """Data validation results"""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    NEEDS_CLEANING = "needs_cleaning"


class CacheConfig:
    """Configuration for caching strategies"""

    def __init__(
        self,
        cache_type: CacheType = CacheType.MEMORY,
        ttl_seconds: int = 3600,
        max_size: int = 10000,
        compression: bool = True,
        parquet_path: Optional[str] = None
    ):
        self.cache_type = cache_type
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.compression = compression
        self.parquet_path = parquet_path
        self.logger = logging.getLogger(self.__class__.__name__)


class StorageConfig:
    """Configuration for storage backends"""

    def __init__(
        self,
        storage_type: StorageType = StorageType.PARQUET,
        base_path: str = "data",
        compression: str = "snappy",
        partition_size: int = 100000,
        format: DataFormat = DataFormat.OHLCV
    ):
        self.storage_type = storage_type
        self.base_path = base_path
        self.compression = compression
        self.partition_size = partition_size
        self.format = format
        self.logger = logging.getLogger(self.__class__.__name__)


class ValidationConfig:
    """Configuration for data validation"""

    def __init__(
        self,
        min_data_points: int = 20,
        max_missing_pct: float = 5.0,
        price_consistency_check: bool = True,
        volume_consistency_check: bool = True,
        timezone_check: bool = True,
        timezone: str = "UTC"
    ):
        self.min_data_points = min_data_points
        self.max_missing_pct = max_missing_pct
        self.price_consistency_check = price_consistency_check
        self.volume_consistency_check = volume_consistency_check
        self.timezone_check = timezone_check
        self.timezone = timezone
        self.logger = logging.getLogger(self.__class__.__name__)


class DataProcessor(ABC):
    """Abstract base class for data processing operations"""

    def __init__(self, config: Optional[Union[CacheConfig, StorageConfig, ValidationConfig]] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def fetch_data_async(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Asynchronously fetch data for a symbol and date range

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with fetched data
        """
        pass

    @abstractmethod
    def fetch_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Synchronously fetch data for a symbol and date range

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with fetched data
        """
        pass

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data (sync version)

        Args:
            data: Raw data to preprocess

        Returns:
            Preprocessed data
        """
        pass

    @abstractmethod
    async def preprocess_data_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Asynchronously preprocess data

        Args:
            data: Raw data to preprocess

        Returns:
            Preprocessed data
        """
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data integrity

        Args:
            data: Data to validate

        Returns:
            ValidationResult indicating data quality
        """
        pass


class DataCache(ABC):
    """Abstract interface for caching strategies"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def get_async(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache (async)

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        pass

    @abstractmethod
    def get_sync(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache (sync)

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        pass

    @abstractmethod
    async def set_async(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set data in cache (async)

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Optional TTL override

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def set_sync(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set data in cache (sync)

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Optional TTL override

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def delete_async(self, key: str) -> bool:
        """
        Delete data from cache (async)

        Args:
            key: Cache key

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def delete_sync(self, key: str) -> bool:
        """
        Delete data from cache (sync)

        Args:
            key: Cache key

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def clear_expired_async(self) -> int:
        """
        Clear expired cache entries (async)

        Returns:
            Number of entries cleared
        """
        pass

    @abstractmethod
    def clear_expired_sync(self) -> int:
        """
        Clear expired cache entries (sync)

        Returns:
            Number of entries cleared
        """
        pass


class DataStorage(ABC):
    """Abstract interface for storage backends"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def store_async(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store data asynchronously

        Args:
            symbol: Stock symbol
            data: Data to store

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def store_sync(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store data synchronously

        Args:
            symbol: Stock symbol
            data: Data to store

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def load_async(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load data asynchronously

        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Loaded data
        """
        pass

    @abstractmethod
    def load_sync(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load data synchronously

        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Loaded data
        """
        pass

    @abstractmethod
    async def delete_async(self, symbol: str) -> bool:
        """
        Delete stored data asynchronously

        Args:
            symbol: Stock symbol

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def delete_sync(self, symbol: str) -> bool:
        """
        Delete stored data synchronously

        Args:
            symbol: Stock symbol

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def get_available_symbols_async(self) -> List[str]:
        """
        Get list of available symbols asynchronously

        Returns:
            List of available symbols
        """
        pass

    @abstractmethod
    def get_available_symbols_sync(self) -> List[str]:
        """
        Get list of available symbols synchronously

        Returns:
            List of available symbols
        """
        pass


class DataValidator(ABC):
    """Abstract interface for data validation"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def validate_async(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data asynchronously

        Args:
            data: Data to validate

        Returns:
            ValidationResult
        """
        pass

    @abstractmethod
    def validate_sync(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data synchronously

        Args:
            data: Data to validate

        Returns:
            ValidationResult
        """
        pass

    @abstractmethod
    async def get_validation_report_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed validation report asynchronously

        Args:
            data: Data to validate

        Returns:
            Detailed validation report
        """
        pass

    @abstractmethod
    def get_validation_report_sync(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed validation report synchronously

        Args:
            data: Data to validate

        Returns:
            Detailed validation report
        """
        pass

    @abstractmethod
    async def auto_clean_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically clean and fix data issues asynchronously

        Args:
            data: Data to clean

        Returns:
            Cleaned data
        """
        pass

    @abstractmethod
    def auto_clean_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically clean and fix data issues synchronously

        Args:
            data: Data to clean

        Returns:
            Cleaned data
        """
        pass


class DataProcessorConfig:
    """Configuration for DataProcessor implementations"""

    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        storage_config: Optional[StorageConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        max_concurrent_requests: int = 10,
        request_timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        self.cache_config = cache_config
        self.storage_config = storage_config
        self.validation_config = validation_config
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(self.__class__.__name__)


class DataProcessorResult:
    """Result wrapper for data processing operations"""

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        success: bool = False,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None
    ):
        self.data = data
        self.success = success
        self.error_message = error_message
        self.metadata = metadata or {}
        self.processing_time = processing_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "data": self.data,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "processing_time": self.processing_time
        }


class DataMetrics:
    """Performance and data quality metrics"""

    def __init__(self):
        self.fetch_times: List[float] = []
        self.process_times: List[float] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.validation_results: List[ValidationResult] = []
        self.error_counts: Dict[str, int] = {}

    def record_fetch_time(self, time_seconds: float):
        """Record data fetch time"""
        self.fetch_times.append(time_seconds)

    def record_process_time(self, time_seconds: float):
        """Record data processing time"""
        self.process_times.append(time_seconds)

    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1

    def record_validation_result(self, result: ValidationResult):
        """Record validation result"""
        self.validation_results.append(result)

    def record_error(self, error_type: str):
        """Record error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_average_fetch_time(self) -> float:
        """Get average fetch time"""
        return sum(self.fetch_times) / len(self.fetch_times) if self.fetch_times else 0.0

    def get_average_process_time(self) -> float:
        """Get average process time"""
        return sum(self.process_times) / len(self.process_times) if self.process_times else 0.0

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_success_rate(self) -> float:
        """Get validation success rate"""
        if not self.validation_results:
            return 0.0

        valid_count = sum(1 for result in self.validation_results if result == ValidationResult.VALID)
        return valid_count / len(self.validation_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "fetch_times": self.fetch_times,
            "process_times": self.process_times,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "validation_results": [result.value for result in self.validation_results],
            "error_counts": self.error_counts,
            "average_fetch_time": self.get_average_fetch_time(),
            "average_process_time": self.get_average_process_time(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "success_rate": self.get_success_rate()
        }