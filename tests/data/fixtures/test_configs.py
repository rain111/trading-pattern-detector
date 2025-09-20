"""
Test configuration fixtures for enhanced data management system.
Provides various configuration scenarios for testing.
"""

import sys
from pathlib import Path
# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

from src.data.core import (
    CacheConfig, StorageConfig, ValidationConfig, DataProcessorConfig,
    CacheType, StorageType
)
from src.data.async_manager import APIConfig


def create_standard_cache_config() -> CacheConfig:
    """Create standard cache configuration for testing."""
    return CacheConfig(
        cache_type=CacheType.MEMORY,
        ttl_seconds=3600,
        max_size=1000
    )


def create_large_cache_config() -> CacheConfig:
    """Create large cache configuration for performance testing."""
    return CacheConfig(
        cache_type=CacheType.MEMORY,
        ttl_seconds=7200,
        max_size=10000
    )


def create_disk_cache_config() -> CacheConfig:
    """Create disk cache configuration."""
    return CacheConfig(
        cache_type=CacheType.DISK,
        ttl_seconds=86400,  # 24 hours
        max_size=50000
    )


def create_strict_cache_config() -> CacheConfig:
    """Create strict cache configuration with small TTL."""
    return CacheConfig(
        cache_type=CacheType.MEMORY,
        ttl_seconds=60,  # 1 minute
        max_size=100
    )


def create_standard_storage_config() -> StorageConfig:
    """Create standard storage configuration."""
    temp_dir = tempfile.mkdtemp()
    return StorageConfig(
        storage_type=StorageType.PARQUET,
        base_path=temp_dir,
        compression="snappy"
    )


def create_json_storage_config() -> StorageConfig:
    """Create JSON storage configuration."""
    temp_dir = tempfile.mkdtemp()
    return StorageConfig(
        storage_type=StorageType.JSON,
        base_path=temp_dir
    )


def create_hdf5_storage_config() -> StorageConfig:
    """Create HDF5 storage configuration."""
    temp_dir = tempfile.mkdtemp()
    return StorageConfig(
        storage_type=StorageType.HDF5,
        base_path=temp_dir
    )


def create_validation_config() -> ValidationConfig:
    """Create standard validation configuration."""
    return ValidationConfig(
        min_data_points=10,
        max_missing_pct=5.0,
        price_consistency_check=True,
        volume_consistency_check=True
    )


def create_strict_validation_config() -> ValidationConfig:
    """Create strict validation configuration."""
    return ValidationConfig(
        min_data_points=30,
        max_missing_pct=1.0,
        price_consistency_check=True,
        volume_consistency_check=True,
        outlier_detection=True,
        trend_validation=True
    )


def create_permissive_validation_config() -> ValidationConfig:
    """Create permissive validation configuration."""
    return ValidationConfig(
        min_data_points=1,
        max_missing_pct=50.0,
        price_consistency_check=False,
        volume_consistency_check=False
    )


def create_standard_data_processor_config() -> DataProcessorConfig:
    """Create standard data processor configuration."""
    temp_dir = tempfile.mkdtemp()
    return DataProcessorConfig(
        cache_config=CacheConfig(
            cache_type=CacheType.MEMORY,
            ttl_seconds=3600,
            max_size=1000
        ),
        storage_config=StorageConfig(
            storage_type=StorageType.PARQUET,
            base_path=temp_dir,
            compression="snappy"
        ),
        validation_config=ValidationConfig(
            min_data_points=10,
            max_missing_pct=5.0,
            price_consistency_check=True,
            volume_consistency_check=True
        ),
        max_concurrent_requests=5,
        request_timeout=30,
        retry_attempts=3,
        retry_delay=1.0
    )


def create_high_performance_config() -> DataProcessorConfig:
    """Create high-performance data processor configuration."""
    temp_dir = tempfile.mkdtemp()
    return DataProcessorConfig(
        cache_config=CacheConfig(
            cache_type=CacheType.MEMORY,
            ttl_seconds=7200,
            max_size=10000
        ),
        storage_config=StorageConfig(
            storage_type=StorageType.PARQUET,
            base_path=temp_dir,
            compression="gzip"
        ),
        validation_config=ValidationConfig(
            min_data_points=5,
            max_missing_pct=10.0,
            price_consistency_check=True,
            volume_consistency_check=True
        ),
        max_concurrent_requests=20,
        request_timeout=10,
        retry_attempts=2,
        retry_delay=0.5
    )


def create_conservative_config() -> DataProcessorConfig:
    """Create conservative data processor configuration."""
    temp_dir = tempfile.mkdtemp()
    return DataProcessorConfig(
        cache_config=CacheConfig(
            cache_type=CacheType.MEMORY,
            ttl_seconds=1800,  # 30 minutes
            max_size=500
        ),
        storage_config=StorageConfig(
            storage_type=StorageType.PARQUET,
            base_path=temp_dir,
            compression="snappy"
        ),
        validation_config=ValidationConfig(
            min_data_points=20,
            max_missing_pct=2.0,
            price_consistency_check=True,
            volume_consistency_check=True
        ),
        max_concurrent_requests=2,
        request_timeout=60,
        retry_attempts=5,
        retry_delay=2.0
    )


def create_api_configs() -> dict:
    """Create various API configurations for testing."""
    return {
        'yahoo_finance': APIConfig(
            name="yahoo_finance",
            base_url="https://query1.finance.yahoo.com",
            timeout=30,
            rate_limit=5,
            retry_attempts=3,
            retry_delay=1.0
        ),
        'alpha_vantage': APIConfig(
            name="alpha_vantage",
            base_url="https://www.alphavantage.co",
            timeout=20,
            rate_limit=25,
            retry_attempts=2,
            retry_delay=0.5,
            api_key="demo_key"
        ),
        'mock_api': APIConfig(
            name="mock_api",
            base_url="https://mock.api.com",
            timeout=10,
            rate_limit=10,
            retry_attempts=1,
            retry_delay=0.1
        ),
        'slow_api': APIConfig(
            name="slow_api",
            base_url="https://slow.api.com",
            timeout=60,
            rate_limit=2,
            retry_attempts=4,
            retry_delay=3.0
        ),
        'unreliable_api': APIConfig(
            name="unreliable_api",
            base_url="https://unreliable.api.com",
            timeout=15,
            rate_limit=3,
            retry_attempts=6,
            retry_delay=1.5
        )
    }


def create_failure_scenario_configs() -> dict:
    """Create configurations that test failure scenarios."""
    temp_dir = tempfile.mkdtemp()
    return {
        'invalid_cache': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type="invalid_cache_type",  # Invalid type
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'invalid_storage': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type="invalid_storage_type",  # Invalid type
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'invalid_validation': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=-1,  # Invalid negative value
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'zero_timeout': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            request_timeout=0,  # Zero timeout
            retry_attempts=3,
            retry_delay=1.0
        ),
        'negative_concurrent': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            max_concurrent_requests=-1,  # Negative concurrent requests
            request_timeout=30,
            retry_attempts=3,
            retry_delay=1.0
        )
    }


def create_edge_case_configs() -> dict:
    """Create configurations that test edge cases."""
    temp_dir = tempfile.mkdtemp()
    return {
        'tiny_cache': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1  # Very small cache
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'huge_ttl': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=31536000,  # 1 year
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'minimal_config': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=0,  # Immediate expiry
                max_size=1
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=1,
                max_missing_pct=100.0,  # Allow all missing
                price_consistency_check=False,
                volume_consistency_check=False
            ),
            max_concurrent_requests=1,
            request_timeout=1,
            retry_attempts=1,
            retry_delay=0
        ),
        'extreme_concurrency': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            max_concurrent_requests=100,  # Very high concurrency
            request_timeout=5,
            retry_attempts=1,
            retry_delay=0.1
        )
    }


def create_performance_test_configs() -> dict:
    """Create configurations for performance testing."""
    temp_dir = tempfile.mkdtemp()
    return {
        'basic_performance': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=5000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            max_concurrent_requests=10,
            request_timeout=20,
            retry_attempts=2,
            retry_delay=0.5
        ),
        'aggressive_performance': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=7200,
                max_size=20000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="lz4"  # Faster compression
            ),
            validation_config=ValidationConfig(
                min_data_points=5,
                max_missing_pct=10.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            max_concurrent_requests=50,
            request_timeout=10,
            retry_attempts=1,
            retry_delay=0.2
        ),
        'memory_optimized': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=1800,
                max_size=1000  # Small memory footprint
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=20,
                max_missing_pct=3.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            max_concurrent_requests=5,
            request_timeout=30,
            retry_attempts=3,
            retry_delay=1.0
        )
    }


class ConfigManager:
    """Manager for test configurations with automatic cleanup."""

    def __init__(self):
        self.temp_dirs = []

    def create_temp_dir(self) -> str:
        """Create a temporary directory and track it for cleanup."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up all temporary directories."""
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def get_all_configs():
    """Get all configuration scenarios organized by category."""
    with ConfigManager() as config_manager:
        return {
            'standard': {
                'cache': create_standard_cache_config(),
                'storage': create_standard_storage_config(),
                'validation': create_validation_config(),
                'data_processor': create_standard_data_processor_config()
            },
            'performance': {
                'cache': create_large_cache_config(),
                'storage': create_standard_storage_config(),
                'validation': create_validation_config(),
                'data_processor': create_high_performance_config()
            },
            'conservative': {
                'cache': create_strict_cache_config(),
                'storage': create_standard_storage_config(),
                'validation': create_strict_validation_config(),
                'data_processor': create_conservative_config()
            },
            'edge_cases': {
                'cache': create_strict_cache_config(),
                'storage': create_standard_storage_config(),
                'validation': create_permissive_validation_config(),
                'data_processor': create_minimal_config()
            },
            'api_configs': create_api_configs(),
            'failure_scenarios': create_failure_scenario_configs(),
            'edge_case_configs': create_edge_case_configs(),
            'performance_test_configs': create_performance_test_configs()
        }


def create_minimal_config():
    """Create minimal configuration for edge case testing."""
    temp_dir = tempfile.mkdtemp()
    return DataProcessorConfig(
        cache_config=CacheConfig(
            cache_type=CacheType.MEMORY,
            ttl_seconds=3600,
            max_size=10
        ),
        storage_config=StorageConfig(
            storage_type=StorageType.PARQUET,
            base_path=temp_dir,
            compression="snappy"
        ),
        validation_config=ValidationConfig(
            min_data_points=1,
            max_missing_pct=100.0,
            price_consistency_check=False,
            volume_consistency_check=False
        ),
        max_concurrent_requests=1,
        request_timeout=10,
        retry_attempts=1,
        retry_delay=0
    )


def create_edge_case_configs():
    """Create configurations that test edge cases."""
    temp_dir = tempfile.mkdtemp()
    return {
        'tiny_cache': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1  # Very small cache
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'huge_ttl': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=31536000,  # 1 year
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            )
        ),
        'minimal_config': create_minimal_config(),
        'extreme_concurrency': DataProcessorConfig(
            cache_config=CacheConfig(
                cache_type=CacheType.MEMORY,
                ttl_seconds=3600,
                max_size=1000
            ),
            storage_config=StorageConfig(
                storage_type=StorageType.PARQUET,
                base_path=temp_dir,
                compression="snappy"
            ),
            validation_config=ValidationConfig(
                min_data_points=10,
                max_missing_pct=5.0,
                price_consistency_check=True,
                volume_consistency_check=True
            ),
            max_concurrent_requests=100,  # Very high concurrency
            request_timeout=5,
            retry_attempts=1,
            retry_delay=0.1
        )
    }


def save_config_to_file(config: dict, file_path: str):
    """Save configuration dictionary to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_config_from_file(file_path: str) -> dict:
    """Load configuration dictionary from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test configuration generation
    print("Testing configuration generation...")

    configs = get_all_configs()
    print(f"Generated {len(configs)} configuration categories")

    # Test individual config creation
    standard_config = create_standard_data_processor_config()
    print(f"Standard config created with max_concurrent_requests={standard_config.max_concurrent_requests}")

    # Test API config creation
    api_configs = create_api_configs()
    print(f"Created {len(api_configs)} API configurations")

    # Test failure scenario config creation
    failure_configs = create_failure_scenario_configs()
    print(f"Created {len(failure_configs)} failure scenario configurations")

    print("Configuration testing completed!")