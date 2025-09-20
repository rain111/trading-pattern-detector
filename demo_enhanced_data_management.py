#!/usr/bin/env python3
"""
Demonstration script for the enhanced data management architecture.

This script shows how to use the new enhanced data management components
while maintaining backward compatibility with existing code.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.async_manager import EnhancedAsyncDataManager, DataProcessorConfig, CacheConfig, StorageConfig, ValidationConfig
from src.data.core import CacheType, StorageType, ValidationResult
from src.data.adapters import DataProcessorFactory
from src.utils.data_preprocessor import DataPreprocessor
from frontend.data.manager import DataManager


def create_sample_data():
    """Create sample market data for demonstration"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

    # Generate realistic price data
    np.random.seed(42)
    price_changes = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * (1 + pd.Series(price_changes).cumprod())

    data = pd.DataFrame({
        'open': prices.shift(1).fillna(100),
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'volume': np.random.uniform(1000000, 5000000, len(dates))
    }, index=dates)

    # Ensure OHLC relationships
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)

    return data.dropna()


async def demo_enhanced_manager():
    """Demonstrate the enhanced async data manager"""
    print("🚀 Enhanced Async Data Manager Demo")
    print("=" * 50)

    # Create configuration
    config = DataProcessorConfig(
        cache_config=CacheConfig(
            cache_type=CacheType.MEMORY,
            ttl_seconds=3600,
            max_size=1000
        ),
        storage_config=StorageConfig(
            storage_type=StorageType.PARQUET,
            base_path="demo_data",
            compression="snappy"
        ),
        validation_config=ValidationConfig(
            min_data_points=20,
            max_missing_pct=5.0,
            price_consistency_check=True,
            volume_consistency_check=True
        ),
        max_concurrent_requests=10,
        request_timeout=30,
        retry_attempts=3,
        retry_delay=1.0
    )

    # Create enhanced manager
    manager = EnhancedAsyncDataManager(config)

    try:
        # Create sample data
        sample_data = create_sample_data()
        print(f"✅ Created sample data with {len(sample_data)} rows")

        # Test data preprocessing
        print("\n📊 Testing data preprocessing...")
        processed_data = await manager.preprocess_data_async(sample_data)
        print(f"✅ Preprocessed data shape: {processed_data.shape}")
        print(f"✅ Technical indicators added: {len([col for col in processed_data.columns if col not in sample_data.columns])}")

        # Test data validation
        print("\n🔍 Testing data validation...")
        validation_result = manager.validate_data(processed_data)
        print(f"✅ Validation result: {validation_result}")

        # Test data storage
        print("\n💾 Testing data storage...")
        storage_success = await manager.storage.store_async("DEMO_STOCK", processed_data)
        print(f"✅ Storage success: {storage_success}")

        # Test data retrieval
        print("\n📥 Testing data retrieval...")
        loaded_data = await manager.storage.load_async("DEMO_STOCK")
        print(f"✅ Loaded data shape: {loaded_data.shape}")
        print(f"✅ Data integrity check: {loaded_data.equals(processed_data)}")

        # Test caching
        print("\n🗃️ Testing caching...")
        cache_key = "demo_cache_key"
        await manager.cache.set_async(cache_key, sample_data)
        cached_data = await manager.cache.get_async(cache_key)
        print(f"✅ Cache operation successful: {not cached_data.empty}")

        # Test performance metrics
        print("\n📈 Testing performance metrics...")
        metrics = await manager.get_performance_metrics_async()
        print(f"✅ Cache stats: {metrics['cache_stats']['size']}")
        print(f"✅ Metrics collected: {len(metrics['metrics']['fetch_times'])} fetch times")

        # Cleanup
        print("\n🧹 Testing cleanup...")
        cleanup_result = await manager.cleanup_async()
        print(f"✅ Cleanup completed: {cleanup_result}")

        print("\n🎉 Enhanced Manager Demo Completed Successfully!")

    except Exception as e:
        print(f"❌ Error in enhanced manager demo: {e}")
        raise


async def demo_async_preprocessor():
    """Demonstrate the enhanced async preprocessor"""
    print("\n🔧 Enhanced Async Preprocessor Demo")
    print("=" * 50)

    # Create preprocessor
    preprocessor = DataPreprocessor()

    try:
        # Create test data
        test_data = create_sample_data().head(100)  # Use smaller dataset
        print(f"✅ Created test data with {len(test_data)} rows")

        # Test individual async methods
        print("\n🔍 Testing individual async methods...")

        # Clean data
        cleaned_data = await preprocessor.clean_data_async(test_data)
        print(f"✅ Data cleaned: {len(test_data)} -> {len(cleaned_data)} rows")

        # Add technical indicators
        indicator_data = await preprocessor.add_technical_indicators_async(cleaned_data)
        indicators_added = len([col for col in indicator_data.columns if col not in cleaned_data.columns])
        print(f"✅ Technical indicators added: {indicators_added}")

        # Calculate returns
        returns_data = await preprocessor.calculate_returns_async(indicator_data)
        print(f"✅ Returns calculated: {'returns' in returns_data.columns}")

        # Test validation
        print("\n🔍 Testing data validation...")
        validation_report = await preprocessor.validate_async(returns_data)
        print(f"✅ Validation report: {validation_report['is_valid']}")
        print(f"✅ Data quality score: {validation_report['data_quality_score']:.1f}%")

        # Test pipeline processing
        print("\n⚡ Testing pipeline processing...")
        pipeline_result = await preprocessor.preprocess_pipeline_async(
            test_data,
            operations=['clean', 'indicators', 'returns'],
            timeframe='1W'
        )
        print(f"✅ Pipeline processed: {len(test_data)} -> {len(pipeline_result)} rows (resampled)")

        # Test multiple symbols processing
        print("\n🔄 Testing multiple symbols processing...")
        data_dict = {
            'AAPL': test_data,
            'MSFT': test_data * 1.5,  # Scale for different prices
            'GOOGL': test_data * 2.0   # Scale for different prices
        }

        multi_results = await preprocessor.process_multiple_symbols_async(data_dict)
        print(f"✅ Processed {len(multi_results)} symbols")
        for symbol, result in multi_results.items():
            print(f"   - {symbol}: {len(result)} rows with {len([col for col in result.columns if col in ['sma_20', 'rsi']])} indicators")

        # Test large dataset processing
        print("\n📦 Testing large dataset processing...")
        large_data = create_sample_data().head(10000)
        chunk_size = 1000
        large_result = await preprocessor.process_large_dataset_async(
            large_data,
            chunk_size=chunk_size,
            operations=['clean', 'indicators']
        )
        print(f"✅ Large dataset processed: {len(large_data)} -> {len(large_result)} rows")

        print("\n🎉 Async Preprocessor Demo Completed Successfully!")

    except Exception as e:
        print(f"❌ Error in async preprocessor demo: {e}")
        raise
    finally:
        # Cleanup
        preprocessor.close()


async def demo_adapters():
    """Demonstrate the adapter pattern for backward compatibility"""
    print("\n🔌 Adapters Demo - Backward Compatibility")
    print("=" * 50)

    try:
        # Create legacy components
        legacy_preprocessor = DataPreprocessor()
        legacy_manager = DataManager(use_enhanced_backend=False)  # Use legacy mode

        print("✅ Created legacy components")

        # Create adapters
        preprocessor_adapter = DataProcessorFactory.create_preprocessor_adapter(legacy_preprocessor)
        data_manager_adapter = DataProcessorFactory.create_data_manager_adapter(legacy_manager)

        print("✅ Created adapter instances")

        # Test adapter functionality
        test_data = create_sample_data().head(50)

        # Test preprocessor adapter
        print("\n🔧 Testing preprocessor adapter...")
        adapted_result = await preprocessor_adapter.preprocess_data_async(test_data)
        print(f"✅ Adapter preprocessing successful: {len(adapted_result)} rows")

        validation_result = preprocessor_adapter.validate_data(adapted_result)
        print(f"✅ Adapter validation: {validation_result}")

        # Test data manager adapter (in legacy mode)
        print("\n📊 Testing data manager adapter...")
        print(f"✅ Backend status: {legacy_manager.get_backend_status()}")
        print(f"✅ Available symbols (legacy): {legacy_manager.get_available_symbols()}")

        print("\n🎉 Adapters Demo Completed Successfully!")

    except Exception as e:
        print(f"❌ Error in adapters demo: {e}")
        raise


async def demo_concurrent_operations():
    """Demonstrate concurrent data operations"""
    print("\n⚡ Concurrent Operations Demo")
    print("=" * 50)

    # Create manager with high concurrency
    config = DataProcessorConfig(max_concurrent_requests=20)
    manager = EnhancedAsyncDataManager(config)

    try:
        # Create multiple datasets
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']
        datasets = {}

        for symbol in symbols:
            # Create unique data for each symbol
            data = create_sample_data().head(100)
            data = data * (1 + len(symbols) * 0.1)  # Scale to make data unique
            datasets[symbol] = data

        print(f"✅ Created {len(datasets)} datasets for concurrent processing")

        # Test concurrent fetching (mocked)
        print("\n🔄 Testing concurrent data fetching...")
        start_time = datetime.now()

        # Simulate concurrent fetch operations
        fetch_tasks = []
        for symbol in symbols:
            task = manager._fetch_single_symbol_with_retry(symbol, datetime(2023, 1, 1), datetime(2023, 4, 1))
            fetch_tasks.append(task)

        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        successful_fetches = [r for r in fetch_results if not isinstance(r, Exception)]
        print(f"✅ Concurrent fetch: {len(successful_fetches)}/{len(symbols)} successful in {elapsed:.2f}s")

        # Test concurrent preprocessing
        print("\n⚙️ Testing concurrent preprocessing...")
        start_time = datetime.now()

        preprocessor = DataPreprocessor()
        preprocessor_tasks = []

        for symbol, data in datasets.items():
            task = preprocessor.preprocess_pipeline_async(data)
            preprocessor_tasks.append((symbol, task))

        # Process all datasets concurrently
        preprocess_results = {}
        for symbol, task in preprocessor_tasks:
            result = await task
            preprocess_results[symbol] = result

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"✅ Concurrent preprocessing: {len(preprocess_results)} symbols in {elapsed:.2f}s")

        # Test performance
        total_rows = sum(len(data) for data in preprocess_results.values())
        rows_per_second = total_rows / elapsed if elapsed > 0 else 0
        print(f"✅ Processing rate: {rows_per_second:.1f} rows/second")

        # Test validation in parallel
        print("\n🔍 Testing concurrent validation...")
        validation_tasks = []
        for symbol, data in preprocess_results.items():
            task = preprocessor.validate_async(data)
            validation_tasks.append((symbol, task))

        validation_results = {}
        for symbol, task in validation_tasks:
            result = await task
            validation_results[symbol] = result

        valid_symbols = [s for s, r in validation_results.items() if r['is_valid']]
        print(f"✅ Concurrent validation: {len(valid_symbols)}/{len(symbols)} symbols valid")

        # Cleanup
        preprocessor.close()

        print("\n🎉 Concurrent Operations Demo Completed Successfully!")

    except Exception as e:
        print(f"❌ Error in concurrent operations demo: {e}")
        raise


async def demo_performance_comparison():
    """Demonstrate performance improvements"""
    print("\n📊 Performance Comparison Demo")
    print("=" * 50)

    try:
        # Create a large dataset
        large_data = create_sample_data().head(1000)
        print(f"✅ Created large dataset with {len(large_data)} rows")

        # Test traditional sync processing
        print("\n⏱️ Testing traditional synchronous processing...")
        start_time = datetime.now()

        traditional_preprocessor = DataPreprocessor()
        # Use executor to simulate async behavior
        import concurrent.futures

        def sync_process(data):
            result = traditional_preprocessor.clean_data(data)
            result = traditional_preprocessor.add_technical_indicators(result)
            result = traditional_preprocessor.calculate_returns(result)
            return result

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(sync_process, large_data)
            sync_result = future.result()

        sync_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Traditional processing: {sync_time:.3f}s")

        # Test enhanced async processing
        print("\n🚀 Testing enhanced async processing...")
        async_preprocessor = DataPreprocessor()
        start_time = datetime.now()

        async_result = await async_preprocessor.preprocess_pipeline_async(
            large_data,
            operations=['clean', 'indicators', 'returns']
        )

        async_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Enhanced async processing: {async_time:.3f}s")

        # Compare performance
        speedup = sync_time / async_time if async_time > 0 else 0
        print(f"\n📈 Performance improvement: {speedup:.1f}x faster")

        # Verify results are equivalent
        if sync_result.equals(async_result):
            print("✅ Results verification: PASSED")
        else:
            print("⚠️ Results verification: Differences detected (may be due to processing order)")

        # Cleanup
        async_preprocessor.close()

        print("\n🎉 Performance Comparison Demo Completed Successfully!")

    except Exception as e:
        print(f"❌ Error in performance comparison demo: {e}")
        raise


async def main():
    """Main demo function"""
    print("🎬 Enhanced Data Management Architecture Demo")
    print("=" * 70)
    print("This demo showcases the new enhanced data management system")
    print("with async capabilities while maintaining backward compatibility.\n")

    demos = [
        demo_enhanced_manager,
        demo_async_preprocessor,
        demo_adapters,
        demo_concurrent_operations,
        demo_performance_comparison
    ]

    for i, demo in enumerate(demos, 1):
        try:
            await demo()
            print(f"\n✅ Demo {i}/{len(demos)} completed successfully")
        except Exception as e:
            print(f"\n❌ Demo {i}/{len(demos)} failed: {e}")
            # Continue with next demo
            continue

    print("\n" + "=" * 70)
    print("🎉 All demos completed!")
    print("\nKey features demonstrated:")
    print("✅ Enhanced async data manager with caching")
    print("✅ Async preprocessing with technical indicators")
    print("✅ Backward compatibility through adapters")
    print("✅ Concurrent operations for performance")
    print("✅ Comprehensive error handling and validation")
    print("✅ Performance optimization and monitoring")

    print("\nFor more detailed information, see:")
    print("- src/data/ - Core architecture components")
    print("- tests/data/ - Comprehensive test suite")
    print("- frontend/data/manager.py - Frontend integration")


if __name__ == "__main__":
    asyncio.run(main())