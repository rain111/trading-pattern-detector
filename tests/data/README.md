# Enhanced Data Management Test Suite

This directory contains comprehensive tests for the enhanced data management architecture.

## Overview

The test suite covers all components of the enhanced data management system:

1. **Core Interfaces** (`test_core_interfaces.py`)
   - Abstract base classes and interfaces
   - Configuration management
   - Data validation
   - Performance metrics

2. **Async Manager** (`test_async_manager.py`)
   - Enhanced async data manager functionality
   - Smart caching
   - Parquet storage
   - Performance optimization

3. **Adapters** (`test_adapters.py`)
   - Backward compatibility adapters
   - Integration patterns
   - Legacy component support

4. **Enhanced Preprocessor** (`test_enhanced_preprocessor.py`)
   - Async preprocessing methods
   - Technical indicators
   - Parallel processing
   - Validation

5. **Integration Tests** (`test_integration.py`)
   - End-to-end workflows
   - Concurrent operations
   - Error handling
   - Performance benchmarks

## Running Tests

### All Tests
```bash
# Run all tests
python run_tests.py
python run_tests.py all
```

### Specific Test Module
```bash
# Run specific test module
python run_tests.py test_core_interfaces.py
python run_tests.py test_async_manager.py
python run_tests.py test_adapters.py
python run_tests.py test_enhanced_preprocessor.py
python run_tests.py test_integration.py
```

### Using pytest directly
```bash
# Run all tests with coverage
pytest -v --cov=src/data --cov=src/utils/data_preprocessor --cov-report=term-missing

# Run specific test file
pytest tests/data/test_core_interfaces.py -v

# Run with verbose output
pytest tests/data/ -v --tb=short
```

### Test Dependencies
The tests require the following dependencies:
- pytest
- pytest-cov (for coverage)
- pytest-asyncio (for async tests)
- pandas
- numpy
- aiohttp
- pyarrow

## Test Structure

### Core Interfaces Tests
- **CacheConfig**: Test cache configuration management
- **StorageConfig**: Test storage configuration management
- **ValidationConfig**: Test validation configuration management
- **DataProcessor**: Test abstract base class functionality
- **DataProcessorResult**: Test result data structures
- **DataMetrics**: Test performance metrics collection

### Async Manager Tests
- **SmartMemoryCache**: Test in-memory caching with TTL and size limits
- **ParquetStorage**: Test parquet file storage operations
- **EnhancedAsyncDataManager**: Test main async manager functionality
- **APIConfig**: Test API configuration management
- **Performance**: Test concurrent operations and caching performance

### Adapter Tests
- **DataPreprocessorAdapter**: Test backward compatibility adapter
- **DataManagerAdapter**: Test legacy manager integration
- **LegacyDataCacheAdapter**: Test cache adapter
- **LegacyDataStorageAdapter**: Test storage adapter
- **DataProcessorFactory**: Test factory methods

### Enhanced Preprocessor Tests
- **Async Methods**: Test all async preprocessing methods
- **Pipeline Processing**: Test preprocessing pipeline
- **Performance Tracking**: Test performance monitoring
- **Validation**: Test data validation with detailed reporting
- **Error Handling**: Test error handling in async operations

### Integration Tests
- **End-to-End**: Test complete data flow from fetch to storage
- **Concurrent Operations**: Test multiple symbol processing
- **Cache Integration**: Test caching with real data flow
- **Storage Integration**: Test storage operations with real data
- **Performance Metrics**: Test metrics collection across components
- **Error Handling**: Test error handling across all components

## Test Data

Tests use both synthetic data and real market data patterns:
- Synthetic OHLCV data for testing edge cases
- Real market data patterns for realistic testing
- Large datasets for performance testing
- Malformed data for error handling testing

## Performance Considerations

Tests are designed to be efficient:
- Mock external APIs for reliable testing
- Use small datasets for unit tests
- Parallel execution for integration tests
- Performance benchmarks for optimization validation

## Coverage Information

Tests provide comprehensive coverage of:
- All public methods and properties
- Error handling paths
- Edge cases and boundary conditions
- Concurrent operation scenarios
- Data validation scenarios

## Continuous Integration

This test suite is designed for continuous integration:
- Fast execution for CI/CD pipelines
- Detailed failure reporting
- Coverage reporting
- Consistent test environments

## Debugging Tips

When debugging tests:
1. Run tests with verbose output: `pytest -v --tb=long`
2. Run specific test methods: `pytest tests/data/test_core_interfaces.py::TestDataProcessor::test_method`
3. Use breakpoints with `pytest --pdb`
4. Check coverage reports to identify untested code
5. Review test data for realistic scenarios

## Adding New Tests

When adding new tests:
1. Follow the existing naming conventions
2. Include both positive and negative test cases
3. Mock external dependencies
4. Test error handling paths
5. Document any special test requirements
6. Ensure tests are independent and can run in any order

## Mock Strategy

Tests use strategic mocking:
- External APIs are mocked to avoid real network calls
- Time-dependent components use fixed timestamps
- Random data is seeded for reproducible tests
- File operations use temporary directories
- Concurrency issues are tested with controlled scenarios