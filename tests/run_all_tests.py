#!/usr/bin/env python3
"""
Comprehensive test runner for the trading pattern detection system.
This script runs all tests and provides a summary of results.
"""

import pytest
import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_all_tests():
    """Run all tests in the test suite"""
    try:
        logger.info("Starting comprehensive test suite...")

        # Get the tests directory
        tests_dir = Path(__file__).parent

        # Run tests with pytest
        exit_code = pytest.main(
            [
                str(tests_dir),
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                "--strict-markers",  # Strict marker handling
                "--disable-warnings",  # Disable warnings for cleaner output
                "-x",  # Stop on first failure
                "--cov=src",  # Coverage analysis
                "--cov-report=html",  # HTML coverage report
                "--cov-report=term",  # Terminal coverage report
                "--cov-fail-under=80",  # Minimum coverage threshold
            ]
        )

        if exit_code == 0:
            logger.info("All tests passed successfully! ✅")
            return True
        else:
            logger.error(f"Tests failed with exit code: {exit_code} ❌")
            return False

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def run_specific_test_module(module_name):
    """Run a specific test module"""
    try:
        logger.info(f"Running test module: {module_name}")

        tests_dir = Path(__file__).parent
        exit_code = pytest.main(
            [str(tests_dir / f"test_{module_name}.py"), "-v", "--tb=short"]
        )

        return exit_code == 0

    except Exception as e:
        logger.error(f"Error running {module_name} tests: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        module_name = sys.argv[1]
        success = run_specific_test_module(module_name)
    else:
        # Run all tests
        success = run_all_tests()

    sys.exit(0 if success else 1)
