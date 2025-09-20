#!/usr/bin/env python3
"""
Test runner for the enhanced data management architecture.
Run all data management tests or specific test modules.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test_module(module_name):
    """Run a specific test module"""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            module_name,
            "-v",
            "--tb=short",
            "--cov=src/data",
            "--cov=src/utils/data_preprocessor",
            "--cov-report=term-missing"
        ], cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)

        print(f"\n{'='*60}")
        print(f"Running tests for: {module_name}")
        print(f"{'='*60}")

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests for {module_name}: {e}")
        return False

def run_all_tests():
    """Run all test modules"""
    test_modules = [
        "test_core_interfaces.py",
        "test_async_manager.py",
        "test_adapters.py",
        "test_enhanced_preprocessor.py",
        "test_integration.py"
    ]

    print("Starting enhanced data management architecture test suite...")
    print("=" * 80)

    total_tests = len(test_modules)
    passed_tests = 0
    failed_tests = []

    for i, module in enumerate(test_modules, 1):
        print(f"\n[{i}/{total_tests}] Running {module}")

        success = run_test_module(f"tests/data/{module}")

        if success:
            passed_tests += 1
            print(f"✅ {module} - PASSED")
        else:
            failed_tests.append(module)
            print(f"❌ {module} - FAILED")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for module in failed_tests:
            print(f"  - {module}")

    success_rate = (passed_tests / total_tests) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("🎉 Test suite PASSED overall!")
    else:
        print("⚠️ Test suite FAILED overall!")

    return len(failed_tests) == 0

def run_single_test():
    """Run a single test module"""
    if len(sys.argv) > 2:
        module_name = sys.argv[2]
        print(f"Running specific test module: {module_name}")
        return run_test_module(f"tests/data/{module_name}")
    else:
        print("Error: Please specify a test module name")
        print("Usage: python run_tests.py <module_name>")
        print("Available modules:")
        modules = [
            "test_core_interfaces.py",
            "test_async_manager.py",
            "test_adapters.py",
            "test_enhanced_preprocessor.py",
            "test_integration.py"
        ]
        for module in modules:
            print(f"  - {module}")
        return False

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        # Run all tests by default
        run_all_tests()
    elif sys.argv[1] == "all":
        run_all_tests()
    else:
        run_single_test()

if __name__ == "__main__":
    main()