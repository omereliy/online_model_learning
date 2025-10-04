#!/usr/bin/env python
"""
Run complete test suite to verify implementation.
Must pass before marking any task as complete.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test suites in order of dependency
TEST_SUITES = [
    ("Basic Metrics - Unit Tests", "tests/experiments/test_metrics.py::TestMetricsCollector", 30),
    ("Mock Environment - Unit Tests", "tests/integration/test_phase3_simple.py::TestPhase3Simple", 30),
    ("Experiment Runner - Unit Tests", "tests/experiments/test_experiment_runner.py::TestExperimentRunner", 60),
    ("Integration Tests - OLAM Simple", "tests/algorithms/test_olam_simple.py::TestOLAMSimple", 60),
    ("Integration Tests - Full Pipeline", "tests/experiments/test_experiment_integration.py::TestExperimentIntegration", 120),
]

# Quick test suite for rapid feedback
QUICK_TESTS = [
    ("Critical Metrics Test", "tests/experiments/test_metrics.py::TestMetricsCollector::test_get_snapshot", 10),
    ("Basic Recording", "tests/experiments/test_metrics.py::TestMetricsCollector::test_record_action", 10),
    ("Mistake Rate Calculation", "tests/experiments/test_metrics.py::TestMetricsCollector::test_compute_mistake_rate", 10),
]


def run_single_test(name: str, test_path: str, timeout: int = 30, verbose: bool = True) -> tuple:
    """
    Run a single test or test suite.

    Returns:
        Tuple of (success: bool, output: str, runtime: float)
    """
    start_time = time.time()

    cmd = [sys.executable, "-m", "pytest", test_path, "-v" if verbose else "-q", "--tb=short"]

    # Add timeout if pytest-timeout is available
    try:
        import pytest_timeout
        cmd.extend([f"--timeout={timeout}"])
    except ImportError:
        pass

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10  # Give extra time for pytest to handle its own timeout
        )

        runtime = time.time() - start_time
        success = result.returncode == 0

        # Combine stdout and stderr for output
        output = result.stdout + result.stderr

        return success, output, runtime

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        return False, f"Test timed out after {timeout} seconds", runtime
    except Exception as e:
        runtime = time.time() - start_time
        return False, f"Error running test: {str(e)}", runtime


def run_test_suite(quick: bool = False):
    """
    Run all tests and report results.

    Args:
        quick: If True, run only quick critical tests
    """
    test_list = QUICK_TESTS if quick else TEST_SUITES

    failed_tests = []
    passed_tests = []
    total_runtime = 0

    print("=" * 80)
    print(f"RUNNING {'QUICK' if quick else 'COMPLETE'} TEST SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for name, test_path, timeout in test_list:
        print(f"\n[{len(passed_tests) + len(failed_tests) + 1}/{len(test_list)}] Running: {name}")
        print("-" * 40)

        success, output, runtime = run_single_test(name, test_path, timeout)
        total_runtime += runtime

        if success:
            print(f"✓ {name} PASSED ({runtime:.2f}s)")
            passed_tests.append((name, runtime))

            # Show test counts from output if available
            if "passed" in output:
                lines = output.split('\n')
                for line in lines:
                    if "passed" in line and "in" in line:
                        print(f"  {line.strip()}")
                        break
        else:
            print(f"✗ {name} FAILED ({runtime:.2f}s)")
            failed_tests.append((name, runtime))

            # Show error details
            if "FAILED" in output or "ERROR" in output:
                print("\n  Error details:")
                lines = output.split('\n')
                error_lines = []
                for i, line in enumerate(lines):
                    if "FAILED" in line or "ERROR" in line or "assert" in line.lower():
                        # Get context around error
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        error_lines.extend(lines[start:end])

                # Show unique error lines (remove duplicates)
                shown = set()
                for line in error_lines[-10:]:  # Show last 10 relevant lines
                    if line.strip() and line not in shown:
                        print(f"    {line[:200]}")  # Truncate long lines
                        shown.add(line)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Passed: {len(passed_tests)}/{len(test_list)}")
    print(f"Failed: {len(failed_tests)}/{len(test_list)}")

    if passed_tests:
        print("\n✓ Passed tests:")
        for test, runtime in passed_tests:
            print(f"  - {test} ({runtime:.2f}s)")

    if failed_tests:
        print("\n✗ Failed tests:")
        for test, runtime in failed_tests:
            print(f"  - {test} ({runtime:.2f}s)")
        print("\nPlease fix the failing tests before marking tasks as complete.")
        return 1
    else:
        print("\n" + "✓" * 40)
        print("✓ ALL TESTS PASSED - Implementation verified! ✓")
        print("✓" * 40)
        return 0


def main():
    """Main entry point with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run test suite for online model learning project')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run only quick critical tests')
    parser.add_argument('--test', '-t', type=str,
                       help='Run a specific test file or test case')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available test suites')

    args = parser.parse_args()

    if args.list:
        print("Available test suites:")
        print("\nComplete suite:")
        for name, path, timeout in TEST_SUITES:
            print(f"  - {name} ({timeout}s timeout)")
            print(f"    Path: {path}")
        print("\nQuick tests:")
        for name, path, timeout in QUICK_TESTS:
            print(f"  - {name} ({timeout}s timeout)")
        return 0

    if args.test:
        # Run specific test
        print(f"Running specific test: {args.test}")
        success, output, runtime = run_single_test("Custom test", args.test, timeout=60)
        print(output)
        print(f"\nTest {'PASSED' if success else 'FAILED'} in {runtime:.2f}s")
        return 0 if success else 1

    # Run test suite
    return run_test_suite(quick=args.quick)


if __name__ == "__main__":
    sys.exit(main())