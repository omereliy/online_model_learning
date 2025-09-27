#!/usr/bin/env python3
"""
Code coverage runner for the online model learning framework.
Analyzes test coverage and generates reports.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Set


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import coverage
        return True
    except ImportError:
        print("Error: coverage package not installed")
        print("Install with: pip install coverage")
        return False


def run_coverage_analysis(test_path: str = "tests/", source_path: str = "src/") -> Dict:
    """
    Run coverage analysis on the codebase.

    Args:
        test_path: Path to test directory
        source_path: Path to source code directory

    Returns:
        Dictionary with coverage statistics
    """
    try:
        import coverage
    except ImportError:
        return {"error": "coverage package not installed"}

    # Initialize coverage
    cov = coverage.Coverage(source=[source_path])

    # Start coverage collection
    cov.start()

    # Run tests
    import pytest
    pytest_args = [test_path, "-q", "--tb=no"]
    result = pytest.main(pytest_args)

    # Stop coverage collection
    cov.stop()
    cov.save()

    # Generate report
    report_data = {}
    try:
        # Get overall statistics
        total = cov.report()
        report_data['total_coverage'] = round(total, 2)

        # Get per-file statistics
        analysis_data = {}
        for filename in cov.get_data().measured_files():
            if source_path in filename:
                analysis = cov.analysis2(filename)
                executed = analysis[1]
                missing = analysis[3]
                if executed or missing:
                    total_lines = len(executed) + len(missing)
                    coverage_pct = (len(executed) / total_lines * 100) if total_lines > 0 else 0
                    relative_path = Path(filename).relative_to(Path.cwd())
                    analysis_data[str(relative_path)] = {
                        'executed_lines': len(executed),
                        'missing_lines': len(missing),
                        'total_lines': total_lines,
                        'coverage_percentage': round(coverage_pct, 2),
                        'missing_line_numbers': missing[:10] if missing else []  # First 10 missing lines
                    }

        report_data['file_coverage'] = analysis_data
        report_data['test_result'] = 'passed' if result == 0 else 'failed'

    except Exception as e:
        report_data['error'] = str(e)

    return report_data


def analyze_test_coverage() -> Dict:
    """
    Analyze test file coverage to identify untested modules.

    Returns:
        Dictionary with test coverage analysis
    """
    src_path = Path("src")
    test_path = Path("tests")

    # Get all Python files in src
    src_files = set()
    for file in src_path.rglob("*.py"):
        if "__pycache__" not in str(file) and file.name != "__init__.py":
            relative = file.relative_to(src_path)
            module_name = str(relative).replace("/", "_").replace(".py", "")
            src_files.add(module_name)

    # Get all test files
    test_files = set()
    tested_modules = set()
    for file in test_path.rglob("test_*.py"):
        if "__pycache__" not in str(file):
            test_files.add(file.name)
            # Extract module name from test file name
            module_name = file.name.replace("test_", "").replace(".py", "")
            tested_modules.add(module_name)

    # Find untested modules
    untested = src_files - tested_modules

    return {
        'total_modules': len(src_files),
        'tested_modules': len(tested_modules),
        'test_files': len(test_files),
        'untested_modules': sorted(list(untested)),
        'test_coverage_percentage': round((len(tested_modules) / len(src_files) * 100), 2) if src_files else 0
    }


def generate_coverage_report(output_format: str = "text") -> None:
    """
    Generate and display coverage report.

    Args:
        output_format: Output format (text, json, html)
    """
    print("=" * 60)
    print("CODE COVERAGE ANALYSIS")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        return

    # Run coverage analysis
    print("\nRunning tests with coverage...")
    coverage_data = run_coverage_analysis()

    if 'error' in coverage_data:
        print(f"Error: {coverage_data['error']}")
        return

    # Display results
    print(f"\n{'='*60}")
    print(f"OVERALL COVERAGE: {coverage_data.get('total_coverage', 0)}%")
    print(f"TEST RESULT: {coverage_data.get('test_result', 'unknown').upper()}")
    print(f"{'='*60}")

    # Display per-file coverage
    if 'file_coverage' in coverage_data:
        print("\nFILE COVERAGE DETAILS:")
        print("-" * 60)

        # Sort by coverage percentage
        sorted_files = sorted(
            coverage_data['file_coverage'].items(),
            key=lambda x: x[1]['coverage_percentage'],
            reverse=True
        )

        for filepath, stats in sorted_files:
            coverage_pct = stats['coverage_percentage']
            status = "✓" if coverage_pct >= 80 else "⚠" if coverage_pct >= 50 else "✗"
            print(f"{status} {filepath:40} {coverage_pct:6.1f}% "
                  f"({stats['executed_lines']}/{stats['total_lines']} lines)")

            if stats['missing_line_numbers'] and coverage_pct < 100:
                missing_preview = stats['missing_line_numbers'][:5]
                missing_str = ", ".join(map(str, missing_preview))
                if len(stats['missing_line_numbers']) > 5:
                    missing_str += "..."
                print(f"    Missing lines: {missing_str}")

    # Test coverage analysis
    print(f"\n{'='*60}")
    print("TEST COVERAGE ANALYSIS")
    print("=" * 60)

    test_analysis = analyze_test_coverage()
    print(f"Modules with tests: {test_analysis['tested_modules']}/{test_analysis['total_modules']} "
          f"({test_analysis['test_coverage_percentage']}%)")
    print(f"Test files: {test_analysis['test_files']}")

    if test_analysis['untested_modules']:
        print("\nModules without dedicated tests:")
        for module in test_analysis['untested_modules'][:10]:
            print(f"  - {module}")
        if len(test_analysis['untested_modules']) > 10:
            print(f"  ... and {len(test_analysis['untested_modules']) - 10} more")

    # Save results if requested
    if output_format == "json":
        output_file = Path("coverage_report.json")
        with open(output_file, 'w') as f:
            json.dump({
                'coverage': coverage_data,
                'test_analysis': test_analysis
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Coverage recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print("=" * 60)

    total_coverage = coverage_data.get('total_coverage', 0)
    if total_coverage < 50:
        print("⚠ Critical: Coverage below 50% - add more tests urgently")
    elif total_coverage < 70:
        print("⚠ Warning: Coverage below 70% - increase test coverage")
    elif total_coverage < 80:
        print("✓ Good: Coverage above 70% - aim for 80%+")
    else:
        print("✓ Excellent: Coverage above 80% - maintain this level")

    # Find files with low coverage
    if 'file_coverage' in coverage_data:
        low_coverage = [
            (f, s['coverage_percentage'])
            for f, s in coverage_data['file_coverage'].items()
            if s['coverage_percentage'] < 50 and s['total_lines'] > 10
        ]

        if low_coverage:
            print("\nFiles needing more tests (< 50% coverage):")
            for filepath, coverage_pct in low_coverage[:5]:
                print(f"  - {filepath}: {coverage_pct}%")


def main():
    """Main entry point for coverage script."""
    parser = argparse.ArgumentParser(description='Run code coverage analysis')
    parser.add_argument('--format', choices=['text', 'json', 'html'],
                       default='text', help='Output format')
    parser.add_argument('--source', default='src/',
                       help='Source code directory')
    parser.add_argument('--tests', default='tests/',
                       help='Test directory')

    args = parser.parse_args()

    generate_coverage_report(args.format)


if __name__ == "__main__":
    main()