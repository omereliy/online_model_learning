#!/usr/bin/env python3
"""
Simple test coverage report without external dependencies.
Analyzes which modules have tests and estimates coverage.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import ast


def get_source_modules() -> Dict[str, Dict]:
    """Get all source modules and their metrics."""
    src_path = Path("src")
    modules = {}

    for file in src_path.rglob("*.py"):
        if "__pycache__" not in str(file):
            relative = file.relative_to(src_path)
            module_path = str(relative)

            # Count lines and functions
            with open(file, 'r') as f:
                content = f.read()
                lines = len(content.splitlines())

                # Count functions and classes
                try:
                    tree = ast.parse(content)
                    functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                    classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                except:
                    functions = 0
                    classes = 0

            modules[module_path] = {
                'lines': lines,
                'functions': functions,
                'classes': classes,
                'has_test': False,
                'test_file': None
            }

    return modules


def get_test_coverage() -> Dict[str, List[str]]:
    """Map test files to source modules they test."""
    test_path = Path("tests")
    test_mapping = {}

    for file in test_path.glob("test_*.py"):
        if "__pycache__" not in str(file):
            test_name = file.stem  # test_module_name

            # Try to determine which modules this test covers
            covered_modules = []

            # Simple heuristic: test_X covers X
            module_name = test_name.replace("test_", "")

            # Look for matching source files
            possible_matches = [
                f"core/{module_name}.py",
                f"algorithms/{module_name}.py",
                f"experiments/{module_name}.py",
                f"environments/{module_name}.py",
                f"planning/{module_name}.py",
                f"sat_integration/{module_name}.py",
                f"{module_name}.py",
            ]

            for match in possible_matches:
                if Path(f"src/{match}").exists():
                    covered_modules.append(match)

            # Also check file content for imports
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    # Look for "from src.X import" patterns
                    import_lines = [line for line in content.splitlines() if 'from src.' in line]
                    for line in import_lines:
                        parts = line.split()
                        if len(parts) >= 2 and parts[0] == 'from':
                            module = parts[1].replace('src.', '').replace('.', '/') + '.py'
                            if module not in covered_modules:
                                covered_modules.append(module)
            except:
                pass

            test_mapping[test_name] = covered_modules

    return test_mapping


def run_tests_and_count() -> Tuple[int, int]:
    """Run tests and count passed/failed."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True,
            text=True
        )

        # Parse output
        lines = result.stdout.splitlines()
        for line in lines:
            if "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part and i > 0:
                        try:
                            passed = int(parts[i-1])
                            failed = 0
                            if "failed" in line:
                                for j, p in enumerate(parts):
                                    if "failed" in p and j > 0:
                                        failed = int(parts[j-1])
                            return passed, failed
                        except:
                            pass

        return 0, 0
    except:
        return 0, 0


def generate_report():
    """Generate and display coverage report."""
    print("=" * 70)
    print(" " * 20 + "TEST COVERAGE REPORT")
    print("=" * 70)

    # Get source modules
    print("\nAnalyzing source code...")
    modules = get_source_modules()

    # Get test mapping
    print("Analyzing test files...")
    test_mapping = get_test_coverage()

    # Update modules with test information
    for test_name, covered_modules in test_mapping.items():
        for module in covered_modules:
            if module in modules:
                modules[module]['has_test'] = True
                modules[module]['test_file'] = test_name

    # Calculate statistics
    total_modules = len(modules)
    tested_modules = sum(1 for m in modules.values() if m['has_test'])
    total_lines = sum(m['lines'] for m in modules.values())
    tested_lines = sum(m['lines'] for m in modules.values() if m['has_test'])
    total_functions = sum(m['functions'] for m in modules.values())
    tested_functions = sum(m['functions'] for m in modules.values() if m['has_test'])

    # Run tests
    print("Running tests...")
    passed, failed = run_tests_and_count()
    total_tests = passed + failed

    # Display summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Value':>15} {'Percentage':>15}")
    print("-" * 60)
    print(f"{'Total Source Files':<30} {total_modules:>15}")
    print(f"{'Files with Tests':<30} {tested_modules:>15} {tested_modules/total_modules*100:>14.1f}%")
    print(f"{'Files without Tests':<30} {total_modules - tested_modules:>15} {(total_modules - tested_modules)/total_modules*100:>14.1f}%")
    print("-" * 60)
    print(f"{'Total Lines of Code':<30} {total_lines:>15}")
    print(f"{'Lines Covered by Tests':<30} {tested_lines:>15} {tested_lines/total_lines*100:>14.1f}%")
    print(f"{'Lines Not Covered':<30} {total_lines - tested_lines:>15} {(total_lines - tested_lines)/total_lines*100:>14.1f}%")
    print("-" * 60)
    print(f"{'Total Functions':<30} {total_functions:>15}")
    print(f"{'Functions in Tested Files':<30} {tested_functions:>15} {tested_functions/total_functions*100 if total_functions else 0:>14.1f}%")
    print("-" * 60)
    print(f"{'Tests Passed':<30} {passed:>15}")
    print(f"{'Tests Failed':<30} {failed:>15}")
    print(f"{'Total Tests':<30} {total_tests:>15}")

    # Display untested modules
    untested = [(path, info) for path, info in modules.items() if not info['has_test']]
    if untested:
        print("\n" + "=" * 70)
        print("MODULES WITHOUT TESTS")
        print("=" * 70)
        print(f"\n{'Module':<40} {'Lines':>10} {'Functions':>10}")
        print("-" * 60)

        # Sort by lines of code (prioritize larger files)
        untested.sort(key=lambda x: x[1]['lines'], reverse=True)

        for path, info in untested[:15]:  # Show top 15
            print(f"{path:<40} {info['lines']:>10} {info['functions']:>10}")

        if len(untested) > 15:
            print(f"\n... and {len(untested) - 15} more files")

    # Display tested modules
    tested = [(path, info) for path, info in modules.items() if info['has_test']]
    if tested:
        print("\n" + "=" * 70)
        print("MODULES WITH TESTS")
        print("=" * 70)
        print(f"\n{'Module':<40} {'Test File':<25}")
        print("-" * 65)

        tested.sort(key=lambda x: x[0])
        for path, info in tested[:10]:  # Show first 10
            test_file = info['test_file'] or 'unknown'
            print(f"{path:<40} {test_file:<25}")

        if len(tested) > 10:
            print(f"\n... and {len(tested) - 10} more files")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    coverage_pct = tested_modules / total_modules * 100

    if coverage_pct < 50:
        print("\n⚠ CRITICAL: Less than 50% of modules have tests!")
        print("  Priority: Add tests for core functionality immediately")
    elif coverage_pct < 70:
        print("\n⚠ WARNING: Test coverage below 70%")
        print("  Priority: Increase test coverage for critical paths")
    elif coverage_pct < 80:
        print("\n✓ GOOD: Test coverage above 70%")
        print("  Suggestion: Aim for 80%+ coverage")
    else:
        print("\n✓ EXCELLENT: Test coverage above 80%")
        print("  Keep up the good work!")

    # Suggest which files to test next
    if untested:
        print("\nPriority files to add tests for (largest first):")
        for path, info in untested[:5]:
            if info['lines'] > 50:  # Only suggest substantial files
                print(f"  - {path} ({info['lines']} lines, {info['functions']} functions)")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    generate_report()


if __name__ == "__main__":
    main()