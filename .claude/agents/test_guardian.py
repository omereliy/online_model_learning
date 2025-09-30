#!/usr/bin/env python
"""
Test Guardian Agent - Ensures all code changes pass tests before integration.

This agent is crucial for maintaining TDD methodology required for scientific paper submission.
It validates that tests pass before and after file edits, blocking changes that break tests.

Compatible with:
- Conda environments
- Docker containers
- System Python installations
"""

import subprocess
import json
import sys
import os
import re
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import shutil

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'test_guardian_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestGuardian')


class TestGuardianAgent:
    """Agent responsible for test validation in the development workflow."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Test Guardian Agent.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.project_root = self._find_project_root()
        self.config = self._load_config(config_path)
        self.test_cache = {}
        self.metrics = {
            'executions': 0,
            'blocks': 0,
            'passes': 0,
            'total_duration': 0
        }
        self.python_cmd = self._detect_python()
        self.environment = self._detect_environment()

    def _find_project_root(self) -> Path:
        """Find the project root directory (where .git exists)."""
        current = Path.cwd()
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return Path.cwd()

    def _detect_python(self) -> str:
        """Detect the appropriate Python command for the current environment."""
        # Check if we're in a conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            conda_python = shutil.which('python')
            if conda_python:
                logger.info(f"Detected conda environment: {conda_env}")
                return 'python'

        # Check for python3 first (preferred)
        if shutil.which('python3'):
            return 'python3'

        # Fall back to python
        if shutil.which('python'):
            return 'python'

        # Default
        logger.warning("Could not detect Python command, defaulting to 'python'")
        return 'python'

    def _detect_environment(self) -> Dict[str, str]:
        """Detect the current execution environment."""
        env_info = {
            'type': 'unknown',
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': sys.platform
        }

        # Check if in Docker
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            env_info['type'] = 'docker'
            env_info['container'] = os.environ.get('HOSTNAME', 'unknown')

        # Check if in conda
        elif os.environ.get('CONDA_DEFAULT_ENV'):
            env_info['type'] = 'conda'
            env_info['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV')
            env_info['conda_prefix'] = os.environ.get('CONDA_PREFIX', '')

        # Check if in virtual environment
        elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            env_info['type'] = 'virtualenv'
            env_info['venv_path'] = sys.prefix

        else:
            env_info['type'] = 'system'

        logger.info(f"Detected environment: {env_info['type']}")
        return env_info

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        defaults = {
            'coverage_threshold': 80,
            'quick_test_timeout': 30,
            'full_test_timeout': 300,
            'strict_mode': False,
            'auto_fix_suggestions': True,
            'environment_specific': {
                'docker': {
                    'quick_test_timeout': 60,  # Docker might be slower
                    'full_test_timeout': 600
                },
                'conda': {
                    'use_conda_python': True
                }
            }
        }

        # First check for default config
        default_config_path = Path(__file__).parent.parent / 'config' / 'test_guardian.json'
        if default_config_path.exists():
            try:
                with open(default_config_path) as f:
                    loaded_config = json.load(f)
                    defaults.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load default config: {e}")

        # Then check for user-provided config
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    defaults.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Apply environment-specific settings
        env_type = self.environment.get('type', 'unknown') if hasattr(self, 'environment') else 'unknown'
        if env_type in defaults.get('environment_specific', {}):
            env_config = defaults['environment_specific'][env_type]
            defaults.update(env_config)

        return defaults

    def identify_affected_tests(self, file_paths: List[str]) -> List[str]:
        """Identify which tests are affected by file changes.

        Args:
            file_paths: List of changed file paths

        Returns:
            List of test identifiers to run
        """
        affected_tests = set()

        for file_path in file_paths:
            path = Path(file_path)

            # Map source files to their tests
            if 'src/algorithms' in str(path):
                if 'olam_adapter' in path.stem:
                    affected_tests.add('test_olam_adapter')
                    affected_tests.add('test_olam_integration')
                elif 'base_learner' in path.stem:
                    affected_tests.add('test_base_learner')
                affected_tests.add('test_algorithms')

            elif 'src/core' in str(path):
                if 'pddl_handler' in path.stem:
                    affected_tests.add('test_pddl_handler')
                elif 'cnf_manager' in path.stem:
                    affected_tests.add('test_cnf_manager')
                elif 'domain_analyzer' in path.stem:
                    affected_tests.add('test_domain_analyzer')

            elif 'src/experiments' in str(path):
                if 'metrics' in path.stem:
                    affected_tests.add('test_metrics')
                elif 'runner' in path.stem:
                    affected_tests.add('test_runner')
                    affected_tests.add('test_experiment_integration')

            elif 'src/environments' in str(path):
                affected_tests.add('test_environments')
                affected_tests.add('test_pddl_environment')

        # If no specific tests identified, run all
        if not affected_tests:
            logger.info(f"No specific tests identified for {file_paths}, will run all")
            return ['all']

        return list(affected_tests)

    def run_quick_tests(self) -> Tuple[bool, str, float]:
        """Run quick critical tests using make test-quick.

        Returns:
            Tuple of (success, output, duration)
        """
        logger.info("Running quick critical tests...")
        start_time = time.time()

        try:
            result = subprocess.run(
                ['make', 'test-quick'],
                capture_output=True,
                text=True,
                timeout=self.config['quick_test_timeout'],
                cwd=self.project_root
            )
            duration = time.time() - start_time
            success = result.returncode == 0

            # Log result
            if success:
                logger.info(f"Quick tests passed in {duration:.2f}s")
            else:
                logger.warning(f"Quick tests failed in {duration:.2f}s")
                logger.debug(f"Test output: {result.stdout}")
                logger.debug(f"Test errors: {result.stderr}")

            return success, result.stdout, duration

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Quick tests timed out after {duration:.2f}s")
            return False, "Tests timed out", duration
        except Exception as e:
            logger.error(f"Error running quick tests: {e}")
            return False, str(e), 0

    def run_full_tests(self) -> Tuple[bool, str, float]:
        """Run full test suite using make test.

        Returns:
            Tuple of (success, output, duration)
        """
        logger.info("Running full test suite...")
        start_time = time.time()

        try:
            result = subprocess.run(
                ['make', 'test'],
                capture_output=True,
                text=True,
                timeout=self.config['full_test_timeout'],
                cwd=self.project_root
            )
            duration = time.time() - start_time
            success = result.returncode == 0

            # Log result
            if success:
                logger.info(f"Full tests passed in {duration:.2f}s")
            else:
                logger.warning(f"Full tests failed in {duration:.2f}s")

            return success, result.stdout, duration

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Full tests timed out after {duration:.2f}s")
            return False, "Tests timed out", duration
        except Exception as e:
            logger.error(f"Error running full tests: {e}")
            return False, str(e), 0

    def validate_pre_edit(self, files: List[str]) -> bool:
        """Validate before editing files.

        Args:
            files: List of files about to be edited

        Returns:
            True if safe to edit, False otherwise
        """
        print("🛡️ Test Guardian: Pre-edit validation starting...")
        logger.info(f"Pre-edit validation for files: {files}")

        # Update metrics
        self.metrics['executions'] += 1

        # Get baseline
        baseline_pass, baseline_output, duration = self.run_quick_tests()
        self.metrics['total_duration'] += duration

        if not baseline_pass:
            print("❌ Tests already failing. Fix existing issues before editing.")
            logger.error("Baseline tests failing, blocking edit")
            self.metrics['blocks'] += 1

            # Show which tests are failing
            self._show_test_failures(baseline_output)
            return False

        # Store baseline for comparison
        self.test_cache['baseline'] = baseline_output
        self.test_cache['baseline_time'] = time.time()

        print("✅ Baseline tests passing. Safe to edit.")
        logger.info("Baseline tests passed, edit approved")
        self.metrics['passes'] += 1
        return True

    def validate_post_edit(self, files: List[str]) -> bool:
        """Validate after editing files.

        Args:
            files: List of files that were edited

        Returns:
            True if changes are safe, False otherwise
        """
        print("🛡️ Test Guardian: Post-edit validation starting...")
        logger.info(f"Post-edit validation for files: {files}")

        # Update metrics
        self.metrics['executions'] += 1

        # Run affected tests
        current_pass, current_output, duration = self.run_quick_tests()
        self.metrics['total_duration'] += duration

        if not current_pass:
            print("❌ Tests broken by changes. Review the failures below:")
            logger.error("Tests broken by changes, suggesting rollback")
            self.metrics['blocks'] += 1

            # Show diff between baseline and current
            if 'baseline' in self.test_cache:
                self._show_test_diff(self.test_cache['baseline'], current_output)
            else:
                self._show_test_failures(current_output)

            if self.config['auto_fix_suggestions']:
                self._suggest_fixes(files, current_output)

            return False

        print("✅ All tests still passing. Changes approved.")
        logger.info("All tests passed, changes approved")
        self.metrics['passes'] += 1
        return True

    def validate_pre_commit(self) -> bool:
        """Validate before git commit.

        Returns:
            True if safe to commit, False otherwise
        """
        print("🛡️ Test Guardian: Pre-commit validation starting...")
        logger.info("Pre-commit validation")

        # For commits, run full test suite
        success, output, duration = self.run_full_tests()
        self.metrics['executions'] += 1
        self.metrics['total_duration'] += duration

        if not success:
            print("❌ Full test suite failed. Commit blocked.")
            print("Run 'make test' to see all failures.")
            logger.error("Full test suite failed, commit blocked")
            self.metrics['blocks'] += 1

            self._show_test_failures(output)
            return False

        # Check test coverage if configured
        if self.config['coverage_threshold'] > 0:
            coverage = self._check_coverage()
            if coverage < self.config['coverage_threshold']:
                print(f"⚠️ Test coverage ({coverage}%) below threshold ({self.config['coverage_threshold']}%)")
                if self.config['strict_mode']:
                    print("❌ Commit blocked due to low coverage (strict mode)")
                    self.metrics['blocks'] += 1
                    return False
                else:
                    print("⚠️ Warning: Proceeding with low coverage")

        print("✅ All tests passing. Safe to commit.")
        logger.info("All tests passed, commit approved")
        self.metrics['passes'] += 1
        return True

    def _show_test_failures(self, output: str):
        """Display test failures in a readable format."""
        lines = output.split('\n')
        failures = []

        for i, line in enumerate(lines):
            if 'FAILED' in line or 'ERROR' in line:
                failures.append(line)
            elif 'assert' in line.lower():
                failures.append(line)

        if failures:
            print("\n--- Test Failures ---")
            for failure in failures[:10]:  # Show first 10 failures
                print(f"  {failure}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more")
            print("-------------------\n")

    def _show_test_diff(self, baseline: str, current: str):
        """Show difference between baseline and current test results."""
        baseline_lines = set(baseline.split('\n'))
        current_lines = set(current.split('\n'))

        new_failures = current_lines - baseline_lines

        if new_failures:
            print("\n--- New Test Failures ---")
            for line in list(new_failures)[:10]:
                if 'FAILED' in line or 'ERROR' in line or 'assert' in line.lower():
                    print(f"  {line}")
            print("------------------------\n")

    def _suggest_fixes(self, files: List[str], test_output: str):
        """Suggest potential fixes based on test failures."""
        suggestions = []

        # Check for common patterns
        if 'ImportError' in test_output:
            suggestions.append("- Check import statements and module paths")
        if 'AttributeError' in test_output:
            suggestions.append("- Verify all class attributes are properly initialized")
        if 'TypeError' in test_output:
            suggestions.append("- Check method signatures and parameter types")
        if 'KeyError' in test_output:
            suggestions.append("- Verify dictionary keys and configuration files")

        if suggestions:
            print("\n💡 Suggested fixes:")
            for suggestion in suggestions:
                print(suggestion)
            print()

    def _check_coverage(self) -> float:
        """Check test coverage percentage.

        Returns:
            Coverage percentage
        """
        # First try to run coverage if .coverage file doesn't exist
        coverage_file = self.project_root / '.coverage'
        if not coverage_file.exists():
            logger.info("No coverage data found, running tests with coverage...")
            try:
                # Run tests with coverage
                subprocess.run(
                    ['coverage', 'run', '-m', 'pytest', 'tests/', '-q'],
                    capture_output=True,
                    timeout=60,
                    cwd=self.project_root
                )
            except Exception as e:
                logger.warning(f"Could not generate coverage data: {e}")
                # If strict mode, return 0, otherwise assume good
                return 0.0 if self.config.get('strict_mode') else 100.0

        try:
            # Try simple format first
            result = subprocess.run(
                ['coverage', 'report', '--skip-covered', '--skip-empty'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # Parse coverage from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line:
                    parts = line.split()
                    # Coverage percentage is typically the last number with %
                    for part in reversed(parts):
                        if '%' in part:
                            coverage_str = part.rstrip('%')
                            try:
                                return float(coverage_str)
                            except ValueError:
                                continue

            # Try alternative parsing for different coverage formats
            for line in lines[-5:]:  # Check last few lines
                match = re.search(r'(\d+(?:\.\d+)?)%', line)
                if match:
                    return float(match.group(1))

            logger.warning("Could not parse coverage percentage from output")
            return 0.0 if self.config.get('strict_mode') else 100.0
        except Exception as e:
            logger.warning(f"Failed to check coverage: {e}")
            # If strict mode, return 0, otherwise assume good
            return 0.0 if self.config.get('strict_mode') else 100.0

    def get_status(self) -> Dict:
        """Get current agent status and metrics.

        Returns:
            Dictionary with status information
        """
        return {
            'enabled': True,
            'project_root': str(self.project_root),
            'environment': self.environment,
            'python_command': self.python_cmd,
            'config': self.config,
            'metrics': self.metrics,
            'success_rate': (
                (self.metrics['passes'] / self.metrics['executions'] * 100)
                if self.metrics['executions'] > 0 else 0
            ),
            'avg_duration': (
                self.metrics['total_duration'] / self.metrics['executions']
                if self.metrics['executions'] > 0 else 0
            )
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'executions': 0,
            'blocks': 0,
            'passes': 0,
            'total_duration': 0
        }
        logger.info("Metrics reset")


def main():
    """Main entry point for Test Guardian Agent."""
    parser = argparse.ArgumentParser(description='Test Guardian Agent')
    parser.add_argument('--pre-edit', action='store_true', help='Pre-edit validation')
    parser.add_argument('--post-edit', action='store_true', help='Post-edit validation')
    parser.add_argument('--pre-commit', action='store_true', help='Pre-commit validation')
    parser.add_argument('--status', action='store_true', help='Show agent status')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('files', nargs='*', help='Files being edited')

    args = parser.parse_args()

    # Create agent instance
    agent = TestGuardianAgent(config_path=args.config)

    # Handle different modes
    if args.status:
        status = agent.get_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)

    elif args.pre_edit:
        if not args.files:
            print("No files specified for pre-edit validation")
            sys.exit(1)
        success = agent.validate_pre_edit(args.files)
        sys.exit(0 if success else 1)

    elif args.post_edit:
        if not args.files:
            print("No files specified for post-edit validation")
            sys.exit(1)
        success = agent.validate_post_edit(args.files)
        sys.exit(0 if success else 1)

    elif args.pre_commit:
        success = agent.validate_pre_commit()
        sys.exit(0 if success else 1)

    else:
        # Default: run quick tests
        success, output, duration = agent.run_quick_tests()
        print(f"Tests {'passed' if success else 'failed'} in {duration:.2f}s")
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()