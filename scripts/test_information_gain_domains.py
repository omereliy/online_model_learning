#!/usr/bin/env python3
"""
Multi-domain testing script for Information Gain Algorithm.

Tests the Information Gain learner on multiple PDDL domains:
1. Blocksworld (OLAM-compatible, no negative preconditions)
2. Gripper (OLAM-compatible, no negative preconditions)
3. Rover (OLAM-incompatible, has negative preconditions)

This validates that the algorithm handles different domain features correctly.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.runner import ExperimentRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DomainTester:
    """Test Information Gain learner on multiple domains."""

    def __init__(self, quick_mode: bool = False):
        """
        Initialize domain tester.

        Args:
            quick_mode: If True, run shorter experiments for quick validation
        """
        self.quick_mode = quick_mode
        self.results: Dict[str, Any] = {}

    def test_blocksworld(self) -> Dict[str, Any]:
        """Test on blocksworld domain (OLAM-compatible)."""
        logger.info("=" * 60)
        logger.info("Testing Information Gain on BLOCKSWORLD domain")
        logger.info("=" * 60)

        config_path = "configs/information_gain_blocksworld.yaml"

        try:
            runner = ExperimentRunner(config_path)
            logger.info(f"Running experiment with {runner.config['stopping_criteria']['max_iterations']} max iterations")

            results = runner.run_experiment()

            logger.info(f"✓ Blocksworld completed: {results['total_iterations']} iterations")
            logger.info(f"  Stopping reason: {results['stopping_reason']}")
            logger.info(f"  Runtime: {results['runtime_seconds']:.2f}s")

            return {
                'domain': 'blocksworld',
                'success': True,
                'iterations': results['total_iterations'],
                'runtime': results['runtime_seconds'],
                'stopping_reason': results['stopping_reason']
            }

        except Exception as e:
            logger.error(f"✗ Blocksworld failed: {e}")
            return {
                'domain': 'blocksworld',
                'success': False,
                'error': str(e)
            }

    def test_gripper(self) -> Dict[str, Any]:
        """Test on gripper domain (OLAM-compatible)."""
        logger.info("=" * 60)
        logger.info("Testing Information Gain on GRIPPER domain")
        logger.info("=" * 60)

        config_path = "configs/information_gain_gripper.yaml"

        try:
            runner = ExperimentRunner(config_path)
            logger.info(f"Running experiment with {runner.config['stopping_criteria']['max_iterations']} max iterations")

            results = runner.run_experiment()

            logger.info(f"✓ Gripper completed: {results['total_iterations']} iterations")
            logger.info(f"  Stopping reason: {results['stopping_reason']}")
            logger.info(f"  Runtime: {results['runtime_seconds']:.2f}s")

            return {
                'domain': 'gripper',
                'success': True,
                'iterations': results['total_iterations'],
                'runtime': results['runtime_seconds'],
                'stopping_reason': results['stopping_reason']
            }

        except Exception as e:
            logger.error(f"✗ Gripper failed: {e}")
            return {
                'domain': 'gripper',
                'success': False,
                'error': str(e)
            }

    def test_rover(self) -> Dict[str, Any]:
        """Test on rover domain (OLAM-incompatible, has negative preconditions)."""
        logger.info("=" * 60)
        logger.info("Testing Information Gain on ROVER domain")
        logger.info("(Has negative preconditions - OLAM incompatible)")
        logger.info("=" * 60)

        config_path = "configs/information_gain_rover.yaml"

        try:
            runner = ExperimentRunner(config_path)
            logger.info(f"Running experiment with {runner.config['stopping_criteria']['max_iterations']} max iterations")

            results = runner.run_experiment()

            logger.info(f"✓ Rover completed: {results['total_iterations']} iterations")
            logger.info(f"  Stopping reason: {results['stopping_reason']}")
            logger.info(f"  Runtime: {results['runtime_seconds']:.2f}s")

            return {
                'domain': 'rover',
                'success': True,
                'iterations': results['total_iterations'],
                'runtime': results['runtime_seconds'],
                'stopping_reason': results['stopping_reason'],
                'has_negative_preconditions': True
            }

        except Exception as e:
            logger.error(f"✗ Rover failed: {e}")
            return {
                'domain': 'rover',
                'success': False,
                'error': str(e),
                'has_negative_preconditions': True
            }

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run tests on all domains."""
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("INFORMATION GAIN ALGORITHM - MULTI-DOMAIN TESTING")
        logger.info("=" * 70)
        logger.info(f"Quick mode: {self.quick_mode}")
        logger.info("\n")

        results = []

        # Test each domain
        results.append(self.test_blocksworld())
        results.append(self.test_gripper())
        results.append(self.test_rover())

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of all test results."""
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - passed_tests

        for result in results:
            domain = result['domain']
            if result['success']:
                logger.info(f"✓ {domain:15} - PASSED ({result['iterations']} iterations, {result['runtime']:.2f}s)")
            else:
                logger.info(f"✗ {domain:15} - FAILED: {result.get('error', 'Unknown error')}")

        logger.info("-" * 70)
        logger.info(f"Total: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}")

        if failed_tests == 0:
            logger.info("=" * 70)
            logger.info("ALL TESTS PASSED ✓")
            logger.info("=" * 70)
        else:
            logger.warning("=" * 70)
            logger.warning(f"SOME TESTS FAILED ({failed_tests}/{total_tests})")
            logger.warning("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Test Information Gain algorithm on multiple domains')
    parser.add_argument('--quick', action='store_true', help='Quick mode with shorter experiments')
    parser.add_argument('--domain', type=str, choices=['blocksworld', 'gripper', 'rover'],
                        help='Test only specific domain')

    args = parser.parse_args()

    tester = DomainTester(quick_mode=args.quick)

    if args.domain:
        # Test specific domain
        if args.domain == 'blocksworld':
            result = tester.test_blocksworld()
        elif args.domain == 'gripper':
            result = tester.test_gripper()
        elif args.domain == 'rover':
            result = tester.test_rover()

        sys.exit(0 if result['success'] else 1)
    else:
        # Test all domains
        results = tester.run_all_tests()
        failed = sum(1 for r in results if not r['success'])
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
