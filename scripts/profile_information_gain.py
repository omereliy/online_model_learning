#!/usr/bin/env python3
"""
Performance profiling script for Information Gain Algorithm.

Profiles key components:
1. CNF construction time
2. SAT solver performance (model counting)
3. Information gain calculation overhead
4. Action selection time

Identifies performance bottlenecks and provides optimization recommendations.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import cProfile
import pstats
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.information_gain import InformationGainLearner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InformationGainProfiler:
    """Profile Information Gain algorithm performance."""

    def __init__(self, domain_file: str, problem_file: str):
        """
        Initialize profiler.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
        """
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.learner = None
        self.profile_results = {}

    def initialize_learner(self):
        """Initialize the Information Gain learner."""
        logger.info("Initializing Information Gain learner...")
        start_time = time.perf_counter()

        self.learner = InformationGainLearner(
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            max_iterations=100,
            selection_strategy='greedy'
        )

        init_time = time.perf_counter() - start_time
        logger.info(f"Initialization completed in {init_time:.3f}s")
        self.profile_results['initialization_time'] = init_time

        return init_time

    def profile_cnf_construction(self, num_trials: int = 10) -> Dict[str, Any]:
        """
        Profile CNF formula construction.

        Args:
            num_trials: Number of trials to average

        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\nProfiling CNF construction ({num_trials} trials)...")

        times = []
        for action_name in list(self.learner.pre.keys())[:3]:  # Test first 3 actions
            # Add some constraints to make it realistic
            self.learner.pre_constraints[action_name] = [
                set(list(self.learner.pre[action_name])[:5]),
                set(list(self.learner.pre[action_name])[5:10])
            ]

            for _ in range(num_trials):
                start = time.perf_counter()
                self.learner._build_cnf_formula(action_name)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

        avg_time = sum(times) / len(times) if times else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0

        logger.info(f"  Average: {avg_time*1000:.3f}ms")
        logger.info(f"  Min:     {min_time*1000:.3f}ms")
        logger.info(f"  Max:     {max_time*1000:.3f}ms")

        results = {
            'average_ms': avg_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'total_trials': len(times)
        }
        self.profile_results['cnf_construction'] = results
        return results

    def profile_sat_solving(self, num_trials: int = 10) -> Dict[str, Any]:
        """
        Profile SAT solver performance (model counting).

        Args:
            num_trials: Number of trials to average

        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\nProfiling SAT solving / model counting ({num_trials} trials)...")

        times = []
        model_counts = []

        for action_name in list(self.learner.pre.keys())[:3]:
            # Build CNF with constraints
            cnf = self.learner.cnf_managers[action_name]
            if len(cnf.cnf.clauses) == 0:
                self.learner._build_cnf_formula(action_name)

            for _ in range(num_trials):
                start = time.perf_counter()
                count = cnf.count_solutions()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                model_counts.append(count)

        avg_time = sum(times) / len(times) if times else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0
        avg_models = sum(model_counts) / len(model_counts) if model_counts else 0

        logger.info(f"  Average: {avg_time*1000:.3f}ms")
        logger.info(f"  Min:     {min_time*1000:.3f}ms")
        logger.info(f"  Max:     {max_time*1000:.3f}ms")
        logger.info(f"  Avg models: {avg_models:.0f}")

        results = {
            'average_ms': avg_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'avg_model_count': avg_models,
            'total_trials': len(times)
        }
        self.profile_results['sat_solving'] = results
        return results

    def profile_information_gain_calculation(self, num_trials: int = 5) -> Dict[str, Any]:
        """
        Profile information gain calculation.

        Args:
            num_trials: Number of trials to average

        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\nProfiling information gain calculation ({num_trials} trials)...")

        # Create a sample state
        state = set(list(self.learner.pddl_handler.problem.initial_values)[:10])
        action_name = list(self.learner.pre.keys())[0]
        objects = ['obj1', 'obj2']

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            try:
                gain = self.learner._calculate_expected_information_gain(action_name, objects, state)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Error calculating gain: {e}")

        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)

            logger.info(f"  Average: {avg_time*1000:.3f}ms")
            logger.info(f"  Min:     {min_time*1000:.3f}ms")
            logger.info(f"  Max:     {max_time*1000:.3f}ms")

            results = {
                'average_ms': avg_time * 1000,
                'min_ms': min_time * 1000,
                'max_ms': max_time * 1000,
                'total_trials': len(times)
            }
        else:
            results = {'error': 'No successful trials'}

        self.profile_results['information_gain_calculation'] = results
        return results

    def profile_action_selection(self, num_trials: int = 5) -> Dict[str, Any]:
        """
        Profile complete action selection process.

        Args:
            num_trials: Number of trials to average

        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"\nProfiling action selection ({num_trials} trials)...")

        # Create a sample state
        state = set(list(self.learner.pddl_handler.problem.initial_values)[:10])

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            try:
                action, objects = self.learner.select_action(state)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Error in action selection: {e}")

        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)

            logger.info(f"  Average: {avg_time*1000:.3f}ms")
            logger.info(f"  Min:     {min_time*1000:.3f}ms")
            logger.info(f"  Max:     {max_time*1000:.3f}ms")

            results = {
                'average_ms': avg_time * 1000,
                'min_ms': min_time * 1000,
                'max_ms': max_time * 1000,
                'total_trials': len(times)
            }
        else:
            results = {'error': 'No successful trials'}

        self.profile_results['action_selection'] = results
        return results

    def run_full_profile(self) -> Dict[str, Any]:
        """Run complete profiling suite."""
        logger.info("=" * 70)
        logger.info("INFORMATION GAIN ALGORITHM - PERFORMANCE PROFILING")
        logger.info("=" * 70)
        logger.info(f"Domain: {self.domain_file}")
        logger.info(f"Problem: {self.problem_file}")
        logger.info("=" * 70)

        # Initialize
        self.initialize_learner()

        # Profile each component
        self.profile_cnf_construction()
        self.profile_sat_solving()
        self.profile_information_gain_calculation()
        self.profile_action_selection()

        # Print summary
        self._print_summary()

        return self.profile_results

    def _print_summary(self):
        """Print summary of profiling results."""
        logger.info("\n" + "=" * 70)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 70)

        for component, results in self.profile_results.items():
            if component == 'initialization_time':
                logger.info(f"{component:30} {results:.3f}s")
            elif 'error' in results:
                logger.info(f"{component:30} ERROR: {results['error']}")
            else:
                logger.info(f"{component:30} {results['average_ms']:.3f}ms (avg)")

        logger.info("=" * 70)

        # Recommendations
        logger.info("\nRECOMMENDATIONS:")
        self._print_recommendations()

    def _print_recommendations(self):
        """Print optimization recommendations based on profiling results."""
        cnf_time = self.profile_results.get('cnf_construction', {}).get('average_ms', 0)
        sat_time = self.profile_results.get('sat_solving', {}).get('average_ms', 0)
        action_time = self.profile_results.get('action_selection', {}).get('average_ms', 0)

        if cnf_time > 10:
            logger.info("  - CNF construction is slow (>10ms). Consider caching CNF formulas.")

        if sat_time > 50:
            logger.info("  - SAT solving is slow (>50ms). Consider approximate model counting for large formulas.")

        if action_time > 1000:
            logger.info("  - Action selection is slow (>1s). Consider reducing grounded action space.")
        elif action_time < 100:
            logger.info("  ✓ Action selection performance is good (<100ms).")

        logger.info("")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Profile Information Gain algorithm performance')
    parser.add_argument('--domain', type=str, default='benchmarks/olam-compatible/blocksworld/domain.pddl',
                        help='Path to PDDL domain file')
    parser.add_argument('--problem', type=str, default='benchmarks/olam-compatible/blocksworld/p01.pddl',
                        help='Path to PDDL problem file')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials for each component')

    args = parser.parse_args()

    profiler = InformationGainProfiler(args.domain, args.problem)
    profiler.run_full_profile()


if __name__ == "__main__":
    main()
