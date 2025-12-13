#!/usr/bin/env python
"""
Performance profiling script for CNF/SAT operations in Information Gain algorithm.

Measures timing of key methods to identify bottlenecks and track optimization progress.

Usage:
    # Run baseline measurements
    python scripts/profile_performance.py --baseline

    # Run after optimizations
    python scripts/profile_performance.py --compare results/baseline_performance.json

    # Custom configuration
    python scripts/profile_performance.py --iterations 50 --domains blocksworld depots
"""

import argparse
import functools
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.information_gain import InformationGainLearner
from src.core.cnf_manager import CNFManager
from src.environments.active_environment import ActiveEnvironment

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during profiling
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Test domains and problems
DEFAULT_TEST_CASES = [
    ("blocksworld", "p09"),
    ("depots", "p06"),
    ("satellite", "p09"),
    ("gold-miner", "p01"),
]


class TimingStats:
    """Collects and reports timing statistics."""

    def __init__(self):
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.total_times: Dict[str, float] = defaultdict(float)
        self.max_times: Dict[str, float] = defaultdict(float)
        self.min_times: Dict[str, float] = defaultdict(lambda: float('inf'))

    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        self.call_counts[name] += 1
        self.total_times[name] += duration
        self.max_times[name] = max(self.max_times[name], duration)
        self.min_times[name] = min(self.min_times[name], duration)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all methods."""
        summary = {}
        for name in self.call_counts:
            count = self.call_counts[name]
            total = self.total_times[name]
            summary[name] = {
                "call_count": count,
                "total_time_ms": total * 1000,
                "avg_time_ms": (total / count * 1000) if count > 0 else 0,
                "min_time_ms": self.min_times[name] * 1000 if self.min_times[name] != float('inf') else 0,
                "max_time_ms": self.max_times[name] * 1000,
            }
        return summary

    def reset(self):
        """Reset all statistics."""
        self.call_counts.clear()
        self.total_times.clear()
        self.max_times.clear()
        self.min_times = defaultdict(lambda: float('inf'))


# Global timing stats
_timing_stats = TimingStats()


def timed(name: Optional[str] = None):
    """Decorator to time function calls."""
    def decorator(func: Callable) -> Callable:
        method_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                _timing_stats.record(method_name, duration)
        return wrapper
    return decorator


def patch_methods_for_profiling():
    """Patch key methods with timing instrumentation."""

    # Patch InformationGainLearner methods
    original_calculate_all_action_gains = InformationGainLearner._calculate_all_action_gains
    original_calculate_applicability = InformationGainLearner._calculate_applicability_probability
    original_calculate_failure_gain = InformationGainLearner._calculate_potential_gain_failure
    original_build_cnf = InformationGainLearner._build_cnf_formula
    original_get_base_count = InformationGainLearner._get_base_model_count

    @functools.wraps(original_calculate_all_action_gains)
    def timed_calculate_all_action_gains(self, state):
        start = time.perf_counter()
        result = original_calculate_all_action_gains(self, state)
        _timing_stats.record("_calculate_all_action_gains", time.perf_counter() - start)
        return result

    @functools.wraps(original_calculate_applicability)
    def timed_calculate_applicability(self, action, objects, state):
        start = time.perf_counter()
        result = original_calculate_applicability(self, action, objects, state)
        _timing_stats.record("_calculate_applicability_probability", time.perf_counter() - start)
        return result

    @functools.wraps(original_calculate_failure_gain)
    def timed_calculate_failure_gain(self, action, objects, state):
        start = time.perf_counter()
        result = original_calculate_failure_gain(self, action, objects, state)
        _timing_stats.record("_calculate_potential_gain_failure", time.perf_counter() - start)
        return result

    @functools.wraps(original_build_cnf)
    def timed_build_cnf(self, action):
        start = time.perf_counter()
        result = original_build_cnf(self, action)
        _timing_stats.record("_build_cnf_formula", time.perf_counter() - start)
        return result

    @functools.wraps(original_get_base_count)
    def timed_get_base_count(self, action):
        start = time.perf_counter()
        result = original_get_base_count(self, action)
        _timing_stats.record("_get_base_model_count", time.perf_counter() - start)
        return result

    InformationGainLearner._calculate_all_action_gains = timed_calculate_all_action_gains
    InformationGainLearner._calculate_applicability_probability = timed_calculate_applicability
    InformationGainLearner._calculate_potential_gain_failure = timed_calculate_failure_gain
    InformationGainLearner._build_cnf_formula = timed_build_cnf
    InformationGainLearner._get_base_model_count = timed_get_base_count

    # Patch CNFManager methods
    original_count_solutions = CNFManager.count_solutions
    original_get_all_solutions = CNFManager.get_all_solutions
    original_count_with_assumptions = CNFManager.count_models_with_assumptions
    original_count_with_temp_clause = CNFManager.count_models_with_temporary_clause

    @functools.wraps(original_count_solutions)
    def timed_count_solutions(self, max_solutions=None):
        start = time.perf_counter()
        result = original_count_solutions(self, max_solutions)
        _timing_stats.record("CNFManager.count_solutions", time.perf_counter() - start)
        return result

    @functools.wraps(original_get_all_solutions)
    def timed_get_all_solutions(self, max_solutions=None):
        start = time.perf_counter()
        result = original_get_all_solutions(self, max_solutions)
        _timing_stats.record("CNFManager.get_all_solutions", time.perf_counter() - start)
        return result

    @functools.wraps(original_count_with_assumptions)
    def timed_count_with_assumptions(self, assumptions, use_cache=True):
        start = time.perf_counter()
        result = original_count_with_assumptions(self, assumptions, use_cache)
        _timing_stats.record("CNFManager.count_models_with_assumptions", time.perf_counter() - start)
        return result

    @functools.wraps(original_count_with_temp_clause)
    def timed_count_with_temp_clause(self, clause_literals):
        start = time.perf_counter()
        result = original_count_with_temp_clause(self, clause_literals)
        _timing_stats.record("CNFManager.count_models_with_temporary_clause", time.perf_counter() - start)
        return result

    CNFManager.count_solutions = timed_count_solutions
    CNFManager.get_all_solutions = timed_get_all_solutions
    CNFManager.count_models_with_assumptions = timed_count_with_assumptions
    CNFManager.count_models_with_temporary_clause = timed_count_with_temp_clause


def run_profiled_experiment(
    domain: str,
    problem: str,
    max_iterations: int = 50
) -> Dict[str, Any]:
    """
    Run a profiled experiment on a single domain/problem.

    Args:
        domain: Domain name (e.g., 'blocksworld')
        problem: Problem name (e.g., 'p09')
        max_iterations: Number of iterations to run

    Returns:
        Dictionary with timing results and experiment metadata
    """
    domain_file = f"benchmarks/olam-compatible/{domain}/domain.pddl"
    problem_file = f"benchmarks/olam-compatible/{domain}/{problem}.pddl"

    # Verify files exist
    if not Path(domain_file).exists():
        print(f"  ERROR: Domain file not found: {domain_file}")
        return {"error": f"Domain file not found: {domain_file}"}
    if not Path(problem_file).exists():
        print(f"  ERROR: Problem file not found: {problem_file}")
        return {"error": f"Problem file not found: {problem_file}"}

    # Reset timing stats
    _timing_stats.reset()

    # Initialize learner
    print(f"  Initializing learner...")
    init_start = time.perf_counter()
    learner = InformationGainLearner(
        domain_file=domain_file,
        problem_file=problem_file,
        max_iterations=max_iterations,
        use_object_subset=True,
        spare_objects_per_type=2,
        selection_strategy='greedy',
        seed=42
    )
    init_time = time.perf_counter() - init_start
    print(f"  Initialization: {init_time:.2f}s")

    # Initialize environment
    env = ActiveEnvironment(domain_file, problem_file)
    state = env.get_state()

    # Run learning loop
    print(f"  Running {max_iterations} iterations...")
    experiment_start = time.perf_counter()
    iteration = 0
    success_count = 0
    failure_count = 0

    while iteration < max_iterations and not learner.has_converged():
        iteration += 1

        # Select action
        action_name, objects = learner.select_action(state)

        if action_name == "no_action":
            break

        # Execute action
        success, _ = env.execute(action_name, objects)

        if success:
            success_count += 1
            next_state = env.get_state()
        else:
            failure_count += 1
            next_state = None

        # Observe result
        learner.observe(
            state=state,
            action=action_name,
            objects=objects,
            success=success,
            next_state=next_state
        )

        # Update state
        if success:
            state = next_state

        # Progress indicator every 10 iterations
        if iteration % 10 == 0:
            print(f"    Iteration {iteration}/{max_iterations} "
                  f"(success: {success_count}, failure: {failure_count})")

    experiment_time = time.perf_counter() - experiment_start
    print(f"  Experiment complete: {iteration} iterations in {experiment_time:.2f}s")

    # Collect results
    timing_summary = _timing_stats.get_summary()

    return {
        "domain": domain,
        "problem": problem,
        "iterations_completed": iteration,
        "success_count": success_count,
        "failure_count": failure_count,
        "total_time_seconds": experiment_time,
        "init_time_seconds": init_time,
        "avg_iteration_time_ms": (experiment_time / iteration * 1000) if iteration > 0 else 0,
        "timing_breakdown": timing_summary,
    }


def format_timing_table(results: List[Dict[str, Any]]) -> str:
    """Format timing results as a table."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TIMING BREAKDOWN BY METHOD")
    lines.append("=" * 80)

    # Aggregate across all experiments
    aggregated: Dict[str, Dict[str, float]] = defaultdict(lambda: {"calls": 0, "total": 0, "max": 0})

    for result in results:
        if "timing_breakdown" not in result:
            continue
        for method, stats in result["timing_breakdown"].items():
            aggregated[method]["calls"] += stats["call_count"]
            aggregated[method]["total"] += stats["total_time_ms"]
            aggregated[method]["max"] = max(aggregated[method]["max"], stats["max_time_ms"])

    # Sort by total time descending
    sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]["total"], reverse=True)

    lines.append(f"{'Method':<45} {'Calls':>10} {'Total(ms)':>12} {'Avg(ms)':>10} {'Max(ms)':>10}")
    lines.append("-" * 90)

    for method, stats in sorted_methods:
        avg = stats["total"] / stats["calls"] if stats["calls"] > 0 else 0
        lines.append(f"{method:<45} {stats['calls']:>10} {stats['total']:>12.1f} {avg:>10.2f} {stats['max']:>10.2f}")

    lines.append("=" * 80)
    return "\n".join(lines)


def compare_results(current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Compare current results with baseline."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("PERFORMANCE COMPARISON (vs baseline)")
    lines.append("=" * 80)

    for domain_result in current.get("experiments", []):
        domain = domain_result["domain"]
        problem = domain_result["problem"]

        # Find matching baseline
        baseline_result = None
        for br in baseline.get("experiments", []):
            if br["domain"] == domain and br["problem"] == problem:
                baseline_result = br
                break

        if not baseline_result:
            lines.append(f"\n{domain}/{problem}: No baseline for comparison")
            continue

        lines.append(f"\n{domain}/{problem}:")

        # Compare key metrics
        curr_iter_time = domain_result.get("avg_iteration_time_ms", 0)
        base_iter_time = baseline_result.get("avg_iteration_time_ms", 0)

        if base_iter_time > 0:
            speedup = base_iter_time / curr_iter_time if curr_iter_time > 0 else float('inf')
            change_pct = (1 - curr_iter_time / base_iter_time) * 100
            lines.append(f"  Avg iteration time: {curr_iter_time:.2f}ms vs {base_iter_time:.2f}ms "
                        f"({change_pct:+.1f}%, {speedup:.2f}x speedup)")

        # Compare method timings
        curr_timing = domain_result.get("timing_breakdown", {})
        base_timing = baseline_result.get("timing_breakdown", {})

        for method in curr_timing:
            if method in base_timing:
                curr_total = curr_timing[method]["total_time_ms"]
                base_total = base_timing[method]["total_time_ms"]
                if base_total > 0:
                    change_pct = (1 - curr_total / base_total) * 100
                    if abs(change_pct) > 5:  # Only show significant changes
                        lines.append(f"    {method}: {change_pct:+.1f}%")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Profile CNF/SAT performance in Information Gain algorithm"
    )
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline measurements and save to results/baseline_performance.json")
    parser.add_argument("--compare", type=str,
                       help="Compare with baseline file (e.g., results/baseline_performance.json)")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of iterations per experiment (default: 50)")
    parser.add_argument("--domains", nargs='+',
                       help="Custom domains to test (default: blocksworld, depots, satellite, gold-miner)")
    parser.add_argument("--output", type=str,
                       help="Output file for results (default: auto-generated)")

    args = parser.parse_args()

    # Determine test cases
    if args.domains:
        test_cases = [(d, "p01") for d in args.domains]  # Default to p01 for custom domains
    else:
        test_cases = DEFAULT_TEST_CASES

    print("=" * 80)
    print("CNF/SAT PERFORMANCE PROFILING")
    print("=" * 80)
    print(f"Test cases: {len(test_cases)}")
    print(f"Iterations per experiment: {args.iterations}")
    print()

    # Patch methods for profiling
    patch_methods_for_profiling()

    # Run experiments
    all_results = []
    total_start = time.perf_counter()

    for i, (domain, problem) in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {domain}/{problem}")
        print("-" * 40)

        result = run_profiled_experiment(domain, problem, args.iterations)
        all_results.append(result)

    total_time = time.perf_counter() - total_start

    # Print timing breakdown
    print(format_timing_table(all_results))

    # Summary
    print("\nSUMMARY")
    print("-" * 40)
    print(f"Total profiling time: {total_time:.2f}s")
    for result in all_results:
        if "error" in result:
            print(f"  {result['domain']}/{result['problem']}: ERROR - {result['error']}")
        else:
            print(f"  {result['domain']}/{result['problem']}: "
                  f"{result['iterations_completed']} iters, "
                  f"{result['avg_iteration_time_ms']:.2f}ms/iter avg")

    # Prepare output
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "iterations_per_experiment": args.iterations,
            "test_cases": test_cases,
        },
        "total_time_seconds": total_time,
        "experiments": all_results,
    }

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif args.baseline:
        output_path = Path("results/baseline_performance.json")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results/performance_{timestamp}.json")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Compare with baseline if requested
    if args.compare:
        baseline_path = Path(args.compare)
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            print(compare_results(output_data, baseline_data))
        else:
            print(f"\nWARNING: Baseline file not found: {baseline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
