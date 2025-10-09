#!/usr/bin/env python3
"""
Performance benchmarking script for the online model learning framework.
Measures performance metrics across different domains and configurations.
"""

import argparse
import json
import time
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, plotting disabled")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pddl_io import PDDLReader
from src.experiments.metrics import MetricsCollector


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    def __init__(self, output_dir: str = "benchmarks/performance"):
        """Initialize performance benchmark."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def benchmark_domain_parsing(self, domains: List[str], repetitions: int = 10) -> Dict[str, Any]:
        """
        Benchmark PDDL domain parsing performance.

        Args:
            domains: List of domain names to benchmark
            repetitions: Number of repetitions for timing

        Returns:
            Dictionary with benchmark results
        """
        results = {}

        for domain in domains:
            domain_file = Path(f"benchmarks/{domain}/domain.pddl")
            problem_file = Path(f"benchmarks/{domain}/p01.pddl")

            if not domain_file.exists() or not problem_file.exists():
                print(f"Warning: Skipping {domain} - files not found")
                continue

            times = []
            for _ in range(repetitions):
                reader = PDDLReader()
                start = time.perf_counter()
                reader.parse_domain_and_problem(str(domain_file), str(problem_file))
                end = time.perf_counter()
                times.append(end - start)

            results[domain] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'repetitions': repetitions
            }

        return results

    def benchmark_action_grounding(self, domains: List[str], repetitions: int = 10) -> Dict[str, Any]:
        """
        Benchmark action grounding performance.

        Args:
            domains: List of domain names to benchmark
            repetitions: Number of repetitions for timing

        Returns:
            Dictionary with benchmark results
        """
        results = {}

        for domain in domains:
            domain_file = Path(f"benchmarks/{domain}/domain.pddl")
            problem_file = Path(f"benchmarks/{domain}/p01.pddl")

            if not domain_file.exists() or not problem_file.exists():
                continue

            reader = PDDLReader()
            domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))

            times = []
            for _ in range(repetitions):
                from src.core import grounding
                start = time.perf_counter()
                grounded_actions = grounding.ground_all_actions(domain)
                end = time.perf_counter()
                times.append(end - start)

            results[domain] = {
                'num_actions': len(grounded_actions),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'repetitions': repetitions
            }

        return results

    def benchmark_metrics_collection(self, num_actions: List[int], repetitions: int = 10) -> Dict[str, Any]:
        """
        Benchmark metrics collection performance.

        Args:
            num_actions: List of different action counts to test
            repetitions: Number of repetitions for timing

        Returns:
            Dictionary with benchmark results
        """
        results = {}

        for n in num_actions:
            times = []
            for _ in range(repetitions):
                collector = MetricsCollector()

                start = time.perf_counter()
                for i in range(n):
                    action_name = f"action_{i % 10}"
                    objects = [f"obj_{j}" for j in range(i % 3 + 1)]
                    success = i % 3 != 0  # Fail every third action
                    collector.record_action(i, action_name, objects, success, 0.001)
                stats = collector.get_summary_statistics()
                end = time.perf_counter()

                times.append(end - start)

            results[f"{n}_actions"] = {
                'num_actions': n,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'time_per_action': np.mean(times) / n,
                'repetitions': repetitions
            }

        return results

    def benchmark_memory_usage(self, domains: List[str]) -> Dict[str, Any]:
        """
        Benchmark memory usage for different domains.

        Args:
            domains: List of domain names to benchmark

        Returns:
            Dictionary with memory usage statistics
        """
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
        except ImportError:
            print("Warning: psutil not installed, skipping memory benchmarks")
            return {}
        results = {}

        for domain in domains:
            domain_file = Path(f"benchmarks/{domain}/domain.pddl")
            problem_file = Path(f"benchmarks/{domain}/p01.pddl")

            if not domain_file.exists() or not problem_file.exists():
                continue

            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            from src.core import grounding
            reader = PDDLReader()
            domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
            grounded_actions = grounding.ground_all_actions(domain)

            # Measure after loading
            loaded_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create metrics collector and add actions
            collector = MetricsCollector()
            for i in range(100):
                collector.record_action(i, "test_action", ["obj1"], True, 0.001)

            # Measure after metrics
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            results[domain] = {
                'initial_memory_mb': initial_memory,
                'loaded_memory_mb': loaded_memory,
                'final_memory_mb': final_memory,
                'parsing_overhead_mb': loaded_memory - initial_memory,
                'metrics_overhead_mb': final_memory - loaded_memory,
                'num_grounded_actions': len(grounded_actions)
            }

        return results

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to file."""
        output_path = self.output_dir / filename

        # Save as JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Convert to DataFrame and save as CSV
        if results:
            df_data = []
            for category, category_results in results.items():
                for key, metrics in category_results.items():
                    row = {'category': category, 'test': key}
                    row.update(metrics)
                    df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(output_path.with_suffix('.csv'), index=False)

        print(f"Results saved to {output_path}")

    def plot_results(self, results: Dict[str, Any]):
        """Create visualization of benchmark results."""
        if not HAS_MATPLOTLIB:
            print("Plotting skipped - matplotlib not installed")
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot parsing times
        if 'domain_parsing' in results:
            ax = axes[0, 0]
            domains = list(results['domain_parsing'].keys())
            mean_times = [results['domain_parsing'][d]['mean_time'] * 1000 for d in domains]
            std_times = [results['domain_parsing'][d]['std_time'] * 1000 for d in domains]

            ax.bar(domains, mean_times, yerr=std_times, capsize=5)
            ax.set_xlabel('Domain')
            ax.set_ylabel('Parsing Time (ms)')
            ax.set_title('Domain Parsing Performance')
            ax.tick_params(axis='x', rotation=45)

        # Plot grounding performance
        if 'action_grounding' in results:
            ax = axes[0, 1]
            domains = list(results['action_grounding'].keys())
            mean_times = [results['action_grounding'][d]['mean_time'] * 1000 for d in domains]
            num_actions = [results['action_grounding'][d]['num_actions'] for d in domains]

            ax2 = ax.twinx()
            bars = ax.bar(domains, mean_times, alpha=0.7, label='Time')
            line = ax2.plot(domains, num_actions, 'r-o', label='# Actions')

            ax.set_xlabel('Domain')
            ax.set_ylabel('Grounding Time (ms)')
            ax2.set_ylabel('Number of Actions')
            ax.set_title('Action Grounding Performance')
            ax.tick_params(axis='x', rotation=45)

        # Plot metrics collection scaling
        if 'metrics_collection' in results:
            ax = axes[1, 0]
            tests = sorted(results['metrics_collection'].keys(),
                          key=lambda x: results['metrics_collection'][x]['num_actions'])
            num_actions = [results['metrics_collection'][t]['num_actions'] for t in tests]
            mean_times = [results['metrics_collection'][t]['mean_time'] * 1000 for t in tests]

            ax.plot(num_actions, mean_times, 'b-o')
            ax.set_xlabel('Number of Actions')
            ax.set_ylabel('Collection Time (ms)')
            ax.set_title('Metrics Collection Scaling')
            ax.grid(True, alpha=0.3)

        # Plot memory usage
        if 'memory_usage' in results:
            ax = axes[1, 1]
            domains = list(results['memory_usage'].keys())
            parsing_mem = [results['memory_usage'][d]['parsing_overhead_mb'] for d in domains]
            metrics_mem = [results['memory_usage'][d]['metrics_overhead_mb'] for d in domains]

            x = np.arange(len(domains))
            width = 0.35

            ax.bar(x - width/2, parsing_mem, width, label='Parsing')
            ax.bar(x + width/2, metrics_mem, width, label='Metrics')

            ax.set_xlabel('Domain')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage by Domain')
            ax.set_xticks(x)
            ax.set_xticklabels(domains, rotation=45)
            ax.legend()

        plt.tight_layout()
        output_path = self.output_dir / 'benchmark_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Performance benchmarking for online model learning')
    parser.add_argument('--domains', nargs='+',
                       default=['blocksworld', 'gripper', 'logistics', 'rover', 'depots'],
                       help='Domains to benchmark')
    parser.add_argument('--repetitions', type=int, default=10,
                       help='Number of repetitions for timing')
    parser.add_argument('--output-dir', default='benchmarks/performance',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of results')

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.output_dir)

    print("Running performance benchmarks...")
    results = {}

    # Benchmark domain parsing
    print("Benchmarking domain parsing...")
    results['domain_parsing'] = benchmark.benchmark_domain_parsing(args.domains, args.repetitions)

    # Benchmark action grounding
    print("Benchmarking action grounding...")
    results['action_grounding'] = benchmark.benchmark_action_grounding(args.domains, args.repetitions)

    # Benchmark metrics collection
    print("Benchmarking metrics collection...")
    results['metrics_collection'] = benchmark.benchmark_metrics_collection(
        [10, 50, 100, 500, 1000], args.repetitions)

    # Benchmark memory usage
    print("Benchmarking memory usage...")
    results['memory_usage'] = benchmark.benchmark_memory_usage(args.domains)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark.save_results(results, f"benchmark_results_{timestamp}")

    # Display summary
    print("\n=== Benchmark Summary ===")
    print("\nDomain Parsing (mean time in ms):")
    for domain, metrics in results.get('domain_parsing', {}).items():
        print(f"  {domain}: {metrics['mean_time']*1000:.2f} ± {metrics['std_time']*1000:.2f}")

    print("\nAction Grounding:")
    for domain, metrics in results.get('action_grounding', {}).items():
        print(f"  {domain}: {metrics['num_actions']} actions in {metrics['mean_time']*1000:.2f} ms")

    print("\nMetrics Collection (time per action):")
    for test, metrics in results.get('metrics_collection', {}).items():
        print(f"  {metrics['num_actions']} actions: {metrics['time_per_action']*1000000:.2f} μs/action")

    # Generate plots if requested
    if args.plot:
        try:
            benchmark.plot_results(results)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()