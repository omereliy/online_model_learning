#!/usr/bin/env python
"""
Debug tool for analyzing experiment logs and reconstructing states.
"""

import sys
import re
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_log_file(log_path: Path, start_line: int = 0, num_lines: int = None) -> list:
    """
    Parse experiment log file.

    Args:
        log_path: Path to log file
        start_line: Line to start from
        num_lines: Number of lines to read (None = all)

    Returns:
        List of parsed log entries
    """
    entries = []

    with open(log_path, 'r') as f:
        lines = f.readlines()[start_line:]

        if num_lines:
            lines = lines[:num_lines]

        for line in lines:
            # Parse timestamp and level
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\S+) - (\w+) - (.*)', line)
            if match:
                timestamp_str, logger_name, level, message = match.groups()
                entries.append({
                    'timestamp': timestamp_str,
                    'logger': logger_name,
                    'level': level,
                    'message': message.strip()
                })

    return entries


def find_failures(log_path: Path) -> list:
    """
    Find all failed actions in the log.

    Args:
        log_path: Path to log file

    Returns:
        List of failed action entries
    """
    failures = []
    entries = parse_log_file(log_path)

    for entry in entries:
        if 'Success: False' in entry['message'] or 'failed' in entry['message'].lower():
            failures.append(entry)

    return failures


def extract_iterations(log_path: Path, target_iterations: list = None) -> dict:
    """
    Extract specific iteration details from log.

    Args:
        log_path: Path to log file
        target_iterations: List of iteration numbers to extract

    Returns:
        Dictionary mapping iteration to log entries
    """
    iterations = {}
    current_iteration = None
    current_entries = []

    entries = parse_log_file(log_path)

    for entry in entries:
        # Check for iteration marker
        iteration_match = re.search(r'step (\d+)', entry['message'])
        if iteration_match:
            iteration = int(iteration_match.group(1))

            # Save previous iteration if needed
            if current_iteration is not None and (target_iterations is None or current_iteration in target_iterations):
                iterations[current_iteration] = current_entries

            current_iteration = iteration
            current_entries = [entry]
        elif current_iteration is not None:
            current_entries.append(entry)

    # Save last iteration
    if current_iteration is not None and (target_iterations is None or current_iteration in target_iterations):
        iterations[current_iteration] = current_entries

    return iterations


def analyze_action_patterns(log_path: Path) -> dict:
    """
    Analyze patterns in action selection and failures.

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with pattern analysis
    """
    patterns = {
        'action_sequences': [],
        'failure_sequences': [],
        'state_sizes': []
    }

    entries = parse_log_file(log_path)

    current_sequence = []
    for entry in entries:
        # Look for action executions
        action_match = re.search(r'Recorded action.*?: (\w+)\((.*?)\) - Success: (\w+)', entry['message'])
        if action_match:
            action, objects, success = action_match.groups()
            action_entry = {
                'action': action,
                'objects': objects,
                'success': success == 'True'
            }
            current_sequence.append(action_entry)

            if len(current_sequence) >= 3:
                # Keep only last 3 actions for pattern detection
                current_sequence = current_sequence[-3:]
                patterns['action_sequences'].append(list(current_sequence))

                # Check for repeated failures
                if all(not a['success'] for a in current_sequence):
                    patterns['failure_sequences'].append(list(current_sequence))

        # Look for state size information
        state_match = re.search(r'state size: (\d+)', entry['message'].lower())
        if state_match:
            patterns['state_sizes'].append(int(state_match.group(1)))

    return patterns


def print_debug_report(experiment_dir: Path, options: dict):
    """
    Print debug report for an experiment.

    Args:
        experiment_dir: Path to experiment directory
        options: Debug options (failures, iterations, patterns)
    """
    log_file = experiment_dir / 'experiment.log'
    debug_file = experiment_dir / 'debug.log'

    # Use debug log if available, otherwise main log
    log_path = debug_file if debug_file.exists() else log_file

    if not log_path.exists():
        print(f"No log file found in {experiment_dir}")
        return

    print("\n" + "=" * 80)
    print(f"DEBUG REPORT: {experiment_dir.name}")
    print(f"Log file: {log_path.name}")
    print("=" * 80)

    if options.get('failures'):
        print("\n" + "-" * 40)
        print("FAILED ACTIONS")
        print("-" * 40)
        failures = find_failures(log_path)
        for i, failure in enumerate(failures[:10], 1):  # Show first 10
            print(f"\n{i}. [{failure['timestamp']}]")
            print(f"   {failure['message'][:200]}")

        if len(failures) > 10:
            print(f"\n... and {len(failures) - 10} more failures")

    if options.get('iterations'):
        print("\n" + "-" * 40)
        print("SPECIFIC ITERATIONS")
        print("-" * 40)
        target_iters = options['iterations']
        iterations = extract_iterations(log_path, target_iters)

        for iter_num, entries in sorted(iterations.items())[:5]:
            print(f"\nIteration {iter_num}:")
            for entry in entries[:5]:  # Show first 5 entries
                print(f"  [{entry['level']}] {entry['message'][:100]}")

    if options.get('patterns'):
        print("\n" + "-" * 40)
        print("ACTION PATTERNS")
        print("-" * 40)
        patterns = analyze_action_patterns(log_path)

        # Show repeated failure patterns
        if patterns['failure_sequences']:
            print("\nRepeated failure sequences:")
            for seq in patterns['failure_sequences'][:5]:
                actions = ' -> '.join([f"{a['action']}({a['objects']})" for a in seq])
                print(f"  {actions}")

        # State size evolution
        if patterns['state_sizes']:
            print(f"\nState size evolution:")
            print(f"  Initial: {patterns['state_sizes'][0]}")
            print(f"  Final: {patterns['state_sizes'][-1]}")
            print(f"  Min: {min(patterns['state_sizes'])}")
            print(f"  Max: {max(patterns['state_sizes'])}")

    print("\n" + "=" * 80)


def extract_timeline(experiment_dir: Path, output_file: str = None):
    """
    Extract a timeline of key events from the experiment.

    Args:
        experiment_dir: Path to experiment directory
        output_file: Optional output file for timeline
    """
    log_file = experiment_dir / 'experiment.log'
    if not log_file.exists():
        print(f"No log file found in {experiment_dir}")
        return

    timeline = []
    entries = parse_log_file(log_file)

    for entry in entries:
        # Key events to extract
        if any(marker in entry['message'] for marker in [
            'Starting experiment',
            'Experiment completed',
            'converged',
            'ERROR',
            'WARNING',
            'milestone',
            'checkpoint'
        ]):
            timeline.append({
                'timestamp': entry['timestamp'],
                'event': entry['message'][:200]
            })

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(timeline, f, indent=2)
        print(f"Timeline saved to {output_file}")
    else:
        print("\nEXPERIMENT TIMELINE:")
        print("-" * 40)
        for event in timeline:
            print(f"[{event['timestamp']}] {event['event']}")


def main():
    parser = argparse.ArgumentParser(description='Debug experiment execution')
    parser.add_argument('experiment', help='Experiment directory name')
    parser.add_argument('--failures', action='store_true',
                       help='Show failed actions')
    parser.add_argument('--iterations', nargs='+', type=int,
                       help='Extract specific iterations')
    parser.add_argument('--patterns', action='store_true',
                       help='Analyze action patterns')
    parser.add_argument('--timeline', action='store_true',
                       help='Extract experiment timeline')
    parser.add_argument('--output', help='Output file for extracted data')

    args = parser.parse_args()

    results_dir = project_root / 'results'
    experiment_dir = results_dir / args.experiment

    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return

    if args.timeline:
        extract_timeline(experiment_dir, args.output)
    else:
        options = {
            'failures': args.failures,
            'iterations': args.iterations,
            'patterns': args.patterns
        }
        print_debug_report(experiment_dir, options)


if __name__ == "__main__":
    main()