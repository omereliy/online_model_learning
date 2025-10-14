#!/usr/bin/env python
"""Debug script to examine precision/recall calculation mismatches."""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.pddl_io import PDDLReader

def normalize_literal(literal: str) -> str:
    """Same normalization as in visualize_paper_results.py"""
    if not literal:
        return literal

    # Handle delete effects with (not ...) wrapper
    if literal.startswith('(not '):
        inner_start = literal.find('(', 4)
        if inner_start != -1:
            inner_end = literal.rfind(')')
            if inner_end != -1:
                literal = literal[inner_start:inner_end]

    # Remove outer parentheses
    literal = literal.strip()
    if literal.startswith('(') and literal.endswith(')'):
        literal = literal[1:-1].strip()

    # Parse predicate and parameters
    if '(' in literal and literal.endswith(')'):
        paren_idx = literal.index('(')
        predicate = literal[:paren_idx]
        params_str = literal[paren_idx+1:-1]
        params = [p.strip() for p in params_str.split(',')] if params_str else []
    else:
        parts = literal.split()
        if not parts:
            return literal
        predicate = parts[0]
        params = parts[1:] if len(parts) > 1 else []

    # Canonicalize variable parameters by order of first appearance
    canonical_vars = ['?x', '?y', '?z', '?w', '?v', '?u', '?t', '?s']
    var_mapping = {}
    next_canonical_idx = 0

    normalized_params = []
    for param in params:
        if param.startswith('?'):
            # Variable parameter - canonicalize by order of first appearance
            if param not in var_mapping:
                var_mapping[param] = canonical_vars[next_canonical_idx] if next_canonical_idx < len(canonical_vars) else f'?v{next_canonical_idx}'
                next_canonical_idx += 1
            normalized_params.append(var_mapping[param])
        else:
            # Non-variable parameter (constant) - keep as-is
            normalized_params.append(param)

    # Reconstruct
    if normalized_params:
        return f"{predicate}({','.join(normalized_params)})"
    else:
        return f"{predicate}()"


def main():
    """Debug Depots domain precision/recall."""

    # Load ground truth
    domain_file = project_root / "benchmarks/olam-compatible/depots/domain.pddl"
    problem_file = project_root / "benchmarks/olam-compatible/depots/p01.pddl"

    reader = PDDLReader()
    ground_truth, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))

    # Load learned model
    learned_model_path = project_root / "results/paper/comparison_20251014_204308/information_gain/depots/p01/experiments/learned_model.json"

    with open(learned_model_path, 'r') as f:
        learned_model = json.load(f)

    # Compare Lift action
    action_name = "lift"
    true_action = ground_truth.get_action(action_name)
    learned_action = learned_model['actions'][action_name]

    print(f"\n{'='*80}")
    print(f"DEBUGGING {action_name.upper()} ACTION")
    print(f"{'='*80}\n")

    # Ground truth preconditions (MUST also normalize these!)
    print("Ground Truth Preconditions (RAW):")
    true_prec_raw = true_action.preconditions
    for p in sorted(true_prec_raw):
        print(f"  - {p}")

    # Normalize ground truth preconditions
    print("\nGround Truth Preconditions (NORMALIZED):")
    print("Individual normalization:")
    for p in sorted(true_prec_raw):
        normalized = normalize_literal(p)
        print(f"  {p} → {normalized}")

    true_prec = set(normalize_literal(lit) for lit in true_prec_raw)
    print("\nUnique normalized preconditions:")
    for p in sorted(true_prec):
        print(f"  - {p}")
    print(f"Total: {len(true_prec)}")

    # Learned preconditions
    print("\nLearned Possible Preconditions (Information Gain):")
    learned_prec_raw = learned_action['preconditions']['possible']
    learned_prec = set(normalize_literal(lit) for lit in learned_prec_raw)
    for p in sorted(learned_prec):
        print(f"  - {p}")
    print(f"Total: {len(learned_prec)}")

    # Calculate precision/recall
    tp = true_prec & learned_prec
    fp = learned_prec - true_prec
    fn = true_prec - learned_prec

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0

    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")

    print(f"True Positives (in both): {len(tp)}")
    for p in sorted(tp):
        print(f"  ✓ {p}")

    print(f"\nFalse Positives (learned but not in ground truth): {len(fp)}")
    for p in sorted(fp):
        print(f"  ✗ {p}")

    print(f"\nFalse Negatives (in ground truth but not learned): {len(fn)}")
    for p in sorted(fn):
        print(f"  ✗ {p}")

    print(f"\nPrecision: {precision:.3f} ({len(tp)}/{len(tp) + len(fp)})")
    print(f"Recall: {recall:.3f} ({len(tp)}/{len(tp) + len(fn)})")

    print(f"\n{'='*80}")
    print("VARIABLE PARAMETER ANALYSIS")
    print(f"{'='*80}\n")

    # Check if the issue is variable names
    print("Checking if variable names are the issue...")
    print("\nGround truth uses these literals:")
    for lit in sorted(true_prec):
        print(f"  {lit}")

    print("\nLearned model uses these literals:")
    for lit in sorted(learned_prec):
        # Only show positive preconditions
        if not lit.startswith('¬'):
            print(f"  {lit}")


if __name__ == "__main__":
    main()
