#!/usr/bin/env python
"""Quick test to verify CNF clauses are being built after fix."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.algorithms.information_gain import InformationGainLearner
from src.environments.active_environment import ActiveEnvironment

def test_cnf_building():
    """Test that CNF clauses are built from observations."""

    # Setup
    domain = 'benchmarks/olam-compatible/blocksworld/domain.pddl'
    problem = 'benchmarks/olam-compatible/blocksworld/p01.pddl'

    learner = InformationGainLearner(
        domain_file=domain,
        problem_file=problem,
        max_iterations=10
    )
    env = ActiveEnvironment(domain, problem)

    print("Initial state:")
    for action_name in learner.pre.keys():
        cnf = learner.cnf_managers[action_name]
        print(f"  {action_name}: {len(cnf.cnf.clauses)} CNF clauses")

    # Run 5 learning iterations
    print("\nRunning 5 iterations...")
    for i in range(5):
        state = env.get_state()
        action, objects = learner.select_action(state)
        success, _ = env.execute(action, objects)
        next_state = env.get_state() if success else state

        # Observe (update_model is now called automatically)
        learner.observe(state, action, objects, success, next_state)

        print(f"Iteration {i+1}: {action}({','.join(objects)}) - {'SUCCESS' if success else 'FAILURE'}")

    print("\nAfter 5 iterations:")
    total_clauses = 0
    for action_name in learner.pre.keys():
        cnf = learner.cnf_managers[action_name]
        clauses = len(cnf.cnf.clauses)
        total_clauses += clauses
        if clauses > 0:
            print(f"  {action_name}: {clauses} CNF clauses ✓")
        else:
            print(f"  {action_name}: {clauses} CNF clauses")

    print(f"\nTotal CNF clauses across all actions: {total_clauses}")

    if total_clauses > 0:
        print("✅ SUCCESS: CNF clauses are being built from observations!")
        return True
    else:
        print("❌ FAILURE: No CNF clauses built - bug still present")
        return False

if __name__ == "__main__":
    success = test_cnf_building()
    sys.exit(0 if success else 1)