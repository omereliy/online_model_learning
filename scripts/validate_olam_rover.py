#!/usr/bin/env python
"""
Validation script for OLAM on Rover domain.
Runs a full experiment with extensive logging to verify actual learning behavior.
"""

import sys
import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.experiments.runner import ExperimentRunner
from src.algorithms.olam_adapter import OLAMAdapter
from src.environments.pddl_environment import PDDLEnvironment
from src.experiments.metrics import MetricsCollector


def setup_logging():
    """Setup detailed logging for validation."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('validation_logs')
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f'olam_rover_validation_{timestamp}.log'

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file


def validate_olam_rover():
    """Run OLAM validation on Rover domain with detailed tracking."""

    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting OLAM Validation on Rover Domain")
    logger.info("=" * 80)

    # Configuration for validation
    config = {
        'experiment': {
            'name': 'olam_rover_validation',
            'algorithm': 'olam',
            'seed': 42
        },
        'domain_problem': {
            'domain': '/home/omer/projects/domains/rover/domain.pddl',
            'problem': '/home/omer/projects/domains/rover/pfile1.pddl'
        },
        'algorithm_params': {
            'olam': {
                'max_iterations': 300,
                'eval_frequency': 10,
                'bypass_java': True  # Use Java bypass mode
            }
        },
        'metrics': {
            'interval': 1,  # Record every action
            'window_size': 50
        },
        'stopping_criteria': {
            'max_iterations': 50,  # Start with fewer iterations for initial validation
            'max_runtime_seconds': 300,  # 5 minutes
            'convergence_check_interval': 10
        },
        'output': {
            'directory': 'validation_results/',
            'formats': ['csv', 'json'],
            'save_learned_model': True
        }
    }

    try:
        # Initialize environment separately to verify it works
        logger.info("\n--- Testing PDDL Environment ---")
        env = PDDLEnvironment(
            config['domain_problem']['domain'],
            config['domain_problem']['problem']
        )

        initial_state = env.get_state()
        logger.info(f"Initial state has {len(initial_state)} fluents")
        logger.info(f"Sample fluents: {list(initial_state)[:5]}")

        # Get applicable actions
        applicable = env.get_applicable_actions()
        logger.info(f"Initially {len(applicable)} applicable actions")
        logger.info(f"Sample actions: {applicable[:3]}")

        # Initialize OLAM adapter directly to verify
        logger.info("\n--- Testing OLAM Adapter ---")
        olam = OLAMAdapter(
            domain_file=config['domain_problem']['domain'],
            problem_file=config['domain_problem']['problem'],
            pddl_environment=env,  # Pass PDDL environment for proper action filtering
            **config['algorithm_params']['olam']
        )

        # Test action selection
        action, objects = olam.select_action(initial_state)
        logger.info(f"OLAM selected action: {action}({','.join(objects)})")

        # Now run a short validation experiment
        logger.info("\n--- Running Validation Experiment ---")

        # Track domain evolution
        domain_snapshots = []

        for iteration in range(10):  # Just 10 iterations for initial validation
            logger.info(f"\n=== Iteration {iteration} ===")

            # Get current state
            state = env.get_state()
            logger.info(f"Current state has {len(state)} true fluents")

            # Select action
            action, objects = olam.select_action(state)
            logger.info(f"OLAM selected: {action}({','.join(objects)})")

            # Log why this action was selected (if available from OLAM)
            if hasattr(olam.learner, 'last_strategy'):
                logger.info(f"Selection strategy: {olam.learner.last_strategy}")

            # Execute action
            old_state = state.copy()
            success, runtime = env.execute(action, objects)
            new_state = env.get_state()

            logger.info(f"Execution result: {'SUCCESS' if success else 'FAILURE'} (runtime: {runtime:.3f}s)")

            # Log state changes
            if success:
                added = new_state - old_state
                removed = old_state - new_state
                if added:
                    logger.info(f"  Added fluents: {added}")
                if removed:
                    logger.info(f"  Removed fluents: {removed}")
            else:
                logger.info("  State unchanged (action failed)")

            # Observe for learning
            olam.observe(state, action, objects, success, new_state if success else None)

            # Every 5 iterations, save OLAM's learned model
            if iteration % 5 == 0 and iteration > 0:
                learned_model = olam.get_learned_model()
                domain_snapshots.append({
                    'iteration': iteration,
                    'model': learned_model,
                    'num_actions': len(learned_model['actions']),
                    'num_predicates': len(learned_model['predicates'])
                })

                logger.info(f"\n--- Domain Snapshot at iteration {iteration} ---")
                logger.info(f"Learned {len(learned_model['actions'])} actions")
                logger.info(f"Tracked {len(learned_model['predicates'])} predicates")

                # Log a sample action's learned preconditions
                sample_actions = list(learned_model['actions'].keys())[:2]
                for action_key in sample_actions:
                    action_data = learned_model['actions'][action_key]
                    logger.info(f"  {action_key}:")
                    logger.info(f"    Certain preconditions: {action_data['preconditions']['certain'][:3]}")
                    logger.info(f"    Uncertain preconditions: {action_data['preconditions']['uncertain'][:3]}")

        # Save final validation results
        logger.info("\n--- Validation Complete ---")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'iterations_run': 10,
            'domain_snapshots': domain_snapshots,
            'log_file': str(log_file)
        }

        results_dir = Path('validation_results')
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f'olam_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Log file: {log_file}")

        return True

    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = validate_olam_rover()

    if success:
        print("\n✅ OLAM validation completed successfully!")
        print("Check validation_logs/ and validation_results/ for details")
    else:
        print("\n❌ OLAM validation failed!")
        print("Check logs for error details")
        sys.exit(1)