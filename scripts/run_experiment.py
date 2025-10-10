#!/usr/bin/env python
"""
Simple script to run a single experiment from a config file.
Usage: python scripts/run_experiment.py configs/experiment_config.yaml
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.runner import ExperimentRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiment.py <config_file.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]

    logger.info(f"Starting experiment with config: {config_file}")

    # Run experiment
    runner = ExperimentRunner(config_file)
    results = runner.run_experiment()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Name: {results['experiment_name']}")
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Iterations: {results['total_iterations']}")
    logger.info(f"Stopping reason: {results['stopping_reason']}")
    logger.info(f"Runtime: {results['runtime_seconds']:.1f} seconds")
    logger.info(f"Success rate: {results['metrics']['success_rate']:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
