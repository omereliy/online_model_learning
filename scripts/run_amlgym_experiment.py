#!/usr/bin/env python3
"""
Run Information Gain experiments via AMLGym benchmarks.

Usage:
    # Single domain, default settings
    python3 scripts/run_amlgym_experiment.py --domain blocksworld

    # Multiple domains
    python3 scripts/run_amlgym_experiment.py --domain blocksworld gripper hanoi

    # All available domains
    python3 scripts/run_amlgym_experiment.py --all-domains

    # Custom settings
    python3 scripts/run_amlgym_experiment.py --domain blocksworld --max-steps 300 --model-mode complete

    # With evaluation
    python3 scripts/run_amlgym_experiment.py --domain blocksworld --evaluate
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from amlgym.algorithms import get_algorithm
from amlgym.benchmarks import (
    get_domain_names,
    get_domain_path,
    get_problems_path,
)
from amlgym.util.util import empty_domain
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import SequentialSimulator

logger = logging.getLogger(__name__)


def run_experiment(
    domain: str,
    max_steps: int = 500,
    model_mode: str = "safe",
    use_object_subset: bool = True,
    spare_objects_per_type: int = 2,
    learn_negative_preconditions: bool = True,
    seed: int = 42,
    output_dir: Path = Path("results/amlgym"),
    evaluate: bool = False,
    selection_strategy: str = "greedy",
    lookahead_depth: int = 2,
    lookahead_top_k: int = 5,
    lookahead_discount: float = 0.9,
    mcts_iterations: int = 50,
    mcts_rollout_depth: int = 5,
) -> dict:
    """Run a single experiment on a domain."""
    logger.info(f"=== {domain} (max_steps={max_steps}, mode={model_mode}, seed={seed}) ===")

    domain_ref_path = get_domain_path(domain)
    input_domain_path = empty_domain(domain_ref_path)

    problem_paths = get_problems_path(domain, kind="learning")
    if not problem_paths:
        logger.warning(f"No learning problems for {domain}, skipping")
        return {}

    problem_path = problem_paths[0]
    logger.info(f"Domain: {domain_ref_path}")
    logger.info(f"Problem: {problem_path}")

    # Create simulator
    problem = PDDLReader().parse_problem(domain_ref_path, problem_path)
    simulator = SequentialSimulator(problem=problem)

    # Run algorithm
    agent = get_algorithm(
        "InformationGainAgent",
        max_steps=max_steps,
        model_mode=model_mode,
        use_object_subset=use_object_subset,
        spare_objects_per_type=spare_objects_per_type,
        learn_negative_preconditions=learn_negative_preconditions,
        selection_strategy=selection_strategy,
        lookahead_depth=lookahead_depth,
        lookahead_top_k=lookahead_top_k,
        lookahead_discount=lookahead_discount,
        mcts_iterations=mcts_iterations,
        mcts_rollout_depth=mcts_rollout_depth,
    )

    start = time.perf_counter()
    model_str, trajectory = agent.learn(simulator, input_domain_path, seed=seed)
    elapsed = time.perf_counter() - start

    n_actions = len(trajectory.actions)
    logger.info(f"Done: {n_actions} actions, {elapsed:.1f}s")

    # Save results
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    model_path = domain_dir / f"learned_domain_{model_mode}.pddl"
    with open(model_path, "w") as f:
        f.write(model_str)
    logger.info(f"Saved model to {model_path}")

    traj_path = domain_dir / "trajectory.txt"
    trajectory.write(str(traj_path))

    result = {
        "domain": domain,
        "steps": n_actions,
        "runtime_s": round(elapsed, 2),
        "model_mode": model_mode,
        "model_path": str(model_path),
    }

    # Evaluation
    if evaluate:
        result.update(run_evaluation(domain, model_path, domain_ref_path))

    return result


def run_evaluation(domain: str, learned_path: Path, ref_path: str) -> dict:
    """Run AMLGym evaluation metrics on a learned model."""
    from amlgym.metrics import syntactic_precision, syntactic_recall

    metrics = {}
    try:
        precision = syntactic_precision(str(learned_path), ref_path)
        recall = syntactic_recall(str(learned_path), ref_path)
        metrics["syntactic_precision"] = precision
        metrics["syntactic_recall"] = recall
        logger.info(f"Syntactic precision: {precision}")
        logger.info(f"Syntactic recall: {recall}")
    except Exception as e:
        logger.warning(f"Syntactic evaluation failed: {e}")

    try:
        from amlgym.benchmarks import get_test_states
        from amlgym.metrics import predictive_power
        from amlgym.modeling.UPEnv import UPEnv

        pp_problems = get_problems_path(domain, kind="predictive_power")
        if pp_problems:
            test_states = get_test_states(domain, kind="predictive_power")
            pp_path = pp_problems[0]
            key = Path(pp_path).name

            sim_learned = UPEnv(str(learned_path), pp_path)
            sim_ref = UPEnv(ref_path, pp_path)
            pp = predictive_power(sim_learned, sim_ref, test_states[key])
            metrics["predictive_power"] = pp
            logger.info(f"Predictive power: {pp}")
    except Exception as e:
        logger.warning(f"Predictive power evaluation failed: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Information Gain via AMLGym")
    parser.add_argument("--domain", nargs="+", help="Domain name(s) to run")
    parser.add_argument("--all-domains", action="store_true", help="Run on all domains")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--model-mode", choices=["safe", "complete"], default="safe")
    parser.add_argument("--no-object-subset", action="store_true",
                        help="Disable object subset selection (use all objects)")
    parser.add_argument("--spare-objects", type=int, default=2,
                        help="Extra objects per type for subset selection (default: 2)")
    parser.add_argument("--no-negative-preconditions", action="store_true",
                        help="Skip negative precondition learning (reduces hypothesis space)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/amlgym")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation metrics")
    parser.add_argument("--selection-strategy",
                        choices=["greedy", "epsilon_greedy", "boltzmann", "lookahead", "mcts"],
                        default="greedy", help="Action selection strategy (default: greedy)")
    parser.add_argument("--lookahead-depth", type=int, default=2,
                        help="Lookahead depth for 'lookahead' strategy (default: 2)")
    parser.add_argument("--lookahead-top-k", type=int, default=5,
                        help="Top-k actions to evaluate in lookahead (default: 5)")
    parser.add_argument("--lookahead-discount", type=float, default=0.9,
                        help="Discount factor for lookahead (default: 0.9)")
    parser.add_argument("--mcts-iterations", type=int, default=50,
                        help="Number of MCTS iterations (default: 50)")
    parser.add_argument("--mcts-rollout-depth", type=int, default=5,
                        help="MCTS rollout depth (default: 5)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all_domains:
        domains = get_domain_names()
    elif args.domain:
        domains = args.domain
    else:
        parser.error("Specify --domain or --all-domains")

    output_dir = Path(args.output_dir)
    results = []

    for domain in domains:
        try:
            result = run_experiment(
                domain=domain,
                max_steps=args.max_steps,
                model_mode=args.model_mode,
                use_object_subset=not args.no_object_subset,
                spare_objects_per_type=args.spare_objects,
                learn_negative_preconditions=not args.no_negative_preconditions,
                seed=args.seed,
                output_dir=output_dir,
                evaluate=args.evaluate,
                selection_strategy=args.selection_strategy,
                lookahead_depth=args.lookahead_depth,
                lookahead_top_k=args.lookahead_top_k,
                lookahead_discount=args.lookahead_discount,
                mcts_iterations=args.mcts_iterations,
                mcts_rollout_depth=args.mcts_rollout_depth,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {domain}: {e}")

    # Print summary
    print("\n=== Summary ===")
    for r in results:
        if r:
            print(f"  {r['domain']:20s}  {r['steps']:4d} steps  {r['runtime_s']:7.1f}s")


if __name__ == "__main__":
    main()
