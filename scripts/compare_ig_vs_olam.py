#!/usr/bin/env python3
"""
Standalone comparison: InformationGainAgent vs OLAM on AMLGym benchmarks.

Runs both algorithms step-by-step with periodic checkpoint evaluation.
All imports are from PyPI packages (amlgym, information-gain-aml, olam).

Setup:
    conda create -n ig_comparison python=3.12 -y
    conda activate ig_comparison
    pip install "amlgym>=1.0.11"

Usage:
    python3 compare_ig_vs_olam.py --domain blocksworld --max-steps 500
    python3 compare_ig_vs_olam.py --domain blocksworld depots --checkpoint-interval 50
    python3 compare_ig_vs_olam.py --all-domains --max-steps 500 --seed 42

Output structure:
    results/<run>/
    ├── config.json
    ├── <domain>/
    │   ├── ig/
    │   │   ├── checkpoints.json    # metrics at every checkpoint
    │   │   └── models/step_050.pddl, step_100.pddl, ...
    │   ├── olam/
    │   │   ├── checkpoints.json
    │   │   └── models/step_050.pddl, ...
    │   └── summary.json            # final side-by-side comparison
    └── all_results.json            # combined results for plotting
"""

import argparse
import json
import logging
import os
import random
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from unified_planning.exceptions import UPInvalidActionError
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.model import Fluent
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import BoolType, SequentialSimulator

from amlgym.benchmarks import (
    get_domain_names,
    get_domain_path,
    get_problems_path,
    get_test_states,
)
from amlgym.metrics import (
    predictive_power,
    problem_solving,
    syntactic_precision,
    syntactic_recall,
)
from amlgym.modeling.UPEnv import UPEnv
from amlgym.util.util import empty_domain

from information_gain_aml.algorithms import InformationGainLearner
from information_gain_aml.core import UPAdapter

from olam.OLAM import OLAM as OLAMBase
from olam.modeling.PDDLenv import PDDLEnv

logger = logging.getLogger(__name__)

SKIP_DOMAINS = {"gripper"}  # Malformed type hierarchy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_str: str, domain: str, domain_ref_path: str,
                   run_predictive: bool = True, run_solving: bool = True) -> dict:
    """Run AMLGym evaluation metrics on a learned model string."""
    metrics = {}
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False)
    tmp.write(model_str)
    tmp.close()
    model_path = tmp.name

    try:
        # Syntactic (fast — always run)
        try:
            prec = syntactic_precision(model_path, domain_ref_path)
            rec = syntactic_recall(model_path, domain_ref_path)
            f1 = {}
            for key in prec:
                p, r = prec[key], rec[key]
                f1[key] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
            metrics["syntactic_precision"] = prec
            metrics["syntactic_recall"] = rec
            metrics["syntactic_f1"] = f1
        except Exception as e:
            logger.warning(f"Syntactic eval failed: {e}")
            metrics["syntactic_error"] = str(e)

        # Predictive power (slow — optional per checkpoint)
        if run_predictive:
            try:
                pp_problems = get_problems_path(domain, kind="predictive_power")
                if pp_problems:
                    test_states = get_test_states(domain, kind="predictive_power")
                    sims_l, sims_r, states = [], [], []
                    for pp_path in pp_problems:
                        key = Path(pp_path).name
                        if key in test_states:
                            sims_l.append(UPEnv(model_path, pp_path))
                            sims_r.append(UPEnv(domain_ref_path, pp_path))
                            states.append(test_states[key])
                    if sims_l:
                        pp = predictive_power(sims_l, sims_r, states,
                                              show_progress=False)
                        metrics["predictive_power"] = pp
            except Exception as e:
                logger.warning(f"Predictive power failed: {e}")
                metrics["predictive_power_error"] = str(e)

        # Problem solving (slow — optional per checkpoint)
        if run_solving:
            try:
                solving_probs = get_problems_path(domain, kind="solving")
                if solving_probs:
                    res = problem_solving(model_path, domain_ref_path, solving_probs,
                                          timeout=60, show_progress=False)
                    metrics["problem_solving"] = res
            except Exception as e:
                logger.warning(f"Problem solving failed: {e}")
                metrics["problem_solving_error"] = str(e)

    finally:
        os.unlink(model_path)

    return metrics


def save_checkpoint(algo_dir: Path, step: int, elapsed: float, model_str: str,
                    metrics: dict, checkpoints: list):
    """Save a single checkpoint: model file + append to checkpoint list."""
    models_dir = algo_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"step_{step:04d}.pddl"
    model_path.write_text(model_str)

    entry = {"step": step, "elapsed_s": round(elapsed, 2), "metrics": metrics}
    checkpoints.append(entry)

    # Flush checkpoint list to disk incrementally (survives crashes)
    with open(algo_dir / "checkpoints.json", "w") as f:
        json.dump(checkpoints, f, indent=2)


# ---------------------------------------------------------------------------
# InformationGain runner (step-by-step via PyPI information-gain-aml)
# ---------------------------------------------------------------------------

def run_ig(input_domain_path: str, simulator: SequentialSimulator,
           domain_name: str, domain_ref_path: str,
           max_steps: int, seed: int, checkpoint_interval: int,
           full_eval_at_checkpoints: bool, output_dir: Path) -> dict:
    """Run InformationGainLearner with periodic checkpoint evaluation."""
    random.seed(seed)
    np.random.seed(seed)

    algo_dir = output_dir / domain_name / "ig"
    algo_dir.mkdir(parents=True, exist_ok=True)

    problem = simulator._problem

    # Write problem to temp file
    tmp_prob = tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False)
    PDDLWriter(problem).write_problem(tmp_prob.name)
    tmp_prob.close()

    checkpoints = []
    t0 = time.perf_counter()
    actual_steps = 0

    try:
        learner = InformationGainLearner(
            domain_file=input_domain_path,
            problem_file=tmp_prob.name,
            max_iterations=max_steps,
            use_object_subset=True,
            spare_objects_per_type=2,
            learn_negative_preconditions=True,
            seed=seed,
        )

        up_state = simulator.get_initial_state()

        for step in range(1, max_steps + 1):
            state_set = UPAdapter.up_state_to_fluent_set(up_state, problem)
            action_name, objects = learner.select_action(state_set)

            if action_name == "no_action":
                logger.info(f"  IG converged at step {step}")
                break

            up_action = problem.action(action_name)
            up_objects = tuple(problem.object(o) for o in objects)
            action_instance = ActionInstance(up_action, up_objects)

            try:
                next_up_state = simulator.apply(up_state, action_instance)
            except UPInvalidActionError:
                next_up_state = None

            if next_up_state is not None:
                next_set = UPAdapter.up_state_to_fluent_set(next_up_state, problem)
                learner.observe(state_set, action_name, objects, True, next_set)
                up_state = next_up_state
            else:
                learner.observe(state_set, action_name, objects, False, None)

            actual_steps = step

            # Checkpoint
            if step % checkpoint_interval == 0:
                elapsed = time.perf_counter() - t0
                model_str = learner.to_pddl_string(mode="safe")
                logger.info(f"  IG checkpoint step={step} ({elapsed:.1f}s)")

                is_final = (step == max_steps)
                run_heavy = full_eval_at_checkpoints or is_final
                metrics = evaluate_model(model_str, domain_name, domain_ref_path,
                                         run_predictive=run_heavy,
                                         run_solving=run_heavy)
                save_checkpoint(algo_dir, step, elapsed, model_str,
                                metrics, checkpoints)

        # Always save final state if not already covered by last checkpoint
        last_saved = checkpoints[-1]["step"] if checkpoints else -1
        if last_saved != actual_steps:
            elapsed = time.perf_counter() - t0
            model_str = learner.to_pddl_string(mode="safe")
            metrics = evaluate_model(model_str, domain_name, domain_ref_path,
                                     run_predictive=True, run_solving=True)
            save_checkpoint(algo_dir, actual_steps, elapsed, model_str,
                            metrics, checkpoints)

    finally:
        os.unlink(tmp_prob.name)

    total_time = time.perf_counter() - t0
    return {
        "algorithm": "InformationGain",
        "steps": actual_steps,
        "runtime_s": round(total_time, 2),
        "checkpoints": len(checkpoints),
        "final_metrics": checkpoints[-1]["metrics"] if checkpoints else {},
    }


# ---------------------------------------------------------------------------
# OLAM runner (override run() loop for checkpoints)
# ---------------------------------------------------------------------------

class CheckpointOLAM(OLAMBase):
    """OLAM subclass that extracts intermediate models during learning."""

    def extract_model_string(self) -> str:
        """Extract current learned model as PDDL string."""
        domain = self.domain.clone()
        dummy = Fluent("dummy", BoolType())
        for a in domain.actions:
            a.add_effect(dummy, True)
        domain_str = PDDLWriter(domain).get_domain()
        return domain_str.replace("(dummy)", "")

    def run_with_checkpoints(self, simulator: SequentialSimulator,
                             max_steps: int = 10000,
                             checkpoint_interval: int = 50,
                             on_checkpoint=None):
        """
        Same as OLAM.run() but calls on_checkpoint(step, model_str) at
        regular intervals. Copied from olam==1.0.3 with minimal additions.
        """
        simulator = PDDLEnv(simulator)
        self.reset(simulator)

        state = self.infer_state_types(self.initial_state.clone())
        from olam.modeling.trajectory import Trajectory as OlamTrajectory
        traj = OlamTrajectory(observations=[state], actions=list())

        plan = None
        actual_steps = 0

        for step in range(1, max_steps + 1):
            # --- Action selection (unchanged from OLAM.run) ---
            executing_learning_action = False
            actions_for_preconditions = (
                self.action_generator.get_learning_actions_precs(
                    state, self.last_failed_action,
                    self.pre_bot, self.pre_bot_ambiguous_types,
                )
            )
            actions_for_effects = self.action_generator.get_learning_actions_effs(
                state, self.uncertain_positive_effects,
                self.uncertain_negative_effects,
            )
            learning_actions = actions_for_preconditions + actions_for_effects

            if len(learning_actions) > 0:
                learning_action_label = learning_actions.pop(0)
                executing_learning_action = True

                learning_action = self.action_generator.parse_textual_action(
                    learning_action_label, self.action_generator.domain
                )
                obj_types = [
                    self.problem.object(str(o)).type
                    for o in learning_action.actual_parameters
                ]
                action_types = [p.type for p in learning_action.action.parameters]
                compatible_types = [
                    p_type.is_compatible(o_type)
                    for o_type, p_type in zip(obj_types, action_types, strict=True)
                ]
                if not np.all(compatible_types):
                    self.executed_actions_ambiguous[str(state)].add(
                        learning_action_label
                    )
                else:
                    self.executed_actions[str(state)].add(learning_action_label)

                op = self.action_generator.parse_textual_action(
                    learning_action_label, self.action_generator.domain
                )
            else:
                if plan is None or len(plan.actions) == 0:
                    operators_to_refine = [
                        a.name
                        for a in sorted(
                            self.domain.actions,
                            key=lambda x: len(x.preconditions),
                        )
                        if a.name not in self.learned_operators
                    ]
                    for operator_name in operators_to_refine:
                        plan = self.learner.plan_for_operator(
                            state,
                            self.domain.action(operator_name),
                            self.uncertain_positive_effects,
                            self.uncertain_negative_effects,
                            self.pre_bot,
                            self.pre_bot_ambiguous_types,
                            self.executed_actions_ambiguous,
                            self.problem,
                            self.invalid_actions,
                            self.PLANNER_CFG,
                        )
                        if plan is None:
                            self.learned_operators.add(operator_name)
                        else:
                            assert len(plan.actions) > 0
                            break
                if plan is None:
                    break  # No more learning possible
                else:
                    assert len(plan.actions) > 0
                    op = plan.actions[0]
                    plan.actions.remove(op)
                    if len(plan.actions) == 0:
                        plan = None

            # --- Execution ---
            operator_name = op.action.name
            actual_params = op.actual_parameters
            next_state, reward, done, truncated, info = self.simulator.step(op)
            next_state = self.infer_state_types(next_state)

            executable = next_state is not None
            operator = self.domain.action(operator_name)
            action_instance = ActionInstance(operator, actual_params)

            if executable:
                traj.add_action(action_instance)
                traj.add_obs(next_state)
                updated_effs = self.learn_effects(action_instance, state, next_state)
                updated_precs = self.learn_preconditions(action_instance, state)
                if updated_precs or updated_effs:
                    self.learner.goal_gen.unsolvable_goals = defaultdict(set)
                    self.learned_operators = set()
                state = next_state
                self.last_failed_action = None
                if executing_learning_action:
                    plan = None
            else:
                params_map = {
                    p: o for p, o in zip(
                        action_instance.action.parameters,
                        action_instance.actual_parameters, strict=True,
                    )
                }
                ground_precs = {
                    str(p.substitute(params_map))
                    for p in action_instance.action.preconditions
                }
                pos_literals = {str(lit) for lit in state.positive_literals}
                if ground_precs.issubset(pos_literals):
                    if str(action_instance) not in {
                        str(a) for a in self.invalid_actions
                    }:
                        self.invalid_actions.add(action_instance)
                else:
                    self.learn_preconditions_from_failed_action(
                        action_instance, state
                    )
                self.last_failed_action = action_instance

            actual_steps = step

            # --- Checkpoint callback ---
            if on_checkpoint and step % checkpoint_interval == 0:
                model_str = self.extract_model_string()
                on_checkpoint(step, model_str)

        else:
            # Loop completed without break — all max_steps used
            actual_steps = max_steps

        # Final model
        final_model = self.extract_model_string()
        return final_model, traj, actual_steps


def run_olam(input_domain_path: str, simulator: SequentialSimulator,
             domain_name: str, domain_ref_path: str,
             max_steps: int, seed: int, checkpoint_interval: int,
             full_eval_at_checkpoints: bool, output_dir: Path) -> dict:
    """Run OLAM with periodic checkpoint evaluation."""
    algo_dir = output_dir / domain_name / "olam"
    algo_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    t0 = time.perf_counter()

    olam = CheckpointOLAM(domain_path=input_domain_path)

    def on_checkpoint(step, model_str):
        elapsed = time.perf_counter() - t0
        logger.info(f"  OLAM checkpoint step={step} ({elapsed:.1f}s)")
        is_final = (step == max_steps)
        run_heavy = full_eval_at_checkpoints or is_final
        metrics = evaluate_model(model_str, domain_name, domain_ref_path,
                                 run_predictive=run_heavy,
                                 run_solving=run_heavy)
        save_checkpoint(algo_dir, step, elapsed, model_str, metrics, checkpoints)

    final_model, traj, actual_steps = olam.run_with_checkpoints(
        simulator,
        max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
        on_checkpoint=on_checkpoint,
    )

    # Always save final state if not already covered by last checkpoint
    last_saved = checkpoints[-1]["step"] if checkpoints else -1
    if last_saved != actual_steps:
        elapsed = time.perf_counter() - t0
        metrics = evaluate_model(final_model, domain_name, domain_ref_path,
                                 run_predictive=True, run_solving=True)
        save_checkpoint(algo_dir, actual_steps, elapsed, final_model,
                        metrics, checkpoints)

    total_time = time.perf_counter() - t0
    return {
        "algorithm": "OLAM",
        "steps": actual_steps,
        "runtime_s": round(total_time, 2),
        "checkpoints": len(checkpoints),
        "final_metrics": checkpoints[-1]["metrics"] if checkpoints else {},
    }


# ---------------------------------------------------------------------------
# Domain runner
# ---------------------------------------------------------------------------

def run_domain(domain: str, max_steps: int, seed: int,
               checkpoint_interval: int, full_eval_at_checkpoints: bool,
               output_dir: Path) -> dict:
    """Run both algorithms on a single domain."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Domain: {domain} | steps={max_steps} | "
                f"ckpt_every={checkpoint_interval} | seed={seed}")
    logger.info(f"{'='*60}")

    domain_ref_path = get_domain_path(domain)
    input_domain_path = empty_domain(
        domain_ref_path,
        os.path.join(tempfile.gettempdir(), f"empty_{domain}.pddl"),
    )

    problem_paths = get_problems_path(domain, kind="learning")
    if not problem_paths:
        logger.warning(f"No learning problems for {domain}, skipping")
        return {"domain": domain, "skipped": True}

    problem_path = problem_paths[0]
    problem = PDDLReader().parse_problem(domain_ref_path, problem_path)
    simulator = SequentialSimulator(problem=problem)

    result = {
        "domain": domain,
        "problem": Path(problem_path).name,
        "max_steps": max_steps,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
    }

    # --- InformationGain ---
    logger.info("Running InformationGain...")
    try:
        result["ig"] = run_ig(
            input_domain_path, simulator, domain, domain_ref_path,
            max_steps, seed, checkpoint_interval,
            full_eval_at_checkpoints, output_dir,
        )
        logger.info(f"  IG finished: {result['ig']['steps']} steps, "
                     f"{result['ig']['runtime_s']}s")
    except Exception as e:
        logger.error(f"  IG failed: {e}\n{traceback.format_exc()}")
        result["ig"] = {"error": str(e)}

    # --- OLAM ---
    logger.info("Running OLAM...")
    try:
        result["olam"] = run_olam(
            input_domain_path, simulator, domain, domain_ref_path,
            max_steps, seed, checkpoint_interval,
            full_eval_at_checkpoints, output_dir,
        )
        logger.info(f"  OLAM finished: {result['olam']['steps']} steps, "
                     f"{result['olam']['runtime_s']}s")
    except Exception as e:
        logger.error(f"  OLAM failed: {e}\n{traceback.format_exc()}")
        result["olam"] = {"error": str(e)}

    # Save domain summary
    summary_path = output_dir / domain / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a comparison table of final metrics."""
    header = (f"{'DOMAIN':<18} {'IG Steps':>8} {'IG Time':>8} {'IG F1':>7} "
              f"{'IG Solv':>8}  {'OL Steps':>8} {'OL Time':>8} {'OL F1':>7} "
              f"{'OL Solv':>8}")
    print(f"\n{'='*98}")
    print(header)
    print(f"{'-'*98}")

    for r in results:
        if r.get("skipped") or "domain" not in r:
            continue

        def fmt(algo_key):
            algo = r.get(algo_key, {})
            if "error" in algo:
                return "ERR", "ERR", "ERR", "ERR"
            steps = str(algo.get("steps", "?"))
            rt = f"{algo.get('runtime_s', 0):.0f}s"
            fm = algo.get("final_metrics", {})
            f1 = fm.get("syntactic_f1", {}).get("mean")
            f1s = f"{f1:.2f}" if f1 is not None else "N/A"
            sv = fm.get("problem_solving", {}).get("solving_ratio")
            svs = f"{sv:.2f}" if sv is not None else "N/A"
            return steps, rt, f1s, svs

        ig = fmt("ig")
        ol = fmt("olam")
        d = r["domain"]
        print(f"{d:<18} {ig[0]:>8} {ig[1]:>8} {ig[2]:>7} {ig[3]:>8}"
              f"  {ol[0]:>8} {ol[1]:>8} {ol[2]:>7} {ol[3]:>8}")

    print(f"{'='*98}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare InformationGain vs OLAM on AMLGym benchmarks "
                    "with checkpoint evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", nargs="+", help="Domain name(s)")
    parser.add_argument("--all-domains", action="store_true")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max interaction steps per algorithm")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--full-eval-at-checkpoints", action="store_true",
                        help="Run predictive_power + problem_solving at "
                             "EVERY checkpoint (slow). Default: only at final.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/ig_vs_olam")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all_domains:
        domains = [d for d in sorted(get_domain_names()) if d not in SKIP_DOMAINS]
    elif args.domain:
        domains = args.domain
    else:
        parser.error("Specify --domain or --all-domains")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "domains": domains,
        "max_steps": args.max_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "full_eval_at_checkpoints": args.full_eval_at_checkpoints,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    results = []
    for domain in domains:
        try:
            result = run_domain(
                domain, args.max_steps, args.seed,
                args.checkpoint_interval, args.full_eval_at_checkpoints,
                output_dir,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Domain {domain} failed: {e}")
            results.append({"domain": domain, "error": str(e)})

    # Save combined results
    combined = output_dir / "all_results.json"
    with open(combined, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {combined}")

    print_summary(results)


if __name__ == "__main__":
    main()
