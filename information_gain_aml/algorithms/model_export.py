"""
Model export functions for Information Gain learner.

BRIDGE module — will be replaced by AMLGym's export interface.

Provides PDDL export, JSON model snapshots, and learning metrics
as standalone functions that operate on the learner's state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional, Set

if TYPE_CHECKING:
    from information_gain_aml.algorithms.information_gain import InformationGainLearner

logger = logging.getLogger(__name__)


# ========== Helpers ==========

def literal_to_pddl(literal: str) -> str:
    """Convert a parameter-bound literal to PDDL syntax.

    Examples:
        on(?x,?y)  -> (on ?x ?y)
        ¬clear(?x) -> (not (clear ?x))
        handempty   -> (handempty)
    """
    negated = literal.startswith('¬')
    if negated:
        literal = literal[1:]

    if '(' in literal:
        pred_name = literal[:literal.index('(')]
        params_str = literal[literal.index('(') + 1:-1]
        params = params_str.replace(',', ' ')
        inner = f"({pred_name} {params})"
    else:
        inner = f"({literal})"

    if negated:
        return f"(not {inner})"
    return inner


def extract_predicate_name(literal: str) -> Optional[str]:
    """Extract predicate name from literal string.

    Args:
        literal: e.g. 'on(?x,?y)' or '¬clear(?x)'

    Returns:
        Predicate name or None
    """
    if literal.startswith('¬'):
        literal = literal[1:]
    if '(' in literal:
        return literal[:literal.index('(')]
    return literal


# ========== Model Export ==========

def get_learned_model(learner: InformationGainLearner) -> Dict[str, Any]:
    """
    Export the current learned model as a dictionary.

    Args:
        learner: InformationGainLearner instance

    Returns:
        Dictionary containing the learned model
    """
    logger.debug("Exporting learned model")

    predicates_set: set[str] = set()
    actions_dict: dict[str, Any] = {}

    for action_name in learner.pre.keys():
        actions_dict[action_name] = {
            'name': action_name,
            'preconditions': {
                'possible': sorted(list(learner.pre[action_name])),
                'constraints': [sorted(list(c)) for c in learner.pre_constraints[action_name]]
            },
            'effects': {
                'add': sorted(list(learner.eff_add[action_name])),
                'delete': sorted(list(learner.eff_del[action_name])),
                'maybe_add': sorted(list(learner.eff_maybe_add[action_name])),
                'maybe_delete': sorted(list(learner.eff_maybe_del[action_name]))
            },
            'observations': len(learner.observation_history[action_name])
        }

        logger.debug(
            f"Exported action '{action_name}': {len(learner.pre[action_name])} preconditions, "
            f"{len(learner.eff_add[action_name])} add effects, "
            f"{len(learner.eff_del[action_name])} delete effects, "
            f"{len(learner.observation_history[action_name])} observations")

        for literal in learner.pre[action_name]:
            pred_name = extract_predicate_name(literal)
            if pred_name:
                predicates_set.add(pred_name)

    model = {
        'actions': actions_dict,
        'predicates': sorted(list(predicates_set)),
        'statistics': learner.get_statistics()
    }

    logger.info(f"Model export complete: {len(model['actions'])} actions, "
                f"{len(model['predicates'])} predicates")
    return model


# ========== PDDL Export ==========

def to_pddl_string(learner: InformationGainLearner, mode: str = "safe") -> str:
    """Export learned model as PDDL domain string.

    Args:
        learner: InformationGainLearner instance
        mode: "safe" or "complete"
            - safe: all possible preconditions (pre) + confirmed effects only.
            - complete: only certain preconditions (singletons) + all possible effects.

    Returns:
        PDDL domain string
    """
    if mode not in ("safe", "complete"):
        raise ValueError(f"mode must be 'safe' or 'complete', got '{mode}'")

    lines = []
    lines.append(f"(define (domain {learner.domain.name})")

    # Check if any negative preconditions exist
    has_negative_precs = False
    for action_name in learner.pre:
        precs = learner.pre[action_name] if mode == "safe" else learner._get_certain_preconditions(action_name)
        if any(lit.startswith('¬') for lit in precs):
            has_negative_precs = True
            break

    requirements = ":strips :typing"
    if has_negative_precs:
        requirements += " :negative-preconditions"
    lines.append(f"  (:requirements {requirements})")

    # Types
    type_strs = []
    for type_name, type_info in learner.domain.types.items():
        if type_name == "object":
            continue
        parent = type_info.parent or "object"
        type_strs.append(f"{type_name} - {parent}")
    if type_strs:
        lines.append(f"  (:types {' '.join(type_strs)})")

    # Predicates
    pred_strs = []
    for pred_name, pred_sig in learner.domain.predicates.items():
        if pred_sig.arity == 0:
            pred_strs.append(f"({pred_name})")
        else:
            params = " ".join(f"?p{i} - {p.type}" for i, p in enumerate(pred_sig.parameters))
            pred_strs.append(f"({pred_name} {params})")
    if pred_strs:
        lines.append("  (:predicates")
        for ps in pred_strs:
            lines.append(f"    {ps}")
        lines.append("  )")

    # Actions
    for action_name, action in learner.domain.lifted_actions.items():
        param_names = learner.domain._generate_parameter_names(action.arity)
        param_strs = " ".join(
            f"{param_names[i]} - {action.parameters[i].type}"
            for i in range(action.arity)
        )

        if mode == "safe":
            precs = learner.pre.get(action_name, set())
            add_effs = learner.eff_add.get(action_name, set())
            del_effs = learner.eff_del.get(action_name, set())
        else:  # complete
            precs = learner._get_certain_preconditions(action_name)
            add_effs = learner.eff_add.get(action_name, set()) | learner.eff_maybe_add.get(action_name, set())
            del_effs = learner.eff_del.get(action_name, set()) | learner.eff_maybe_del.get(action_name, set())

        # Filter out negative literals from effects
        add_effs = {e for e in add_effs if not e.startswith('¬')}
        del_effs = {e for e in del_effs if not e.startswith('¬')}

        lines.append(f"  (:action {action_name}")
        lines.append(f"    :parameters ({param_strs})")

        # Precondition
        pddl_precs = sorted(literal_to_pddl(lit) for lit in precs)
        if not pddl_precs:
            lines.append("    :precondition ()")
        elif len(pddl_precs) == 1:
            lines.append(f"    :precondition {pddl_precs[0]}")
        else:
            lines.append(f"    :precondition (and")
            for p in pddl_precs:
                lines.append(f"      {p}")
            lines.append("    )")

        # Effect
        pddl_adds = sorted(literal_to_pddl(lit) for lit in add_effs)
        pddl_dels = sorted(f"(not {literal_to_pddl(lit)})" for lit in del_effs)
        pddl_effects = pddl_adds + pddl_dels
        if not pddl_effects:
            lines.append("    :effect ()")
        elif len(pddl_effects) == 1:
            lines.append(f"    :effect {pddl_effects[0]}")
        else:
            lines.append(f"    :effect (and")
            for e in pddl_effects:
                lines.append(f"      {e}")
            lines.append("    )")

        lines.append("  )")

    lines.append(")")
    return "\n".join(lines)


# ========== Metrics ==========

def get_action_model_metrics(learner: InformationGainLearner) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed learning metrics for each action showing what has been learned.

    For each action, computes certain/excluded/uncertain counts for
    preconditions, add effects, and delete effects.

    Args:
        learner: InformationGainLearner instance

    Returns:
        Dict[action_name -> metrics]
    """
    action_metrics: dict[str, dict[str, Any]] = {}

    for action_name in learner.pre.keys():
        La = learner._get_parameter_bound_literals(action_name)
        la_size = len(La)

        # Preconditions
        certain_pre: set[str] = set()
        if learner.pre_constraints[action_name]:
            constraint_sets = [set(c) for c in learner.pre_constraints[action_name]]
            certain_pre = set.intersection(*constraint_sets) if constraint_sets else set()
        excluded_pre = La - learner.pre[action_name]
        uncertain_pre = learner.pre[action_name] - certain_pre

        # Add effects
        certain_eff_add = learner.eff_add[action_name]
        excluded_eff_add = La - (learner.eff_add[action_name] | learner.eff_maybe_add[action_name])
        uncertain_eff_add = learner.eff_maybe_add[action_name]

        # Delete effects
        certain_eff_del = learner.eff_del[action_name]
        excluded_eff_del = La - (learner.eff_del[action_name] | learner.eff_maybe_del[action_name])
        uncertain_eff_del = learner.eff_maybe_del[action_name]

        def _pct(n: int) -> float:
            return (n / la_size * 100) if la_size > 0 else 0

        action_metrics[action_name] = {
            'La_size': la_size,
            'observations': len(learner.observation_history[action_name]),
            'preconditions': {
                'certain_count': len(certain_pre),
                'excluded_count': len(excluded_pre),
                'uncertain_count': len(uncertain_pre),
                'certain_percent': _pct(len(certain_pre)),
                'excluded_percent': _pct(len(excluded_pre)),
                'uncertain_percent': _pct(len(uncertain_pre)),
            },
            'add_effects': {
                'certain_count': len(certain_eff_add),
                'excluded_count': len(excluded_eff_add),
                'uncertain_count': len(uncertain_eff_add),
                'certain_percent': _pct(len(certain_eff_add)),
                'excluded_percent': _pct(len(excluded_eff_add)),
                'uncertain_percent': _pct(len(uncertain_eff_add)),
            },
            'delete_effects': {
                'certain_count': len(certain_eff_del),
                'excluded_count': len(excluded_eff_del),
                'uncertain_count': len(uncertain_eff_del),
                'certain_percent': _pct(len(certain_eff_del)),
                'excluded_percent': _pct(len(excluded_eff_del)),
                'uncertain_percent': _pct(len(uncertain_eff_del)),
            },
            'learning_progress': {
                'total_certain': len(certain_pre) + len(certain_eff_add) + len(certain_eff_del),
                'total_excluded': len(excluded_pre) + len(excluded_eff_add) + len(excluded_eff_del),
                'total_uncertain': len(uncertain_pre) + len(uncertain_eff_add) + len(uncertain_eff_del),
                'explored_percent': (
                    (len(certain_pre) + len(excluded_pre) +
                     len(certain_eff_add) + len(excluded_eff_add) +
                     len(certain_eff_del) + len(excluded_eff_del)) / (3 * la_size) * 100
                ) if la_size > 0 else 0,
            }
        }

    return action_metrics


# ========== Snapshot Export ==========

def export_model_snapshot(learner: InformationGainLearner, iteration: int, output_dir: Path) -> None:
    """
    Export model snapshot at checkpoint.

    Args:
        learner: InformationGainLearner instance
        iteration: Current iteration number
        output_dir: Directory to export the model snapshot to
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    domain_name = Path(learner.domain_file).stem
    problem_name = Path(learner.problem_file).stem

    snapshot: Dict[str, Any] = {
        "iteration": iteration,
        "algorithm": "information_gain",
        "actions": {},
        "metadata": {
            "domain": domain_name,
            "problem": problem_name,
            "export_timestamp": datetime.now().isoformat()
        }
    }

    for action_name in learner.pre.keys():
        possible_precs = set(learner.pre.get(action_name, set()))
        certain_precs = learner._get_certain_preconditions(action_name)
        uncertain_precs = possible_precs - certain_precs

        action = learner.domain.lifted_actions.get(action_name)
        parameters = [p.name for p in action.parameters] if action else []

        snapshot["actions"][action_name] = {
            "parameters": parameters,
            "possible_preconditions": sorted(list(possible_precs)),
            "certain_preconditions": sorted(list(certain_precs)),
            "uncertain_preconditions": sorted(list(uncertain_precs)),
            "confirmed_add_effects": sorted(list(learner.eff_add.get(action_name, set()))),
            "confirmed_del_effects": sorted(list(learner.eff_del.get(action_name, set()))),
            "possible_add_effects": sorted(list(learner.eff_maybe_add.get(action_name, set()))),
            "possible_del_effects": sorted(list(learner.eff_maybe_del.get(action_name, set()))),
            "constraint_sets": [sorted(list(cs)) for cs in learner.pre_constraints.get(action_name, set())]
        }

    snapshot["statistics"] = {
        "iterations": learner.iteration_count,
        "observations": learner.observation_count,
        "converged": learner._converged,
        "max_information_gain": learner._last_max_gain,
        "action_model_metrics": get_action_model_metrics(learner)
    }

    output_path = models_dir / f"model_iter_{iteration:03d}.json"
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    logger.debug(f"Exported model snapshot at iteration {iteration} to {output_path}")
