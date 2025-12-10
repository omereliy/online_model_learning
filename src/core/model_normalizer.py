"""
Model normalizer for fair comparison between OLAM and Information Gain algorithms.

This module provides utilities to normalize models from different algorithms
so they can be fairly compared. It addresses semantic differences in how
each algorithm represents learned knowledge.

Key normalization strategies:
1. Safe model preconditions: Ensure both use same "all possible" universe
2. Complete model preconditions: Document semantic differences
3. Contradiction handling: Configurable strategy for effect conflicts
"""

import logging
from dataclasses import dataclass
from typing import Set, Dict, Tuple, Optional, List
from pathlib import Path

from src.core.model_reconstructor import ReconstructedModel, ReconstructedAction
from src.core.model_validator import normalize_predicate_parameters

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of normalizing a model for comparison."""
    model: ReconstructedModel
    strategy_used: str
    modifications: Dict[str, List[str]]


class ModelNormalizer:
    """Normalizes models from different algorithms for fair comparison."""

    @staticmethod
    def normalize_safe_preconditions(
        model: ReconstructedModel,
        algorithm: str,
        all_possible_literals: Set[str],
        strategy: str = "infogain_style"
    ) -> ReconstructedModel:
        """
        Normalize safe model preconditions to ensure fair comparison.

        The key issue: For unexecuted actions, InfoGain has La (all possible)
        while OLAM has empty set. This normalizes both to the same representation.

        Args:
            model: The reconstructed model to normalize
            algorithm: "information_gain" or "olam"
            all_possible_literals: La - all possible parameter-bound literals
            strategy:
                - "infogain_style": Use all non-ruled-out (La for unexecuted)
                - "olam_style": Use only observed (certain+uncertain)
                - "intersection": Use only literals present in both algorithms

        Returns:
            Normalized ReconstructedModel
        """
        normalized_actions = {}

        for action_name, action in model.actions.items():
            if strategy == "infogain_style":
                # For unexecuted/unlearned actions, use full La
                if len(action.preconditions) == 0:
                    preconditions = all_possible_literals.copy()
                    logger.debug(f"Normalized {action_name} preconditions to La ({len(preconditions)} literals)")
                else:
                    preconditions = action.preconditions
            elif strategy == "olam_style":
                # Use only observed preconditions (may be empty)
                preconditions = action.preconditions
            else:
                # intersection or unknown - keep as is
                preconditions = action.preconditions

            normalized_actions[action_name] = ReconstructedAction(
                name=action_name,
                parameters=action.parameters,
                preconditions=preconditions,
                add_effects=action.add_effects,
                del_effects=action.del_effects
            )

        return ReconstructedModel(
            model_type=model.model_type,
            algorithm=model.algorithm,
            iteration=model.iteration,
            actions=normalized_actions
        )

    @staticmethod
    def normalize_contradiction_handling(
        add_effects: Set[str],
        del_effects: Set[str],
        strategy: str = "remove_both"
    ) -> Tuple[Set[str], Set[str]]:
        """
        Normalize contradiction handling between add and delete effects.

        Different algorithms handle contradictions (same fluent in both add and del)
        differently. This applies a consistent strategy.

        Args:
            add_effects: Set of add effect literals
            del_effects: Set of delete effect literals
            strategy:
                - "remove_both": Remove from both (InfoGain style, conservative)
                - "prefer_add": Keep in add, remove from del (OLAM style)
                - "prefer_del": Keep in del, remove from add

        Returns:
            Tuple of (normalized_add_effects, normalized_del_effects)
        """
        # Normalize all literals for comparison
        add_normalized = {normalize_predicate_parameters(p) for p in add_effects}
        del_normalized = {normalize_predicate_parameters(p) for p in del_effects}

        contradictions = add_normalized & del_normalized

        if not contradictions:
            return add_effects, del_effects

        logger.debug(f"Found {len(contradictions)} contradictions: {contradictions}")

        if strategy == "remove_both":
            # Conservative: remove contradictions from both
            return (
                {p for p in add_effects if normalize_predicate_parameters(p) not in contradictions},
                {p for p in del_effects if normalize_predicate_parameters(p) not in contradictions}
            )
        elif strategy == "prefer_add":
            # Keep in add, remove from del
            return (
                add_effects,
                {p for p in del_effects if normalize_predicate_parameters(p) not in contradictions}
            )
        elif strategy == "prefer_del":
            # Keep in del, remove from add
            return (
                {p for p in add_effects if normalize_predicate_parameters(p) not in contradictions},
                del_effects
            )
        else:
            logger.warning(f"Unknown contradiction strategy: {strategy}, using remove_both")
            return ModelNormalizer.normalize_contradiction_handling(
                add_effects, del_effects, "remove_both"
            )

    @staticmethod
    def normalize_model_for_comparison(
        model: ReconstructedModel,
        all_possible_literals: Optional[Set[str]] = None,
        precondition_strategy: str = "infogain_style",
        contradiction_strategy: str = "remove_both"
    ) -> ReconstructedModel:
        """
        Fully normalize a model for fair cross-algorithm comparison.

        Args:
            model: The model to normalize
            all_possible_literals: La - required for infogain_style precondition strategy
            precondition_strategy: How to handle preconditions
            contradiction_strategy: How to handle effect contradictions

        Returns:
            Fully normalized ReconstructedModel
        """
        # First normalize preconditions
        if all_possible_literals:
            model = ModelNormalizer.normalize_safe_preconditions(
                model, model.algorithm, all_possible_literals, precondition_strategy
            )

        # Then normalize effect contradictions
        normalized_actions = {}
        for action_name, action in model.actions.items():
            add_eff, del_eff = ModelNormalizer.normalize_contradiction_handling(
                action.add_effects, action.del_effects, contradiction_strategy
            )

            normalized_actions[action_name] = ReconstructedAction(
                name=action_name,
                parameters=action.parameters,
                preconditions=action.preconditions,
                add_effects=add_eff,
                del_effects=del_eff
            )

        return ReconstructedModel(
            model_type=model.model_type,
            algorithm=model.algorithm,
            iteration=model.iteration,
            actions=normalized_actions
        )

    @staticmethod
    def generate_all_possible_literals(
        domain_file: Path,
        action_name: str
    ) -> Set[str]:
        """
        Generate all type-compatible literals (La) for an action.

        This is needed to normalize OLAM's empty preconditions for unexecuted
        actions to match InfoGain's La representation.

        Args:
            domain_file: Path to PDDL domain file
            action_name: Name of the action

        Returns:
            Set of all possible parameter-bound literals for the action
        """
        from unified_planning.io import PDDLReader as UPReader
        from itertools import permutations

        try:
            reader = UPReader()
            problem = reader.parse_problem(str(domain_file))

            # Find the action
            up_action = None
            for action in problem.actions:
                if action.name.lower() == action_name.lower():
                    up_action = action
                    break

            if not up_action:
                logger.warning(f"Action '{action_name}' not found in domain")
                return set()

            possible_literals = set()
            action_params = list(up_action.parameters)
            num_params = len(action_params)

            # For each predicate/fluent in the domain
            for fluent in problem.fluents:
                fluent_arity = len(list(fluent.signature))

                if fluent_arity == 0:
                    # 0-arity predicate
                    possible_literals.add(f"{fluent.name}()")
                elif fluent_arity <= num_params:
                    # Generate all permutations of action parameters
                    for perm in permutations(range(num_params), fluent_arity):
                        param_str = ','.join([f"?{idx}" for idx in perm])
                        possible_literals.add(f"{fluent.name}({param_str})")

            logger.debug(f"Generated {len(possible_literals)} possible literals for {action_name}")
            return possible_literals

        except Exception as e:
            logger.error(f"Failed to generate possible literals: {e}")
            return set()

    @staticmethod
    def compare_models_side_by_side(
        infogain_model: ReconstructedModel,
        olam_model: ReconstructedModel,
        normalize: bool = True,
        all_possible_literals: Optional[Set[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare two models side-by-side and report differences.

        Args:
            infogain_model: Information Gain model
            olam_model: OLAM model
            normalize: Whether to normalize before comparison
            all_possible_literals: La for normalization

        Returns:
            Dictionary with per-action comparison results
        """
        if normalize and all_possible_literals:
            infogain_model = ModelNormalizer.normalize_model_for_comparison(
                infogain_model, all_possible_literals
            )
            olam_model = ModelNormalizer.normalize_model_for_comparison(
                olam_model, all_possible_literals
            )

        comparison = {}
        all_actions = set(infogain_model.actions.keys()) | set(olam_model.actions.keys())

        for action_name in all_actions:
            ig_action = infogain_model.actions.get(action_name)
            olam_action = olam_model.actions.get(action_name)

            if not ig_action or not olam_action:
                comparison[action_name] = {
                    "status": "MISSING",
                    "in_infogain": ig_action is not None,
                    "in_olam": olam_action is not None
                }
                continue

            # Normalize literals for comparison
            ig_precs = {normalize_predicate_parameters(p) for p in ig_action.preconditions}
            olam_precs = {normalize_predicate_parameters(p) for p in olam_action.preconditions}

            ig_add = {normalize_predicate_parameters(p) for p in ig_action.add_effects}
            olam_add = {normalize_predicate_parameters(p) for p in olam_action.add_effects}

            ig_del = {normalize_predicate_parameters(p) for p in ig_action.del_effects}
            olam_del = {normalize_predicate_parameters(p) for p in olam_action.del_effects}

            comparison[action_name] = {
                "preconditions": {
                    "infogain": sorted(ig_precs),
                    "olam": sorted(olam_precs),
                    "only_in_infogain": sorted(ig_precs - olam_precs),
                    "only_in_olam": sorted(olam_precs - ig_precs),
                    "common": sorted(ig_precs & olam_precs),
                    "equivalent": ig_precs == olam_precs
                },
                "add_effects": {
                    "infogain": sorted(ig_add),
                    "olam": sorted(olam_add),
                    "only_in_infogain": sorted(ig_add - olam_add),
                    "only_in_olam": sorted(olam_add - ig_add),
                    "common": sorted(ig_add & olam_add),
                    "equivalent": ig_add == olam_add
                },
                "del_effects": {
                    "infogain": sorted(ig_del),
                    "olam": sorted(olam_del),
                    "only_in_infogain": sorted(ig_del - olam_del),
                    "only_in_olam": sorted(olam_del - ig_del),
                    "common": sorted(ig_del & olam_del),
                    "equivalent": ig_del == olam_del
                },
                "status": "EQUIVALENT" if (ig_precs == olam_precs and ig_add == olam_add and ig_del == olam_del) else "DIVERGENT"
            }

        return comparison


def print_comparison_table(comparison: Dict[str, Dict]) -> str:
    """
    Format comparison results as a readable table.

    Args:
        comparison: Result from compare_models_side_by_side

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COMPARISON RESULTS")
    lines.append("=" * 80)

    for action_name, data in sorted(comparison.items()):
        lines.append(f"\nAction: {action_name}")
        lines.append("-" * 40)

        if data.get("status") == "MISSING":
            lines.append(f"  STATUS: MISSING")
            lines.append(f"  In InfoGain: {data.get('in_infogain')}")
            lines.append(f"  In OLAM: {data.get('in_olam')}")
            continue

        status = data.get("status", "UNKNOWN")
        lines.append(f"  STATUS: {status}")

        for component in ["preconditions", "add_effects", "del_effects"]:
            comp_data = data.get(component, {})
            equiv = "YES" if comp_data.get("equivalent") else "NO"
            lines.append(f"\n  {component.upper()}:")
            lines.append(f"    Equivalent: {equiv}")
            lines.append(f"    InfoGain: {comp_data.get('infogain', [])}")
            lines.append(f"    OLAM: {comp_data.get('olam', [])}")

            if not comp_data.get("equivalent"):
                if comp_data.get("only_in_infogain"):
                    lines.append(f"    Only in InfoGain: {comp_data.get('only_in_infogain')}")
                if comp_data.get("only_in_olam"):
                    lines.append(f"    Only in OLAM: {comp_data.get('only_in_olam')}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
