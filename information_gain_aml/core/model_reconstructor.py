"""
Model reconstructor for safe and complete models from exported snapshots.

This module provides functionality to reconstruct action models from exported
JSON snapshots, creating both safe (conservative) and complete (optimistic)
versions for precision/recall analysis.
"""

from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReconstructedAction:
    """Reconstructed action model."""
    name: str
    parameters: List[str]
    preconditions: Set[str]
    add_effects: Set[str]
    del_effects: Set[str]


@dataclass
class ReconstructedModel:
    """Complete reconstructed model."""
    model_type: str  # "safe" or "complete"
    algorithm: str
    iteration: int
    actions: Dict[str, ReconstructedAction]


class ModelReconstructor:
    """Reconstructs safe and complete models from exported snapshots."""

    @staticmethod
    def reconstruct_information_gain_safe(snapshot: Dict, domain_file: Optional[str] = None) -> ReconstructedModel:
        """
        Reconstruct safe model from Information Gain snapshot.

        Safe model semantics (guarantees recall=1.0 for preconditions):
        - Preconditions: La (all type-compatible predicates) - safe has no false negatives
        - Effects: Only certain (confirmed) - safe has no false positives for effects

        For unobserved actions:
        - Preconditions: recall=1, precision=low (La)
        - Effects: recall=0, precision=1 (empty)

        Args:
            snapshot: Dictionary containing exported model snapshot
            domain_file: Path to PDDL domain file (required for La generation)

        Returns:
            ReconstructedModel with safe configuration
        """
        # Generate La (all possible predicates) for safe model preconditions
        all_possible = {}
        if domain_file:
            all_possible = ModelReconstructor._generate_all_possible_predicates(
                domain_file, snapshot["actions"]
            )
            if all_possible:
                logger.debug(f"Generated La for {len(all_possible)} actions for safe model")
        else:
            logger.warning("No domain_file provided for safe model, recall may be < 1.0")

        actions = {}
        for action_name, action_data in snapshot["actions"].items():
            # Safe model preconditions: Use La (all type-compatible predicates)
            # This guarantees recall=1.0 (no false negatives)
            if action_name in all_possible:
                preconditions = all_possible[action_name].copy()
            else:
                # Fallback if La not available: use possible_preconditions
                # Filter out negated preconditions (¬pred) - PDDL preconditions are positive
                preconditions = {p for p in action_data["possible_preconditions"] if not p.startswith('¬')}

            # Safe: only confirmed effects (filter negated literals)
            # This guarantees precision=1.0 for effects (no false positives)
            add_effects = {e for e in action_data["confirmed_add_effects"] if not e.startswith('¬')}
            del_effects = {e for e in action_data["confirmed_del_effects"] if not e.startswith('¬')}

            actions[action_name] = ReconstructedAction(
                name=action_name,
                parameters=action_data["parameters"],
                preconditions=preconditions,
                add_effects=add_effects,
                del_effects=del_effects
            )

        return ReconstructedModel(
            model_type="safe",
            algorithm=snapshot["algorithm"],
            iteration=snapshot["iteration"],
            actions=actions
        )

    @staticmethod
    def reconstruct_information_gain_complete(snapshot: Dict) -> ReconstructedModel:
        """
        Reconstruct complete model from Information Gain snapshot.

        Complete model:
        - Preconditions: Only certain (less restrictive)
        - Effects: Certain + possible (no contradictions)

        Args:
            snapshot: Dictionary containing exported model snapshot

        Returns:
            ReconstructedModel with complete configuration
        """
        actions = {}
        for action_name, action_data in snapshot["actions"].items():
            # Complete: only certain preconditions
            # Filter out negated preconditions (¬pred) - PDDL preconditions are positive
            preconditions = {p for p in action_data["certain_preconditions"] if not p.startswith('¬')}

            # Complete: confirmed + possible effects
            # Filter out negated literals (¬pred) - PDDL effects don't have negations
            # Add effects make predicates TRUE, delete effects make them FALSE
            def filter_negated(literals):
                return {lit for lit in literals if not lit.startswith('¬')}

            # DON'T remove contradictions - for complete model we want recall=1
            # Contradictions just mean we don't know yet if a literal is add or delete,
            # which is fine for complete model (we're being optimistic)
            add_effects = filter_negated(
                set(action_data["confirmed_add_effects"]) |
                set(action_data["possible_add_effects"])
            )
            del_effects = filter_negated(
                set(action_data["confirmed_del_effects"]) |
                set(action_data["possible_del_effects"])
            )

            actions[action_name] = ReconstructedAction(
                name=action_name,
                parameters=action_data["parameters"],
                preconditions=preconditions,
                add_effects=add_effects,
                del_effects=del_effects
            )

        return ReconstructedModel(
            model_type="complete",
            algorithm=snapshot["algorithm"],
            iteration=snapshot["iteration"],
            actions=actions
        )

    @staticmethod
    def _generate_all_possible_predicates(domain_file: str, actions_data: Dict) -> Dict[str, Set[str]]:
        """
        Generate all type-compatible predicates for each action.

        For the safe model, we need ALL possible preconditions (not just observed ones).
        This generates all predicates that are type-compatible with each action's parameters.

        Args:
            domain_file: Path to PDDL domain file
            actions_data: Dictionary of action data from snapshot

        Returns:
            Dictionary mapping action names to sets of all possible preconditions (normalized)
        """
        from unified_planning.io import PDDLReader
        from information_gain_aml.core.model_validator import normalize_predicate_parameters

        try:
            # Parse domain to get predicates and actions
            reader = PDDLReader()
            problem = reader.parse_problem(domain_file)

            all_possible = {}

            # For each action in the snapshot
            for action_name in actions_data.keys():
                # Find matching action in domain
                up_action = None
                for action in problem.actions:
                    if action.name.lower() == action_name.lower():
                        up_action = action
                        break

                if not up_action:
                    logger.warning(f"Action '{action_name}' not found in domain {domain_file}")
                    continue

                # Generate all type-compatible predicates for this action
                possible_precs = set()

                # Get action parameters
                action_params = list(up_action.parameters)

                # For each predicate in the domain
                for fluent in problem.fluents:
                    fluent_params = list(fluent.signature)

                    # Skip if arity doesn't match any subset of action parameters
                    if len(fluent_params) == 0:
                        # 0-arity predicate - always possible
                        possible_precs.add(f"{fluent.name}()")
                    else:
                        # Generate all type-compatible parameter bindings
                        from itertools import permutations

                        # Try all permutations of action parameters
                        # For safe model, we overapproximate - include all possible groundings
                        for perm in permutations(range(len(action_params)), len(fluent_params)):
                            # Generate predicate with positional parameters
                            param_str = ','.join([f"?{idx}" for idx in perm])
                            possible_precs.add(f"{fluent.name}({param_str})")

                all_possible[action_name] = possible_precs
                logger.debug(f"Generated {len(possible_precs)} possible preconditions for {action_name}")

            return all_possible

        except Exception as e:
            logger.error(f"Failed to generate all possible predicates: {e}")
            return {}

    @staticmethod
    def _remove_contradictions(add_effects: Set[str],
                              del_effects: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Remove contradictions where same fluent appears in both add and delete.

        Strategy: Remove from both sets (conservative approach).

        Args:
            add_effects: Set of add effects
            del_effects: Set of delete effects

        Returns:
            Tuple of (filtered_add_effects, filtered_del_effects)
        """
        contradictions = add_effects & del_effects
        if contradictions:
            logger.debug(f"Removing contradictions: {contradictions}")
            add_effects = add_effects - contradictions
            del_effects = del_effects - contradictions
        return add_effects, del_effects

    @staticmethod
    def load_and_reconstruct(snapshot_path: Path, domain_file: Optional[str] = None) -> List[ReconstructedModel]:
        """
        Load snapshot file and reconstruct both safe and complete models.

        Args:
            snapshot_path: Path to the model snapshot JSON file
            domain_file: Path to PDDL domain file (optional, for safe model generation)

        Returns:
            List of 2 reconstructed models: [safe_model, complete_model]

        Raises:
            ValueError: If algorithm is unknown
            FileNotFoundError: If snapshot file doesn't exist
        """
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)

        algorithm = snapshot["algorithm"]

        if algorithm == "information_gain":
            safe = ModelReconstructor.reconstruct_information_gain_safe(snapshot)
            complete = ModelReconstructor.reconstruct_information_gain_complete(snapshot)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.debug(f"Reconstructed {algorithm} models from iteration {snapshot['iteration']}")
        return [safe, complete]

    @staticmethod
    def reconstruct_from_directory(models_dir: Path,
                                  iteration: int) -> List[ReconstructedModel]:
        """
        Reconstruct models from a specific checkpoint in a directory.

        Args:
            models_dir: Directory containing model snapshots
            iteration: Iteration number to reconstruct

        Returns:
            List of 2 reconstructed models: [safe_model, complete_model]

        Raises:
            FileNotFoundError: If snapshot for iteration doesn't exist
        """
        snapshot_path = models_dir / f"model_iter_{iteration:03d}.json"
        return ModelReconstructor.load_and_reconstruct(snapshot_path)

