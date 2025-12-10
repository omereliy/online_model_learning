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
    def reconstruct_olam_safe(snapshot: Dict, domain_file: Optional[str] = None) -> ReconstructedModel:
        """
        Reconstruct safe model from OLAM snapshot.

        Safe model semantics:
        - Preconditions: certain + uncertain (what OLAM considers possible)
        - Effects: ONLY certain effects (confirmed)
        - Fallback to La (all possible) when certain+uncertain is empty (unobserved action)

        The safe model uses OLAM's tracked preconditions. When an action hasn't been
        observed yet (both certain and uncertain empty), we fall back to La (all
        type-compatible predicates) to guarantee recall=1.0.

        Args:
            snapshot: Dictionary containing exported model snapshot
            domain_file: Path to PDDL domain file (for La fallback on unobserved actions)

        Returns:
            ReconstructedModel with safe configuration
        """
        actions = {}

        # Generate La (all possible predicates) as fallback for unobserved actions
        all_possible = {}
        if domain_file:
            all_possible = ModelReconstructor._generate_all_possible_predicates(
                domain_file, snapshot["actions"]
            )
            if all_possible:
                logger.debug(f"Generated La for {len(all_possible)} actions (fallback for unobserved)")

        for action_name, action_data in snapshot["actions"].items():
            # Primary: certain + uncertain preconditions (OLAM's tracked hypothesis)
            certain_precs = action_data.get("certain_preconditions", [])
            uncertain_precs = action_data.get("uncertain_preconditions", [])

            preconditions = set()
            for lit in (certain_precs + uncertain_precs):
                normalized = ModelReconstructor._normalize_olam_literal(lit)
                if normalized:
                    preconditions.add(normalized)

            # Fallback: If action has no observations (empty preconditions), use La
            if not preconditions and action_name in all_possible:
                preconditions = all_possible[action_name].copy()
                logger.debug(f"Safe model {action_name}: Using La fallback ({len(preconditions)} predicates)")

            # Safe: ONLY certain effects - normalized
            # Filter out negated literals (¬pred) - same as Information Gain
            add_effects = set()
            for lit in action_data.get("certain_add_effects", []):
                normalized = ModelReconstructor._normalize_olam_literal(lit)
                if normalized and not normalized.startswith('¬'):
                    add_effects.add(normalized)

            # Delete effects: strip negation from literals (OLAM stores delete effects
            # with (not ...) wrapper which normalizes to ¬prefix, but PDDL effects
            # are positive predicates - the delete list indicates what becomes FALSE)
            # Filter out non-negated literals that shouldn't be in del_effects
            del_effects = set()
            for lit in action_data.get("certain_del_effects", []):
                normalized = ModelReconstructor._normalize_olam_literal(lit)
                if normalized:
                    # Strip ¬ prefix if present - delete effects are positive predicates
                    if normalized.startswith('¬'):
                        normalized = normalized[1:]
                        del_effects.add(normalized)
                    # If not negated, it's already a positive predicate in del list - keep it
                    else:
                        del_effects.add(normalized)

            actions[action_name] = ReconstructedAction(
                name=action_name,
                parameters=action_data.get("parameters", []),
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
    def reconstruct_olam_complete(snapshot: Dict) -> ReconstructedModel:
        """
        Reconstruct complete model from OLAM snapshot.

        Complete model:
        - Preconditions: Only certain (less restrictive)
        - Effects: Certain + uncertain (optimistic)

        Args:
            snapshot: Dictionary containing exported model snapshot

        Returns:
            ReconstructedModel with complete configuration
        """
        actions = {}
        for action_name, action_data in snapshot["actions"].items():
            # Complete: only certain preconditions - normalized
            preconditions = set()
            for lit in action_data.get("certain_preconditions", []):
                normalized = ModelReconstructor._normalize_olam_literal(lit)
                if normalized:
                    preconditions.add(normalized)

            # Complete: certain + uncertain effects - normalized
            # DON'T remove contradictions - for complete model we want recall=1
            # Contradictions just mean we don't know yet if a literal is add or delete
            all_add_literals = (action_data.get("certain_add_effects", []) +
                               action_data.get("uncertain_add_effects", []))
            all_del_literals = (action_data.get("certain_del_effects", []) +
                               action_data.get("uncertain_del_effects", []))

            # Add effects: include all (certain + uncertain), normalized
            # Filter out negated literals (¬pred) - same as Information Gain
            add_effects = set()
            for lit in all_add_literals:
                normalized = ModelReconstructor._normalize_olam_literal(lit)
                if normalized and not normalized.startswith('¬'):
                    add_effects.add(normalized)

            # Delete effects: include all (certain + uncertain), normalized
            # Strip ¬ prefix - PDDL delete effects are positive predicates
            del_effects = set()
            for lit in all_del_literals:
                normalized = ModelReconstructor._normalize_olam_literal(lit)
                if normalized:
                    # Strip ¬ prefix if present - delete effects are positive predicates
                    if normalized.startswith('¬'):
                        normalized = normalized[1:]
                    del_effects.add(normalized)

            actions[action_name] = ReconstructedAction(
                name=action_name,
                parameters=action_data.get("parameters", []),
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
        from src.core.model_validator import normalize_predicate_parameters

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
                        # Type checking would be done by OLAM during exploration (and marked as useless if wrong)
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
    def _normalize_olam_literal(literal: str) -> str:
        """
        Normalize OLAM PDDL-style literal to match ground truth format.

        Examples:
            "(clear ?x)" -> "clear(?x)"
            "(on ?x ?y)" -> "on(?x,?y)"
            "(not (clear ?x))" -> "¬clear(?x)"
            "(handempty)" -> "handempty"
        """
        literal = literal.strip()

        # Handle negation
        if literal.startswith('(not '):
            inner = literal[5:-1].strip()  # Remove "(not " and ")"
            return '¬' + ModelReconstructor._normalize_olam_literal(inner)

        # Remove outer parentheses
        if literal.startswith('(') and literal.endswith(')'):
            literal = literal[1:-1].strip()

        # Split predicate and arguments
        parts = literal.split()
        if not parts:
            return ""

        predicate = parts[0]
        args = parts[1:]

        if args:
            # Join arguments with commas (no spaces)
            return f"{predicate}({','.join(args)})"
        else:
            return predicate

    @staticmethod
    def load_and_reconstruct(snapshot_path: Path, domain_file: Optional[str] = None) -> List[ReconstructedModel]:
        """
        Load snapshot file and reconstruct both safe and complete models.

        Args:
            snapshot_path: Path to the model snapshot JSON file
            domain_file: Path to PDDL domain file (optional, for OLAM safe model generation)

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
        elif algorithm == "olam":
            safe = ModelReconstructor.reconstruct_olam_safe(snapshot, domain_file=domain_file)
            complete = ModelReconstructor.reconstruct_olam_complete(snapshot)
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

    @staticmethod
    def reconstruct_from_olam_exports(exports_dir: Path, iteration: int) -> List[ReconstructedModel]:
        """
        Reconstruct models from OLAM's native JSON exports.

        OLAM exports model components as separate JSON files:
        - operator_certain_predicates.json
        - operator_uncertain_predicates.json
        - operator_negative_preconditions.json
        - certain_positive_effects.json
        - certain_negative_effects.json

        Args:
            exports_dir: Directory containing OLAM's JSON exports
            iteration: Iteration number for metadata

        Returns:
            List of 2 reconstructed models: [safe_model, complete_model]
        """
        # Load OLAM's JSON exports
        certain_precs = ModelReconstructor._load_json_or_empty(
            exports_dir / "operator_certain_predicates.json"
        )
        uncertain_precs = ModelReconstructor._load_json_or_empty(
            exports_dir / "operator_uncertain_predicates.json"
        )
        neg_precs = ModelReconstructor._load_json_or_empty(
            exports_dir / "operator_negative_preconditions.json"
        )
        add_effects = ModelReconstructor._load_json_or_empty(
            exports_dir / "certain_positive_effects.json"
        )
        del_effects = ModelReconstructor._load_json_or_empty(
            exports_dir / "certain_negative_effects.json"
        )

        # Convert to our snapshot format
        snapshot = {
            "algorithm": "olam",
            "iteration": iteration,
            "actions": {}
        }

        # Collect all operators
        all_operators = set()
        all_operators.update(certain_precs.keys())
        all_operators.update(uncertain_precs.keys())
        all_operators.update(add_effects.keys())
        all_operators.update(del_effects.keys())

        for operator in all_operators:
            snapshot["actions"][operator] = {
                "parameters": [],  # Would need domain file to extract
                "certain_preconditions": certain_precs.get(operator, []),
                "uncertain_preconditions": uncertain_precs.get(operator, []),
                "negative_preconditions": neg_precs.get(operator, []),
                "add_effects": add_effects.get(operator, []),
                "del_effects": del_effects.get(operator, [])
            }

        # Reconstruct using existing OLAM methods
        safe = ModelReconstructor.reconstruct_olam_safe(snapshot)
        complete = ModelReconstructor.reconstruct_olam_complete(snapshot)

        logger.info(f"Reconstructed OLAM models from native exports at iteration {iteration}")
        return [safe, complete]

    @staticmethod
    def _load_json_or_empty(path: Path) -> Dict:
        """
        Load JSON file or return empty dict if not found.

        Args:
            path: Path to JSON file

        Returns:
            Loaded dictionary or empty dict
        """
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {path}: {e}")
        return {}