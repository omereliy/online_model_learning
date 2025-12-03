"""Model validation infrastructure for comparing learned models against ground truth PDDL.

This module provides functionality to compute precision, recall, and F1 scores
for learned action models compared to ground truth from PDDL domain files.
"""

import logging
import re
from dataclasses import dataclass
from typing import Set, Dict, Optional, Any, Tuple

from src.core.pddl_io import PDDLReader
from src.core.expression_converter import ExpressionConverter

logger = logging.getLogger(__name__)


def normalize_predicate_parameters(predicate: str) -> str:
    """
    Normalize predicate parameters to positional format for comparison.

    This enables comparison between predicates with different parameter naming schemes
    and different formats (OLAM vs Information Gain). Produces canonical format that
    matches Information Gain's ExpressionConverter.to_parameter_bound_string().

    OLAM format: "(at ?param_1 ?param_2)" - outer parens, spaces, no commas
    Info Gain format: "at(?x,?y)" - no outer parens, commas between params
    Both normalize to: "at(?0,?1)"

    Args:
        predicate: Predicate string in any format

    Returns:
        Normalized predicate: "predicate_name(?0,?1,...)" or "predicate_name()" for 0-arity

    Examples:
        >>> normalize_predicate_parameters("(at ?param_1 ?param_2)")
        'at(?0,?1)'
        >>> normalize_predicate_parameters("at(?x,?y)")
        'at(?0,?1)'
        >>> normalize_predicate_parameters("clear(?x)")
        'clear(?0)'
        >>> normalize_predicate_parameters("(clear ?param_1)")
        'clear(?0)'
        >>> normalize_predicate_parameters("handempty()")
        'handempty()'
    """
    # Strip leading/trailing whitespace
    predicate = predicate.strip()

    # Strip outer PDDL-style parentheses if present: "(at ?x ?y)" → "at ?x ?y"
    if predicate.startswith('(') and not predicate.startswith('(?'):
        # Find matching closing paren
        paren_count = 0
        for i, c in enumerate(predicate):
            if c == '(':
                paren_count += 1
            elif c == ')':
                paren_count -= 1
                if paren_count == 0 and i == len(predicate) - 1:
                    # Outer parens wrap the whole thing
                    predicate = predicate[1:-1].strip()
                    break

    # Extract all parameters in order of appearance
    params = re.findall(r'\?[\w_]+', predicate)

    # Extract predicate name (before '(' or before first '?')
    if '(' in predicate:
        pred_name = predicate[:predicate.index('(')]
    elif params:
        # Space-separated format: "at ?x ?y"
        pred_name = predicate[:predicate.index(params[0])].strip()
    else:
        # No parameters - just the predicate name
        pred_name = predicate

    # Build normalized format matching Information Gain
    if not params:
        # 0-arity predicate
        return f"{pred_name}()"
    else:
        # Create positional parameter mapping
        positional_params = [f"?{i}" for i in range(len(params))]
        return f"{pred_name}({','.join(positional_params)})"


@dataclass
class ModelComparisonResult:
    """Stores results from comparing a learned action model to ground truth."""

    action_name: str

    # Precondition metrics
    precondition_precision: float
    precondition_recall: float
    precondition_f1: float
    precondition_false_positives: Set[str]
    precondition_false_negatives: Set[str]

    # Add effect metrics
    add_effect_precision: float
    add_effect_recall: float
    add_effect_f1: float
    add_effect_false_positives: Set[str]
    add_effect_false_negatives: Set[str]

    # Delete effect metrics
    delete_effect_precision: float
    delete_effect_recall: float
    delete_effect_f1: float
    delete_effect_false_positives: Set[str]
    delete_effect_false_negatives: Set[str]

    # Overall accuracy
    overall_f1: float


class ModelValidator:
    """Validates learned models against ground truth from PDDL specifications."""

    def __init__(self, domain_file: Optional[str] = None, problem_file: Optional[str] = None):
        """Initialize ModelValidator.

        Args:
            domain_file: Path to PDDL domain file (optional)
            problem_file: Path to PDDL problem file (optional)
        """
        self.ground_truth_models: Dict[str, Dict[str, Set[str]]] = {}  # TODO: inspect dataclass instead of dict

        if domain_file and problem_file:
            self._parse_ground_truth(domain_file, problem_file)

    def _parse_ground_truth(self, domain_file: str, problem_file: str):
        """Parse ground truth models from PDDL files.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
        """
        # Parse PDDL files with new architecture
        reader = PDDLReader()
        domain, _ = reader.parse_domain_and_problem(domain_file, problem_file)

        # Get UP problem for accessing actions (for compatibility)
        problem = reader.get_up_problem()

        # Extract ground truth for each action
        for action in problem.actions:
            # Normalize action name to lowercase for consistent comparison
            action_name = action.name.lower()

            # Extract preconditions
            preconditions = self._extract_literals(action.preconditions, action.parameters)

            # Extract effects (positive and negative)
            add_effects, delete_effects = self._extract_effects(action.effects, action.parameters)

            # Store in ground truth
            self.ground_truth_models[action_name] = {
                "preconditions": preconditions,
                "add_effects": add_effects,
                "delete_effects": delete_effects
            }

            logger.debug(f"Parsed ground truth for {action_name}:")
            logger.debug(f"  Preconditions: {preconditions}")
            logger.debug(f"  Add effects: {add_effects}")
            logger.debug(f"  Delete effects: {delete_effects}")

    def _extract_literals(self, expression, parameters) -> Set[str]:
        """Extract literals from a UP expression.

        Args:
            expression: UP FNode expression or list of expressions
            parameters: Action parameters

        Returns:
            Set of parameter-bound literal strings
        """
        literals = set()

        if expression is None:
            return literals

        # Handle list of preconditions (UP returns list for preconditions)
        if isinstance(expression, list):
            for expr in expression:
                literals.update(self._extract_literals(expr, parameters))
            return literals

        # Handle different expression types
        if hasattr(expression, 'is_and') and expression.is_and():
            # AND expression - extract all conjuncts
            for arg in expression.args:
                literals.update(self._extract_literals(arg, parameters))
        elif hasattr(expression, 'is_fluent_exp') and expression.is_fluent_exp():
            # Single fluent
            literal = ExpressionConverter.to_parameter_bound_string(expression, parameters)
            if literal:
                literals.add(literal)
        else:
            # Try to convert directly
            literal = ExpressionConverter.to_parameter_bound_string(expression, parameters)
            if literal:
                literals.add(literal)

        return literals

    def _extract_effects(self, effects, parameters) -> Tuple[Set[str], Set[str]]:
        """Extract positive and negative effects from UP effects.

        Args:
            effects: List of UP Effect objects
            parameters: Action parameters

        Returns:
            Tuple of (add_effects, delete_effects)
        """
        add_effects = set()
        delete_effects = set()

        for effect in effects:
            fluent_expr = effect.fluent

            # Check if it's a positive or negative effect
            if effect.is_increase() or not hasattr(effect, 'value') or effect.value.is_true():
                # Positive effect (add)
                literal = ExpressionConverter.to_parameter_bound_string(fluent_expr, parameters)
                if literal:
                    add_effects.add(literal)
            elif effect.value.is_false():
                # Negative effect (delete)
                literal = ExpressionConverter.to_parameter_bound_string(fluent_expr, parameters)
                if literal:
                    delete_effects.add(literal)

        return add_effects, delete_effects

    def compare_action(
        self,
        action_name: str,
        learned_preconditions: Set[str],
        learned_add_effects: Set[str],
        learned_delete_effects: Set[str]
    ) -> ModelComparisonResult:
        """Compare a learned action model to ground truth.

        Args:
            action_name: Name of the action
            learned_preconditions: Set of learned precondition literals
            learned_add_effects: Set of learned positive effect literals
            learned_delete_effects: Set of learned negative effect literals

        Returns:
            ModelComparisonResult with precision/recall/F1 scores

        Raises:
            ValueError: If action not found in ground truth
        """
        if action_name not in self.ground_truth_models:
            raise ValueError(f"Action '{action_name}' not found in ground truth")

        ground_truth = self.ground_truth_models[action_name]

        # Compare preconditions
        prec_metrics = self.compare_preconditions(
            learned_preconditions,
            ground_truth["preconditions"]
        )

        # Compare effects
        add_metrics, delete_metrics = self.compare_effects(
            learned_add_effects,
            learned_delete_effects,
            ground_truth["add_effects"],
            ground_truth["delete_effects"]
        )

        # Calculate overall F1 (average of all three F1 scores)
        overall_f1 = (
            prec_metrics["f1"] +
            add_metrics["f1"] +
            delete_metrics["f1"]
        ) / 3.0

        return ModelComparisonResult(
            action_name=action_name,
            # Preconditions
            precondition_precision=prec_metrics["precision"],
            precondition_recall=prec_metrics["recall"],
            precondition_f1=prec_metrics["f1"],
            precondition_false_positives=prec_metrics["false_positives"],
            precondition_false_negatives=prec_metrics["false_negatives"],
            # Add effects
            add_effect_precision=add_metrics["precision"],
            add_effect_recall=add_metrics["recall"],
            add_effect_f1=add_metrics["f1"],
            add_effect_false_positives=add_metrics["false_positives"],
            add_effect_false_negatives=add_metrics["false_negatives"],
            # Delete effects
            delete_effect_precision=delete_metrics["precision"],
            delete_effect_recall=delete_metrics["recall"],
            delete_effect_f1=delete_metrics["f1"],
            delete_effect_false_positives=delete_metrics["false_positives"],
            delete_effect_false_negatives=delete_metrics["false_negatives"],
            # Overall
            overall_f1=overall_f1
        )

    def compare_preconditions(
        self,
        learned: Set[str],
        ground_truth: Set[str]
    ) -> Dict[str, Any]:
        """Compare learned preconditions to ground truth.

        Args:
            learned: Set of learned precondition literals
            ground_truth: Set of ground truth precondition literals

        Returns:
            Dictionary with precision, recall, F1, and error sets
        """
        return self._calculate_metrics(learned, ground_truth)

    def compare_effects(
        self,
        learned_add: Set[str],
        learned_delete: Set[str],
        ground_truth_add: Set[str],
        ground_truth_delete: Set[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compare learned effects to ground truth.

        Args:
            learned_add: Set of learned positive effect literals
            learned_delete: Set of learned negative effect literals
            ground_truth_add: Set of ground truth positive effect literals
            ground_truth_delete: Set of ground truth negative effect literals

        Returns:
            Tuple of (add_metrics, delete_metrics)
        """
        add_metrics = self._calculate_metrics(learned_add, ground_truth_add)
        delete_metrics = self._calculate_metrics(learned_delete, ground_truth_delete)
        return add_metrics, delete_metrics

    def _calculate_metrics(
        self,
        learned: Set[str],
        ground_truth: Set[str]
    ) -> Dict[str, Any]:
        """Calculate precision, recall, F1 for set comparison.

        Normalizes predicate parameters before comparison to handle different
        parameter naming schemes (e.g., OLAM's ?param_N vs ground truth's ?x, ?y).

        Args:
            learned: Set of learned items
            ground_truth: Set of ground truth items

        Returns:
            Dictionary with metrics and error sets
        """
        # Normalize parameters in both sets for comparison
        learned_normalized = {normalize_predicate_parameters(p) for p in learned}
        ground_truth_normalized = {normalize_predicate_parameters(p) for p in ground_truth}

        # True positives: in both learned and ground truth (after normalization)
        true_positives = learned_normalized & ground_truth_normalized

        # False positives: in learned but not in ground truth (after normalization)
        false_positives = learned_normalized - ground_truth_normalized

        # False negatives: in ground truth but not in learned (after normalization)
        false_negatives = ground_truth_normalized - learned_normalized

        # Calculate metrics
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)

        # Handle edge cases for empty sets
        if tp_count == 0 and fp_count == 0 and fn_count == 0:
            # Both sets are empty - perfect match
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        elif tp_count == 0 and fp_count == 0:
            # Learned set is empty (no predictions made)
            # Making no claims → no false claims → precision = 1.0
            # But missing everything → recall = 0.0
            # Important for safe models with no effects learned yet
            precision = 1.0
            recall = 0.0
            f1 = 0.0
        elif tp_count == 0:
            # No true positives (but have false positives)
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # Normal calculation
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }