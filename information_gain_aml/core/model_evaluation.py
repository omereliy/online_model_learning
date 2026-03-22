"""
Model evaluation: validation, reconstruction, and metrics.

BRIDGE module — will be replaced by AMLGym's evaluation interface.

Merged from model_validator.py + model_reconstructor.py + model_metrics.py.
Provides precision/recall/F1 computation for learned action models against
ground truth PDDL, plus safe/complete model reconstruction from snapshots.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional, Any

from information_gain_aml.core.pddl_io import PDDLReader
from information_gain_aml.core.expression_converter import ExpressionConverter

logger = logging.getLogger(__name__)


# ========== Predicate Normalization ==========

def normalize_predicate_parameters(predicate: str) -> str:
    """
    Normalize predicate parameters to positional format for comparison.

    Produces canonical format: "predicate_name(?0,?1,...)" or "predicate_name()" for 0-arity.

    Examples:
        >>> normalize_predicate_parameters("at(?x,?y)")
        'at(?0,?1)'
        >>> normalize_predicate_parameters("clear(?x)")
        'clear(?0)'
        >>> normalize_predicate_parameters("handempty()")
        'handempty()'
    """
    predicate = predicate.strip()

    # Strip outer PDDL-style parentheses if present
    if predicate.startswith('(') and not predicate.startswith('(?'):
        paren_count = 0
        for i, c in enumerate(predicate):
            if c == '(':
                paren_count += 1
            elif c == ')':
                paren_count -= 1
                if paren_count == 0 and i == len(predicate) - 1:
                    predicate = predicate[1:-1].strip()
                    break

    params = re.findall(r'\?[\w_]+', predicate)

    if '(' in predicate:
        pred_name = predicate[:predicate.index('(')]
    elif params:
        pred_name = predicate[:predicate.index(params[0])].strip()
    else:
        pred_name = predicate

    if not params:
        return f"{pred_name}()"
    else:
        positional_params = [f"?{i}" for i in range(len(params))]
        return f"{pred_name}({','.join(positional_params)})"


# ========== Data Classes ==========

@dataclass
class ModelComparisonResult:
    """Stores results from comparing a learned action model to ground truth."""

    action_name: str

    precondition_precision: float
    precondition_recall: float
    precondition_f1: float
    precondition_false_positives: Set[str]
    precondition_false_negatives: Set[str]

    add_effect_precision: float
    add_effect_recall: float
    add_effect_f1: float
    add_effect_false_positives: Set[str]
    add_effect_false_negatives: Set[str]

    delete_effect_precision: float
    delete_effect_recall: float
    delete_effect_f1: float
    delete_effect_false_positives: Set[str]
    delete_effect_false_negatives: Set[str]

    overall_f1: float


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


# ========== Model Validator ==========

class ModelValidator:
    """Validates learned models against ground truth from PDDL specifications."""

    def __init__(self, domain_file: Optional[str] = None, problem_file: Optional[str] = None):
        self.ground_truth_models: Dict[str, Dict[str, Set[str]]] = {}

        if domain_file and problem_file:
            self._parse_ground_truth(domain_file, problem_file)

    def _parse_ground_truth(self, domain_file: str, problem_file: str):
        reader = PDDLReader()
        domain, _ = reader.parse_domain_and_problem(domain_file, problem_file)

        problem = reader.get_up_problem()
        assert problem is not None, "UP problem must be available after parsing"

        for action in problem.actions:
            action_name = action.name.lower()
            preconditions = self._extract_literals(action.preconditions, action.parameters)
            add_effects, delete_effects = self._extract_effects(action.effects, action.parameters)

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
        literals: set[str] = set()
        if expression is None:
            return literals

        if isinstance(expression, list):
            for expr in expression:
                literals.update(self._extract_literals(expr, parameters))
            return literals

        if hasattr(expression, 'is_and') and expression.is_and():
            for arg in expression.args:
                literals.update(self._extract_literals(arg, parameters))
        elif hasattr(expression, 'is_fluent_exp') and expression.is_fluent_exp():
            literal = ExpressionConverter.to_parameter_bound_string(expression, parameters)
            if literal:
                literals.add(literal)
        else:
            literal = ExpressionConverter.to_parameter_bound_string(expression, parameters)
            if literal:
                literals.add(literal)

        return literals

    def _extract_effects(self, effects, parameters) -> Tuple[Set[str], Set[str]]:
        add_effects: set[str] = set()
        delete_effects: set[str] = set()

        for effect in effects:
            fluent_expr = effect.fluent
            if effect.is_increase() or not hasattr(effect, 'value') or effect.value.is_true():
                literal = ExpressionConverter.to_parameter_bound_string(fluent_expr, parameters)
                if literal:
                    add_effects.add(literal)
            elif effect.value.is_false():
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
        if action_name not in self.ground_truth_models:
            raise ValueError(f"Action '{action_name}' not found in ground truth")

        ground_truth = self.ground_truth_models[action_name]

        prec_metrics = self.compare_preconditions(
            learned_preconditions, ground_truth["preconditions"]
        )
        add_metrics, delete_metrics = self.compare_effects(
            learned_add_effects, learned_delete_effects,
            ground_truth["add_effects"], ground_truth["delete_effects"]
        )

        overall_f1 = (prec_metrics["f1"] + add_metrics["f1"] + delete_metrics["f1"]) / 3.0

        return ModelComparisonResult(
            action_name=action_name,
            precondition_precision=prec_metrics["precision"],
            precondition_recall=prec_metrics["recall"],
            precondition_f1=prec_metrics["f1"],
            precondition_false_positives=prec_metrics["false_positives"],
            precondition_false_negatives=prec_metrics["false_negatives"],
            add_effect_precision=add_metrics["precision"],
            add_effect_recall=add_metrics["recall"],
            add_effect_f1=add_metrics["f1"],
            add_effect_false_positives=add_metrics["false_positives"],
            add_effect_false_negatives=add_metrics["false_negatives"],
            delete_effect_precision=delete_metrics["precision"],
            delete_effect_recall=delete_metrics["recall"],
            delete_effect_f1=delete_metrics["f1"],
            delete_effect_false_positives=delete_metrics["false_positives"],
            delete_effect_false_negatives=delete_metrics["false_negatives"],
            overall_f1=overall_f1
        )

    def compare_preconditions(self, learned: Set[str], ground_truth: Set[str]) -> Dict[str, Any]:
        return self._calculate_metrics(learned, ground_truth)

    def compare_effects(
        self, learned_add: Set[str], learned_delete: Set[str],
        ground_truth_add: Set[str], ground_truth_delete: Set[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._calculate_metrics(learned_add, ground_truth_add), \
               self._calculate_metrics(learned_delete, ground_truth_delete)

    def _calculate_metrics(self, learned: Set[str], ground_truth: Set[str]) -> Dict[str, Any]:
        learned_normalized = {normalize_predicate_parameters(p) for p in learned}
        ground_truth_normalized = {normalize_predicate_parameters(p) for p in ground_truth}

        true_positives = learned_normalized & ground_truth_normalized
        false_positives = learned_normalized - ground_truth_normalized
        false_negatives = ground_truth_normalized - learned_normalized

        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)

        if tp_count == 0 and fp_count == 0 and fn_count == 0:
            precision, recall, f1 = 1.0, 1.0, 1.0
        elif tp_count == 0 and fp_count == 0:
            precision, recall, f1 = 1.0, 0.0, 0.0
        elif tp_count == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return {
            "precision": precision, "recall": recall, "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }


# ========== Model Reconstructor ==========

class ModelReconstructor:
    """Reconstructs safe and complete models from exported snapshots."""

    @staticmethod
    def reconstruct_information_gain_safe(snapshot: Dict, domain_file: Optional[str] = None) -> ReconstructedModel:
        all_possible = {}
        if domain_file:
            all_possible = ModelReconstructor._generate_all_possible_predicates(
                domain_file, snapshot["actions"]
            )

        actions = {}
        for action_name, action_data in snapshot["actions"].items():
            if action_name in all_possible:
                preconditions = all_possible[action_name].copy()
            else:
                preconditions = {p for p in action_data["possible_preconditions"] if not p.startswith('¬')}

            add_effects = {e for e in action_data["confirmed_add_effects"] if not e.startswith('¬')}
            del_effects = {e for e in action_data["confirmed_del_effects"] if not e.startswith('¬')}

            actions[action_name] = ReconstructedAction(
                name=action_name, parameters=action_data["parameters"],
                preconditions=preconditions, add_effects=add_effects, del_effects=del_effects
            )

        return ReconstructedModel(
            model_type="safe", algorithm=snapshot["algorithm"],
            iteration=snapshot["iteration"], actions=actions
        )

    @staticmethod
    def reconstruct_information_gain_complete(snapshot: Dict) -> ReconstructedModel:
        actions = {}
        for action_name, action_data in snapshot["actions"].items():
            preconditions = {p for p in action_data["certain_preconditions"] if not p.startswith('¬')}

            def filter_negated(literals):
                return {lit for lit in literals if not lit.startswith('¬')}

            add_effects = filter_negated(
                set(action_data["confirmed_add_effects"]) | set(action_data["possible_add_effects"])
            )
            del_effects = filter_negated(
                set(action_data["confirmed_del_effects"]) | set(action_data["possible_del_effects"])
            )

            actions[action_name] = ReconstructedAction(
                name=action_name, parameters=action_data["parameters"],
                preconditions=preconditions, add_effects=add_effects, del_effects=del_effects
            )

        return ReconstructedModel(
            model_type="complete", algorithm=snapshot["algorithm"],
            iteration=snapshot["iteration"], actions=actions
        )

    @staticmethod
    def _generate_all_possible_predicates(domain_file: str, actions_data: Dict) -> Dict[str, Set[str]]:
        from unified_planning.io import PDDLReader
        try:
            reader = PDDLReader()
            problem = reader.parse_problem(domain_file)

            all_possible = {}
            for action_name in actions_data.keys():
                up_action = None
                for action in problem.actions:
                    if action.name.lower() == action_name.lower():
                        up_action = action
                        break

                if not up_action:
                    logger.warning(f"Action '{action_name}' not found in domain {domain_file}")
                    continue

                possible_precs: set[str] = set()
                action_params = list(up_action.parameters)

                for fluent in problem.fluents:
                    fluent_params = list(fluent.signature)
                    if len(fluent_params) == 0:
                        possible_precs.add(f"{fluent.name}()")
                    else:
                        from itertools import permutations
                        for perm in permutations(range(len(action_params)), len(fluent_params)):
                            param_str = ','.join([f"?{idx}" for idx in perm])
                            possible_precs.add(f"{fluent.name}({param_str})")

                all_possible[action_name] = possible_precs

            return all_possible
        except Exception as e:
            logger.error(f"Failed to generate all possible predicates: {e}")
            return {}

    @staticmethod
    def _remove_contradictions(add_effects: Set[str], del_effects: Set[str]) -> Tuple[Set[str], Set[str]]:
        contradictions = add_effects & del_effects
        if contradictions:
            add_effects = add_effects - contradictions
            del_effects = del_effects - contradictions
        return add_effects, del_effects

    @staticmethod
    def load_and_reconstruct(snapshot_path: Path, domain_file: Optional[str] = None) -> List[ReconstructedModel]:
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

        return [safe, complete]

    @staticmethod
    def reconstruct_from_directory(models_dir: Path, iteration: int) -> List[ReconstructedModel]:
        snapshot_path = models_dir / f"model_iter_{iteration:03d}.json"
        return ModelReconstructor.load_and_reconstruct(snapshot_path)


# ========== Model Metrics ==========

class ModelMetrics:
    """Computes aggregate metrics for learned models against ground truth."""

    def __init__(self, ground_truth_domain_file: Path, problem_file: Optional[Path] = None):
        self.ground_truth_file = Path(ground_truth_domain_file)

        if problem_file is None:
            problem_dir = self.ground_truth_file.parent
            problem_files = list(problem_dir.glob("p*.pddl"))
            if problem_files:
                problem_file = problem_files[0]
                logger.info(f"Using problem file: {problem_file}")
            else:
                raise ValueError(f"No problem file found in {problem_dir}")

        self.problem_file = Path(problem_file)
        self.validator = ModelValidator(
            domain_file=str(self.ground_truth_file),
            problem_file=str(self.problem_file)
        )

    def compute_metrics(
        self, model: ReconstructedModel,
        observation_counts: Optional[Dict[str, int]] = None,
        min_observations: int = 0
    ) -> Dict[str, Any]:
        if not model.actions:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'precondition_precision': 0.0, 'precondition_recall': 0.0,
                'effect_precision': 0.0, 'effect_recall': 0.0,
                'detailed_per_action': {}
            }

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_prec_precision = 0.0
        total_prec_recall = 0.0
        total_effect_tp = 0
        total_effect_fp = 0
        total_effect_fn = 0
        action_count = 0
        detailed_per_action: dict[str, Any] = {}
        excluded_actions: list[dict[str, Any]] = []

        for action_name, action_model in model.actions.items():
            if observation_counts is not None and min_observations > 0:
                obs_count = observation_counts.get(action_name, 0)
                if obs_count < min_observations:
                    excluded_actions.append({
                        'action': action_name, 'observations': obs_count,
                        'reason': f'observations ({obs_count}) < min_observations ({min_observations})'
                    })
                    continue

            try:
                result = self.validator.compare_action(
                    action_name=action_name,
                    learned_preconditions=action_model.preconditions,
                    learned_add_effects=action_model.add_effects,
                    learned_delete_effects=action_model.del_effects
                )

                action_precision = (
                    result.precondition_precision + result.add_effect_precision + result.delete_effect_precision
                ) / 3.0
                action_recall = (
                    result.precondition_recall + result.add_effect_recall + result.delete_effect_recall
                ) / 3.0

                total_precision += action_precision
                total_recall += action_recall
                total_f1 += result.overall_f1
                total_prec_precision += result.precondition_precision
                total_prec_recall += result.precondition_recall

                prec_metrics = self.validator.compare_preconditions(
                    action_model.preconditions,
                    self.validator.ground_truth_models[action_name]["preconditions"]
                )
                add_metrics = self.validator.compare_effects(
                    action_model.add_effects, action_model.del_effects,
                    self.validator.ground_truth_models[action_name]["add_effects"],
                    self.validator.ground_truth_models[action_name]["delete_effects"]
                )

                add_tp = len(add_metrics[0]["true_positives"])
                add_fp = len(add_metrics[0]["false_positives"])
                add_fn = len(add_metrics[0]["false_negatives"])
                del_tp = len(add_metrics[1]["true_positives"])
                del_fp = len(add_metrics[1]["false_positives"])
                del_fn = len(add_metrics[1]["false_negatives"])

                total_effect_tp += add_tp + del_tp
                total_effect_fp += add_fp + del_fp
                total_effect_fn += add_fn + del_fn

                detailed_per_action[action_name] = {
                    'prec_tp': len(prec_metrics["true_positives"]),
                    'prec_fp': len(prec_metrics["false_positives"]),
                    'prec_fn': len(prec_metrics["false_negatives"]),
                    'prec_precision': result.precondition_precision,
                    'prec_recall': result.precondition_recall,
                    'add_tp': add_tp, 'add_fp': add_fp, 'add_fn': add_fn,
                    'add_precision': result.add_effect_precision,
                    'add_recall': result.add_effect_recall,
                    'del_tp': del_tp, 'del_fp': del_fp, 'del_fn': del_fn,
                    'del_precision': result.delete_effect_precision,
                    'del_recall': result.delete_effect_recall
                }
                action_count += 1

            except ValueError as e:
                logger.warning(f"Could not validate action {action_name}: {e}")
                continue

        if action_count == 0:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'precondition_precision': 0.0, 'precondition_recall': 0.0,
                'effect_precision': 0.0, 'effect_recall': 0.0,
                'detailed_per_action': {}, 'excluded_actions': excluded_actions,
                'actions_evaluated': 0
            }

        effect_precision = total_effect_tp / (total_effect_tp + total_effect_fp) if (total_effect_tp + total_effect_fp) > 0 else 1.0
        effect_recall = total_effect_tp / (total_effect_tp + total_effect_fn) if (total_effect_tp + total_effect_fn) > 0 else 0.0

        return {
            'precision': total_precision / action_count,
            'recall': total_recall / action_count,
            'f1': total_f1 / action_count,
            'precondition_precision': total_prec_precision / action_count,
            'precondition_recall': total_prec_recall / action_count,
            'effect_precision': effect_precision,
            'effect_recall': effect_recall,
            'detailed_per_action': detailed_per_action,
            'excluded_actions': excluded_actions,
            'actions_evaluated': action_count
        }

    def compute_metrics_from_pddl(self, learned_domain_file: Path) -> Dict[str, float]:
        reader = PDDLReader()
        learned_domain, _ = reader.parse_domain_and_problem(
            str(learned_domain_file), str(self.problem_file)
        )

        if not learned_domain.lifted_actions:
            return {
                'precision': 0.0, 'recall': 0.0,
                'precondition_precision': 0.0, 'precondition_recall': 0.0,
                'effect_precision': 0.0, 'effect_recall': 0.0
            }

        reconstructed_actions = {}
        for action_name, action in learned_domain.lifted_actions.items():
            reconstructed_actions[action_name] = ReconstructedAction(
                name=action_name, parameters=[p.name for p in action.parameters],
                preconditions=action.preconditions, add_effects=action.add_effects,
                del_effects=action.del_effects
            )

        model = ReconstructedModel(
            model_type="learned_pddl", algorithm="pddl_import",
            iteration=0, actions=reconstructed_actions
        )

        total_prec_precision = 0.0
        total_prec_recall = 0.0
        total_eff_precision = 0.0
        total_eff_recall = 0.0
        action_count = 0

        for action_name, action_model in model.actions.items():
            try:
                result = self.validator.compare_action(
                    action_name=action_name,
                    learned_preconditions=action_model.preconditions,
                    learned_add_effects=action_model.add_effects,
                    learned_delete_effects=action_model.del_effects
                )
                total_prec_precision += result.precondition_precision
                total_prec_recall += result.precondition_recall
                total_eff_precision += (result.add_effect_precision + result.delete_effect_precision) / 2.0
                total_eff_recall += (result.add_effect_recall + result.delete_effect_recall) / 2.0
                action_count += 1
            except ValueError as e:
                logger.warning(f"Could not validate action {action_name}: {e}")
                continue

        if action_count == 0:
            return {
                'precision': 0.0, 'recall': 0.0,
                'precondition_precision': 0.0, 'precondition_recall': 0.0,
                'effect_precision': 0.0, 'effect_recall': 0.0
            }

        pp = total_prec_precision / action_count
        pr = total_prec_recall / action_count
        ep = total_eff_precision / action_count
        er = total_eff_recall / action_count

        return {
            'precision': (pp + ep) / 2.0, 'recall': (pr + er) / 2.0,
            'precondition_precision': pp, 'precondition_recall': pr,
            'effect_precision': ep, 'effect_recall': er
        }
