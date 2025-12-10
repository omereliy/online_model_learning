"""Model metrics computation for comparing learned models against ground truth.

Provides a simplified interface for computing aggregate precision/recall/F1 metrics
across all actions in a learned model.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from src.core.model_validator import ModelValidator
from src.core.model_reconstructor import ReconstructedModel

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Computes aggregate metrics for learned models against ground truth."""

    def __init__(self, ground_truth_domain_file: Path, problem_file: Path = None):
        """Initialize ModelMetrics.

        Args:
            ground_truth_domain_file: Path to ground truth PDDL domain file
            problem_file: Path to PDDL problem file (optional, will try to infer if not provided)
        """
        self.ground_truth_file = Path(ground_truth_domain_file)

        # If problem file not provided, try to find one in same directory
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
        self,
        model: ReconstructedModel,
        observation_counts: Dict[str, int] = None,
        min_observations: int = 0
    ) -> Dict[str, Any]:
        """Compute aggregate precision, recall, and F1 for a reconstructed model.

        Args:
            model: ReconstructedModel to evaluate
            observation_counts: Optional dict mapping action names to observation counts.
                               Used to filter out unexecuted actions.
            min_observations: Minimum observations required to include action in metrics.
                             Default 0 means include all actions.

        Returns:
            Dictionary with 'precision', 'recall', 'f1', 'precondition_precision',
            'precondition_recall', 'effect_precision', 'effect_recall' keys, plus
            'detailed_per_action' with per-action breakdown, and 'excluded_actions'
            listing any actions filtered out.
        """
        if not model.actions:
            logger.warning("Model has no actions, returning zero metrics")
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

        # For effect metrics: sum TP/FP/FN across all actions
        total_effect_tp = 0
        total_effect_fp = 0
        total_effect_fn = 0

        action_count = 0
        detailed_per_action = {}
        excluded_actions = []

        for action_name, action_model in model.actions.items():
            # Filter out actions with insufficient observations
            if observation_counts is not None and min_observations > 0:
                obs_count = observation_counts.get(action_name, 0)
                if obs_count < min_observations:
                    excluded_actions.append({
                        'action': action_name,
                        'observations': obs_count,
                        'reason': f'observations ({obs_count}) < min_observations ({min_observations})'
                    })
                    logger.debug(f"Excluding action {action_name}: {obs_count} < {min_observations} observations")
                    continue

            try:
                result = self.validator.compare_action(
                    action_name=action_name,
                    learned_preconditions=action_model.preconditions,
                    learned_add_effects=action_model.add_effects,
                    learned_delete_effects=action_model.del_effects
                )

                # Average across preconditions, add effects, and delete effects
                action_precision = (
                    result.precondition_precision +
                    result.add_effect_precision +
                    result.delete_effect_precision
                ) / 3.0

                action_recall = (
                    result.precondition_recall +
                    result.add_effect_recall +
                    result.delete_effect_recall
                ) / 3.0

                action_f1 = result.overall_f1

                total_precision += action_precision
                total_recall += action_recall
                total_f1 += action_f1

                # Accumulate separated precondition metrics
                total_prec_precision += result.precondition_precision
                total_prec_recall += result.precondition_recall

                # Get precondition metrics with TP/FP/FN counts
                prec_metrics = self.validator.compare_preconditions(
                    action_model.preconditions,
                    self.validator.ground_truth_models[action_name]["preconditions"]
                )
                prec_tp = len(prec_metrics["true_positives"])
                prec_fp = len(prec_metrics["false_positives"])
                prec_fn = len(prec_metrics["false_negatives"])

                # Sum TP/FP/FN for effects (instead of averaging)
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

                # Sum effects
                total_effect_tp += add_tp + del_tp
                total_effect_fp += add_fp + del_fp
                total_effect_fn += add_fn + del_fn

                # Store detailed per-action metrics
                detailed_per_action[action_name] = {
                    'prec_tp': prec_tp, 'prec_fp': prec_fp, 'prec_fn': prec_fn,
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

                logger.debug(f"Metrics for {action_name}: P={action_precision:.3f}, R={action_recall:.3f}, F1={action_f1:.3f}")

            except ValueError as e:
                logger.warning(f"Could not validate action {action_name}: {e}")
                continue

        if action_count == 0:
            logger.warning("No actions could be validated, returning zero metrics")
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'precondition_precision': 0.0, 'precondition_recall': 0.0,
                'effect_precision': 0.0, 'effect_recall': 0.0,
                'detailed_per_action': {},
                'excluded_actions': excluded_actions,
                'actions_evaluated': 0
            }

        # Calculate effect precision/recall from summed TP/FP/FN
        effect_precision = total_effect_tp / (total_effect_tp + total_effect_fp) if (total_effect_tp + total_effect_fp) > 0 else 1.0
        effect_recall = total_effect_tp / (total_effect_tp + total_effect_fn) if (total_effect_tp + total_effect_fn) > 0 else 0.0

        # Return aggregate and separated metrics
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
        """Compute metrics from a PDDL domain file with separated prec/effect metrics.

        Args:
            learned_domain_file: Path to learned PDDL domain file

        Returns:
            Dictionary with overall, precondition, and effect metrics
        """
        from src.core.pddl_io import PDDLReader

        # Parse learned domain
        reader = PDDLReader()
        learned_domain, _ = reader.parse_domain_and_problem(
            str(learned_domain_file),
            str(self.problem_file)
        )

        if not learned_domain.actions:
            logger.warning("Learned domain has no actions, returning zero metrics")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'precondition_precision': 0.0,
                'precondition_recall': 0.0,
                'effect_precision': 0.0,
                'effect_recall': 0.0
            }

        # Convert to ReconstructedModel format
        reconstructed_actions = {}
        for action_name, action in learned_domain.actions.items():
            from src.core.model_reconstructor import ReconstructedActionModel
            reconstructed_actions[action_name] = ReconstructedActionModel(
                preconditions=action.preconditions,
                add_effects=action.add_effects,
                del_effects=action.delete_effects
            )

        from src.core.model_reconstructor import ReconstructedModel
        model = ReconstructedModel(actions=reconstructed_actions)

        # Compute separated metrics
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

                # Precondition metrics (SEPARATED)
                total_prec_precision += result.precondition_precision
                total_prec_recall += result.precondition_recall

                # Effect metrics (SEPARATED - average of add and delete)
                effect_precision = (result.add_effect_precision + result.delete_effect_precision) / 2.0
                effect_recall = (result.add_effect_recall + result.delete_effect_recall) / 2.0
                total_eff_precision += effect_precision
                total_eff_recall += effect_recall

                action_count += 1

            except ValueError as e:
                logger.warning(f"Could not validate action {action_name}: {e}")
                continue

        if action_count == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'precondition_precision': 0.0,
                'precondition_recall': 0.0,
                'effect_precision': 0.0,
                'effect_recall': 0.0
            }

        # Compute aggregates
        prec_precision = total_prec_precision / action_count
        prec_recall = total_prec_recall / action_count
        eff_precision = total_eff_precision / action_count
        eff_recall = total_eff_recall / action_count

        # Overall metrics (average of prec and eff)
        overall_precision = (prec_precision + eff_precision) / 2.0
        overall_recall = (prec_recall + eff_recall) / 2.0

        return {
            'precision': overall_precision,
            'recall': overall_recall,
            'precondition_precision': prec_precision,
            'precondition_recall': prec_recall,
            'effect_precision': eff_precision,
            'effect_recall': eff_recall
        }
