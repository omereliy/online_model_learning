"""Tests for ModelValidator class."""

import pytest
from typing import Set, Dict
from dataclasses import dataclass

from src.core.model_validator import ModelValidator, ModelComparisonResult


class TestModelComparisonResult:
    """Test the ModelComparisonResult dataclass."""

    def test_dataclass_fields(self):
        """Test that ModelComparisonResult contains all required fields."""
        result = ModelComparisonResult(
            action_name="pick-up",
            # Precondition metrics
            precondition_precision=0.75,
            precondition_recall=0.60,
            precondition_f1=0.67,
            precondition_false_positives={"extra(?x)"},
            precondition_false_negatives={"missing(?x)"},
            # Add effect metrics
            add_effect_precision=0.80,
            add_effect_recall=0.90,
            add_effect_f1=0.85,
            add_effect_false_positives=set(),
            add_effect_false_negatives={"forgotten_effect(?x)"},
            # Delete effect metrics
            delete_effect_precision=1.0,
            delete_effect_recall=0.50,
            delete_effect_f1=0.67,
            delete_effect_false_positives=set(),
            delete_effect_false_negatives={"missing_delete(?x)"},
            # Overall accuracy
            overall_f1=0.73
        )

        assert result.action_name == "pick-up"
        assert result.precondition_precision == 0.75
        assert result.precondition_recall == 0.60
        assert result.precondition_f1 == 0.67
        assert result.precondition_false_positives == {"extra(?x)"}
        assert result.precondition_false_negatives == {"missing(?x)"}
        assert result.add_effect_precision == 0.80
        assert result.add_effect_recall == 0.90
        assert result.add_effect_f1 == 0.85
        assert result.delete_effect_precision == 1.0
        assert result.delete_effect_recall == 0.50
        assert result.delete_effect_f1 == 0.67
        assert result.overall_f1 == 0.73


class TestModelValidator:
    """Test ModelValidator functionality."""

    def test_perfect_match(self):
        """Test when learned model perfectly matches ground truth."""
        # Create validator with hand-crafted ground truth
        validator = ModelValidator()

        # Manually set ground truth for testing
        validator.ground_truth_models = {
            "pick-up": {
                "preconditions": {"clear(?x)", "ontable(?x)", "handempty()"},
                "add_effects": {"holding(?x)"},
                "delete_effects": {"clear(?x)", "ontable(?x)", "handempty()"}
            }
        }

        # Test perfect match
        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions={"clear(?x)", "ontable(?x)", "handempty()"},
            learned_add_effects={"holding(?x)"},
            learned_delete_effects={"clear(?x)", "ontable(?x)", "handempty()"}
        )

        assert result.precondition_precision == 1.0
        assert result.precondition_recall == 1.0
        assert result.precondition_f1 == 1.0
        assert result.precondition_false_positives == set()
        assert result.precondition_false_negatives == set()

        assert result.add_effect_precision == 1.0
        assert result.add_effect_recall == 1.0
        assert result.add_effect_f1 == 1.0

        assert result.delete_effect_precision == 1.0
        assert result.delete_effect_recall == 1.0
        assert result.delete_effect_f1 == 1.0

        assert result.overall_f1 == 1.0

    def test_missing_preconditions(self):
        """Test false negatives in preconditions."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "pick-up": {
                "preconditions": {"clear(?x)", "ontable(?x)", "handempty()"},
                "add_effects": {"holding(?x)"},
                "delete_effects": {"clear(?x)", "ontable(?x)", "handempty()"}
            }
        }

        # Learned model missing "handempty()"
        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions={"clear(?x)", "ontable(?x)"},  # Missing handempty()
            learned_add_effects={"holding(?x)"},
            learned_delete_effects={"clear(?x)", "ontable(?x)", "handempty()"}
        )

        # Precision: 2/2 = 1.0 (all learned are correct)
        # Recall: 2/3 = 0.667 (missing one)
        assert result.precondition_precision == 1.0
        assert result.precondition_recall == pytest.approx(0.667, rel=1e-2)
        assert result.precondition_f1 == pytest.approx(0.8, rel=1e-2)
        assert result.precondition_false_positives == set()
        assert result.precondition_false_negatives == {"handempty()"}

    def test_extra_preconditions(self):
        """Test false positives in preconditions."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "pick-up": {
                "preconditions": {"clear(?x)", "ontable(?x)", "handempty()"},
                "add_effects": {"holding(?x)"},
                "delete_effects": {"clear(?x)", "ontable(?x)", "handempty()"}
            }
        }

        # Learned model has extra precondition
        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions={"clear(?x)", "ontable(?x)", "handempty()", "extra(?x)"},
            learned_add_effects={"holding(?x)"},
            learned_delete_effects={"clear(?x)", "ontable(?x)", "handempty()"}
        )

        # Precision: 3/4 = 0.75 (one extra)
        # Recall: 3/3 = 1.0 (all found)
        assert result.precondition_precision == 0.75
        assert result.precondition_recall == 1.0
        assert result.precondition_f1 == pytest.approx(0.857, rel=1e-2)
        assert result.precondition_false_positives == {"extra(?x)"}
        assert result.precondition_false_negatives == set()

    def test_missing_effects(self):
        """Test false negatives in effects."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "stack": {
                "preconditions": {"holding(?x)", "clear(?y)"},
                "add_effects": {"on(?x,?y)", "clear(?x)", "handempty()"},
                "delete_effects": {"holding(?x)", "clear(?y)"}
            }
        }

        # Missing some add effects
        result = validator.compare_action(
            action_name="stack",
            learned_preconditions={"holding(?x)", "clear(?y)"},
            learned_add_effects={"on(?x,?y)"},  # Missing clear(?x) and handempty()
            learned_delete_effects={"holding(?x)", "clear(?y)"}
        )

        # Add effects: precision 1/1 = 1.0, recall 1/3 = 0.333
        assert result.add_effect_precision == 1.0
        assert result.add_effect_recall == pytest.approx(0.333, rel=1e-2)
        assert result.add_effect_false_positives == set()
        assert result.add_effect_false_negatives == {"clear(?x)", "handempty()"}

    def test_extra_effects(self):
        """Test false positives in effects."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "unstack": {
                "preconditions": {"on(?x,?y)", "clear(?x)", "handempty()"},
                "add_effects": {"holding(?x)", "clear(?y)"},
                "delete_effects": {"on(?x,?y)", "clear(?x)", "handempty()"}
            }
        }

        # Extra delete effect
        result = validator.compare_action(
            action_name="unstack",
            learned_preconditions={"on(?x,?y)", "clear(?x)", "handempty()"},
            learned_add_effects={"holding(?x)", "clear(?y)"},
            learned_delete_effects={"on(?x,?y)", "clear(?x)", "handempty()", "extra_delete(?y)"}
        )

        # Delete effects: precision 3/4 = 0.75, recall 3/3 = 1.0
        assert result.delete_effect_precision == 0.75
        assert result.delete_effect_recall == 1.0
        assert result.delete_effect_false_positives == {"extra_delete(?y)"}
        assert result.delete_effect_false_negatives == set()

    def test_precision_recall_calculation(self):
        """Test precise calculation of precision/recall/F1."""
        validator = ModelValidator()

        # Test with known values
        ground_truth = {"a", "b", "c", "d"}
        learned = {"a", "b", "e", "f"}

        metrics = validator._calculate_metrics(learned, ground_truth)

        # TP = 2 (a, b), FP = 2 (e, f), FN = 2 (c, d)
        # Precision = 2/4 = 0.5
        # Recall = 2/4 = 0.5
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5
        assert metrics["f1"] == 0.5
        assert metrics["false_positives"] == {"e", "f"}
        assert metrics["false_negatives"] == {"c", "d"}

    def test_empty_learned_model(self):
        """Test edge case with empty learned model (all false negatives)."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "pick-up": {
                "preconditions": {"clear(?x)", "ontable(?x)", "handempty()"},
                "add_effects": {"holding(?x)"},
                "delete_effects": {"clear(?x)", "ontable(?x)", "handempty()"}
            }
        }

        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions=set(),
            learned_add_effects=set(),
            learned_delete_effects=set()
        )

        # Everything is a false negative
        assert result.precondition_precision == 0.0  # No true positives
        assert result.precondition_recall == 0.0
        assert result.precondition_f1 == 0.0
        assert result.precondition_false_negatives == {"clear(?x)", "ontable(?x)", "handempty()"}

    def test_over_general_learned_model(self):
        """Test edge case with over-general learned model (many false positives)."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "noop": {
                "preconditions": set(),  # No preconditions
                "add_effects": set(),
                "delete_effects": set()
            }
        }

        # Learned model has many unnecessary preconditions
        result = validator.compare_action(
            action_name="noop",
            learned_preconditions={"extra1(?x)", "extra2(?y)", "extra3(?z)"},
            learned_add_effects={"effect1(?x)"},
            learned_delete_effects={"effect2(?y)"}
        )

        # Everything is a false positive
        assert result.precondition_precision == 0.0
        assert result.precondition_false_positives == {"extra1(?x)", "extra2(?y)", "extra3(?z)"}
        assert result.add_effect_false_positives == {"effect1(?x)"}
        assert result.delete_effect_false_positives == {"effect2(?y)"}

    def test_action_with_no_preconditions(self):
        """Test action with no preconditions (propositional)."""
        validator = ModelValidator()

        validator.ground_truth_models = {
            "always_applicable": {
                "preconditions": set(),
                "add_effects": {"result()"},
                "delete_effects": set()
            }
        }

        result = validator.compare_action(
            action_name="always_applicable",
            learned_preconditions=set(),
            learned_add_effects={"result()"},
            learned_delete_effects=set()
        )

        # Perfect match on empty preconditions
        assert result.precondition_precision == 1.0
        assert result.precondition_recall == 1.0
        assert result.precondition_f1 == 1.0
        assert result.add_effect_f1 == 1.0

    def test_compare_preconditions_method(self):
        """Test the compare_preconditions method directly."""
        validator = ModelValidator()

        ground_truth = {"clear(?x)", "ontable(?x)", "handempty()"}
        learned = {"clear(?x)", "ontable(?x)", "extra(?y)"}

        metrics = validator.compare_preconditions(learned, ground_truth)

        # TP = 2, FP = 1, FN = 1
        # Precision = 2/3 = 0.667, Recall = 2/3 = 0.667
        assert metrics["precision"] == pytest.approx(0.667, rel=1e-2)
        assert metrics["recall"] == pytest.approx(0.667, rel=1e-2)
        assert metrics["false_positives"] == {"extra(?y)"}
        assert metrics["false_negatives"] == {"handempty()"}

    def test_compare_effects_method(self):
        """Test the compare_effects method directly."""
        validator = ModelValidator()

        ground_truth_add = {"on(?x,?y)", "clear(?x)"}
        learned_add = {"on(?x,?y)", "extra_add(?z)"}

        ground_truth_delete = {"holding(?x)", "clear(?y)"}
        learned_delete = {"holding(?x)"}

        add_metrics, delete_metrics = validator.compare_effects(
            learned_add, learned_delete,
            ground_truth_add, ground_truth_delete
        )

        # Add effects: TP = 1, FP = 1, FN = 1
        assert add_metrics["precision"] == 0.5
        assert add_metrics["recall"] == 0.5
        assert add_metrics["false_positives"] == {"extra_add(?z)"}
        assert add_metrics["false_negatives"] == {"clear(?x)"}

        # Delete effects: TP = 1, FP = 0, FN = 1
        assert delete_metrics["precision"] == 1.0
        assert delete_metrics["recall"] == 0.5
        assert delete_metrics["false_negatives"] == {"clear(?y)"}


class TestModelValidatorWithPDDL:
    """Test ModelValidator with actual PDDL parsing."""

    @pytest.fixture
    def blocksworld_domain_file(self, tmp_path):
        """Create a simple blocksworld domain file."""
        domain = tmp_path / "domain.pddl"
        domain.write_text("""
(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)

  (:predicates
    (clear ?x - block)
    (ontable ?x - block)
    (handempty)
    (holding ?x - block)
    (on ?x ?y - block)
  )

  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (holding ?x)
                 (not (clear ?x))
                 (not (ontable ?x))
                 (not (handempty)))
  )

  (:action stack
    :parameters (?x ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y)
                 (clear ?x)
                 (handempty)
                 (not (holding ?x))
                 (not (clear ?y)))
  )
)
        """)
        return str(domain)

    @pytest.fixture
    def blocksworld_problem_file(self, tmp_path):
        """Create a simple blocksworld problem file."""
        problem = tmp_path / "problem.pddl"
        problem.write_text("""
(define (problem blocks-problem)
  (:domain blocksworld)
  (:objects a b c - block)

  (:init
    (clear a)
    (clear b)
    (clear c)
    (ontable a)
    (ontable b)
    (ontable c)
    (handempty)
  )

  (:goal (and (on a b) (on b c)))
)
        """)
        return str(problem)

    def test_parse_ground_truth_from_pddl(self, blocksworld_domain_file, blocksworld_problem_file):
        """Test parsing ground truth from PDDL files."""
        validator = ModelValidator(blocksworld_domain_file, blocksworld_problem_file)

        # Check that ground truth was parsed correctly
        assert "pick-up" in validator.ground_truth_models
        assert "stack" in validator.ground_truth_models

        # Check pick-up action
        pickup_model = validator.ground_truth_models["pick-up"]
        assert "clear(?x)" in pickup_model["preconditions"]
        assert "ontable(?x)" in pickup_model["preconditions"]
        assert "handempty()" in pickup_model["preconditions"]
        assert "holding(?x)" in pickup_model["add_effects"]
        assert "clear(?x)" in pickup_model["delete_effects"]

    def test_integration_with_real_domain(self, blocksworld_domain_file, blocksworld_problem_file):
        """Test integration with real blocksworld domain."""
        validator = ModelValidator(blocksworld_domain_file, blocksworld_problem_file)

        # Test with partially learned model (realistic scenario)
        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions={"clear(?x)", "ontable(?x)"},  # Missing handempty
            learned_add_effects={"holding(?x)"},
            learned_delete_effects={"clear(?x)", "ontable(?x)"}  # Missing handempty
        )

        # Should identify missing precondition and delete effect
        assert result.precondition_false_negatives == {"handempty()"}
        assert result.delete_effect_false_negatives == {"handempty()"}
        assert result.overall_f1 < 1.0  # Not perfect match

    def test_action_not_in_ground_truth(self):
        """Test comparing an action that doesn't exist in ground truth."""
        validator = ModelValidator()
        validator.ground_truth_models = {
            "pick-up": {
                "preconditions": {"clear(?x)"},
                "add_effects": {"holding(?x)"},
                "delete_effects": {"clear(?x)"}
            }
        }

        with pytest.raises(ValueError, match="Action 'unknown-action' not found"):
            validator.compare_action(
                action_name="unknown-action",
                learned_preconditions={"some(?x)"},
                learned_add_effects={"effect(?x)"},
                learned_delete_effects=set()
            )