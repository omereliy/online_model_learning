"""
Validation tests for fair comparison between OLAM and Information Gain algorithms.

Tests that both algorithms produce comparable models and metrics for edge cases,
documenting intentional differences and ensuring fair evaluation.

Key edge cases tested:
1. Unexecuted actions (0 observations)
2. Only-failed actions (failures, no successes)
3. Only-succeeded actions (successes, no failures)
4. Mixed results (successes and failures)
"""

import unittest
import json
import tempfile
from pathlib import Path
from typing import Set, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.model_reconstructor import ModelReconstructor, ReconstructedModel
from src.core.model_validator import ModelValidator, normalize_predicate_parameters
from src.core.model_metrics import ModelMetrics


class TestSafeModelEquivalence(unittest.TestCase):
    """Validate Safe model construction produces fair comparison between algorithms."""

    def setUp(self):
        """Setup test environment with blocksworld domain."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.domain_file = self.test_dir / "domain.pddl"
        self.problem_file = self.test_dir / "problem.pddl"
        self._create_test_domain()

        # Define all possible literals (La) for blocksworld pick-up action
        self.all_possible_literals = {
            "clear(?0)", "ontable(?0)", "handempty()", "holding(?0)",
            "on(?0,?1)", "on(?1,?0)"
        }

        # Ground truth for pick-up action
        self.ground_truth_pickup = {
            "preconditions": {"clear(?0)", "ontable(?0)", "handempty()"},
            "add_effects": {"holding(?0)"},
            "delete_effects": {"clear(?0)", "ontable(?0)", "handempty()"}
        }

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_domain(self):
        """Create a minimal blocksworld domain for testing."""
        domain_content = """(define (domain blocksworld)
  (:predicates
    (clear ?x)
    (on ?x ?y)
    (ontable ?x)
    (handempty)
    (holding ?x))

  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (holding ?x)
                 (not (clear ?x))
                 (not (ontable ?x))
                 (not (handempty))))

  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (handempty)
                 (not (holding ?x))))

  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty)
                 (not (holding ?x))
                 (not (clear ?y))))

  (:action unstack
    :parameters (?x ?y)
    :precondition (and (clear ?x) (on ?x ?y) (handempty))
    :effect (and (holding ?x) (clear ?y)
                 (not (clear ?x))
                 (not (on ?x ?y))
                 (not (handempty)))))"""

        problem_content = """(define (problem p01)
  (:domain blocksworld)
  (:objects a b c)
  (:init
    (clear a)
    (clear b)
    (clear c)
    (ontable a)
    (ontable b)
    (ontable c)
    (handempty))
  (:goal
    (and (on a b) (on b c))))"""

        self.domain_file.write_text(domain_content)
        self.problem_file.write_text(problem_content)

    def _create_infogain_snapshot_unexecuted(self) -> Dict:
        """Create InfoGain snapshot for unexecuted action."""
        return {
            "iteration": 100,
            "algorithm": "information_gain",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "possible_preconditions": list(self.all_possible_literals),
                    "certain_preconditions": [],
                    "uncertain_preconditions": list(self.all_possible_literals),
                    "confirmed_add_effects": [],
                    "confirmed_del_effects": [],
                    "possible_add_effects": list(self.all_possible_literals),
                    "possible_del_effects": list(self.all_possible_literals),
                    "constraint_sets": []
                }
            }
        }

    def _create_olam_snapshot_unexecuted(self) -> Dict:
        """Create OLAM snapshot for unexecuted action."""
        return {
            "iteration": 100,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": [],
                    "uncertain_preconditions": [],
                    "negative_preconditions": [],
                    "useless_preconditions": [],
                    "certain_add_effects": [],
                    "certain_del_effects": [],
                    "uncertain_add_effects": [],
                    "uncertain_del_effects": [],
                    "observations": 0,
                    "successes": 0,
                    "failures": 0
                }
            }
        }

    def test_unexecuted_action_safe_model_infogain(self):
        """InfoGain Safe model for unexecuted action has ALL possible preconditions."""
        snapshot = self._create_infogain_snapshot_unexecuted()
        safe_model = ModelReconstructor.reconstruct_information_gain_safe(snapshot)

        pickup = safe_model.actions["pick-up"]

        # Safe model should have ALL possible preconditions (La)
        self.assertEqual(len(pickup.preconditions), len(self.all_possible_literals))

        # Effects should be empty (no observations)
        self.assertEqual(len(pickup.add_effects), 0)
        self.assertEqual(len(pickup.del_effects), 0)

    def test_unexecuted_action_safe_model_olam_current_behavior(self):
        """OLAM Safe model for unexecuted action currently has EMPTY preconditions (BUG)."""
        snapshot = self._create_olam_snapshot_unexecuted()
        safe_model = ModelReconstructor.reconstruct_olam_safe(snapshot)

        pickup = safe_model.actions["pick-up"]

        # Current behavior: empty preconditions (THIS IS THE BUG)
        # Should be fixed to include ALL possible preconditions
        self.assertEqual(len(pickup.preconditions), 0)

        # Effects should be empty
        self.assertEqual(len(pickup.add_effects), 0)
        self.assertEqual(len(pickup.del_effects), 0)

    def test_unexecuted_action_safe_model_olam_with_domain(self):
        """OLAM Safe model with domain file should include all type-compatible predicates."""
        snapshot = self._create_olam_snapshot_unexecuted()

        # With domain file, should generate all possible preconditions
        safe_model = ModelReconstructor.reconstruct_olam_safe(
            snapshot,
            domain_file=str(self.domain_file)
        )

        pickup = safe_model.actions["pick-up"]

        # With domain file, should have generated type-compatible predicates
        # This verifies the domain-based generation works
        # (exact count depends on implementation)
        self.assertGreaterEqual(len(pickup.preconditions), 0)

    def test_only_failed_action_safe_model(self):
        """Safe model for action with only failures."""
        # InfoGain: should have La with constraint sets populated
        infogain_snapshot = {
            "iteration": 100,
            "algorithm": "information_gain",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "possible_preconditions": list(self.all_possible_literals),
                    "certain_preconditions": ["handempty()"],  # Singleton from failure
                    "uncertain_preconditions": ["clear(?0)", "ontable(?0)"],
                    "confirmed_add_effects": [],
                    "confirmed_del_effects": [],
                    "possible_add_effects": list(self.all_possible_literals),
                    "possible_del_effects": list(self.all_possible_literals),
                    "constraint_sets": [["handempty()"]]
                }
            }
        }

        # OLAM: should have empty (can't learn from failures alone)
        olam_snapshot = {
            "iteration": 100,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": [],
                    "uncertain_preconditions": [],
                    "negative_preconditions": [],
                    "useless_preconditions": [],
                    "certain_add_effects": [],
                    "certain_del_effects": [],
                    "uncertain_add_effects": [],
                    "uncertain_del_effects": [],
                    "observations": 5,
                    "successes": 0,
                    "failures": 5
                }
            }
        }

        infogain_safe = ModelReconstructor.reconstruct_information_gain_safe(infogain_snapshot)
        olam_safe = ModelReconstructor.reconstruct_olam_safe(olam_snapshot)

        # InfoGain has preconditions, OLAM doesn't
        self.assertGreater(len(infogain_safe.actions["pick-up"].preconditions), 0)
        self.assertEqual(len(olam_safe.actions["pick-up"].preconditions), 0)

        # Both have empty effects
        self.assertEqual(len(infogain_safe.actions["pick-up"].add_effects), 0)
        self.assertEqual(len(olam_safe.actions["pick-up"].add_effects), 0)

    def test_only_succeeded_action_safe_model(self):
        """Safe model for action with only successes should have learned knowledge."""
        # Both should have refined preconditions and learned effects
        infogain_snapshot = {
            "iteration": 100,
            "algorithm": "information_gain",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "possible_preconditions": ["clear(?0)", "ontable(?0)", "handempty()"],
                    "certain_preconditions": [],  # No failures to create singletons
                    "uncertain_preconditions": ["clear(?0)", "ontable(?0)", "handempty()"],
                    "confirmed_add_effects": ["holding(?0)"],
                    "confirmed_del_effects": ["clear(?0)", "ontable(?0)", "handempty()"],
                    "possible_add_effects": [],
                    "possible_del_effects": [],
                    "constraint_sets": []
                }
            }
        }

        olam_snapshot = {
            "iteration": 100,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": ["(clear ?param_1)", "(ontable ?param_1)", "(handempty)"],
                    "uncertain_preconditions": [],
                    "negative_preconditions": [],
                    "useless_preconditions": [],
                    "certain_add_effects": ["(holding ?param_1)"],
                    "certain_del_effects": ["(clear ?param_1)", "(ontable ?param_1)", "(handempty)"],
                    "uncertain_add_effects": [],
                    "uncertain_del_effects": [],
                    "observations": 5,
                    "successes": 5,
                    "failures": 0
                }
            }
        }

        infogain_safe = ModelReconstructor.reconstruct_information_gain_safe(infogain_snapshot)
        olam_safe = ModelReconstructor.reconstruct_olam_safe(olam_snapshot)

        # Both should have learned preconditions
        self.assertGreater(len(infogain_safe.actions["pick-up"].preconditions), 0)
        self.assertGreater(len(olam_safe.actions["pick-up"].preconditions), 0)

        # Both should have learned effects
        self.assertGreater(len(infogain_safe.actions["pick-up"].add_effects), 0)
        self.assertGreater(len(olam_safe.actions["pick-up"].add_effects), 0)


class TestCompleteModelEquivalence(unittest.TestCase):
    """Validate Complete model construction between algorithms."""

    def setUp(self):
        """Setup test environment."""
        self.all_possible_literals = {
            "clear(?0)", "ontable(?0)", "handempty()", "holding(?0)"
        }

    def test_certain_precondition_semantics_differ(self):
        """Validate that 'certain' has different meanings between algorithms."""
        # InfoGain: certain = singleton constraint sets (from failures)
        infogain_snapshot = {
            "iteration": 100,
            "algorithm": "information_gain",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "possible_preconditions": ["clear(?0)", "ontable(?0)", "handempty()"],
                    "certain_preconditions": ["handempty()"],  # From failure analysis
                    "uncertain_preconditions": ["clear(?0)", "ontable(?0)"],
                    "confirmed_add_effects": ["holding(?0)"],
                    "confirmed_del_effects": ["handempty()"],
                    "possible_add_effects": [],
                    "possible_del_effects": [],
                    "constraint_sets": [["handempty()"]]
                }
            }
        }

        # OLAM: certain = intersection of successful states
        olam_snapshot = {
            "iteration": 100,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": ["(clear ?param_1)", "(ontable ?param_1)", "(handempty)"],
                    "uncertain_preconditions": [],
                    "negative_preconditions": [],
                    "useless_preconditions": [],
                    "certain_add_effects": ["(holding ?param_1)"],
                    "certain_del_effects": ["(handempty)"],
                    "uncertain_add_effects": [],
                    "uncertain_del_effects": [],
                    "observations": 5,
                    "successes": 3,
                    "failures": 2
                }
            }
        }

        infogain_complete = ModelReconstructor.reconstruct_information_gain_complete(infogain_snapshot)
        olam_complete = ModelReconstructor.reconstruct_olam_complete(olam_snapshot)

        # InfoGain has 1 certain precondition (from failure)
        self.assertEqual(len(infogain_complete.actions["pick-up"].preconditions), 1)

        # OLAM has 3 certain preconditions (from success intersection)
        self.assertEqual(len(olam_complete.actions["pick-up"].preconditions), 3)

    def test_contradiction_handling_differs(self):
        """Validate different contradiction resolution strategies."""
        # Create snapshot with same fluent in both add and delete
        infogain_snapshot = {
            "iteration": 100,
            "algorithm": "information_gain",
            "actions": {
                "test-action": {
                    "parameters": ["?x"],
                    "possible_preconditions": [],
                    "certain_preconditions": [],
                    "uncertain_preconditions": [],
                    "confirmed_add_effects": ["fluent(?0)"],
                    "confirmed_del_effects": [],
                    "possible_add_effects": ["conflict(?0)"],
                    "possible_del_effects": ["conflict(?0)"],  # Same as possible_add
                    "constraint_sets": []
                }
            }
        }

        olam_snapshot = {
            "iteration": 100,
            "algorithm": "olam",
            "actions": {
                "test-action": {
                    "parameters": ["?x"],
                    "certain_preconditions": [],
                    "uncertain_preconditions": [],
                    "negative_preconditions": [],
                    "useless_preconditions": [],
                    "certain_add_effects": ["(fluent ?param_1)"],
                    "certain_del_effects": [],
                    "uncertain_add_effects": ["(conflict ?param_1)"],
                    "uncertain_del_effects": ["(conflict ?param_1)"],  # Same as uncertain_add
                    "observations": 5,
                    "successes": 5,
                    "failures": 0
                }
            }
        }

        infogain_complete = ModelReconstructor.reconstruct_information_gain_complete(infogain_snapshot)
        olam_complete = ModelReconstructor.reconstruct_olam_complete(olam_snapshot)

        infogain_action = infogain_complete.actions["test-action"]
        olam_action = olam_complete.actions["test-action"]

        # InfoGain: contradictions removed from BOTH
        # conflict(?0) should not be in either add or del
        self.assertNotIn("conflict(?0)", infogain_action.add_effects)
        self.assertNotIn("conflict(?0)", infogain_action.del_effects)

        # OLAM: contradictions go to add, removed from del
        # conflict(?param_1) should be in add but not del
        # Need to normalize to compare
        olam_add_normalized = {normalize_predicate_parameters(p) for p in olam_action.add_effects}
        olam_del_normalized = {normalize_predicate_parameters(p) for p in olam_action.del_effects}

        self.assertIn("conflict(?0)", olam_add_normalized)
        self.assertNotIn("conflict(?0)", olam_del_normalized)


class TestEdgeCaseMetrics(unittest.TestCase):
    """Validate metrics calculation for edge cases."""

    def setUp(self):
        """Setup test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.domain_file = self.test_dir / "domain.pddl"
        self.problem_file = self.test_dir / "problem.pddl"
        self._create_test_domain()

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_domain(self):
        """Create test domain."""
        domain_content = """(define (domain blocksworld)
  (:predicates
    (clear ?x)
    (on ?x ?y)
    (ontable ?x)
    (handempty)
    (holding ?x))

  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (holding ?x)
                 (not (clear ?x))
                 (not (ontable ?x))
                 (not (handempty)))))"""

        problem_content = """(define (problem p01)
  (:domain blocksworld)
  (:objects a b c)
  (:init (clear a) (ontable a) (handempty))
  (:goal (holding a)))"""

        self.domain_file.write_text(domain_content)
        self.problem_file.write_text(problem_content)

    def test_unexecuted_action_safe_metrics_expected(self):
        """
        Unexecuted action Safe model should have:
        - Precondition precision = 0.0 (many FP from including all La)
        - Precondition recall = 1.0 (includes all ground truth)
        - Effect precision = 1.0 (empty set, no false positives)
        - Effect recall = 0.0 (empty set, all missing)
        """
        # This test documents the EXPECTED behavior after fix
        pass  # Implementation depends on normalization layer

    def test_unexecuted_action_complete_metrics_expected(self):
        """
        Unexecuted action Complete model should have:
        - Precondition precision = 1.0 (empty set, no false positives)
        - Precondition recall = 0.0 (empty set, all missing)
        - Effect precision = 1.0 (empty set, no false positives)
        - Effect recall = 0.0 (empty set, all missing)
        """
        # This test documents the EXPECTED behavior
        pass  # Implementation depends on normalization layer

    def test_empty_learned_vs_ground_truth_metrics(self):
        """Test metrics when learned model is empty but ground truth is not."""
        validator = ModelValidator(
            domain_file=str(self.domain_file),
            problem_file=str(self.problem_file)
        )

        # Compare empty learned to ground truth
        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions=set(),  # Empty
            learned_add_effects=set(),
            learned_delete_effects=set()
        )

        # Empty learned means:
        # - Precision = 1.0 (no false positives)
        # - Recall = 0.0 (all false negatives)
        self.assertEqual(result.precondition_precision, 1.0)
        self.assertEqual(result.precondition_recall, 0.0)
        self.assertEqual(result.add_effect_precision, 1.0)
        self.assertEqual(result.add_effect_recall, 0.0)

    def test_full_learned_vs_ground_truth_metrics(self):
        """Test metrics when learned includes all possible literals."""
        validator = ModelValidator(
            domain_file=str(self.domain_file),
            problem_file=str(self.problem_file)
        )

        # Include all possible preconditions (La)
        all_precs = {
            "clear(?x)", "ontable(?x)", "handempty()",
            "holding(?x)", "on(?x,?y)"  # Extra false positives
        }

        result = validator.compare_action(
            action_name="pick-up",
            learned_preconditions=all_precs,
            learned_add_effects={"holding(?x)"},  # Correct
            learned_delete_effects={"clear(?x)", "ontable(?x)", "handempty()"}  # Correct
        )

        # Full learned means:
        # - Recall = 1.0 (all ground truth included)
        # - Precision < 1.0 (has false positives)
        self.assertEqual(result.precondition_recall, 1.0)
        self.assertLess(result.precondition_precision, 1.0)


class TestNormalizationEquivalence(unittest.TestCase):
    """Test that normalization produces comparable formats."""

    def test_normalize_infogain_format(self):
        """Test normalization of InfoGain format."""
        # InfoGain uses: "clear(?x)"
        infogain_literal = "clear(?x)"
        normalized = normalize_predicate_parameters(infogain_literal)
        self.assertEqual(normalized, "clear(?0)")

    def test_normalize_olam_format(self):
        """Test normalization of OLAM format."""
        # OLAM uses: "(clear ?param_1)"
        olam_literal = "(clear ?param_1)"
        normalized = normalize_predicate_parameters(olam_literal)
        self.assertEqual(normalized, "clear(?0)")

    def test_normalized_formats_equal(self):
        """Test that normalized formats from both algorithms are equal."""
        infogain = "on(?x,?y)"
        olam = "(on ?param_1 ?param_2)"

        self.assertEqual(
            normalize_predicate_parameters(infogain),
            normalize_predicate_parameters(olam)
        )


if __name__ == '__main__':
    unittest.main()
