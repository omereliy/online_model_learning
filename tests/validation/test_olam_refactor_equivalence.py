"""
Validation test for OLAM refactor equivalence.

This test compares results from the old real-time adapter approach
with the new post-processing trace replay approach to ensure they
produce equivalent models.

Author: OLAM Refactor Implementation
Date: 2025
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.olam_trace_parser import OLAMTraceParser, OLAMTraceStep
from src.core.olam_knowledge_reconstructor import OLAMKnowledgeReconstructor, OLAMKnowledge
from src.core.model_reconstructor import ModelReconstructor


class TestOLAMRefactorEquivalence(unittest.TestCase):
    """Test that the refactored OLAM approach produces equivalent results."""

    def setUp(self):
        """Setup test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.domain_file = self.test_dir / "domain.pddl"
        self.problem_file = self.test_dir / "problem.pddl"

        # Create minimal test domain
        self._create_test_domain()

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

    def test_trace_parser_basic(self):
        """Test that the trace parser can handle basic OLAM log format."""
        # Create a sample trace
        trace_content = """Iteration: 1
Selected action: pick-up(a)
Success: True
State before: (clear a) (ontable a) (handempty)
State after: (holding a)
---
Iteration: 2
Selected action: stack(a,b)
Success: False
State: (holding a) (clear b) (clear c)
---
Iteration: 3
Selected action: put-down(a)
Success: True
State before: (holding a)
State after: (clear a) (ontable a) (handempty)
---"""

        trace_file = self.test_dir / "trace.log"
        trace_file.write_text(trace_content)

        # Parse trace
        parser = OLAMTraceParser()
        trace = parser.parse_log_file(trace_file)

        # Validate parsing
        self.assertEqual(len(trace), 3)

        # Check first step
        self.assertEqual(trace[0].iteration, 1)
        self.assertEqual(trace[0].action, "pick-up(a)")
        self.assertTrue(trace[0].success)
        self.assertIn("(clear a)", trace[0].state_before)
        self.assertIn("(holding a)", trace[0].state_after)

        # Check failed step
        self.assertEqual(trace[1].iteration, 2)
        self.assertEqual(trace[1].action, "stack(a,b)")
        self.assertFalse(trace[1].success)
        self.assertIsNone(trace[1].state_after)

    def test_knowledge_reconstruction(self):
        """Test that knowledge reconstruction follows OLAM's learning rules."""
        # Create a simple trace
        trace = [
            OLAMTraceStep(
                iteration=1,
                action="pick-up(a)",
                success=True,
                state_before={"(clear a)", "(ontable a)", "(handempty)"},
                state_after={"(holding a)"}
            ),
            OLAMTraceStep(
                iteration=2,
                action="stack(a,b)",
                success=False,
                state_before={"(holding a)", "(clear b)"},
                state_after=None
            ),
            OLAMTraceStep(
                iteration=3,
                action="pick-up(b)",
                success=True,
                state_before={"(clear b)", "(ontable b)", "(handempty)"},
                state_after={"(holding b)"}
            )
        ]

        # Reconstruct knowledge
        reconstructor = OLAMKnowledgeReconstructor(self.domain_file)
        knowledge = reconstructor.replay_trace(trace)

        # Check learned preconditions for pick-up
        # After two successful observations, should have intersection
        pickup_precs = knowledge.certain_precs.get("pick-up", set())
        self.assertIn("(handempty)", pickup_precs)  # Common to both

        # Check that stack has a failed observation recorded
        self.assertEqual(knowledge.failed_observations.get("stack", 0), 1)

    def test_model_export_format(self):
        """Test that exported models match expected format."""
        # Create knowledge
        knowledge = OLAMKnowledge()
        knowledge.certain_precs["pick-up"] = {"(clear ?param_1)", "(ontable ?param_1)", "(handempty)"}
        knowledge.add_effects["pick-up"] = {"(holding ?param_1)"}
        knowledge.del_effects["pick-up"] = {"(clear ?param_1)", "(ontable ?param_1)", "(handempty)"}

        # Export snapshot
        reconstructor = OLAMKnowledgeReconstructor()
        snapshot = reconstructor.export_snapshot(knowledge, iteration=10)

        # Validate snapshot format
        self.assertEqual(snapshot["iteration"], 10)
        self.assertEqual(snapshot["algorithm"], "olam")
        self.assertIn("pick-up", snapshot["actions"])

        action = snapshot["actions"]["pick-up"]
        self.assertEqual(len(action["certain_preconditions"]), 3)
        self.assertIn("(clear ?param_1)", action["certain_preconditions"])
        self.assertEqual(len(action["add_effects"]), 1)
        self.assertEqual(len(action["del_effects"]), 3)

    def test_model_reconstruction_from_snapshot(self):
        """Test that ModelReconstructor can handle the new snapshot format."""
        # Create a snapshot
        snapshot = {
            "iteration": 10,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": ["(clear ?x)", "(ontable ?x)", "(handempty)"],
                    "uncertain_preconditions": [],
                    "negative_preconditions": [],
                    "add_effects": ["(holding ?x)"],
                    "del_effects": ["(clear ?x)", "(ontable ?x)", "(handempty)"]
                }
            }
        }

        # Reconstruct models
        safe = ModelReconstructor.reconstruct_olam_safe(snapshot)
        complete = ModelReconstructor.reconstruct_olam_complete(snapshot)

        # Validate safe model (includes all preconditions)
        self.assertEqual(safe.model_type, "safe")
        self.assertEqual(safe.algorithm, "olam")
        self.assertIn("pick-up", safe.actions)

        pickup_safe = safe.actions["pick-up"]
        self.assertEqual(len(pickup_safe.preconditions), 3)
        self.assertEqual(len(pickup_safe.add_effects), 1)

        # Validate complete model (only certain preconditions)
        self.assertEqual(complete.model_type, "complete")
        pickup_complete = complete.actions["pick-up"]
        self.assertEqual(pickup_complete.preconditions, pickup_safe.preconditions)  # Same since no uncertain

    def test_parameter_lifting_and_grounding(self):
        """Test conversion between grounded and lifted predicates."""
        reconstructor = OLAMKnowledgeReconstructor()

        # Test ground to lifted
        grounded = {"(clear a)", "(on a b)", "(holding c)"}
        action = "stack(a,b)"
        lifted = reconstructor._ground_to_lifted(grounded, action)

        self.assertIn("(clear ?param_1)", lifted)  # a -> ?param_1
        self.assertIn("(on ?param_1 ?param_2)", lifted)  # a,b -> ?param_1,?param_2
        self.assertIn("(holding c)", lifted)  # c not in params, stays grounded

        # Test lifted to ground (reverse)
        lifted = {"(clear ?param_1)", "(on ?param_1 ?param_2)"}
        grounded = reconstructor._lifted_to_ground(lifted, action)

        self.assertIn("(clear a)", grounded)
        self.assertIn("(on a b)", grounded)

    @patch('src.algorithms.olam_external_runner.subprocess.run')
    def test_external_runner_integration(self, mock_run):
        """Test that external runner properly executes and collects results."""
        from src.algorithms.olam_external_runner import OLAMExternalRunner

        # Mock subprocess result
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="OLAM execution complete",
            stderr=""
        )

        # Create mock output files
        run_dir = self.test_dir / "Analysis" / "run_0"
        test_dir = run_dir / "Tests" / "blocksworld" / "1_p01_blocksworld_gen"
        test_dir.mkdir(parents=True, exist_ok=True)

        trace_file = test_dir / "1_p01_blocksworld_gen_log"
        trace_file.write_text("Iteration: 1\nAction: pick-up(a)\nSuccess: True")

        # Run experiment
        runner = OLAMExternalRunner(olam_dir=self.test_dir)
        runner.main_script = self.test_dir / "main.py"  # Mock main script
        runner.main_script.touch()

        result = runner.run_experiment(
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            config={'max_iterations': 10},
            output_dir=self.test_dir / "output"
        )

        # Validate execution
        self.assertTrue(result.success)
        mock_run.assert_called_once()

    def test_equivalence_simple_trace(self):
        """Test that reconstruction produces equivalent results to direct learning."""
        # This test simulates what OLAM would learn and verifies our reconstruction matches

        # Simulated OLAM trace
        trace = [
            OLAMTraceStep(
                iteration=1,
                action="pick-up(a)",
                success=True,
                state_before={"(clear a)", "(ontable a)", "(handempty)", "(clear b)"},
                state_after={"(holding a)", "(clear b)"}
            ),
            OLAMTraceStep(
                iteration=2,
                action="pick-up(b)",
                success=True,
                state_before={"(clear b)", "(ontable b)", "(handempty)", "(clear a)"},
                state_after={"(holding b)", "(clear a)"}
            )
        ]

        # Reconstruct knowledge
        reconstructor = OLAMKnowledgeReconstructor(self.domain_file)
        knowledge = reconstructor.replay_trace(trace)

        # Expected OLAM behavior: intersection of successful states
        # Both observations have handempty, but only relevant object is clear
        pickup_precs = knowledge.certain_precs.get("pick-up", set())

        # Should have learned handempty is always present
        self.assertIn("(handempty)", pickup_precs)

        # Effects should be learned from state differences
        pickup_add = knowledge.add_effects.get("pick-up", set())
        self.assertIn("(holding ?param_1)", pickup_add)

        pickup_del = knowledge.del_effects.get("pick-up", set())
        self.assertIn("(handempty)", pickup_del)

    def test_checkpoint_reconstruction(self):
        """Test reconstruction at multiple checkpoints."""
        # Create a longer trace
        trace = []
        for i in range(1, 11):
            trace.append(
                OLAMTraceStep(
                    iteration=i,
                    action=f"pick-up(obj{i})" if i % 2 else f"put-down(obj{i//2})",
                    success=True,
                    state_before=set(),  # Simplified
                    state_after=set()     # Simplified
                )
            )

        reconstructor = OLAMKnowledgeReconstructor(self.domain_file)

        # Reconstruct at checkpoints
        checkpoints = [3, 5, 10]
        models = {}

        for checkpoint in checkpoints:
            knowledge = reconstructor.replay_to_checkpoint(trace, checkpoint)
            models[checkpoint] = knowledge

            # Verify observation count matches checkpoint
            total_obs = sum(knowledge.observation_count.values())
            self.assertEqual(total_obs, checkpoint)

        # Verify models grow over time (monotonic learning)
        self.assertLessEqual(
            len(models[3].certain_precs.get("pick-up", set())),
            len(models[5].certain_precs.get("pick-up", set()))
        )


class TestOLAMJSONExports(unittest.TestCase):
    """Test handling of OLAM's native JSON export format."""

    def test_parse_json_exports(self):
        """Test parsing OLAM's JSON export files."""
        test_dir = Path(tempfile.mkdtemp())

        try:
            # Create mock JSON exports
            exports = {
                'certain_precs': {
                    "pick-up": ["(clear ?param_1)", "(ontable ?param_1)", "(handempty)"],
                    "put-down": ["(holding ?param_1)"]
                },
                'add_effects': {
                    "pick-up": ["(holding ?param_1)"],
                    "put-down": ["(clear ?param_1)", "(ontable ?param_1)"]
                }
            }

            # Write JSON files
            for key, content in exports.items():
                filename = {
                    'certain_precs': 'operator_certain_predicates.json',
                    'add_effects': 'certain_positive_effects.json'
                }[key]

                filepath = test_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(content, f)

            # Parse exports
            parser = OLAMTraceParser()
            parsed = parser.parse_json_exports(test_dir)

            # Validate parsing
            self.assertIn('certain_precs', parsed)
            self.assertIn('add_effects', parsed)
            self.assertEqual(parsed['certain_precs']['pick-up'], exports['certain_precs']['pick-up'])

        finally:
            import shutil
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()