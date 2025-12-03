"""
Integration tests for the post-processing model reconstruction and analysis pipeline.

Tests the full workflow:
1. Model export during experiments
2. Model reconstruction from snapshots
3. Metrics calculation
4. Visualization generation
"""

import pytest
import json
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Set, List
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.model_reconstructor import ModelReconstructor, ReconstructedModel
from src.algorithms.information_gain import InformationGainLearner
from src.experiments.runner import CHECKPOINTS


class TestModelExport:
    """Test model export functionality."""

    def test_information_gain_export(self, tmp_path):
        """Test that Information Gain learner exports model snapshots correctly."""
        # Setup learner
        domain_file = "benchmarks/olam-compatible/blocksworld/domain.pddl"
        problem_file = "benchmarks/olam-compatible/blocksworld/p00.pddl"

        if not Path(domain_file).exists() or not Path(problem_file).exists():
            pytest.skip("Benchmark files not found")

        learner = InformationGainLearner(domain_file, problem_file, max_iterations=10)

        # Export model at iteration 5
        learner.export_model_snapshot(5, tmp_path)

        # Verify file exists
        snapshot_file = tmp_path / "models" / "model_iter_005.json"
        assert snapshot_file.exists(), "Model snapshot not exported"

        # Verify JSON structure
        with open(snapshot_file, 'r') as f:
            snapshot = json.load(f)

        assert snapshot["iteration"] == 5
        assert snapshot["algorithm"] == "information_gain"
        assert "actions" in snapshot
        assert "metadata" in snapshot

        # Check action structure
        for action_name, action_data in snapshot["actions"].items():
            assert "parameters" in action_data
            assert "possible_preconditions" in action_data
            assert "certain_preconditions" in action_data
            assert "confirmed_add_effects" in action_data
            assert "confirmed_del_effects" in action_data

    def test_checkpoint_iterations(self):
        """Test that checkpoint iterations are properly defined."""
        assert len(CHECKPOINTS) == 21, "Should have 21 checkpoint iterations"
        assert CHECKPOINTS[0] == 5, "First checkpoint should be at iteration 5"
        assert CHECKPOINTS[-1] == 400, "Last checkpoint should be at iteration 400"
        assert all(CHECKPOINTS[i] < CHECKPOINTS[i+1] for i in range(len(CHECKPOINTS)-1)), \
            "Checkpoints should be in ascending order"


class TestModelReconstruction:
    """Test model reconstruction from snapshots."""

    def create_test_snapshot(self, algorithm: str = "information_gain") -> Dict:
        """Create a test snapshot for reconstruction."""
        return {
            "iteration": 50,
            "algorithm": algorithm,
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "possible_preconditions": ["clear(?x)", "ontable(?x)", "handempty"],
                    "certain_preconditions": ["clear(?x)", "handempty"],
                    "uncertain_preconditions": ["ontable(?x)"],
                    "confirmed_add_effects": ["holding(?x)"],
                    "confirmed_del_effects": ["handempty"],
                    "possible_add_effects": ["holding(?x)", "lifted(?x)"],
                    "possible_del_effects": ["handempty", "clear(?x)", "ontable(?x)"]
                }
            },
            "metadata": {
                "domain": "blocksworld",
                "problem": "p00"
            }
        }

    def test_information_gain_safe_reconstruction(self, tmp_path):
        """Test safe model reconstruction for Information Gain."""
        # Create test snapshot
        snapshot = self.create_test_snapshot()
        snapshot_file = tmp_path / "test_snapshot.json"

        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f)

        # Reconstruct models
        models = ModelReconstructor.load_and_reconstruct(snapshot_file)

        # Check we got both models
        assert len(models) == 2
        safe_model = models[0]
        complete_model = models[1]

        # Verify safe model
        assert safe_model.model_type == "safe"
        assert safe_model.algorithm == "information_gain"
        assert safe_model.iteration == 50

        pick_up = safe_model.actions["pick-up"]
        # Safe model should have all possible preconditions (certain + uncertain)
        assert pick_up.preconditions == {"clear(?x)", "ontable(?x)", "handempty"}
        # Safe model should have only confirmed effects
        assert pick_up.add_effects == {"holding(?x)"}
        assert pick_up.del_effects == {"handempty"}

    def test_information_gain_complete_reconstruction(self, tmp_path):
        """Test complete model reconstruction for Information Gain."""
        # Create test snapshot
        snapshot = self.create_test_snapshot()
        snapshot_file = tmp_path / "test_snapshot.json"

        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f)

        # Reconstruct models
        models = ModelReconstructor.load_and_reconstruct(snapshot_file)
        complete_model = models[1]

        # Verify complete model
        assert complete_model.model_type == "complete"
        assert complete_model.algorithm == "information_gain"

        pick_up = complete_model.actions["pick-up"]
        # Complete model should have only certain preconditions
        assert pick_up.preconditions == {"clear(?x)", "handempty"}
        # Complete model should have confirmed + possible effects
        expected_add = {"holding(?x)", "lifted(?x)"}
        expected_del = {"handempty", "clear(?x)", "ontable(?x)"}
        assert pick_up.add_effects == expected_add
        assert pick_up.del_effects == expected_del

    def test_contradiction_removal(self):
        """Test that contradictions are properly removed."""
        # Create sets with contradictions
        add_effects = {"on(?x,?y)", "clear(?x)", "holding(?x)"}
        del_effects = {"clear(?y)", "holding(?x)", "handempty"}  # holding(?x) is contradiction

        filtered_add, filtered_del = ModelReconstructor._remove_contradictions(
            add_effects, del_effects
        )

        # holding(?x) should be removed from both
        assert "holding(?x)" not in filtered_add
        assert "holding(?x)" not in filtered_del
        assert "on(?x,?y)" in filtered_add
        assert "clear(?y)" in filtered_del

    def test_olam_reconstruction(self, tmp_path):
        """Test OLAM model reconstruction."""
        snapshot = {
            "iteration": 30,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": ["clear(?x)", "handempty"],
                    "uncertain_preconditions": ["ontable(?x)"],
                    "add_effects": [],  # OLAM doesn't easily expose effects
                    "del_effects": []
                }
            }
        }

        snapshot_file = tmp_path / "olam_snapshot.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f)

        models = ModelReconstructor.load_and_reconstruct(snapshot_file)

        # Check safe model
        safe_model = models[0]
        assert safe_model.actions["pick-up"].preconditions == {"clear(?x)", "handempty", "ontable(?x)"}

        # Check complete model
        complete_model = models[1]
        assert complete_model.actions["pick-up"].preconditions == {"clear(?x)", "handempty"}


class TestMetricsCalculation:
    """Test precision/recall calculation from reconstructed models."""

    def test_precision_recall_calculation(self):
        """Test basic precision/recall calculation."""
        from scripts.recalculate_model_metrics import calculate_precision_recall

        learned = {"a", "b", "c"}
        ground_truth = {"a", "b", "d"}

        metrics = calculate_precision_recall(learned, ground_truth)

        # TP=2 (a,b), FP=1 (c), FN=1 (d)
        assert abs(metrics["precision"] - 2/3) < 0.01  # 2/(2+1) = 0.667
        assert abs(metrics["recall"] - 2/3) < 0.01     # 2/(2+1) = 0.667
        assert abs(metrics["f1"] - 2/3) < 0.01         # 2*0.667*0.667/(0.667+0.667)

    def test_empty_sets_handling(self):
        """Test handling of empty sets in metrics calculation."""
        from scripts.recalculate_model_metrics import calculate_precision_recall

        # Both empty (perfect match)
        metrics = calculate_precision_recall(set(), set())
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

        # Learned empty, ground truth non-empty
        metrics = calculate_precision_recall(set(), {"a", "b"})
        assert metrics["precision"] == 0.0  # No false positives but also no true positives
        assert metrics["recall"] == 0.0      # Missing all ground truth

        # Learned non-empty, ground truth empty
        metrics = calculate_precision_recall({"a", "b"}, set())
        assert metrics["precision"] == 0.0  # All are false positives
        assert metrics["recall"] == 1.0     # Nothing to recall


class TestFullPipeline:
    """Test the complete post-processing pipeline."""

    def test_pipeline_with_mock_data(self, tmp_path):
        """Test full pipeline with mock experiment data."""
        # Create mock experiment directory structure
        exp_dir = tmp_path / "experiment"
        algo_dir = exp_dir / "information_gain"
        domain_dir = algo_dir / "blocksworld"
        problem_dir = domain_dir / "p00"
        models_dir = problem_dir / "models"
        models_dir.mkdir(parents=True)

        # Create mock snapshots at checkpoints
        for iteration in [5, 10, 15]:
            snapshot = {
                "iteration": iteration,
                "algorithm": "information_gain",
                "actions": {
                    "pick-up": {
                        "parameters": ["?x"],
                        "possible_preconditions": ["clear(?x)", "handempty"],
                        "certain_preconditions": ["clear(?x)"] if iteration > 10 else [],
                        "confirmed_add_effects": ["holding(?x)"],
                        "confirmed_del_effects": ["handempty"],
                        "possible_add_effects": [],
                        "possible_del_effects": []
                    }
                }
            }

            snapshot_file = models_dir / f"model_iter_{iteration:03d}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f)

        # Verify snapshots can be loaded
        from scripts.recalculate_model_metrics import find_model_snapshots
        snapshots = find_model_snapshots(problem_dir)
        assert len(snapshots) == 3

        # Verify reconstruction works
        for snapshot_path in snapshots:
            models = ModelReconstructor.load_and_reconstruct(snapshot_path)
            assert len(models) == 2
            assert models[0].model_type == "safe"
            assert models[1].model_type == "complete"


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_missing_snapshot_file(self):
        """Test handling of missing snapshot files."""
        with pytest.raises(FileNotFoundError):
            ModelReconstructor.load_and_reconstruct(Path("nonexistent.json"))

    def test_corrupted_json(self, tmp_path):
        """Test handling of corrupted JSON files."""
        bad_file = tmp_path / "bad.json"
        with open(bad_file, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            ModelReconstructor.load_and_reconstruct(bad_file)

    def test_unknown_algorithm(self, tmp_path):
        """Test handling of unknown algorithm in snapshot."""
        snapshot = {
            "iteration": 10,
            "algorithm": "unknown_algorithm",
            "actions": {}
        }

        snapshot_file = tmp_path / "unknown.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f)

        with pytest.raises(ValueError, match="Unknown algorithm"):
            ModelReconstructor.load_and_reconstruct(snapshot_file)

    def test_missing_fields_graceful_handling(self, tmp_path):
        """Test graceful handling of missing fields in snapshots."""
        # OLAM snapshot with missing effects (expected)
        snapshot = {
            "iteration": 10,
            "algorithm": "olam",
            "actions": {
                "pick-up": {
                    "parameters": ["?x"],
                    "certain_preconditions": ["clear(?x)"],
                    "uncertain_preconditions": []
                    # Missing add_effects and del_effects
                }
            }
        }

        snapshot_file = tmp_path / "partial.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f)

        # Should handle gracefully with empty sets
        models = ModelReconstructor.load_and_reconstruct(snapshot_file)
        assert len(models) == 2
        assert models[0].actions["pick-up"].add_effects == set()
        assert models[0].actions["pick-up"].del_effects == set()


@pytest.fixture
def cleanup_test_files():
    """Cleanup any test files created during tests."""
    yield
    # Cleanup happens after test
    test_dirs = [
        Path("test_output"),
        Path("checkpoint_metrics.csv"),
        Path("test_checkpoint_metrics.csv")
    ]
    for path in test_dirs:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from src.core.model_reconstructor import ModelReconstructor
        from src.algorithms.information_gain import InformationGainLearner
        from src.experiments.runner import ExperimentRunner
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])