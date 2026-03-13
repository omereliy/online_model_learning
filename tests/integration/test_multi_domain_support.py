"""
Test suite for multi-domain support in the online model learning framework.
Tests various PDDL domains to ensure the framework works beyond blocksworld.
"""
from information_gain_aml.experiments.metrics import MetricsCollector
from information_gain_aml.core.pddl_io import parse_pddl
from information_gain_aml.core import grounding
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Project-root benchmarks directory
BENCHMARKS_PATH = Path(__file__).parent.parent.parent / "benchmarks"


class TestMultiDomainSupport:
    """Test that all domains are properly supported."""

    @pytest.fixture
    def benchmarks_path(self):
        """Path to benchmarks directory."""
        return BENCHMARKS_PATH

    @pytest.mark.parametrize("domain_name,expected_actions,compatibility",
                             [("blocksworld",
                               ["pick-up",
                                "put-down",
                                "stack",
                                "unstack"],
                                 "olam-compatible"),
                                 ("gripper",
                                  ["move",
                                   "pick",
                                   "drop"],
                                  "olam-compatible"),
                                 ("rover",
                                  ["navigate",
                                   "sample_soil",
                                   "sample_rock",
                                   "drop",
                                   "calibrate",
                                   "take_image",
                                   "communicate_soil_data",
                                   "communicate_rock_data",
                                   "communicate_image_data"],
                                  "olam-compatible"),
                                 ("depots",
                                  ["drive",
                                   "lift",
                                   "drop",
                                   "load",
                                   "unload"],
                                  "olam-compatible")])
    def test_domain_parsing(self, benchmarks_path, domain_name, expected_actions, compatibility):
        """Test that each domain can be parsed correctly."""
        domain_file = benchmarks_path / compatibility / domain_name / "domain.pddl"
        problem_file = benchmarks_path / compatibility / domain_name / "p01.pddl"

        # Check files exist
        assert domain_file.exists(), f"Domain file missing: {domain_file}"
        assert problem_file.exists(), f"Problem file missing: {problem_file}"

        # Parse the domain
        domain, initial_state = parse_pddl(str(domain_file), str(problem_file))

        # Check actions can be parsed
        action_names = list(domain.lifted_actions.keys())
        for action in expected_actions:
            assert action in action_names, f"Missing action {action} in {domain_name}"

    @pytest.mark.parametrize("domain_name,compatibility", [
        ("blocksworld", "olam-compatible"), ("gripper", "olam-compatible"),
        ("rover", "olam-compatible"),
        ("depots", "olam-compatible")
    ])
    def test_domain_grounding(self, benchmarks_path, domain_name, compatibility):
        """Test that domains can be grounded with their problem files."""
        domain_file = benchmarks_path / compatibility / domain_name / "domain.pddl"
        problem_file = benchmarks_path / compatibility / domain_name / "p01.pddl"

        domain, initial_state = parse_pddl(str(domain_file), str(problem_file))

        # Get grounded actions
        grounded_actions = grounding.ground_all_actions(domain, require_injective=False)
        assert len(grounded_actions) > 0, f"No grounded actions for {domain_name}"

        # Get initial state
        assert len(initial_state) > 0, f"Empty initial state for {domain_name}"

    @pytest.mark.parametrize("domain_name,min_actions", [
        ("blocksworld", 10),  # At least 10 grounded actions expected
        ("gripper", 8),       # Move between rooms + pick/drop actions
        ("rover", 20),        # Complex domain with many actions
        ("depots", 10)        # Logistics-like domain
    ])
    def test_domain_complexity(self, benchmarks_path, domain_name, min_actions):
        """Test that domains have reasonable complexity."""
        for compat in ["olam-compatible"]:
            domain_file = benchmarks_path / compat / domain_name / "domain.pddl"
            problem_file = benchmarks_path / compat / domain_name / "p01.pddl"
            if domain_file.exists():
                break
        else:
            pytest.skip(f"Domain {domain_name} not found")

        domain, _ = parse_pddl(str(domain_file), str(problem_file))

        grounded_actions = grounding.ground_all_actions(domain, require_injective=False)
        assert len(grounded_actions) >= min_actions, \
            f"{domain_name} has only {len(grounded_actions)} actions, expected at least {min_actions}"

    def test_metrics_collector_multi_domain(self):
        """Test that metrics collector works with different action types."""
        collector = MetricsCollector()

        # Simulate actions from different domains
        domain_actions = [
            # Blocksworld actions
            ("pick-up", ["a"], True),
            ("stack", ["a", "b"], True),
            ("unstack", ["c", "d"], False),
            # Gripper actions
            ("move", ["room-a", "room-b"], True),
            ("pick", ["ball1", "room-a", "left"], True),
            ("drop", ["ball1", "room-b", "left"], True),
            # Rover actions
            ("navigate", ["rover1", "wp1", "wp2"], True),
            ("sample_soil", ["rover1", "store1", "wp1"], True),
            ("communicate_soil_data", ["rover1", "lander1", "wp1", "wp2", "wp3"], False),
        ]

        for i, (action, objects, success) in enumerate(domain_actions):
            collector.record_action(i, action, objects, success, 0.01)

        # Check metrics
        stats = collector.get_summary_statistics()
        assert stats['total_actions'] == len(domain_actions)
        assert stats['failed_actions'] == 2  # Two failures

        # Check per-action distribution
        action_dist = collector.get_action_distribution()
        assert 'pick-up' in action_dist
        assert action_dist['pick-up'] == 1
        assert 'move' in action_dist
        assert action_dist['move'] == 1


class TestDomainSpecificFeatures:
    """Test domain-specific features and characteristics."""

    def test_gripper_two_grippers(self):
        """Test that gripper domain correctly handles two grippers."""
        domain_file = BENCHMARKS_PATH / "olam-compatible/gripper/domain.pddl"
        problem_file = BENCHMARKS_PATH / "olam-compatible/gripper/p01.pddl"

        domain, _ = parse_pddl(str(domain_file), str(problem_file))

        # Check that gripper type is defined
        gripper_objects = [name for name, obj_info in domain.objects.items()
                           if obj_info.type.lower() == 'gripper']
        assert len(gripper_objects) == 2, "Should have exactly 2 grippers"
        assert 'lgripper1' in gripper_objects
        assert 'rgripper1' in gripper_objects

    def test_rover_specialized_equipment(self):
        """Test that rover domain has specialized equipment predicates."""
        domain_file = BENCHMARKS_PATH / "olam-compatible/rover/domain.pddl"
        problem_file = BENCHMARKS_PATH / "olam-compatible/rover/p01.pddl"

        domain, _ = parse_pddl(str(domain_file), str(problem_file))

        # Check for equipment fluents in domain
        fluent_names = list(domain.predicates.keys())

        # Check for equipment predicates
        equipment_preds = ["equipped_for_soil_analysis",
                           "equipped_for_rock_analysis",
                           "equipped_for_imaging"]

        for equip in equipment_preds:
            assert equip in fluent_names, f"Missing equipment predicate: {equip}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
