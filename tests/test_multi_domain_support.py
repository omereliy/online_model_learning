"""
Test suite for multi-domain support in the online model learning framework.
Tests various PDDL domains to ensure the framework works beyond blocksworld.
"""
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pddl_handler import PDDLHandler
from src.experiments.metrics import MetricsCollector


class TestMultiDomainSupport:
    """Test that all domains are properly supported."""

    @pytest.fixture
    def benchmarks_path(self):
        """Path to benchmarks directory."""
        return Path(__file__).parent.parent / "benchmarks"

    @pytest.mark.parametrize("domain_name,expected_actions,compatibility", [
        ("blocksworld", ["pick-up", "put-down", "stack", "unstack"], "olam-compatible"),
        ("gripper", ["move", "pick", "drop"], "olam-compatible"),
        ("logistics", ["load-truck", "unload-truck", "drive-truck", "load-airplane", "unload-airplane", "fly-airplane"], "olam-compatible"),
        ("rover", ["navigate", "sample_soil", "sample_rock", "drop", "calibrate",
                   "take_image", "communicate_soil_data", "communicate_rock_data",
                   "communicate_image_data"], "olam-incompatible"),
        ("depots", ["drive", "lift", "drop", "load", "unload"], "olam-compatible")
    ])
    def test_domain_parsing(self, benchmarks_path, domain_name, expected_actions, compatibility):
        """Test that each domain can be parsed correctly."""
        domain_file = benchmarks_path / compatibility / domain_name / "domain.pddl"
        problem_file = benchmarks_path / compatibility / domain_name / "p01.pddl"

        # Check files exist
        assert domain_file.exists(), f"Domain file missing: {domain_file}"
        assert problem_file.exists(), f"Problem file missing: {problem_file}"

        # Parse the domain
        handler = PDDLHandler()
        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        # Check actions can be parsed
        lifted_actions = handler.problem.actions if handler.problem else []
        action_names = [a.name for a in lifted_actions]
        for action in expected_actions:
            assert action in action_names, f"Missing action {action} in {domain_name}"

    @pytest.mark.parametrize("domain_name,compatibility", [
        ("blocksworld", "olam-compatible"), ("gripper", "olam-compatible"),
        ("logistics", "olam-compatible"), ("rover", "olam-incompatible"),
        ("depots", "olam-compatible")
    ])
    def test_domain_grounding(self, benchmarks_path, domain_name, compatibility):
        """Test that domains can be grounded with their problem files."""
        domain_file = benchmarks_path / compatibility / domain_name / "domain.pddl"
        problem_file = benchmarks_path / compatibility / domain_name / "p01.pddl"

        handler = PDDLHandler()
        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        # Get grounded actions
        grounded_actions = handler.get_grounded_actions()
        assert len(grounded_actions) > 0, f"No grounded actions for {domain_name}"

        # Get initial state
        initial_state = handler.get_initial_state()
        assert len(initial_state) > 0, f"Empty initial state for {domain_name}"

        # Get goal
        goal = handler.get_goal()
        assert goal is not None, f"No goal for {domain_name}"

    @pytest.mark.parametrize("domain_name,min_actions", [
        ("blocksworld", 10),  # At least 10 grounded actions expected
        ("gripper", 8),       # Move between rooms + pick/drop actions
        ("logistics", 15),    # Multiple vehicles and locations
        ("rover", 20),        # Complex domain with many actions
        ("depots", 10)        # Logistics-like domain
    ])
    def test_domain_complexity(self, benchmarks_path, domain_name, min_actions):
        """Test that domains have reasonable complexity."""
        domain_file = benchmarks_path / domain_name / "domain.pddl"
        problem_file = benchmarks_path / domain_name / "p01.pddl"

        handler = PDDLHandler()
        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        grounded_actions = handler.get_grounded_actions()
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
            # Logistics actions
            ("load-truck", ["pkg1", "truck1", "loc1"], True),
            ("drive-truck", ["truck1", "loc1", "loc2", "city1"], True),
            ("unload-truck", ["pkg1", "truck1", "loc2"], False),
        ]

        for i, (action, objects, success) in enumerate(domain_actions):
            collector.record_action(i, action, objects, success, 0.01)

        # Check metrics
        stats = collector.get_summary_statistics()
        assert stats['total_actions'] == len(domain_actions)
        assert stats['cumulative_mistakes'] == 2  # Two failures

        # Check per-action distribution
        action_dist = collector.get_action_distribution()
        assert 'pick-up' in action_dist
        assert action_dist['pick-up'] == 1
        assert 'move' in action_dist
        assert action_dist['move'] == 1

    @pytest.mark.parametrize("config_file", [
        "experiment_blocksworld.yaml",
        "experiment_gripper.yaml",
        "experiment_logistics.yaml",
        "experiment_rover.yaml",
        "experiment_depots.yaml"
    ])
    def test_experiment_configs_exist(self, config_file):
        """Test that experiment configuration files exist for all domains."""
        config_path = Path(__file__).parent.parent / "configs" / config_file
        assert config_path.exists(), f"Config file missing: {config_path}"

        # Try to load the config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Basic validation
        if 'experiment' in config:
            assert 'domain' in config['experiment'] or 'domain' in config.get('domain_problem', {})
            assert 'problem' in config['experiment'] or 'problem' in config.get('domain_problem', {})
        elif 'domain_problem' in config:
            assert 'domain' in config['domain_problem']
            assert 'problem' in config['domain_problem']


class TestDomainSpecificFeatures:
    """Test domain-specific features and characteristics."""

    def test_gripper_two_grippers(self):
        """Test that gripper domain correctly handles two grippers."""
        handler = PDDLHandler()
        domain_file = Path(__file__).parent.parent / "benchmarks/gripper/domain.pddl"
        problem_file = Path(__file__).parent.parent / "benchmarks/gripper/p01.pddl"

        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        # Check that gripper type is defined
        if handler.problem:
            gripper_objects = [o.name for o in handler.problem.objects
                             if str(o.type).lower() == 'gripper']
            assert len(gripper_objects) == 2, "Should have exactly 2 grippers"
            assert 'left' in gripper_objects
            assert 'right' in gripper_objects

    def test_logistics_vehicles(self):
        """Test that logistics domain has both trucks and airplanes."""
        handler = PDDLHandler()
        domain_file = Path(__file__).parent.parent / "benchmarks/logistics/domain.pddl"
        problem_file = Path(__file__).parent.parent / "benchmarks/logistics/p01.pddl"

        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        # Check vehicle types in problem
        if handler.problem:
            trucks = [o.name for o in handler.problem.objects
                     if str(o.type).lower() == 'truck']
            airplanes = [o.name for o in handler.problem.objects
                        if str(o.type).lower() == 'airplane']

            assert len(trucks) > 0, "Should have at least one truck"
            assert len(airplanes) > 0, "Should have at least one airplane"

    def test_rover_specialized_equipment(self):
        """Test that rover domain has specialized equipment predicates."""
        handler = PDDLHandler()
        domain_file = Path(__file__).parent.parent / "benchmarks/rover/domain.pddl"
        problem_file = Path(__file__).parent.parent / "benchmarks/rover/p01.pddl"

        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        # Check for equipment fluents in domain
        if handler.problem and handler.problem.fluents:
            fluent_names = [f.name for f in handler.problem.fluents]

            # Check for equipment predicates
            equipment_preds = ["equipped_for_soil_analysis",
                             "equipped_for_rock_analysis",
                             "equipped_for_imaging"]

            for equip in equipment_preds:
                assert equip in fluent_names, f"Missing equipment predicate: {equip}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])