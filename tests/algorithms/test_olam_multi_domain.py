"""
Test OLAM adapter with multiple domains to ensure it's not hardcoded for blocksworld.
"""
from src.core.pddl_handler import PDDLHandler
from src.algorithms.olam_adapter import OLAMAdapter
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class TestOLAMMultiDomain:
    """Test OLAM adapter with various domains."""

    @pytest.fixture
    def domain_paths(self):
        """Get paths to different domain files."""
        base_path = Path(__file__).parent.parent / "benchmarks"
        return {
            'blocksworld': {
                'domain': base_path / "blocksworld" / "domain.pddl",
                'problem': base_path / "blocksworld" / "p01.pddl",
                'expected_action_types': ['pick-up', 'put-down', 'stack', 'unstack'],
                'expected_min_actions': 10  # At least 10 grounded actions
            },
            'gripper': {
                'domain': base_path / "gripper" / "domain.pddl",
                'problem': base_path / "gripper" / "p01.pddl",
                'expected_action_types': ['move', 'pick', 'drop'],
                'expected_min_actions': 8  # Multiple rooms and balls
            },
            'rover': {
                'domain': base_path / "rover" / "domain.pddl",
                'problem': base_path / "rover" / "p01.pddl",
                'expected_action_types': ['navigate', 'sample_soil', 'sample_rock', 'calibrate', 'take_image'],
                'expected_min_actions': 20  # Complex domain
            },
            'logistics': {
                'domain': base_path / "logistics" / "domain.pddl",
                'problem': base_path / "logistics" / "p01.pddl",
                'expected_action_types': ['load-truck', 'unload-truck', 'drive-truck', 'fly-airplane'],
                'expected_min_actions': 15  # Multiple vehicles and locations
            }
        }

    def test_olam_grounding_blocksworld(self, domain_paths):
        """Test that OLAM correctly grounds blocksworld actions."""
        domain_info = domain_paths['blocksworld']

        # Initialize PDDLHandler
        pddl_handler = PDDLHandler()
        pddl_handler.parse_domain_and_problem(
            str(domain_info['domain']),
            str(domain_info['problem'])
        )

        # Initialize OLAM with PDDLHandler
        adapter = OLAMAdapter(
            str(domain_info['domain']),
            str(domain_info['problem']),
            pddl_handler=pddl_handler
        )

        # Get grounded actions
        actions = adapter._extract_action_list()

        # Verify we have actions
        assert len(actions) >= domain_info['expected_min_actions'], \
            f"Expected at least {domain_info['expected_min_actions']} actions, got {len(actions)}"

        # Check that expected action types are present
        action_types_found = set()
        for action in actions:
            action_name = action.split('(')[0]
            action_types_found.add(action_name)

        for expected_type in domain_info['expected_action_types']:
            assert expected_type in action_types_found, \
                f"Expected action type '{expected_type}' not found in {action_types_found}"

        # Check format is correct (should have parentheses)
        for action in actions:
            assert '(' in action and ')' in action, f"Action '{action}' not in correct format"

    def test_olam_grounding_gripper(self, domain_paths):
        """Test that OLAM correctly grounds gripper domain actions."""
        domain_info = domain_paths['gripper']

        # Initialize PDDLHandler
        pddl_handler = PDDLHandler()
        pddl_handler.parse_domain_and_problem(
            str(domain_info['domain']),
            str(domain_info['problem'])
        )

        # Initialize OLAM with PDDLHandler
        adapter = OLAMAdapter(
            str(domain_info['domain']),
            str(domain_info['problem']),
            pddl_handler=pddl_handler
        )

        # Get grounded actions
        actions = adapter._extract_action_list()

        # Verify we have actions
        assert len(actions) >= domain_info['expected_min_actions'], \
            f"Expected at least {domain_info['expected_min_actions']} actions, got {len(actions)}"

        # Check that we have multi-parameter actions (pick has 3 params)
        pick_actions = [a for a in actions if a.startswith('pick(')]
        assert len(pick_actions) > 0, "No pick actions found"

        # Pick should have parameters (ball, room, gripper)
        for pick_action in pick_actions[:1]:  # Check at least one
            params = pick_action[5:-1].split(',')  # Extract params from pick(...)
            assert len(
                params) == 3, f"Pick action should have 3 parameters, got {len(params)}: {pick_action}"

    def test_olam_grounding_rover(self, domain_paths):
        """Test that OLAM correctly grounds rover domain actions."""
        domain_info = domain_paths['rover']

        if not domain_info['domain'].exists():
            pytest.skip("Rover domain not found")

        # Initialize PDDLHandler
        pddl_handler = PDDLHandler()
        pddl_handler.parse_domain_and_problem(
            str(domain_info['domain']),
            str(domain_info['problem'])
        )

        # Initialize OLAM with PDDLHandler
        adapter = OLAMAdapter(
            str(domain_info['domain']),
            str(domain_info['problem']),
            pddl_handler=pddl_handler
        )

        # Get grounded actions
        actions = adapter._extract_action_list()

        # Verify we have actions
        assert len(actions) >= domain_info['expected_min_actions'], \
            f"Expected at least {domain_info['expected_min_actions']} actions, got {len(actions)}"

        # Check navigate actions have multiple parameters
        navigate_actions = [a for a in actions if a.startswith('navigate(')]
        if navigate_actions:
            # Navigate should have parameters (rover, from, to)
            params = navigate_actions[0][9:-1].split(',')
            assert len(
                params) >= 3, f"Navigate should have at least 3 parameters: {navigate_actions[0]}"

    def test_olam_grounding_logistics(self, domain_paths):
        """Test that OLAM correctly grounds logistics domain actions."""
        domain_info = domain_paths['logistics']

        if not domain_info['domain'].exists():
            pytest.skip("Logistics domain not found")

        # Initialize PDDLHandler
        pddl_handler = PDDLHandler()
        pddl_handler.parse_domain_and_problem(
            str(domain_info['domain']),
            str(domain_info['problem'])
        )

        # Initialize OLAM with PDDLHandler
        adapter = OLAMAdapter(
            str(domain_info['domain']),
            str(domain_info['problem']),
            pddl_handler=pddl_handler
        )

        # Get grounded actions
        actions = adapter._extract_action_list()

        # Verify we have actions
        assert len(actions) >= domain_info['expected_min_actions'], \
            f"Expected at least {domain_info['expected_min_actions']} actions, got {len(actions)}"

        # Check drive-truck actions have multiple parameters
        drive_actions = [a for a in actions if a.startswith('drive-truck(')]
        if drive_actions:
            # drive-truck should have parameters (truck, from, to, city)
            params = drive_actions[0][12:-1].split(',')
            assert len(params) == 4, f"drive-truck should have 4 parameters: {drive_actions[0]}"

    def test_action_format_consistency(self, domain_paths):
        """Test that action format is consistent across all domains."""
        for domain_name, domain_info in domain_paths.items():
            if not domain_info['domain'].exists():
                continue

            # Initialize PDDLHandler
            pddl_handler = PDDLHandler()
            pddl_handler.parse_domain_and_problem(
                str(domain_info['domain']),
                str(domain_info['problem'])
            )

            # Initialize OLAM with PDDLHandler
            adapter = OLAMAdapter(
                str(domain_info['domain']),
                str(domain_info['problem']),
                pddl_handler=pddl_handler
            )

            # Get grounded actions
            actions = adapter._extract_action_list()

            # All actions should follow format: name(params) or name()
            for action in actions:
                assert '(' in action, f"Action '{action}' missing opening parenthesis in {domain_name}"
                assert ')' in action, f"Action '{action}' missing closing parenthesis in {domain_name}"
                assert action.endswith(')'), f"Action '{action}' should end with ) in {domain_name}"

                # Check no spaces in parameter list
                if '(' in action and ')' in action:
                    params_str = action[action.index('(') + 1:action.rindex(')')]
                    if params_str:  # If there are parameters
                        assert ' ' not in params_str, f"Spaces in parameters: '{action}' in {domain_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
