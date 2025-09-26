"""
Basic integration test for OLAM adapter.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.olam_adapter import OLAMAdapter


class TestOLAMIntegration:
    """Basic integration tests for OLAM adapter."""

    def test_basic_workflow(self, sample_domain_problem):
        """Test basic workflow: initialize, select action, observe."""
        domain_file, problem_file = sample_domain_problem

        # Initialize adapter
        adapter = OLAMAdapter(domain_file, problem_file)
        assert adapter is not None

        # Test initial state
        initial_state = {'clear_a', 'on_a_b', 'on_b_c', 'ontable_c', 'handempty'}

        # Select an action
        action, objects = adapter.select_action(initial_state)
        assert isinstance(action, str)
        assert isinstance(objects, list)
        print(f"Selected action: {action}({','.join(objects)})")

        # Simulate successful execution
        next_state = initial_state.copy()
        if action == 'unstack' and objects == ['a', 'b']:
            # Successful unstack
            next_state = {'holding_a', 'clear_b', 'on_b_c', 'ontable_c'}
            adapter.observe(initial_state, action, objects, True, next_state)

        # Get learned model
        model = adapter.get_learned_model()
        assert 'actions' in model
        assert 'predicates' in model
        assert 'statistics' in model

        # Check statistics
        stats = adapter.get_statistics()
        assert stats['iterations'] == 1
        assert stats['observations'] == 1

    def test_state_conversion(self):
        """Test state format conversions."""
        from src.algorithms.olam_adapter import OLAMAdapter

        # Create a mock adapter to test conversion methods
        # We'll use a simplified initialization to avoid full OLAM setup
        class TestAdapter(OLAMAdapter):
            def __init__(self):
                # Skip full initialization
                pass

        adapter = TestAdapter()

        # Test fluent to PDDL conversion
        assert adapter._fluent_to_pddl_string('clear_a') == '(clear a)'
        assert adapter._fluent_to_pddl_string('on_a_b') == '(on a b)'
        assert adapter._fluent_to_pddl_string('handempty') == '(handempty)'

        # Test PDDL to fluent conversion
        assert adapter._pddl_string_to_fluent('(clear a)') == 'clear_a'
        assert adapter._pddl_string_to_fluent('(on a b)') == 'on_a_b'
        assert adapter._pddl_string_to_fluent('(handempty)') == 'handempty'

        # Test state conversion
        up_state = {'clear_a', 'on_a_b', 'handempty'}
        olam_state = adapter._up_state_to_olam(up_state)
        assert '(clear a)' in olam_state
        assert '(on a b)' in olam_state
        assert '(handempty)' in olam_state

        # Test reverse conversion
        converted_back = adapter._olam_state_to_up(olam_state)
        assert converted_back == up_state

    def test_action_conversion(self):
        """Test action format conversions."""
        from src.algorithms.olam_adapter import OLAMAdapter

        class TestAdapter(OLAMAdapter):
            def __init__(self):
                # Skip full initialization
                pass

        adapter = TestAdapter()

        # Test action to OLAM format
        assert adapter._up_action_to_olam('pick-up', ['a']) == 'pick-up(a)'
        assert adapter._up_action_to_olam('stack', ['a', 'b']) == 'stack(a,b)'
        assert adapter._up_action_to_olam('handempty', []) == 'handempty()'

        # Test OLAM to action format
        assert adapter._olam_action_to_up('pick-up(a)') == ('pick-up', ['a'])
        assert adapter._olam_action_to_up('stack(a,b)') == ('stack', ['a', 'b'])
        assert adapter._olam_action_to_up('handempty()') == ('handempty', [])

    @pytest.fixture
    def sample_domain_problem(self, temp_dir, blocksworld_domain, blocksworld_problem):
        """Create temporary PDDL files for testing."""
        domain_file = temp_dir / "domain.pddl"
        problem_file = temp_dir / "problem.pddl"

        domain_file.write_text(blocksworld_domain)
        problem_file.write_text(blocksworld_problem)

        return str(domain_file), str(problem_file)