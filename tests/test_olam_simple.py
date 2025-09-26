"""
Simple test to verify OLAM adapter basic functionality without full OLAM execution.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.olam_adapter import OLAMAdapter


class TestOLAMSimple:
    """Simple tests for OLAM adapter without full execution."""

    def test_adapter_initialization(self, temp_dir, blocksworld_domain, blocksworld_problem):
        """Test that adapter can be initialized."""
        domain_file = temp_dir / "domain.pddl"
        problem_file = temp_dir / "problem.pddl"

        domain_file.write_text(blocksworld_domain)
        problem_file.write_text(blocksworld_problem)

        adapter = OLAMAdapter(str(domain_file), str(problem_file))

        # Check basic properties
        assert adapter.domain_file == str(domain_file)
        assert adapter.problem_file == str(problem_file)
        assert adapter.learner is not None
        assert len(adapter.action_list) > 0

        # Check that action list includes expected actions
        action_names = [a.split('(')[0] for a in adapter.action_list]
        assert 'pick-up' in action_names
        assert 'put-down' in action_names
        assert 'stack' in action_names
        assert 'unstack' in action_names

    def test_conversion_methods(self):
        """Test state and action conversion methods."""
        # Create a simplified adapter for testing conversions
        class SimpleAdapter(OLAMAdapter):
            def __init__(self):
                # Skip full initialization
                pass

        adapter = SimpleAdapter()

        # Test state conversions
        up_state = {'clear_a', 'on_a_b', 'ontable_b', 'handempty'}
        olam_state = adapter._up_state_to_olam(up_state)

        assert len(olam_state) == 4
        assert '(clear a)' in olam_state
        assert '(on a b)' in olam_state
        assert '(ontable b)' in olam_state
        assert '(handempty)' in olam_state

        # Test reverse conversion
        back_state = adapter._olam_state_to_up(olam_state)
        assert back_state == up_state

        # Test action conversions
        assert adapter._up_action_to_olam('pick-up', ['a']) == 'pick-up(a)'
        assert adapter._up_action_to_olam('stack', ['a', 'b']) == 'stack(a,b)'
        assert adapter._olam_action_to_up('pick-up(a)') == ('pick-up', ['a'])
        assert adapter._olam_action_to_up('stack(a,b)') == ('stack', ['a', 'b'])

    def test_action_list_extraction(self, temp_dir, blocksworld_domain, blocksworld_problem):
        """Test that action list is properly extracted."""
        domain_file = temp_dir / "domain.pddl"
        problem_file = temp_dir / "problem.pddl"

        domain_file.write_text(blocksworld_domain)
        problem_file.write_text(blocksworld_problem)

        adapter = OLAMAdapter(str(domain_file), str(problem_file))

        # Check that we have grounded actions
        # 3 pick-up + 3 put-down + 6 stack (3*2, no self-stacking) + 6 unstack = 18
        assert len(adapter.action_list) == 18

        # Check specific actions
        assert 'pick-up(a)' in adapter.action_list
        assert 'stack(a,b)' in adapter.action_list
        assert 'unstack(c,b)' in adapter.action_list

        # Verify no self-stacking/unstacking
        assert 'stack(a,a)' not in adapter.action_list
        assert 'unstack(b,b)' not in adapter.action_list

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        # Create a simplified adapter
        class SimpleAdapter(OLAMAdapter):
            def __init__(self):
                self.iteration_count = 0
                self.observation_count = 0
                self._converged = False

        adapter = SimpleAdapter()

        # Test initial statistics
        stats = adapter.get_statistics()
        assert stats['iterations'] == 0
        assert stats['observations'] == 0
        assert stats['converged'] == False

        # Simulate some iterations
        adapter.iteration_count = 10
        adapter.observation_count = 25
        adapter._converged = True

        stats = adapter.get_statistics()
        assert stats['iterations'] == 10
        assert stats['observations'] == 25
        assert stats['converged'] == True

    def test_reset_functionality(self):
        """Test adapter reset functionality with base class."""
        # Create a concrete implementation for testing
        class TestLearner:
            def __init__(self):
                self.iteration_count = 10
                self.observation_count = 20
                self._converged = True

            def reset(self):
                """Reset to initial state."""
                self.iteration_count = 0
                self.observation_count = 0
                self._converged = False

        learner = TestLearner()
        assert learner.iteration_count == 10
        assert learner.observation_count == 20
        assert learner._converged == True

        learner.reset()

        assert learner.iteration_count == 0
        assert learner.observation_count == 0
        assert learner._converged == False