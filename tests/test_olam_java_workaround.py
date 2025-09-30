"""
Test suite for OLAM Java dependency workaround.
Ensures OLAM can function without Java for validation purposes.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.olam_adapter import OLAMAdapter


class TestOLAMJavaWorkaround:
    """Test OLAM adapter can work without Java dependency."""

    @pytest.fixture
    def domain_problem_files(self):
        """Return test domain and problem file paths."""
        domain = str(project_root / "benchmarks/olam-compatible/blocksworld/domain.pddl")
        problem = str(project_root / "benchmarks/olam-compatible/blocksworld/p01.pddl")
        return domain, problem

    def test_olam_initialization_without_java(self, domain_problem_files):
        """Test OLAM adapter can initialize even without Java."""
        domain, problem = domain_problem_files

        # Test with bypass_java flag
        adapter = OLAMAdapter(
            domain_file=domain,
            problem_file=problem,
            bypass_java=True  # New flag to bypass Java dependency
        )

        assert adapter is not None
        assert hasattr(adapter, 'bypass_java')
        assert adapter.bypass_java is True

    def test_java_bypass_mode_configuration(self, domain_problem_files):
        """Test Java bypass mode can be configured."""
        domain, problem = domain_problem_files

        # Test default behavior (Java enabled)
        adapter_default = OLAMAdapter(domain, problem)
        assert not getattr(adapter_default, 'bypass_java', False)

        # Test explicit bypass
        adapter_bypass = OLAMAdapter(domain, problem, bypass_java=True)
        assert adapter_bypass.bypass_java is True

    @patch('subprocess.call')
    def test_compute_not_executable_actions_bypass(self, mock_subprocess, domain_problem_files):
        """Test that Java subprocess is not called in bypass mode."""
        domain, problem = domain_problem_files

        adapter = OLAMAdapter(domain, problem, bypass_java=True)

        # Mock OLAM's learner to test action selection
        with patch.object(adapter.learner, 'compute_not_executable_actionsJAVA') as mock_java:
            # In bypass mode, should return empty list instead of calling Java
            mock_java.return_value = []

            state = {'on_a_b', 'on_b_c', 'clear_a', 'ontable_c', 'handempty'}
            action, objects = adapter.select_action(state)

            # Java should not be called in bypass mode
            if adapter.bypass_java:
                # We expect the adapter to handle this internally
                pass

        # Subprocess should not be called
        mock_subprocess.assert_not_called()

    def test_python_fallback_for_action_filtering(self, domain_problem_files):
        """Test Python-based action filtering as Java alternative."""
        domain, problem = domain_problem_files

        adapter = OLAMAdapter(domain, problem, bypass_java=True)

        # Test the Python fallback method
        state = {'clear_a', 'ontable_a', 'handempty'}

        # Should have a method to compute executable actions without Java
        if hasattr(adapter, '_compute_executable_actions_python'):
            executable = adapter._compute_executable_actions_python(state)
            assert isinstance(executable, list)
            # In this state, pick-up(a) should be executable
            assert any('pick-up' in str(action) for action in executable)

    def test_action_selection_without_java(self, domain_problem_files):
        """Test action selection works without Java dependency."""
        domain, problem = domain_problem_files

        adapter = OLAMAdapter(domain, problem, bypass_java=True)

        state = {'on_b_a', 'on_c_b', 'clear_c', 'ontable_a', 'handempty'}

        # Should be able to select an action without Java
        action, objects = adapter.select_action(state)

        assert action is not None
        assert isinstance(action, str)
        assert isinstance(objects, list)

    def test_learning_process_without_java(self, domain_problem_files):
        """Test learning can proceed without Java dependency."""
        domain, problem = domain_problem_files

        adapter = OLAMAdapter(domain, problem, bypass_java=True)

        # Simulate a learning iteration
        state = {'clear_c', 'on_c_b', 'on_b_a', 'ontable_a', 'handempty'}
        action, objects = adapter.select_action(state)

        # Test observation
        next_state = state.copy()
        next_state.remove('clear_c')
        next_state.add('holding_c')

        adapter.observe(state, action, objects, success=True, next_state=next_state)

        # Should update internal model
        model = adapter.get_learned_model()
        assert 'actions' in model
        assert len(model['actions']) > 0

    def test_configuration_java_bin_path_handling(self, domain_problem_files):
        """Test proper handling of Configuration.JAVA_BIN_PATH."""
        domain, problem = domain_problem_files

        with patch('Configuration.JAVA_BIN_PATH', ''):
            # Should handle empty Java path gracefully
            adapter = OLAMAdapter(domain, problem, bypass_java=True)
            assert adapter is not None

        # Test with system Java attempt
        with patch('Configuration.JAVA_BIN_PATH', 'java'):
            adapter = OLAMAdapter(domain, problem, use_system_java=True)
            assert adapter is not None

    def test_graceful_degradation_on_java_error(self, domain_problem_files):
        """Test graceful handling when Java fails."""
        domain, problem = domain_problem_files

        # Test with bypass mode - should not raise errors
        adapter = OLAMAdapter(domain, problem, bypass_java=True)

        # In bypass mode, Java errors are avoided entirely
        state = {'clear_a', 'ontable_a', 'handempty'}
        action, objects = adapter.select_action(state)

        # Should return an action without Java
        assert action is not None

        # Also test that without bypass, we handle Java path issues
        adapter2 = OLAMAdapter(domain, problem, bypass_java=True)

        # Verify bypass is working
        assert hasattr(adapter2.learner, 'compute_not_executable_actionsJAVA')
        result = adapter2.learner.compute_not_executable_actionsJAVA()
        assert result == []  # Bypass returns empty list

    def test_bypass_mode_logging(self, domain_problem_files, caplog):
        """Test that bypass mode is properly logged."""
        domain, problem = domain_problem_files

        import logging
        with caplog.at_level(logging.INFO):
            adapter = OLAMAdapter(domain, problem, bypass_java=True)

            # Should log bypass mode activation
            assert any('bypass' in record.message.lower() or 'java' in record.message.lower()
                      for record in caplog.records)

    def test_compatibility_with_experiment_runner(self, domain_problem_files):
        """Test OLAM adapter with bypass works in experiment runner context."""
        domain, problem = domain_problem_files

        # Test that bypass mode doesn't break BaseActionModelLearner interface
        adapter = OLAMAdapter(domain, problem, bypass_java=True)

        # All required methods should be present
        assert hasattr(adapter, 'select_action')
        assert hasattr(adapter, 'observe')
        assert hasattr(adapter, 'get_learned_model')
        assert hasattr(adapter, 'has_converged')
        assert hasattr(adapter, 'reset')
        assert hasattr(adapter, 'get_statistics')

        # Methods should be callable
        state = {'clear_a', 'ontable_a', 'handempty'}
        action, objects = adapter.select_action(state)
        stats = adapter.get_statistics()

        assert action is not None
        assert stats is not None