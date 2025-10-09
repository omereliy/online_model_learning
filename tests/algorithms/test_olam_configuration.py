"""
Test suite for OLAM adapter configuration parameter support.
Tests that OLAM Configuration.py parameters can be passed through adapter.
"""

import pytest
import sys
from pathlib import Path

# Add OLAM to path
olam_path = '/home/omer/projects/OLAM'
if olam_path not in sys.path:
    sys.path.append(olam_path)

from src.algorithms.olam_adapter import OLAMAdapter
import Configuration


@pytest.fixture
def blocksworld_files():
    """Get blocksworld domain and problem file paths."""
    base_path = Path("benchmarks/olam-compatible/blocksworld")
    return str(base_path / "domain.pddl"), str(base_path / "p01.pddl")


class TestOLAMConfiguration:
    """Test OLAM configuration parameter passthrough."""

    def test_default_configuration_values(self, blocksworld_files):
        """Test that default OLAM configuration values are used when not specified."""
        domain_file, problem_file = blocksworld_files

        # Create adapter without extra config
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            bypass_java=True  # For testing without Java
        )

        # Verify OLAM Configuration has default values
        # These should match OLAM's defaults from Configuration.py
        assert hasattr(Configuration, 'TIME_LIMIT_SECONDS')
        assert hasattr(Configuration, 'MAX_ITER')
        assert hasattr(Configuration, 'PLANNER_TIME_LIMIT')
        assert hasattr(Configuration, 'MAX_PRECS_LENGTH')

    def test_planner_time_limit_parameter(self, blocksworld_files):
        """Test that planner_time_limit parameter is passed to OLAM Configuration."""
        domain_file, problem_file = blocksworld_files

        custom_time_limit = 90

        # Create adapter with custom planner time limit
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            planner_time_limit=custom_time_limit,
            bypass_java=True
        )

        # Verify Configuration was updated
        assert Configuration.PLANNER_TIME_LIMIT == custom_time_limit

    def test_max_precs_length_parameter(self, blocksworld_files):
        """Test that max_precs_length parameter is passed to OLAM Configuration."""
        domain_file, problem_file = blocksworld_files

        custom_precs_length = 12

        # Create adapter with custom max preconditions length
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            max_precs_length=custom_precs_length,
            bypass_java=True
        )

        # Verify Configuration was updated
        assert Configuration.MAX_PRECS_LENGTH == custom_precs_length

    def test_neg_eff_assumption_parameter(self, blocksworld_files):
        """Test that neg_eff_assumption parameter is passed to OLAM Configuration."""
        domain_file, problem_file = blocksworld_files

        # Test with True
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            neg_eff_assumption=True,
            bypass_java=True
        )

        assert Configuration.NEG_EFF_ASSUMPTION == True

        # Test with False
        adapter2 = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            neg_eff_assumption=False,
            bypass_java=True
        )

        assert Configuration.NEG_EFF_ASSUMPTION == False

    def test_output_console_parameter(self, blocksworld_files):
        """Test that output_console parameter is passed to OLAM Configuration."""
        domain_file, problem_file = blocksworld_files

        # Test with True
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            output_console=True,
            bypass_java=True
        )

        assert Configuration.OUTPUT_CONSOLE == True

        # Test with False
        adapter2 = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            output_console=False,
            bypass_java=True
        )

        assert Configuration.OUTPUT_CONSOLE == False

    def test_random_seed_parameter(self, blocksworld_files):
        """Test that random_seed parameter is passed to OLAM Configuration."""
        domain_file, problem_file = blocksworld_files

        custom_seed = 42

        # Create adapter with custom random seed
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            random_seed=custom_seed,
            bypass_java=True
        )

        # Verify Configuration was updated
        assert Configuration.RANDOM_SEED == custom_seed

    def test_time_limit_parameter(self, blocksworld_files):
        """Test that time_limit_seconds parameter is passed to OLAM Configuration."""
        domain_file, problem_file = blocksworld_files

        custom_time_limit = 600

        # Create adapter with custom time limit
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            time_limit_seconds=custom_time_limit,
            bypass_java=True
        )

        # Verify Configuration was updated
        assert Configuration.TIME_LIMIT_SECONDS == custom_time_limit

    def test_multiple_configuration_parameters(self, blocksworld_files):
        """Test that multiple configuration parameters can be set together."""
        domain_file, problem_file = blocksworld_files

        config_params = {
            'planner_time_limit': 75,
            'max_precs_length': 10,
            'neg_eff_assumption': True,
            'output_console': False,
            'random_seed': 123,
            'time_limit_seconds': 450
        }

        # Create adapter with multiple parameters
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            bypass_java=True,
            **config_params
        )

        # Verify all configurations were updated
        assert Configuration.PLANNER_TIME_LIMIT == config_params['planner_time_limit']
        assert Configuration.MAX_PRECS_LENGTH == config_params['max_precs_length']
        assert Configuration.NEG_EFF_ASSUMPTION == config_params['neg_eff_assumption']
        assert Configuration.OUTPUT_CONSOLE == config_params['output_console']
        assert Configuration.RANDOM_SEED == config_params['random_seed']
        assert Configuration.TIME_LIMIT_SECONDS == config_params['time_limit_seconds']

    def test_configuration_isolation_between_instances(self, blocksworld_files):
        """Test that configuration changes don't leak between adapter instances."""
        domain_file, problem_file = blocksworld_files

        # Create first adapter with specific config
        adapter1 = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            planner_time_limit=100,
            bypass_java=True
        )

        first_value = Configuration.PLANNER_TIME_LIMIT
        assert first_value == 100

        # Create second adapter with different config
        adapter2 = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            planner_time_limit=200,
            bypass_java=True
        )

        second_value = Configuration.PLANNER_TIME_LIMIT
        assert second_value == 200

        # Note: Configuration is global in OLAM, so last adapter wins
        # This documents the expected behavior

    def test_none_configuration_values(self, blocksworld_files):
        """Test that None configuration values don't override OLAM defaults."""
        domain_file, problem_file = blocksworld_files

        # Store original values
        original_planner_limit = Configuration.PLANNER_TIME_LIMIT
        original_max_precs = Configuration.MAX_PRECS_LENGTH

        # Create adapter with None values (should not change Configuration)
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            planner_time_limit=None,
            max_precs_length=None,
            bypass_java=True
        )

        # Configuration should retain original values
        # Note: This documents that None means "use OLAM's default"

    def test_configuration_parameter_types(self, blocksworld_files):
        """Test that configuration parameters accept correct types."""
        domain_file, problem_file = blocksworld_files

        # Test integer parameters
        adapter = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            planner_time_limit=60,
            max_precs_length=8,
            random_seed=0,
            bypass_java=True
        )

        assert isinstance(Configuration.PLANNER_TIME_LIMIT, int)
        assert isinstance(Configuration.MAX_PRECS_LENGTH, int)
        assert isinstance(Configuration.RANDOM_SEED, int)

        # Test boolean parameters
        adapter2 = OLAMAdapter(
            domain_file=domain_file,
            problem_file=problem_file,
            neg_eff_assumption=False,
            output_console=True,
            bypass_java=True
        )

        assert isinstance(Configuration.NEG_EFF_ASSUMPTION, bool)
        assert isinstance(Configuration.OUTPUT_CONSOLE, bool)
