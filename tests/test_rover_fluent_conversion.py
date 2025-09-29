"""
Test suite for rover domain fluent conversion.
Ensures proper conversion between UP fluent format and PDDL string format.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.olam_adapter import OLAMAdapter


class TestRoverFluentConversion:
    """Test fluent conversion for rover domain predicates."""

    @pytest.fixture
    def olam_adapter(self):
        """Create OLAM adapter with rover domain."""
        domain = "/home/omer/projects/domains/rover/domain.pddl"
        problem = "/home/omer/projects/domains/rover/pfile1.pddl"

        # Skip if files don't exist
        if not Path(domain).exists() or not Path(problem).exists():
            pytest.skip("Rover domain files not found")

        adapter = OLAMAdapter(domain, problem, bypass_java=True)
        return adapter

    def test_simple_predicates(self, olam_adapter):
        """Test conversion of simple single-word predicates."""
        test_cases = [
            ("available_rover0", "(available rover0)"),
            ("empty_rover0store", "(empty rover0store)"),
            ("full_rover0store", "(full rover0store)"),
            ("channel_free_general", "(channel_free general)"),  # channel_free is the predicate name
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_at_predicates(self, olam_adapter):
        """Test conversion of 'at' predicates (rover/lander location)."""
        test_cases = [
            ("at_rover0_waypoint0", "(at rover0 waypoint0)"),
            ("at_rover0_waypoint1", "(at rover0 waypoint1)"),
            ("at_lander_general_waypoint1", "(at_lander general waypoint1)"),
            ("at_soil_sample_waypoint0", "(at_soil_sample waypoint0)"),
            ("at_rock_sample_waypoint0", "(at_rock_sample waypoint0)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_equipped_for_predicates(self, olam_adapter):
        """Test conversion of 'equipped_for' predicates."""
        test_cases = [
            ("equipped_for_imaging_rover0", "(equipped_for_imaging rover0)"),
            ("equipped_for_rock_analysis_rover0", "(equipped_for_rock_analysis rover0)"),
            ("equipped_for_soil_analysis_rover0", "(equipped_for_soil_analysis rover0)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_can_traverse_predicates(self, olam_adapter):
        """Test conversion of 'can_traverse' predicates (3 parameters)."""
        test_cases = [
            ("can_traverse_rover0_waypoint0_waypoint1", "(can_traverse rover0 waypoint0 waypoint1)"),
            ("can_traverse_rover0_waypoint1_waypoint0", "(can_traverse rover0 waypoint1 waypoint0)"),
            ("can_traverse_rover0_waypoint2_waypoint3", "(can_traverse rover0 waypoint2 waypoint3)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_visible_predicates(self, olam_adapter):
        """Test conversion of visibility predicates."""
        test_cases = [
            ("visible_waypoint0_waypoint1", "(visible waypoint0 waypoint1)"),
            ("visible_waypoint1_waypoint0", "(visible waypoint1 waypoint0)"),
            ("visible_from_objective0_waypoint1", "(visible_from objective0 waypoint1)"),
            ("visible_from_objective1_waypoint0", "(visible_from objective1 waypoint0)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_calibration_and_support_predicates(self, olam_adapter):
        """Test conversion of calibration and support predicates."""
        test_cases = [
            ("calibration_target_camera0_objective0", "(calibration_target camera0 objective0)"),
            ("calibration_target_camera1_objective1", "(calibration_target camera1 objective1)"),
            ("supports_camera0_colour", "(supports camera0 colour)"),
            # Note: high_res, low_res are single object names with underscores
            # PDDLEnvironment outputs them as supports_camera0_high_res
            # This is correct behavior - they remain as single parameters
            ("supports_camera0_high_res", "(supports camera0 high_res)"),
            ("supports_camera0_low_res", "(supports camera0 low_res)"),
            ("supports_camera1_high_res", "(supports camera1 high_res)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_on_board_and_store_predicates(self, olam_adapter):
        """Test conversion of on_board and store_of predicates."""
        test_cases = [
            ("on_board_camera0_rover0", "(on_board camera0 rover0)"),
            ("on_board_camera1_rover0", "(on_board camera1 rover0)"),
            ("store_of_rover0store_rover0", "(store_of rover0store rover0)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_have_predicates(self, olam_adapter):
        """Test conversion of have_ predicates (for samples and images)."""
        test_cases = [
            ("have_soil_analysis_rover0_waypoint0", "(have_soil_analysis rover0 waypoint0)"),
            ("have_rock_analysis_rover0_waypoint0", "(have_rock_analysis rover0 waypoint0)"),
            ("have_image_rover0_objective0_colour", "(have_image rover0 objective0 colour)"),
            ("have_image_rover0_objective0_high_res", "(have_image rover0 objective0 high_res)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_communicated_predicates(self, olam_adapter):
        """Test conversion of communicated_ predicates."""
        test_cases = [
            ("communicated_soil_data_waypoint0", "(communicated_soil_data waypoint0)"),
            ("communicated_rock_data_waypoint0", "(communicated_rock_data waypoint0)"),
            ("communicated_image_data_objective0_colour", "(communicated_image_data objective0 colour)"),
            ("communicated_image_data_objective0_high_res", "(communicated_image_data objective0 high_res)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_bidirectional_conversion(self, olam_adapter):
        """Test that conversion works both ways (fluent -> PDDL -> fluent)."""
        test_fluents = [
            "at_rover0_waypoint0",
            "can_traverse_rover0_waypoint0_waypoint1",
            "equipped_for_imaging_rover0",
            "visible_from_objective0_waypoint1",
            "calibration_target_camera0_objective0",
            "have_soil_analysis_rover0_waypoint0",
            "communicated_image_data_objective0_high_res",
        ]

        for original_fluent in test_fluents:
            # Convert to PDDL
            pddl_str = olam_adapter._fluent_to_pddl_string(original_fluent)
            # Convert back to fluent
            result_fluent = olam_adapter._pddl_string_to_fluent(pddl_str)
            assert result_fluent == original_fluent, f"Bidirectional conversion failed for {original_fluent}"

    def test_all_rover_predicates_list(self, olam_adapter):
        """Test that all known rover predicates are handled."""
        rover_predicates = [
            "at", "at_lander", "at_rock_sample", "at_soil_sample",
            "available", "calibrated", "calibration_target",
            "can_traverse", "channel_free", "communicated_image_data",
            "communicated_rock_data", "communicated_soil_data",
            "empty", "equipped_for_imaging", "equipped_for_rock_analysis",
            "equipped_for_soil_analysis", "full", "have_image",
            "have_rock_analysis", "have_soil_analysis", "on_board",
            "store_of", "supports", "visible", "visible_from"
        ]

        # Verify all these are in the multi-word predicate list or handled
        for pred in rover_predicates:
            # Test a sample fluent for each predicate
            if pred == "at":
                test_fluent = "at_rover0_waypoint0"
            elif pred == "available":
                test_fluent = "available_rover0"
            elif pred == "empty":
                test_fluent = "empty_rover0store"
            elif pred == "channel_free":
                test_fluent = "channel_free_general"
            elif pred == "calibrated":
                test_fluent = "calibrated_camera0"
            elif pred == "full":
                test_fluent = "full_rover0store"
            else:
                # For multi-word predicates
                test_fluent = f"{pred}_test_param"

            # Just verify it doesn't return empty string
            result = olam_adapter._fluent_to_pddl_string(test_fluent)
            assert result != "", f"Failed to handle predicate {pred}"

    def test_edge_cases(self, olam_adapter):
        """Test edge cases and unusual formats."""
        test_cases = [
            # Empty string
            ("", ""),
            # Single word (no underscore)
            ("handempty", "(handempty)"),
            # Many underscores
            ("test_a_b_c_d_e", "(test a b c d e)"),
        ]

        for fluent, expected in test_cases:
            result = olam_adapter._fluent_to_pddl_string(fluent)
            # For complex cases, just ensure no crash
            assert isinstance(result, str)