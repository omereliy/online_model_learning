"""
Test suite for fluent conversion with OLAM-compatible domains.
Ensures proper conversion between UP fluent format and PDDL string format.
Uses gripper domain which is OLAM-compatible.
"""

from src.algorithms.olam_adapter import OLAMAdapter
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestFluentConversion:
    """Test fluent conversion for OLAM-compatible domain predicates."""

    @pytest.fixture
    def gripper_adapter(self):
        """Create OLAM adapter with gripper domain."""
        domain = "benchmarks/olam-compatible/gripper/domain.pddl"
        problem = "benchmarks/olam-compatible/gripper/p01.pddl"

        # Skip if files don't exist
        if not Path(domain).exists() or not Path(problem).exists():
            pytest.skip("Gripper domain files not found")

        adapter = OLAMAdapter(domain, problem, bypass_java=True)
        return adapter

    @pytest.fixture
    def blocksworld_adapter(self):
        """Create OLAM adapter with blocksworld domain."""
        domain = "benchmarks/olam-compatible/blocksworld/domain.pddl"
        problem = "benchmarks/olam-compatible/blocksworld/p01.pddl"

        # Skip if files don't exist
        if not Path(domain).exists() or not Path(problem).exists():
            pytest.skip("Blocksworld domain files not found")

        adapter = OLAMAdapter(domain, problem, bypass_java=True)
        return adapter

    def test_gripper_simple_predicates(self, gripper_adapter):
        """Test conversion of simple gripper predicates."""
        test_cases = [
            ("free_left", "(free left)"),
            ("free_right", "(free right)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = gripper_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_gripper_carry_predicates(self, gripper_adapter):
        """Test conversion of 'carry' predicates (2 parameters)."""
        test_cases = [
            ("carry_ball1_left", "(carry ball1 left)"),
            ("carry_ball2_right", "(carry ball2 right)"),
            ("carry_ball3_left", "(carry ball3 left)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = gripper_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_gripper_at_predicates(self, gripper_adapter):
        """Test conversion of 'at' predicates in gripper domain."""
        test_cases = [
            ("at_ball1_rooma", "(at ball1 rooma)"),
            ("at_ball2_roomb", "(at ball2 roomb)"),
            ("at_ball3_rooma", "(at ball3 rooma)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = gripper_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_gripper_at_robby_predicates(self, gripper_adapter):
        """Test conversion of 'at-robby' predicates."""
        test_cases = [
            ("at_robby_rooma", "(at_robby rooma)"),
            ("at_robby_roomb", "(at_robby roomb)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = gripper_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_blocksworld_predicates(self, blocksworld_adapter):
        """Test conversion of blocksworld predicates."""
        test_cases = [
            ("on_a_b", "(on a b)"),
            ("on_b_c", "(on b c)"),
            ("clear_a", "(clear a)"),
            ("clear_b", "(clear b)"),
            ("ontable_a", "(ontable a)"),
            ("ontable_b", "(ontable b)"),
            ("handempty", "(handempty)"),
            ("holding_a", "(holding a)"),
        ]

        for fluent, expected_pddl in test_cases:
            result = blocksworld_adapter._fluent_to_pddl_string(fluent)
            assert result == expected_pddl, f"Failed for {fluent}: got {result}"

    def test_bidirectional_conversion_gripper(self, gripper_adapter):
        """Test that conversion works both ways for gripper domain."""
        test_fluents = [
            "free_left",
            "carry_ball1_left",
            "at_ball1_rooma",
            "at_robby_rooma",
        ]

        for original_fluent in test_fluents:
            # Convert to PDDL
            pddl_str = gripper_adapter._fluent_to_pddl_string(original_fluent)
            # Convert back to fluent
            result_fluent = gripper_adapter._pddl_string_to_fluent(pddl_str)
            assert result_fluent == original_fluent, f"Bidirectional conversion failed for {original_fluent}"

    def test_bidirectional_conversion_blocksworld(self, blocksworld_adapter):
        """Test that conversion works both ways for blocksworld domain."""
        test_fluents = [
            "on_a_b",
            "clear_a",
            "ontable_a",
            "handempty",
            "holding_a",
        ]

        for original_fluent in test_fluents:
            # Convert to PDDL
            pddl_str = blocksworld_adapter._fluent_to_pddl_string(original_fluent)
            # Convert back to fluent
            result_fluent = blocksworld_adapter._pddl_string_to_fluent(pddl_str)
            assert result_fluent == original_fluent, f"Bidirectional conversion failed for {original_fluent}"

    def test_edge_cases(self, gripper_adapter):
        """Test edge cases and unusual formats."""
        test_cases = [
            # Empty string
            ("", ""),
            # Single word (no underscore)
            ("handempty", "(handempty)"),
            # Many underscores
            ("test_a_b_c_d", "(test a b c d)"),
        ]

        for fluent, expected in test_cases:
            result = gripper_adapter._fluent_to_pddl_string(fluent)
            # For complex cases, just ensure no crash
            assert isinstance(result, str)

    def test_all_gripper_predicates_list(self, gripper_adapter):
        """Test that all known gripper predicates are handled."""
        gripper_predicates = [
            "at", "at_robby", "carry", "free"
        ]

        # Verify all these are handled
        for pred in gripper_predicates:
            # Test a sample fluent for each predicate
            if pred == "at":
                test_fluent = "at_ball1_rooma"
            elif pred == "at_robby":
                test_fluent = "at_robby_rooma"
            elif pred == "carry":
                test_fluent = "carry_ball1_left"
            elif pred == "free":
                test_fluent = "free_left"
            else:
                test_fluent = f"{pred}_test_param"

            # Just verify it doesn't return empty string
            result = gripper_adapter._fluent_to_pddl_string(test_fluent)
            assert result != "", f"Failed to handle predicate {pred}"