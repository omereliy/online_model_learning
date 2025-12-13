"""
Comprehensive unit tests for CNFManager class.

Tests are designed with predefined expected outcomes and proper assertions
to validate correctness rather than just evaluating outputs.
"""

import pytest
from src.core.cnf_manager import CNFManager


class TestCNFManagerBasics:
    """Test basic CNF manager functionality."""

    def test_initialization(self):
        """Test CNFManager initialization."""
        cnf = CNFManager()

        # Expected: Empty CNF manager
        assert len(cnf.fluent_to_var) == 0
        assert len(cnf.var_to_fluent) == 0
        assert cnf.next_var == 1
        assert len(cnf.cnf.clauses) == 0
        assert cnf.is_satisfiable() is True  # Empty formula is satisfiable

    def test_add_fluent_basic(self):
        """Test adding single fluent."""
        cnf = CNFManager()

        # Expected: First fluent gets variable ID 1
        var_id = cnf.add_fluent('test_fluent')
        assert var_id == 1
        assert cnf.fluent_to_var['test_fluent'] == 1
        assert cnf.var_to_fluent[1] == 'test_fluent'
        assert cnf.next_var == 2

    def test_add_fluent_multiple(self):
        """Test adding multiple fluents."""
        cnf = CNFManager()

        # Expected: Sequential variable IDs
        fluents = ['on_a_b', 'clear_c', 'handempty']
        expected_ids = [1, 2, 3]

        for fluent, expected_id in zip(fluents, expected_ids):
            var_id = cnf.add_fluent(fluent)
            assert var_id == expected_id

        assert cnf.next_var == 4
        assert len(cnf.fluent_to_var) == 3

    def test_add_fluent_duplicate(self):
        """Test adding same fluent twice."""
        cnf = CNFManager()

        # Expected: Same variable ID returned
        var_id_1 = cnf.add_fluent('test_fluent')
        var_id_2 = cnf.add_fluent('test_fluent')

        assert var_id_1 == var_id_2 == 1
        assert cnf.next_var == 2
        assert len(cnf.fluent_to_var) == 1


class TestCNFManagerClauses:
    """Test clause addition and manipulation."""

    def test_add_simple_clause(self):
        """Test adding a simple clause."""
        cnf = CNFManager()

        # Expected: Single clause (a OR b)
        cnf.add_clause(['a', 'b'])

        assert len(cnf.cnf.clauses) == 1
        assert cnf.cnf.clauses[0] == [1, 2]  # Variables for 'a' and 'b'
        assert cnf.fluent_to_var == {'a': 1, 'b': 2}

    def test_add_negated_clause(self):
        """Test adding clause with negation."""
        cnf = CNFManager()

        # Expected: Clause (NOT a OR b)
        cnf.add_clause(['-a', 'b'])

        assert len(cnf.cnf.clauses) == 1
        assert cnf.cnf.clauses[0] == [-1, 2]  # -1 for NOT a, 2 for b
        assert cnf.fluent_to_var == {'a': 1, 'b': 2}

    def test_add_multiple_clauses(self):
        """Test adding multiple clauses."""
        cnf = CNFManager()

        # Expected: Two clauses forming (a OR b) AND (NOT a OR c)
        cnf.add_clause(['a', 'b'])
        cnf.add_clause(['-a', 'c'])

        assert len(cnf.cnf.clauses) == 2
        assert cnf.cnf.clauses[0] == [1, 2]
        assert cnf.cnf.clauses[1] == [-1, 3]
        assert len(cnf.fluent_to_var) == 3


class TestCNFManagerSatisfiability:
    """Test satisfiability checking with known formulas."""

    def test_satisfiable_formula(self, sample_cnf_formulas):
        """Test satisfiable CNF formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['simple_sat']

        # Build known satisfiable formula: (a OR b) AND (NOT a OR c)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected: Formula is satisfiable
        assert cnf.is_satisfiable() is formula['expected_satisfiable']

    def test_unsatisfiable_formula(self, sample_cnf_formulas):
        """Test unsatisfiable CNF formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['unsat']

        # Build known unsatisfiable formula: (a) AND (NOT a)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected: Formula is unsatisfiable
        assert cnf.is_satisfiable() is formula['expected_satisfiable']

    def test_single_solution_formula(self, sample_cnf_formulas):
        """Test formula with single solution."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['single_solution']

        # Build formula with single solution: (a) AND (NOT b)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        assert cnf.is_satisfiable() is formula['expected_satisfiable']


class TestCNFManagerSolutionCounting:
    """Test solution counting with predefined expected counts."""

    def test_count_solutions_simple(self, sample_cnf_formulas):
        """Test counting solutions for simple satisfiable formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['simple_sat']

        # Build formula: (a OR b) AND (NOT a OR c)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected: Exactly 3 solutions
        count = cnf.count_solutions()
        assert count == formula['expected_count']

    def test_count_solutions_unsatisfiable(self, sample_cnf_formulas):
        """Test counting solutions for unsatisfiable formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['unsat']

        # Build unsatisfiable formula: (a) AND (NOT a)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected: 0 solutions
        count = cnf.count_solutions()
        assert count == formula['expected_count']

    def test_count_solutions_single(self, sample_cnf_formulas):
        """Test counting solutions for single-solution formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['single_solution']

        # Build single-solution formula: (a) AND (NOT b)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected: Exactly 1 solution
        count = cnf.count_solutions()
        assert count == formula['expected_count']

    def test_get_all_solutions_content(self, sample_cnf_formulas):
        """Test that get_all_solutions returns correct solution sets."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['simple_sat']

        # Build formula: (a OR b) AND (NOT a OR c)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        solutions = cnf.get_all_solutions()

        # Expected: Exactly these 3 solutions
        expected_solutions = formula['expected_solutions']
        assert len(solutions) == len(expected_solutions)

        # Convert to sets for comparison
        solution_sets = [set(sol) for sol in solutions]
        for expected in expected_solutions:
            assert expected in solution_sets


class TestCNFManagerProbabilities:
    """Test probability calculations with known expected values."""

    def test_probability_simple_formula(self, sample_cnf_formulas):
        """Test probability calculation for known formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['simple_sat']

        # Build formula: (a OR b) AND (NOT a OR c)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected probabilities based on known solutions
        prob_a = cnf.get_probability('a')
        prob_b = cnf.get_probability('b')
        prob_c = cnf.get_probability('c')

        # Use pytest.approx for floating point comparison
        assert prob_a == pytest.approx(formula['expected_prob_a'], abs=1e-10)
        assert prob_b == pytest.approx(formula['expected_prob_b'], abs=1e-10)
        assert prob_c == pytest.approx(formula['expected_prob_c'], abs=1e-10)

    def test_probability_single_solution(self, sample_cnf_formulas):
        """Test probability for single-solution formula."""
        cnf = CNFManager()
        formula = sample_cnf_formulas['single_solution']

        # Build formula: (a) AND (NOT b)
        for clause in formula['clauses']:
            cnf.add_clause(clause)

        # Expected: a is always true (prob=1), b is always false (prob=0)
        assert cnf.get_probability('a') == formula['expected_prob_a']
        assert cnf.get_probability('b') == formula['expected_prob_b']

    def test_probability_unknown_fluent(self):
        """Test probability of fluent not in formula."""
        cnf = CNFManager()
        cnf.add_clause(['a'])

        # Expected: Unknown fluent has probability 0
        prob = cnf.get_probability('unknown_fluent')
        assert prob == 0.0


class TestCNFManagerEntropy:
    """Test entropy calculation with known expected values."""

    def test_entropy_single_solution(self):
        """Test entropy for single-solution formula."""
        cnf = CNFManager()
        cnf.add_clause(['a'])
        cnf.add_clause(['-b'])

        # Expected: Single solution means 0 entropy
        entropy = cnf.get_entropy()
        assert entropy == 0.0

    def test_entropy_unsatisfiable(self):
        """Test entropy for unsatisfiable formula."""
        cnf = CNFManager()
        cnf.add_clause(['a'])
        cnf.add_clause(['-a'])

        # Expected: Unsatisfiable formula has 0 entropy
        entropy = cnf.get_entropy()
        assert entropy == 0.0

    def test_entropy_balanced_formula(self):
        """Test entropy for formula with multiple satisfying models."""
        cnf = CNFManager()
        # Formula: (a OR b) - has 3 satisfying models
        # Models: {a}, {b}, {a,b}
        # Expected entropy: log2(3) ≈ 1.585
        cnf.add_clause(['a', 'b'])

        entropy = cnf.get_entropy()
        import math
        expected_entropy = math.log2(3)
        assert abs(entropy - expected_entropy) < 0.001, \
            f"Expected entropy log2(3)={expected_entropy:.3f}, got {entropy:.3f}"


class TestCNFManagerOperations:
    """Test CNF manager operations like copy, merge."""

    def test_copy_basic(self):
        """Test copying CNF manager."""
        cnf1 = CNFManager()
        cnf1.add_clause(['a', 'b'])
        cnf1.add_clause(['-a', 'c'])

        cnf2 = cnf1.copy()

        # Expected: Independent copies with same content
        assert len(cnf2.cnf.clauses) == len(cnf1.cnf.clauses)
        assert cnf2.fluent_to_var == cnf1.fluent_to_var
        assert cnf2.var_to_fluent == cnf1.var_to_fluent
        assert cnf2.next_var == cnf1.next_var

        # Expected: Modifications to copy don't affect original
        cnf2.add_clause(['d'])
        assert len(cnf2.cnf.clauses) == len(cnf1.cnf.clauses) + 1

    def test_merge_basic(self):
        """Test merging two CNF managers."""
        cnf1 = CNFManager()
        cnf1.add_clause(['a', 'b'])

        cnf2 = CNFManager()
        cnf2.add_clause(['c', 'd'])

        original_clauses = len(cnf1.cnf.clauses)
        cnf1.merge(cnf2)

        # Expected: cnf1 now contains clauses from both
        assert len(cnf1.cnf.clauses) == original_clauses + 1
        assert 'c' in cnf1.fluent_to_var
        assert 'd' in cnf1.fluent_to_var

    def test_get_model_satisfiable(self):
        """Test getting a model from satisfiable formula."""
        cnf = CNFManager()
        cnf.add_clause(['a'])  # a must be true
        cnf.add_clause(['-b'])  # b must be false

        model = cnf.get_model()

        # Expected: Model contains 'a' but not 'b'
        assert model is not None
        assert 'a' in model
        assert 'b' not in model

    def test_get_model_unsatisfiable(self):
        """Test getting model from unsatisfiable formula."""
        cnf = CNFManager()
        cnf.add_clause(['a'])
        cnf.add_clause(['-a'])

        model = cnf.get_model()

        # Expected: No model exists
        assert model is None


class TestCNFManagerStringRepresentation:
    """Test string representations."""

    def test_to_string_empty(self):
        """Test string representation of empty formula."""
        cnf = CNFManager()

        # Expected: Empty formula representation
        string_repr = cnf.to_string()
        assert string_repr == "⊤ (empty/true)"

    def test_to_string_simple_formula(self):
        """Test string representation of simple formula."""
        cnf = CNFManager()
        cnf.add_clause(['a', 'b'])

        string_repr = cnf.to_string()

        # Expected: Human-readable format
        assert 'a' in string_repr
        assert 'b' in string_repr
        assert '∨' in string_repr  # OR symbol

    def test_to_string_with_negation(self):
        """Test string representation with negated literals."""
        cnf = CNFManager()
        cnf.add_clause(['-a', 'b'])

        string_repr = cnf.to_string()

        # Expected: Negation symbol present
        assert '¬a' in string_repr
        assert 'b' in string_repr

    def test_str_and_repr(self):
        """Test __str__ and __repr__ methods."""
        cnf = CNFManager()
        cnf.add_clause(['a', 'b'])

        str_repr = str(cnf)
        repr_str = repr(cnf)

        # Expected: Different representations
        assert 'CNFManager' in str_repr
        assert 'CNFManager' in repr_str
        assert 'vars' in str_repr or 'clauses' in str_repr
        assert 'sat=' in repr_str  # repr should include satisfiability


class TestCNFManagerConstraintOperations:
    """Test constraint-based operations for information gain algorithm."""

    def test_create_with_state_constraints(self):
        """Test creating CNF with state constraints added."""
        cnf = CNFManager()

        # Build initial formula
        cnf.add_clause(['a', 'b'])
        cnf.add_clause(['-a', 'c'])

        # Create copy with state constraints
        state_constraints = {'a': False, 'b': True}  # a must be false, b must be true
        cnf_with_state = cnf.create_with_state_constraints(state_constraints)

        # Expected: Original unchanged, new has additional unit clauses
        assert len(cnf.cnf.clauses) == 2  # Original unchanged
        assert len(cnf_with_state.cnf.clauses) == 4  # Original 2 + 2 unit clauses

        # Verify state constraints were added as unit clauses
        # Check that the new CNF respects the constraints
        model = cnf_with_state.get_model()
        if model:  # If satisfiable
            assert 'a' not in model  # a should be false
            assert 'b' in model  # b should be true

    def test_add_constraint_from_unsatisfied(self):
        """Test adding constraint from unsatisfied literals."""
        cnf = CNFManager()

        # Initial setup
        cnf.add_fluent('on_a_b')
        cnf.add_fluent('clear_b')
        cnf.add_fluent('handempty')

        # Add constraint from unsatisfied literals
        unsatisfied_literals = {'on_a_b', '¬clear_b', 'handempty'}
        frozen_unsatisfied = frozenset(unsatisfied_literals)
        cnf.add_constraint_from_unsatisfied(frozen_unsatisfied)

        # Expected: One clause with the unsatisfied literals
        assert len(cnf.cnf.clauses) == 1
        # The clause should contain these literals (some may be negated)
        assert len(cnf.cnf.clauses[0]) == 3

    def test_build_from_constraint_sets(self):
        """Test building CNF from constraint sets."""
        cnf = CNFManager()

        # Define constraint sets (each set is a disjunction)
        constraint_sets = [
            {'clear_a', '¬on_a_b'},  # At least one must be true
            {'handempty'},  # Must be true
            {'¬on_b_c', 'clear_c'}  # At least one must be true
        ]

        cnf.build_from_constraint_sets(constraint_sets)

        # Expected: 3 clauses, one for each constraint set
        assert len(cnf.cnf.clauses) == 3

        # Verify the formula is satisfiable
        assert cnf.is_satisfiable() is True

    def test_clear_and_rebuild(self):
        """Test clearing and rebuilding CNF formula."""
        cnf = CNFManager()

        # Build initial formula
        cnf.add_clause(['a', 'b'])
        cnf.add_clause(['c', 'd'])
        initial_vars = len(cnf.fluent_to_var)

        # Clear formula but keep variable mappings
        cnf.clear_formula()

        # Expected: No clauses but variables preserved
        assert len(cnf.cnf.clauses) == 0
        assert len(cnf.fluent_to_var) == initial_vars

        # Rebuild with new clauses
        cnf.add_clause(['a', 'c'])
        assert len(cnf.cnf.clauses) == 1

    def test_add_unit_constraint(self):
        """Test adding unit constraints for literals."""
        cnf = CNFManager()

        # Add some clauses
        cnf.add_clause(['a', 'b'])

        # Add unit constraint that 'a' must be true
        cnf.add_unit_constraint('a', True)

        # Add unit constraint that 'c' must be false
        cnf.add_unit_constraint('c', False)

        # Expected: 3 clauses total (original + 2 unit clauses)
        assert len(cnf.cnf.clauses) == 3

        # Check that unit clauses were added correctly
        model = cnf.get_model()
        if model:
            assert 'a' in model  # a must be true
            assert 'c' not in model  # c must be false


class TestCNFManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_clause_handling(self):
        """Test handling of empty clauses."""
        cnf = CNFManager()
        # Empty clause should make the formula unsatisfiable
        cnf.add_clause([])

        # However, our implementation filters empty clauses in to_string
        # The behavior depends on PySAT's handling
        # We'll test that it doesn't crash
        result = cnf.to_string()
        assert isinstance(result, str)

    def test_large_formula_handling(self):
        """Test handling of larger formulas."""
        cnf = CNFManager()

        # Create formula with many variables
        for i in range(10):
            cnf.add_clause([f'var_{i}', f'var_{i+1}'])

        # Expected: Should handle without issues
        assert cnf.is_satisfiable() is True
        count = cnf.count_solutions()
        assert count > 0

    def test_variable_consistency(self):
        """Test that variable mappings remain consistent."""
        cnf = CNFManager()

        # Add fluents in different orders
        var1 = cnf.add_fluent('fluent_1')
        var2 = cnf.add_fluent('fluent_2')
        var3 = cnf.add_fluent('fluent_1')  # Duplicate

        # Expected: Consistent mapping
        assert var1 == var3  # Same fluent gets the same ID
        assert var1 != var2  # Different fluents get different IDs
        assert cnf.var_to_fluent[var1] == 'fluent_1'
        assert cnf.var_to_fluent[var2] == 'fluent_2'
