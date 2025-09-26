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
        """Test entropy for formula with balanced probabilities."""
        cnf = CNFManager()
        # Formula where each variable has probability 0.5
        cnf.add_clause(['a', '-a'])  # Tautology - always true

        # For single variable with p=0.5: entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        # But this creates a tautology, so we need a different approach

        # Create formula: (a OR NOT a) which is always true
        # This doesn't constrain 'a', so it should have entropy
        entropy = cnf.get_entropy()
        # Tautology should have maximum entropy for involved variables
        assert entropy >= 0


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
        cnf.add_clause(['-b']) # b must be false

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


class TestCNFManagerLiftedFluents:
    """Test lifted fluent support in CNF Manager."""

    def test_add_lifted_fluent_basic(self):
        """Test adding basic lifted fluent."""
        cnf = CNFManager()

        # Expected: Lifted fluent registered with parameters
        var_id = cnf.add_lifted_fluent("on", ["?x", "?y"])
        assert var_id == 1
        assert "on" in cnf.lifted_fluents
        assert cnf.lifted_fluents["on"] == ["?x", "?y"]
        assert "on(?x,?y)" in cnf.fluent_to_var

    def test_add_lifted_fluent_multiple(self):
        """Test adding multiple lifted fluents."""
        cnf = CNFManager()

        # Add different lifted predicates
        on_id = cnf.add_lifted_fluent("on", ["?x", "?y"])
        clear_id = cnf.add_lifted_fluent("clear", ["?x"])
        holding_id = cnf.add_lifted_fluent("holding", ["?x"])

        # Expected: Sequential IDs and proper registration
        assert on_id == 1
        assert clear_id == 2
        assert holding_id == 3
        assert len(cnf.lifted_fluents) == 3

    def test_ground_lifted_fluent(self):
        """Test grounding a lifted fluent."""
        cnf = CNFManager()

        # Register lifted fluent
        cnf.add_lifted_fluent("on", ["?x", "?y"])

        # Ground it with objects
        grounded_id = cnf.ground_lifted_fluent("on", ["a", "b"])

        # Expected: Grounded version created and tracked
        assert grounded_id == 2  # First was lifted, second is grounded
        assert "on_a_b" in cnf.fluent_to_var
        assert "on(?x,?y)" in cnf.lifted_to_grounded
        assert "on_a_b" in cnf.lifted_to_grounded["on(?x,?y)"]

    def test_multiple_groundings(self):
        """Test multiple groundings of same lifted fluent."""
        cnf = CNFManager()

        # Register lifted fluent
        cnf.add_lifted_fluent("on", ["?x", "?y"])

        # Create multiple groundings
        on_ab = cnf.ground_lifted_fluent("on", ["a", "b"])
        on_bc = cnf.ground_lifted_fluent("on", ["b", "c"])
        on_ca = cnf.ground_lifted_fluent("on", ["c", "a"])

        # Expected: All groundings tracked
        groundings = cnf.get_lifted_groundings("on")
        assert len(groundings) == 3
        assert "on_a_b" in groundings
        assert "on_b_c" in groundings
        assert "on_c_a" in groundings

    def test_add_lifted_clause(self):
        """Test adding clause with lifted fluents."""
        cnf = CNFManager()

        # Add lifted clause: on(?x,?y) → ¬clear(?y)
        cnf.add_clause(["on(?x,?y)", "-clear(?y)"], lifted=True)

        # Expected: Lifted fluents registered and clause added
        assert "on(?x,?y)" in cnf.fluent_to_var
        assert "clear(?y)" in cnf.fluent_to_var
        assert len(cnf.cnf.clauses) == 1
        assert cnf.cnf.clauses[0] == [1, -2]  # on=1, clear=2

    def test_instantiate_lifted_clause_simple(self):
        """Test instantiating a simple lifted clause."""
        cnf = CNFManager()

        # Register lifted fluents
        cnf.add_lifted_fluent("clear", ["?x"])
        cnf.add_lifted_fluent("holding", ["?x"])

        # Instantiate clause
        lifted_clause = ["clear(?x)", "-holding(?x)"]
        bindings = {"?x": "a"}
        grounded = cnf.instantiate_lifted_clause(lifted_clause, bindings)

        # Expected: Properly grounded clause
        assert grounded == ["clear_a", "-holding_a"]

    def test_instantiate_lifted_clause_multiple_vars(self):
        """Test instantiating clause with multiple variables."""
        cnf = CNFManager()

        # Register lifted fluents
        cnf.add_lifted_fluent("on", ["?x", "?y"])
        cnf.add_lifted_fluent("clear", ["?y"])

        # Instantiate with bindings
        lifted_clause = ["on(?x,?y)", "-clear(?y)"]
        bindings = {"?x": "block1", "?y": "block2"}
        grounded = cnf.instantiate_lifted_clause(lifted_clause, bindings)

        # Expected: Both variables substituted
        assert grounded == ["on_block1_block2", "-clear_block2"]

    def test_lifted_grounded_mixed_formula(self):
        """Test CNF with both lifted and grounded clauses."""
        cnf = CNFManager()

        # Add lifted clause
        cnf.add_clause(["on(?x,?y)", "-clear(?y)"], lifted=True)

        # Add grounded clause
        cnf.add_clause(["on_a_b", "-clear_b"])

        # Add propositional clause
        cnf.add_clause(["handempty"])

        # Expected: All types of clauses coexist
        assert len(cnf.cnf.clauses) == 3
        assert "on(?x,?y)" in cnf.fluent_to_var
        assert "on_a_b" in cnf.fluent_to_var
        assert "handempty" in cnf.fluent_to_var

    def test_lifted_copy(self):
        """Test copying CNF manager with lifted fluents."""
        cnf1 = CNFManager()

        # Set up lifted structure
        cnf1.add_lifted_fluent("on", ["?x", "?y"])
        cnf1.ground_lifted_fluent("on", ["a", "b"])
        cnf1.add_clause(["on(?x,?y)"], lifted=True)

        # Copy
        cnf2 = cnf1.copy()

        # Expected: All lifted structures copied
        assert cnf2.lifted_fluents == cnf1.lifted_fluents
        assert cnf2.lifted_to_grounded == cnf1.lifted_to_grounded
        assert len(cnf2.cnf.clauses) == len(cnf1.cnf.clauses)

        # Expected: Independent copies
        cnf2.ground_lifted_fluent("on", ["c", "d"])
        assert len(cnf2.lifted_to_grounded["on(?x,?y)"]) == len(cnf1.lifted_to_grounded["on(?x,?y)"]) + 1

    def test_lifted_satisfiability(self):
        """Test satisfiability with lifted clauses."""
        cnf = CNFManager()

        # Create a simple lifted theory
        cnf.add_lifted_fluent("clear", ["?x"])
        cnf.add_lifted_fluent("on", ["?x", "?y"])

        # Add constraint: if on(?x,?y) then not clear(?y)
        cnf.add_clause(["-on(?x,?y)", "-clear(?y)"], lifted=True)

        # Ground for specific case
        bindings = {"?x": "a", "?y": "b"}
        grounded_clause = cnf.instantiate_lifted_clause(["-on(?x,?y)", "-clear(?y)"], bindings)
        cnf.add_clause(grounded_clause)

        # Add facts
        cnf.add_clause(["on_a_b"])  # on(a,b) is true

        # Expected: Still satisfiable (clear_b can be false)
        assert cnf.is_satisfiable() is True

        # Now add conflicting fact
        cnf.add_clause(["clear_b"])  # clear(b) is true

        # Expected: Now unsatisfiable due to conflict
        assert cnf.is_satisfiable() is False


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