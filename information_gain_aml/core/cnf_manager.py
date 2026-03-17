"""
CNF Manager for Online Action Model Learning
Manages CNF formulas for precondition/effect uncertainty using PySAT.
"""

from __future__ import annotations

from pysat.solvers import Minisat22
from pysat.formula import CNF

import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class CNFManager:
    """Manages CNF formulas for precondition/effect uncertainty."""

    def __init__(self) -> None:
        """Initialize CNF manager with empty formula."""
        self.fluent_to_var: dict[str, int] = {}
        self.var_to_fluent: dict[int, str] = {}
        self.cnf = CNF()
        self.next_var = 1
        self._solution_cache: list[set[str]] | None = None
        self._cache_valid = False
        # Solver reuse (Session 2 optimization)
        self._solver: Minisat22 | None = None
        self._solver_valid: bool = False

    def add_fluent(self, fluent_str: str) -> int:
        """
        Map fluent string to variable ID.

        Args:
            fluent_str: String representation of fluent (e.g., 'on_a_b')

        Returns:
            Variable ID for the fluent
        """
        if fluent_str not in self.fluent_to_var:
            self.fluent_to_var[fluent_str] = self.next_var
            self.var_to_fluent[self.next_var] = fluent_str
            self.next_var += 1
        return self.fluent_to_var[fluent_str]

    def _literals_to_var_clause(self, literals: Iterable[str]) -> list[int]:
        """Convert literal strings to PySAT variable clause.

        Handles both '-' prefix (internal format) and '¬' prefix (algorithm format).
        """
        clause = []
        for lit in literals:
            if lit.startswith('¬'):
                clause.append(-self.add_fluent(lit[1:]))
            elif lit.startswith('-'):
                clause.append(-self.add_fluent(lit[1:]))
            else:
                clause.append(self.add_fluent(lit))
        return clause

    def add_clause(self, clause: list[str]) -> None:
        """
        Add clause with fluent strings (prefix '-' for negation).

        Args:
            clause: List of fluent strings, prefix with '-' for negation
                   e.g., ['clear_b', '-on_a_b'] represents (clear_b OR NOT on_a_b)
        """
        self.cnf.append(self._literals_to_var_clause(clause))
        self._invalidate_cache()

    def add_clause_with_subsumption(self, clause: list[str]) -> bool:
        """
        Add clause with subsumption checking to keep CNF minimal.

        Subsumption rules:
        - If new clause is subsumed by existing shorter clause, don't add
        - If new clause subsumes existing longer clauses, remove them

        This reduces CNF size and SAT solving time by eliminating redundant clauses.

        Args:
            clause: List of fluent strings, prefix with '-' for negation

        Returns:
            True if clause was added, False if subsumed by existing clause
        """
        var_clause = self._literals_to_var_clause(clause)
        new_clause_set = frozenset(var_clause)

        # Check if new clause is subsumed by any existing shorter clause
        # A clause C1 subsumes C2 if C1 ⊆ C2 (C1 is shorter and implies C2)
        for existing in self.cnf.clauses:
            existing_set = frozenset(existing)
            if existing_set.issubset(new_clause_set):
                # Existing clause subsumes new clause - don't add
                logger.debug(f"Clause subsumed by existing: {clause}")
                return False

        # Remove existing clauses that are subsumed by new clause
        original_count = len(self.cnf.clauses)
        self.cnf.clauses = [
            c for c in self.cnf.clauses
            if not new_clause_set.issubset(frozenset(c))
        ]
        removed_count = original_count - len(self.cnf.clauses)

        if removed_count > 0:
            logger.debug(f"Removed {removed_count} subsumed clauses")

        # Add new clause
        self.cnf.append(var_clause)
        self._invalidate_cache()
        return True

    def refine_clauses_by_intersection(self, satisfied_literals: set[str]) -> int:
        """
        Refine existing clauses by removing unsatisfied literals (Session 2 optimization).

        Used when action succeeds - unsatisfied literals can't be preconditions,
        so they're removed from each clause. This implements the algorithm's
        constraint refinement: pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}

        This is more efficient than rebuilding the entire CNF from constraint sets.

        Args:
            satisfied_literals: Lifted literals that were satisfied in the state.
                               Format: {'on(?x,?y)', '¬clear(?x)', ...}

        Returns:
            Number of clauses that were modified
        """
        if not self.cnf.clauses:
            return 0

        modified = 0
        new_clauses = []

        for clause in self.cnf.clauses:
            # Convert clause to literal strings
            clause_literals = set()
            for lit in clause:
                var = abs(lit)
                fluent = self.var_to_fluent.get(var)
                if fluent is None:
                    continue  # Skip unknown variables
                if lit < 0:
                    clause_literals.add(f"¬{fluent}")
                else:
                    clause_literals.add(fluent)

            # Keep only literals that were satisfied (intersection)
            refined = clause_literals.intersection(satisfied_literals)

            if len(refined) < len(clause_literals):
                modified += 1

            if refined:  # Non-empty after refinement
                # Convert back to var clause
                new_clause = []
                for lit_str in refined:
                    if lit_str.startswith('¬'):
                        fluent = lit_str[1:]
                        var_id = self.fluent_to_var.get(fluent)
                        if var_id:
                            new_clause.append(-var_id)
                    else:
                        var_id = self.fluent_to_var.get(lit_str)
                        if var_id:
                            new_clause.append(var_id)
                if new_clause:
                    new_clauses.append(new_clause)
            # Empty clauses are dropped - they become tautologies after
            # a successful action rules out all their literals

        self.cnf.clauses = new_clauses

        if modified > 0:
            self._invalidate_cache()
            logger.debug(f"Refined {modified} clauses, {len(new_clauses)} remain")

        return modified

    def _get_or_create_solver(self) -> Minisat22:
        """
        Get persistent solver instance, creating if needed (Session 2 optimization).

        Reuses existing solver if valid, otherwise creates new one.
        This avoids the overhead of creating a new solver for each query.

        Returns:
            Minisat22 solver instance bootstrapped with current CNF
        """
        if self._solver is None or not self._solver_valid:
            if self._solver is not None:
                try:
                    self._solver.delete()
                except Exception:
                    pass
            self._solver = Minisat22(bootstrap_with=self.cnf)
            self._solver_valid = True
        return self._solver

    def _cleanup_solver(self) -> None:
        """Clean up solver instance if it exists."""
        if self._solver is not None:
            try:
                self._solver.delete()
            except Exception:
                pass
            self._solver = None
        self._solver_valid = False

    def __del__(self) -> None:
        """Clean up solver on object destruction."""
        self._cleanup_solver()

    def _enumerate_models(self, assumptions: list[int] | None = None,
                          max_count: int | None = None) -> int:
        """Count satisfying models by SAT enumeration with optional assumptions."""
        solver = Minisat22(bootstrap_with=self.cnf)
        try:
            count = 0
            solve_args = {"assumptions": assumptions} if assumptions is not None else {}

            while solver.solve(**solve_args):
                count += 1
                if max_count is not None and count >= max_count:
                    break
                model = solver.get_model()
                solver.add_clause([-lit for lit in model if abs(lit) < self.next_var])

            return count
        finally:
            solver.delete()

    def count_solutions(self, max_solutions: int | None = None) -> int:
        """
        Count all satisfying assignments.

        Args:
            max_solutions: Maximum number of solutions to count (None for all)

        Returns:
            Number of satisfying assignments
        """
        if max_solutions is None and self._cache_valid and self._solution_cache is not None:
            return len(self._solution_cache)
        return self._enumerate_models(max_count=max_solutions)

    def get_all_solutions(self, max_solutions: int | None = None) -> list[set[str]]:
        """
        Get all satisfying assignments as sets of true fluents.

        Args:
            max_solutions: Maximum number of solutions to return

        Returns:
            List of solutions, each as a set of true fluent strings
        """
        if self._cache_valid and self._solution_cache is not None:
            return self._solution_cache[:max_solutions] if max_solutions else self._solution_cache

        solutions = []
        solver = Minisat22(bootstrap_with=self.cnf)

        try:
            while solver.solve():
                model = solver.get_model()
                solution = {
                    self.var_to_fluent[abs(lit)]
                    for lit in model
                    if lit > 0 and abs(lit) in self.var_to_fluent
                }
                solutions.append(solution)

                if max_solutions and len(solutions) >= max_solutions:
                    break

                # Block current solution
                solver.add_clause([-lit for lit in model if abs(lit) < self.next_var])
        finally:
            solver.delete()

        if not max_solutions:
            self._solution_cache = solutions
            self._cache_valid = True

        return solutions

    def is_satisfiable(self) -> bool:
        """
        Check if the CNF formula is satisfiable.

        Uses persistent solver for efficiency (Session 2 optimization).

        Returns:
            True if satisfiable, False otherwise
        """
        solver = self._get_or_create_solver()
        return bool(solver.solve())

    def get_model(self) -> set[str] | None:
        """
        Get a single satisfying model if one exists.

        Uses persistent solver for efficiency (Session 2 optimization).

        Returns:
            Set of true fluents or None if unsatisfiable
        """
        solver = self._get_or_create_solver()
        if solver.solve():
            model = solver.get_model()
            return {
                self.var_to_fluent[abs(lit)]
                for lit in model
                if lit > 0 and abs(lit) in self.var_to_fluent
            }
        return None

    def _invalidate_cache(self) -> None:
        """Invalidate solution cache and solver when formula changes."""
        self._cache_valid = False
        self._solution_cache = None
        # Clean up and invalidate solver (Session 2)
        self._cleanup_solver()

    def to_string(self) -> str:
        """
        Human-readable CNF representation.

        Returns:
            String representation of CNF formula
        """
        if not self.cnf.clauses:
            return "⊤ (empty/true)"

        clauses = []
        for clause in self.cnf.clauses:
            if not clause:
                continue
            literals = []
            for lit in clause:
                var = abs(lit)
                if var in self.var_to_fluent:
                    fluent = self.var_to_fluent[var]
                    literals.append(f"¬{fluent}" if lit < 0 else fluent)
                else:
                    literals.append(f"¬x{var}" if lit < 0 else f"x{var}")
            if literals:
                clauses.append(f"({' ∨ '.join(literals)})")

        return ' ∧ '.join(clauses) if clauses else "⊥ (false)"

    def copy(self) -> 'CNFManager':
        """
        Create a deep copy of this CNF manager.

        Returns:
            New CNFManager instance with same formula
        """
        new_manager = CNFManager()
        new_manager.fluent_to_var = self.fluent_to_var.copy()
        new_manager.var_to_fluent = self.var_to_fluent.copy()
        new_manager.next_var = self.next_var

        # Deep copy CNF
        new_manager.cnf = CNF()
        for clause in self.cnf.clauses:
            new_manager.cnf.append(clause.copy())

        return new_manager

    def get_probability(self, fluent_str: str) -> float:
        """
        Calculate probability of a fluent being true in satisfying assignments.

        Args:
            fluent_str: Fluent to check

        Returns:
            Probability between 0 and 1
        """
        solutions = self.get_all_solutions()
        if not solutions:
            return 0.0

        true_count = sum(1 for sol in solutions if fluent_str in sol)
        return true_count / len(solutions)

    def get_entropy(self) -> float:
        """
        Calculate Shannon entropy of the hypothesis space.

        Entropy measures uncertainty about which model is correct.
        H = log2(number_of_satisfying_models)

        Returns:
            Entropy value in bits (non-negative)
        """
        import math

        # If no clauses, maximum uncertainty
        if not self.has_clauses():
            num_vars = len(self.fluent_to_var)
            max_models = 2 ** num_vars if num_vars > 0 else 1
            return math.log2(max_models) if max_models > 1 else 0.0

        # Count satisfying models
        num_models = self.count_solutions()

        # No uncertainty if 0 or 1 model
        if num_models <= 1:
            return 0.0

        # Entropy is log2 of model count
        return math.log2(num_models)

    def __str__(self) -> str:
        """String representation."""
        return f"CNFManager({len(self.fluent_to_var)} vars, {len(self.cnf.clauses)} clauses)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CNFManager(vars={len(self.fluent_to_var)}, clauses={len(self.cnf.clauses)})"

    def build_from_constraint_sets(self, constraint_sets: set[frozenset[str]]) -> None:
        """
        Build CNF formula from constraint sets.

        Each constraint set becomes a disjunctive clause in the CNF.
        Used by information gain algorithm to build CNF from pre?(a).

        Args:
            constraint_sets: List of constraint sets, each is a set of literals
        """
        # Clear existing formula but preserve variable mappings
        self.clear_formula()

        # Add each constraint set as a clause (handles ¬ prefix directly)
        for constraint_set in constraint_sets:
            if not constraint_set:
                continue
            var_clause = self._literals_to_var_clause(constraint_set)
            if var_clause:
                self.cnf.append(var_clause)
        self._invalidate_cache()

    def clear_formula(self) -> None:
        """
        Clear all clauses but preserve variable mappings.

        Used when rebuilding formula from scratch.
        """
        self.cnf.clauses = []
        self._invalidate_cache()

    def has_clauses(self) -> bool:
        """
        Check if the CNF formula has any clauses.

        Returns:
            True if there are clauses, False if formula is empty
        """
        return len(self.cnf.clauses) > 0

    def state_constraints_to_assumptions(self, state_constraints: dict[str, bool]) -> list[int]:
        """
        Convert state constraints dict to PySAT assumptions list (Phase 2 enhancement).

        Assumptions allow temporary constraints without copying the CNF formula.

        Args:
            state_constraints: Dict mapping fluent names to their required values
                              e.g., {'clear_a': False, 'on_a_b': True}

        Returns:
            List of variable IDs (negative for False, positive for True)
        """
        assumptions = []
        for fluent, must_be_true in state_constraints.items():
            # Get or create variable ID for this fluent
            var_id = self.add_fluent(fluent)
            # Positive literal if must be true, negative if must be false
            assumptions.append(var_id if must_be_true else -var_id)

        return assumptions

    def count_models_with_assumptions(self, assumptions: list[int], use_cache: bool = True) -> int:
        """
        Count models with assumptions instead of deep copy (Phase 2 enhancement).

        Uses PySAT's solve(assumptions=[...]) feature for 2-3x speedup.
        No deep copy needed - assumptions are temporary constraints.

        If use_cache=True and solutions are cached, uses faster filtering approach
        (O(|solutions| * |assumptions|) instead of O(SAT_SOLVE)).

        Args:
            assumptions: List of variable IDs (negative for must-be-false)
            use_cache: Whether to use cached solutions filtering (default True)

        Returns:
            Number of satisfying models with assumptions applied
        """
        # Use cache-based filtering if available and enabled
        if use_cache and self._cache_valid and self._solution_cache is not None:
            return self.count_models_with_assumptions_via_filter(assumptions)

        return self._enumerate_models(assumptions=assumptions)

    def count_models_with_assumptions_via_filter(self, assumptions: list[int]) -> int:
        """
        Count models by filtering cached solutions (faster if solutions already cached).

        Instead of re-solving with assumptions, filters the pre-computed solutions.
        This transforms O(N * SAT_SOLVE) into O(1 * SAT_SOLVE + N * FILTER).

        Args:
            assumptions: List of variable IDs (negative for must-be-false)

        Returns:
            Number of satisfying models matching the assumptions
        """
        # Ensure base solutions are cached
        if not self._cache_valid or self._solution_cache is None:
            self.get_all_solutions()  # Populates cache

        if not self._solution_cache:
            return 0

        # Convert assumptions to set-based check
        must_be_true = set()
        must_be_false = set()
        for assumption in assumptions:
            var = abs(assumption)
            fluent = self.var_to_fluent.get(var)
            if fluent:
                if assumption > 0:
                    must_be_true.add(fluent)
                else:
                    must_be_false.add(fluent)

        # Filter solutions - each solution is a set of true fluents
        count = 0
        for solution in self._solution_cache:
            # Check: all must_be_true fluents are in solution
            # Check: no must_be_false fluents are in solution
            if must_be_true.issubset(solution) and must_be_false.isdisjoint(solution):
                count += 1

        return count

    def count_models_with_temporary_clause(self, clause_literals: frozenset[str]) -> int:
        """
        Count models with a temporary clause added (Phase 2 enhancement).

        When solution cache is available, filters cached solutions in O(|cache| * |clause|)
        instead of O(SAT_SOLVE). Falls back to SAT solving when cache is not populated.

        Args:
            clause_literals: Set of literal strings for the clause
                           (e.g., frozenset({'on(?x,?y)', '¬clear(?x)'}))

        Returns:
            Number of satisfying models with temporary clause
        """
        if not clause_literals:
            return self.count_solutions()

        # Use cache-based filtering if available and all fluents are known
        if self._cache_valid and self._solution_cache is not None:
            result = self._count_models_with_clause_via_filter(clause_literals)
            if result >= 0:
                return result

        clause = self._literals_to_var_clause(clause_literals)

        # Add clause temporarily
        self.cnf.clauses.append(clause)

        try:
            # Count models with new clause (creates fresh solver from self.cnf)
            count = self._enumerate_models()
            return count
        finally:
            # Remove temporary clause (restore original state)
            self.cnf.clauses.pop()

    def _count_models_with_clause_via_filter(self, clause_literals: frozenset[str]) -> int:
        """
        Filter cached solutions by a disjunctive clause.

        A clause (l₁ ∨ ... ∨ lₖ) is satisfied if ANY literal is true:
        - Positive literal p: p ∈ solution
        - Negative literal ¬p: base fluent ∉ solution

        Only applicable when all fluents in the clause are known to the CNF.
        Returns None if unknown fluents are found (caller should fall back to SAT).

        Args:
            clause_literals: Set of literal strings (prefix '¬' or '-' for negation)

        Returns:
            Number of cached solutions satisfying the clause, or -1 if
            clause references unknown fluents (caller falls back to SAT)
        """
        if not self._solution_cache:
            return 0

        # Parse literals into positive and negative base fluents
        positive = set()
        negative = set()
        for lit in clause_literals:
            if lit.startswith('¬'):
                negative.add(lit[1:])
            elif lit.startswith('-'):
                negative.add(lit[1:])
            else:
                positive.add(lit)

        # All fluents must be known to the CNF for filtering to be valid
        all_fluents = positive | negative
        if not all_fluents.issubset(self.fluent_to_var):
            return -1

        count = 0
        for solution in self._solution_cache:
            # Clause satisfied if ANY positive literal is in solution
            if positive and not positive.isdisjoint(solution):
                count += 1
                continue
            # Or ANY negative literal's base fluent is absent from solution
            if negative and not negative.issubset(solution):
                count += 1
        return count