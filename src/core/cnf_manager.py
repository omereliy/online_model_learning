"""
CNF Manager for Online Action Model Learning
Manages CNF formulas for precondition/effect uncertainty using PySAT.
"""

from pysat.solvers import Glucose4
from pysat.formula import CNF

import itertools
import logging
from typing import List, Dict, Set, Optional, FrozenSet

logger = logging.getLogger(__name__)


class CNFManager:
    """Manages CNF formulas for precondition/effect uncertainty."""

    def __init__(self):
        """Initialize CNF manager with empty formula."""
        self.fluent_to_var: Dict[str, int] = {}
        self.var_to_fluent: Dict[int, str] = {}
        self.cnf = CNF()
        self.next_var = 1
        self._solution_cache = None
        self._cache_valid = False
        # Solver reuse (Session 2 optimization)
        self._solver: Optional[Glucose4] = None
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

    def get_variable(self, fluent_str: str) -> Optional[int]:
        """
        Get variable ID for fluent if it exists.

        Args:
            fluent_str: String representation of fluent

        Returns:
            Variable ID or None if fluent not mapped
        """
        return self.fluent_to_var.get(fluent_str)

    def _literal_to_var(self, literal: str) -> tuple:
        """
        Convert a literal string to (variable_id, is_negated) tuple.

        Handles negation:
        - '-' (ASCII): CNF negation prefix meaning "variable = FALSE"
        - '¬' (Unicode): Part of fluent name, treated as DISTINCT variable

        IMPORTANT: p(?x) and ¬p(?x) are treated as TWO SEPARATE VARIABLES.
        This is required by the information gain algorithm which models:
        - "is p(?x) a precondition?" (yes/no)
        - "is ¬p(?x) a precondition?" (yes/no)
        as independent questions in the hypothesis space.

        Args:
            literal: Literal string, possibly with '-' prefix for CNF negation
                    The '¬' prefix is kept as part of the fluent name.

        Returns:
            Tuple of (variable_id, is_negated) where is_negated indicates
            whether the variable should be negated in the CNF clause
        """
        cnf_negated = False
        fluent = literal

        # Check for CNF negation prefix '-'
        if fluent.startswith('-'):
            cnf_negated = True
            fluent = fluent[1:]

        # IMPORTANT: Keep '¬' as part of the fluent name!
        # '¬clear(?x)' is a distinct fluent from 'clear(?x)'
        # Do NOT strip '¬' - it represents a negative precondition literal

        # Register the fluent (including any ¬ prefix) as a variable
        var_id = self.add_fluent(fluent)

        return var_id, cnf_negated

    def add_clause(self, clause: List[str]):
        """
        Add clause with fluent strings (prefix '-' for CNF negation).

        - '-' (ASCII): CNF negation prefix meaning "variable = FALSE"
        - '¬' (Unicode): Part of fluent name (distinct variable from positive form)

        IMPORTANT: p(?x) and ¬p(?x) are treated as separate variables.
        - '-clear(?x)' means "clear(?x) variable = FALSE"
        - '-¬clear(?x)' means "¬clear(?x) variable = FALSE" (different variable!)

        Args:
            clause: List of fluent strings, prefix with '-' for negation
                   e.g., ['clear_b', '-on_a_b'] represents (clear_b OR NOT on_a_b)
        """
        var_clause = []
        for lit in clause:
            var_id, is_negated = self._literal_to_var(lit)
            var_clause.append(-var_id if is_negated else var_id)

        self.cnf.append(var_clause)
        self._invalidate_cache()

    def add_var_clause(self, var_clause: List[int]):
        """
        Add clause using variable IDs directly.

        Args:
            var_clause: List of variable IDs (negative for negation)
        """
        self.cnf.append(var_clause)
        self._invalidate_cache()

    def add_clause_with_subsumption(self, clause: List[str]) -> bool:
        """
        Add clause with subsumption checking to keep CNF minimal.

        Subsumption rules:
        - If new clause is subsumed by existing shorter clause, don't add
        - If new clause subsumes existing longer clauses, remove them

        This reduces CNF size and SAT solving time by eliminating redundant clauses.

        Negation handling via _literal_to_var():
        - '-' (ASCII): CNF negation prefix meaning "variable = FALSE"
        - '¬' (Unicode): Part of fluent name (distinct variable from positive form)

        Args:
            clause: List of fluent strings, prefix with '-' for negation

        Returns:
            True if clause was added, False if subsumed by existing clause
        """
        # Convert to variable IDs using unified negation handling
        var_clause = []
        for lit in clause:
            var_id, is_negated = self._literal_to_var(lit)
            var_clause.append(-var_id if is_negated else var_id)

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

    def refine_clauses_by_intersection(self, satisfied_literals: Set[str]) -> int:
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
            # Skip unit negative clauses - these are "NOT a precondition" constraints
            # from successful actions that should not be refined. They represent
            # permanent exclusions, and refining them causes double negation issues
            # when the fluent name already contains ¬ (e.g., '¬clear(?x)').
            if len(clause) == 1 and clause[0] < 0:
                new_clauses.append(clause)
                continue

            # Convert clause to literal strings
            # Only include POSITIVE CNF literals - these represent "fluent IS a precondition"
            # Negative CNF literals mean "fluent is NOT a precondition" and should be skipped
            # because refining by intersection only applies to the disjunction of possible preconditions
            clause_literals = set()
            for lit in clause:
                if lit > 0:  # Only positive CNF literals
                    fluent = self.var_to_fluent.get(lit)
                    if fluent is not None:
                        clause_literals.add(fluent)

            # Keep only literals that were satisfied (intersection)
            refined = clause_literals.intersection(satisfied_literals)

            if len(refined) < len(clause_literals):
                modified += 1

            if refined:  # Non-empty after refinement
                # Convert back to var clause
                # IMPORTANT: ¬p(?x) is a DIFFERENT variable from p(?x)
                # Use literal as-is (including any ¬ prefix)
                new_clause = []
                for lit_str in refined:
                    var_id = self.fluent_to_var.get(lit_str)
                    if var_id:
                        new_clause.append(var_id)
                if new_clause:
                    new_clauses.append(new_clause)
            else:
                # Empty clause after refinement - all literals were unsatisfied
                # This could indicate model inconsistency (constraint that can't be satisfied)
                logger.warning(
                    f"Empty clause after refinement - original had {len(clause_literals)} literals, "
                    f"none were satisfied. This may indicate model inconsistency."
                )

        self.cnf.clauses = new_clauses

        if modified > 0:
            self._invalidate_cache()
            logger.debug(f"Refined {modified} clauses, {len(new_clauses)} remain")

        return modified

    def _get_or_create_solver(self) -> Glucose4:
        """
        Get persistent solver instance, creating if needed (Session 2 optimization).

        Reuses existing solver if valid, otherwise creates new one.
        This avoids the overhead of creating a new solver for each query.

        Returns:
            Glucose4 solver instance bootstrapped with current CNF
        """
        if self._solver is None or not self._solver_valid:
            if self._solver is not None:
                try:
                    self._solver.delete()
                except Exception:
                    pass
            self._solver = Glucose4(bootstrap_with=self.cnf)
            self._solver_valid = True
        return self._solver

    def _cleanup_solver(self):
        """Clean up solver instance if it exists."""
        if self._solver is not None:
            try:
                self._solver.delete()
            except Exception:
                pass
            self._solver = None
        self._solver_valid = False

    def __del__(self):
        """Clean up solver on object destruction."""
        self._cleanup_solver()

    def remove_clause(self, clause_index: int):
        """
        Remove clause at given index.

        Args:
            clause_index: Index of clause to remove
        """
        if 0 <= clause_index < len(self.cnf.clauses):
            del self.cnf.clauses[clause_index]
            self._invalidate_cache()

    def count_solutions(self, max_solutions: int = None) -> int:
        """
        Count all satisfying assignments.

        Args:
            max_solutions: Maximum number of solutions to count (None for all)

        Returns:
            Number of satisfying assignments
        """
        solver = Glucose4(bootstrap_with=self.cnf)
        try:
            count = 0

            while solver.solve():
                count += 1
                if max_solutions and count >= max_solutions:
                    break

                # Block current solution to find next one
                model = solver.get_model()
                solver.add_clause([-lit for lit in model if abs(lit) < self.next_var])

            return count
        finally:
            solver.delete()

    def get_all_solutions(self, max_solutions: int = None) -> List[Set[str]]:
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
        solver = Glucose4(bootstrap_with=self.cnf)

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
        return solver.solve()

    def get_model(self) -> Optional[Set[str]]:
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

    # TODO: CNF Minimization Methods (minimize_qm, _rebuild_from_solutions, minimize_espresso)
    # These methods are currently unused but could potentially improve model counting performance.
    # Current limitation: They call get_all_solutions() first (which is expensive), then rebuild.
    # For proper integration, they would need incremental CNF simplification (subsumption
    # elimination, resolution) rather than solution-based rebuilding.
    # Consider connecting to the model counting pipeline if performance becomes an issue.

    def minimize_qm(self):
        """
        Minimize CNF formula using Quine-McCluskey algorithm.
        This is a simplified version for demonstration.
        """
        # Get all solutions as minterms
        solutions = self.get_all_solutions()
        if not solutions:
            return

        # Convert solutions to binary representation
        all_vars = sorted(self.fluent_to_var.keys())
        minterms = []

        for solution in solutions:
            term = []
            for var in all_vars:
                term.append(1 if var in solution else 0)
            minterms.append(tuple(term))

        # Simplified minimization (full QM would be more complex)
        # For now, just rebuild CNF from solutions
        self._rebuild_from_solutions(solutions)

    def _rebuild_from_solutions(self, solutions: List[Set[str]]):
        """
        Rebuild CNF from list of solutions.

        Args:
            solutions: List of solution sets
        """
        if not solutions:
            # Unsatisfiable formula
            self.cnf = CNF()
            self.cnf.append([1, -1])  # Add contradiction
            return

        # Create new CNF that excludes non-solutions
        # This is a simplified approach
        new_cnf = CNF()

        # For each variable combination not in solutions, add blocking clause
        all_vars = sorted(self.fluent_to_var.keys())
        solution_tuples = set()

        for sol in solutions:
            term = tuple(var in sol for var in all_vars)
            solution_tuples.add(term)

        # Add clauses to block non-solutions (simplified for small formulas)
        if len(all_vars) <= 10:  # Only for small problems
            for assignment in itertools.product([False, True], repeat=len(all_vars)):
                if assignment not in solution_tuples:
                    # Block this assignment
                    clause = []
                    for i, var in enumerate(all_vars):
                        var_id = self.fluent_to_var[var]
                        if assignment[i]:
                            clause.append(-var_id)
                        else:
                            clause.append(var_id)
                    if clause:
                        new_cnf.append(clause)

        self.cnf = new_cnf
        self._invalidate_cache()

    def minimize_espresso(self):
        """
        Minimize CNF using Espresso algorithm.
        Requires pyeda package for full implementation.
        """
        try:
            from pyeda.inter import espresso_exprs
            from pyeda.inter import expr

            # Convert CNF to pyeda expression
            solutions = self.get_all_solutions()
            if not solutions:
                return

            # Build truth table and minimize
            # This is a placeholder - full implementation would be more complex
            logger.info("[APPROX] Espresso minimization not fully implemented")
            self.minimize_qm()  # Fall back to QM

        except ImportError:
            logger.warning("[APPROX] pyeda not installed, using QM minimization instead")
            self.minimize_qm()

    def _invalidate_cache(self):
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

    def merge(self, other: 'CNFManager'):
        """
        Merge another CNF manager's formula into this one.

        Args:
            other: CNFManager to merge
        """
        # Map variables from other to this
        var_mapping = {}
        for fluent, var in other.fluent_to_var.items():
            var_mapping[var] = self.add_fluent(fluent)

        # Add mapped clauses
        for clause in other.cnf.clauses:
            mapped_clause = []
            for lit in clause:
                var = abs(lit)
                sign = 1 if lit > 0 else -1
                if var in var_mapping:
                    mapped_clause.append(sign * var_mapping[var])
            if mapped_clause:
                self.cnf.append(mapped_clause)

        self._invalidate_cache()

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

    def count_solutions_approximate(self, epsilon: float = 0.3, delta: float = 0.05) -> int:
        """
        Approximate model count using pyapproxmc with PAC guarantees.

        Args:
            epsilon: Tolerance (0.3 = within 1.3x of true count)
            delta: Confidence (0.05 = 95% confident)

        Returns:
            Approximate number of satisfying models

        Raises:
            ImportError: If pyapproxmc is not installed (REQUIRED dependency)
        """
        import pyapproxmc  # REQUIRED - fail if not installed

        if not self.cnf.clauses:
            num_vars = len(self.fluent_to_var)
            return 2 ** num_vars if num_vars > 0 else 1

        if not self.is_satisfiable():
            return 0

        counter = pyapproxmc.Counter(epsilon=epsilon, delta=delta)
        for clause in self.cnf.clauses:
            counter.add_clause(clause)

        cells, hashes = counter.count()
        return max(1, cells * (2 ** hashes))

    def count_solutions_adaptive(
        self,
        threshold: int = 15,
        use_approximate: bool = True,
        epsilon: float = 0.3,
        delta: float = 0.05
    ) -> int:
        """
        Adaptive counting: exact for simple formulas, approximate for complex.

        Args:
            threshold: Variable count above which to use approximate
            use_approximate: Whether approximate counting is enabled
            epsilon: Tolerance for approximate counting
            delta: Confidence for approximate counting

        Returns:
            Model count (exact or approximate)
        """
        num_vars = len(self.fluent_to_var)

        if num_vars <= threshold or not use_approximate:
            return self.count_solutions()

        # Try approximate counting, fall back to upper bound if not available
        try:
            logger.info(f"[APPROX] Using approximate counting: {num_vars} vars > {threshold} threshold")
            return self.count_solutions_approximate(epsilon, delta)
        except ImportError:
            # pyapproxmc not installed - use upper bound estimate
            # This is conservative but avoids exponential enumeration
            logger.warning(f"[APPROX] pyapproxmc not available, using upper bound 2^{num_vars}")
            return 2 ** num_vars

    def __str__(self) -> str:
        """String representation."""
        return f"CNFManager({len(self.fluent_to_var)} vars, {len(self.cnf.clauses)} clauses)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CNFManager(vars={len(self.fluent_to_var)}, clauses={len(self.cnf.clauses)}, sat={self.is_satisfiable()})"

    def create_with_state_constraints(self, state_constraints: Dict[str, bool]) -> 'CNFManager':
        """
        Create a new CNF manager with additional state constraints.

        This method is used by information gain algorithm to add constraints
        for unsatisfied literals in the current state.

        Args:
            state_constraints: Dict mapping fluent names to their required values
                              e.g., {'a': True, 'b': False} means a must be true, b must be false

        Returns:
            New CNFManager with original clauses plus unit clauses for constraints
        """
        # Create a deep copy
        new_cnf = self.copy()

        # Add unit clauses for each state constraint
        for fluent, must_be_true in state_constraints.items():
            new_cnf.add_unit_constraint(fluent, must_be_true)

        return new_cnf

    def add_unit_constraint(self, fluent: str, must_be_true: bool):
        """
        Add a unit clause constraining a fluent to a specific value.

        Args:
            fluent: Fluent name to constrain
            must_be_true: True if fluent must be true, False if it must be false
        """
        if must_be_true:
            # Add unit clause [fluent] (fluent must be true)
            self.add_clause([fluent])
        else:
            # Add unit clause [-fluent] (fluent must be false)
            self.add_clause(['-' + fluent])

    def add_constraint_from_unsatisfied(self, unsatisfied_literals: FrozenSet[str]):
        """
        Add a constraint clause from a set of unsatisfied literals.

        Used by information gain algorithm when action fails to add
        constraint that at least one unsatisfied literal must be a precondition.

        IMPORTANT: p(?x) and ¬p(?x) are separate variables.
        Each literal is added as a positive occurrence in the disjunction.

        Args:
            unsatisfied_literals: Set of literals not satisfied in state
                                 (can include negated literals like '¬clear_a')
        """
        if not unsatisfied_literals:
            return

        # Build clause from unsatisfied literals
        # Each literal is treated as a distinct variable (including any ¬ prefix)
        clause = list(unsatisfied_literals)

        if clause:
            self.add_clause(clause)

    def build_from_constraint_sets(self, constraint_sets: Set[FrozenSet[str]]):
        """
        Build CNF formula from constraint sets.

        Each constraint set becomes a disjunctive clause in the CNF.
        Used by information gain algorithm to build CNF from pre?(a).

        Args:
            constraint_sets: List of constraint sets, each is a set of literals
        """
        # Clear existing formula but preserve variable mappings
        self.clear_formula()

        # Add each constraint set as a clause
        # Each constraint: "at least one of these literals IS a precondition"
        # Literals are variable names (¬ is part of the name, not CNF negation)
        for constraint_set in constraint_sets:
            if not constraint_set:
                continue

            # Keep literals as-is - they are variable names
            clause = list(constraint_set)
            if clause:
                self.add_clause(clause)

    def clear_formula(self):
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

    def state_constraints_to_assumptions(self, state_constraints: Dict[str, bool]) -> List[int]:
        """
        Convert state constraints dict to PySAT assumptions list (Phase 2 enhancement).

        Assumptions allow temporary constraints without copying the CNF formula.

        IMPORTANT: p(?x) and ¬p(?x) are separate variables (consistent with _literal_to_var).
        The fluent name is used as-is, including any '¬' prefix.

        Args:
            state_constraints: Dict mapping fluent names to their required values
                              e.g., {'clear_a': False, 'on_a_b': True, '¬holding_a': True}

        Returns:
            List of variable IDs (negative for False, positive for True)
        """
        assumptions = []
        for fluent, must_be_true in state_constraints.items():
            # Use get_variable() to avoid creating new variables as side effect
            var_id = self.get_variable(fluent)
            if var_id is not None:
                assumptions.append(var_id if must_be_true else -var_id)
            # Skip unknown fluents - they're not in the hypothesis space

        return assumptions

    def count_models_with_assumptions(self, assumptions: List[int], use_cache: bool = True) -> int:
        """
        Count models with assumptions instead of deep copy (Phase 2 enhancement).

        Uses adaptive counting: for formulas with many variables, uses approximate
        counting to avoid exponential enumeration.

        If use_cache=True and solutions are cached, uses faster filtering approach.

        Args:
            assumptions: List of variable IDs (negative for must-be-false)
            use_cache: Whether to use cached solutions filtering (default True)

        Returns:
            Number of satisfying models with assumptions applied
        """
        # Use cache-based filtering if available and enabled
        if use_cache and self._cache_valid and self._solution_cache is not None:
            return self.count_models_with_assumptions_via_filter(assumptions)

        # Add assumptions as temporary unit clauses for adaptive counting
        # Each assumption becomes a unit clause: [assumption]
        num_added = 0
        for assumption in assumptions:
            self.cnf.clauses.append([assumption])
            num_added += 1

        try:
            # Use adaptive counting to avoid exponential enumeration
            count = self.count_solutions_adaptive()
            return count
        finally:
            # Remove temporary unit clauses
            for _ in range(num_added):
                self.cnf.clauses.pop()

    def count_models_with_assumptions_via_filter(self, assumptions: List[int]) -> int:
        """
        Count models by filtering cached solutions (faster if solutions already cached).

        Instead of re-solving with assumptions, filters the pre-computed solutions.
        This transforms O(N * SAT_SOLVE) into O(1 * SAT_SOLVE + N * FILTER).

        IMPORTANT: Only call this when cache is already valid! Does NOT populate cache
        automatically to avoid exponential enumeration.

        Args:
            assumptions: List of variable IDs (negative for must-be-false)

        Returns:
            Number of satisfying models matching the assumptions
        """
        # SAFETY: Do NOT populate cache here - that could trigger exponential enumeration
        # Caller must ensure cache is valid before calling this method
        if not self._cache_valid or self._solution_cache is None:
            logger.warning("count_models_with_assumptions_via_filter called without valid cache, returning 0")
            return 0

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

    def count_models_with_temporary_clause(self, clause_literals: FrozenSet[str]) -> int:
        """
        Count models with a temporary clause added (Phase 2 enhancement).

        Adds clause, counts models, removes clause - no deep copy needed!
        Much faster than cnf.copy() for temporary constraints.

        IMPORTANT: p(?x) and ¬p(?x) are separate variables (consistent with _literal_to_var).
        The fluent name is used as-is, including any '¬' prefix.

        Args:
            clause_literals: Set of literal strings for the clause
                           (e.g., frozenset({'on(?x,?y)', '¬clear(?x)'}))

        Returns:
            Number of satisfying models with temporary clause
        """
        if not clause_literals:
            return self.count_solutions()

        # Convert literals to variable IDs for PySAT clause
        # Each literal is added as a positive occurrence in the disjunction
        # IMPORTANT: Use get_variable() to avoid creating new variables as side effect
        clause = []
        for literal in clause_literals:
            var_id = self.get_variable(literal)
            if var_id is not None:
                clause.append(var_id)
            # Skip unknown literals - they're not in the hypothesis space

        # Add clause temporarily
        self.cnf.clauses.append(clause)

        try:
            # Count models with new clause (creates fresh solver from self.cnf)
            # Use adaptive counting to avoid exponential enumeration for large formulas
            count = self.count_solutions_adaptive()
            return count
        finally:
            # Remove temporary clause (restore original state)
            self.cnf.clauses.pop()