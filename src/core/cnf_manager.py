"""
CNF Manager for Online Action Model Learning
Manages CNF formulas for precondition/effect uncertainty using PySAT.
"""

try:
    from pysat.solvers import Minisat22
    from pysat.formula import CNF
except ImportError:
    # python-sat package uses different import path
    from pysat.solvers import Minisat22
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
        # Support for lifted fluents
        self.lifted_fluents: Dict[str, List[str]] = {}  # predicate -> [param_types]
        self.lifted_to_grounded: Dict[str, Set[str]] = {}  # lifted -> grounded instances

    def add_fluent(self, fluent_str: str, is_lifted: bool = False) -> int:
        """
        Map fluent string to variable ID.

        Args:
            fluent_str: String representation of fluent (e.g., 'on_a_b' or 'on(?x,?y)')
            is_lifted: Whether this is a lifted fluent with variables

        Returns:
            Variable ID for the fluent
        """
        if fluent_str not in self.fluent_to_var:
            self.fluent_to_var[fluent_str] = self.next_var
            self.var_to_fluent[self.next_var] = fluent_str
            self.next_var += 1

            # Track lifted fluents
            if is_lifted:
                self._register_lifted_fluent(fluent_str)
        return self.fluent_to_var[fluent_str]

    def _register_lifted_fluent(self, lifted_str: str):
        """
        Register a lifted fluent and parse its structure.

        Args:
            lifted_str: Lifted fluent string (e.g., 'on(?x,?y)')
        """
        import re
        # Parse lifted fluent: predicate(var1,var2,...)
        match = re.match(r'([^(]+)\(([^)]*)\)', lifted_str)
        if match:
            predicate = match.group(1)
            params = [p.strip() for p in match.group(2).split(',') if p.strip()]
            self.lifted_fluents[predicate] = params
            self.lifted_to_grounded[lifted_str] = set()

    def add_lifted_fluent(self, predicate: str, param_types: List[str]) -> int:
        """
        Add a lifted fluent with its parameter types.

        Args:
            predicate: Predicate name (e.g., 'on')
            param_types: List of parameter types (e.g., ['?x', '?y'])

        Returns:
            Variable ID for the lifted fluent
        """
        # Create lifted fluent string representation
        lifted_str = f"{predicate}({','.join(param_types)})"
        return self.add_fluent(lifted_str, is_lifted=True)

    def ground_lifted_fluent(self, predicate: str, objects: List[str]) -> int:
        """
        Create grounded instance of a lifted fluent.

        Args:
            predicate: Predicate name
            objects: List of object names to bind

        Returns:
            Variable ID for the grounded fluent
        """
        # Create grounded fluent string
        grounded_str = f"{predicate}_{'_'.join(objects)}"

        # Track relationship to lifted version
        lifted_pattern = f"{predicate}("
        for lifted_str in self.lifted_to_grounded:
            if lifted_str.startswith(lifted_pattern):
                self.lifted_to_grounded[lifted_str].add(grounded_str)
                break

        return self.add_fluent(grounded_str)

    def get_variable(self, fluent_str: str) -> Optional[int]:
        """
        Get variable ID for fluent if it exists.

        Args:
            fluent_str: String representation of fluent

        Returns:
            Variable ID or None if fluent not mapped
        """
        return self.fluent_to_var.get(fluent_str)

    def add_clause(self, clause: List[str], lifted: bool = False):
        """
        Add clause with fluent strings (prefix '-' for negation).

        Args:
            clause: List of fluent strings, prefix with '-' for negation
                   e.g., ['clear_b', '-on_a_b'] represents (clear_b OR NOT on_a_b)
            lifted: Whether clause contains lifted fluents
        """
        var_clause = []
        for lit in clause:
            negated = lit.startswith('-')
            fluent = lit[1:] if negated else lit

            # Check if this is a lifted fluent pattern
            is_lifted = lifted or '?' in fluent or '(' in fluent
            var_id = self.add_fluent(fluent, is_lifted=is_lifted)

            var_clause.append(-var_id if negated else var_id)

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
        solver = Minisat22(bootstrap_with=self.cnf)
        count = 0

        while solver.solve():
            count += 1
            if max_solutions and count >= max_solutions:
                break

            # Block current solution to find next one
            model = solver.get_model()
            solver.add_clause([-lit for lit in model if abs(lit) < self.next_var])

        return count

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
        solver = Minisat22(bootstrap_with=self.cnf)

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

        if not max_solutions:
            self._solution_cache = solutions
            self._cache_valid = True

        return solutions

    def is_satisfiable(self) -> bool:
        """
        Check if the CNF formula is satisfiable.

        Returns:
            True if satisfiable, False otherwise
        """
        solver = Minisat22(bootstrap_with=self.cnf)
        return solver.solve()

    def get_model(self) -> Optional[Set[str]]:
        """
        Get a single satisfying model if one exists.

        Returns:
            Set of true fluents or None if unsatisfiable
        """
        solver = Minisat22(bootstrap_with=self.cnf)
        if solver.solve():
            model = solver.get_model()
            return {
                self.var_to_fluent[abs(lit)]
                for lit in model
                if lit > 0 and abs(lit) in self.var_to_fluent
            }
        return None

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
            logger.info("Espresso minimization not fully implemented")
            self.minimize_qm()  # Fall back to QM

        except ImportError:
            logger.warning("pyeda not installed, using QM minimization instead")
            self.minimize_qm()

    def _invalidate_cache(self):
        """Invalidate solution cache when formula changes."""
        self._cache_valid = False
        self._solution_cache = None

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

    def get_lifted_groundings(self, predicate: str) -> Set[str]:
        """
        Get all grounded instances of a lifted predicate.

        Args:
            predicate: Predicate name

        Returns:
            Set of grounded fluent strings
        """
        groundings = set()
        for lifted_str, grounded_set in self.lifted_to_grounded.items():
            if lifted_str.startswith(f"{predicate}("):
                groundings.update(grounded_set)
        return groundings

    def instantiate_lifted_clause(self, lifted_clause: List[str], bindings: Dict[str, str]) -> List[str]:
        """
        Instantiate a lifted clause with variable bindings.

        Args:
            lifted_clause: Clause with variables (e.g., ['on(?x,?y)', '-clear(?y)'])
            bindings: Variable bindings (e.g., {'?x': 'a', '?y': 'b'})

        Returns:
            Grounded clause with substitutions
        """
        grounded_clause = []
        for lit in lifted_clause:
            negated = lit.startswith('-')
            fluent = lit[1:] if negated else lit

            # Substitute variables in fluent
            grounded_fluent = fluent
            for var, obj in bindings.items():
                grounded_fluent = grounded_fluent.replace(var, obj)

            # Convert to standard grounded format
            import re
            match = re.match(r'([^(]+)\(([^)]*)\)', grounded_fluent)
            if match:
                pred = match.group(1)
                args = [a.strip() for a in match.group(2).split(',')]
                grounded_fluent = f"{pred}_{'_'.join(args)}"

            grounded_clause.append(('-' if negated else '') + grounded_fluent)

        return grounded_clause

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
        new_manager.lifted_fluents = {k: v.copy() for k, v in self.lifted_fluents.items()}
        new_manager.lifted_to_grounded = {k: v.copy() for k, v in self.lifted_to_grounded.items()}

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

        Args:
            unsatisfied_literals: Set of literals not satisfied in state
                                 (can include negated literals like '¬clear_a')
        """
        if not unsatisfied_literals:
            return

        # Build clause from unsatisfied literals
        clause = []
        for literal in unsatisfied_literals:
            if literal.startswith('¬'):
                # Negative literal ¬p becomes -p in clause
                positive = literal[1:]
                clause.append('-' + positive)
            else:
                # Positive literal stays positive
                clause.append(literal)

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
        for constraint_set in constraint_sets:
            if not constraint_set:
                continue

            clause = []
            for literal in constraint_set:
                if literal.startswith('¬'):
                    # Negative literal
                    positive = literal[1:]
                    clause.append('-' + positive)
                else:
                    # Positive literal
                    clause.append(literal)

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

    def count_models_with_constraints(self, state_constraints: Dict[str, bool]) -> int:
        """
        Count models with additional state constraints applied.

        Does not modify the original formula.

        Args:
            state_constraints: Dict mapping fluent names to required values

        Returns:
            Number of satisfying models with constraints
        """
        # Create temporary CNF with constraints
        temp_cnf = self.create_with_state_constraints(state_constraints)
        return temp_cnf.count_solutions()

    def state_constraints_to_assumptions(self, state_constraints: Dict[str, bool]) -> List[int]:
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

    def count_models_with_assumptions(self, assumptions: List[int]) -> int:
        """
        Count models with assumptions instead of deep copy (Phase 2 enhancement).

        Uses PySAT's solve(assumptions=[...]) feature for 2-3x speedup.
        No deep copy needed - assumptions are temporary constraints.

        Args:
            assumptions: List of variable IDs (negative for must-be-false)

        Returns:
            Number of satisfying models with assumptions applied
        """
        solver = Minisat22(bootstrap_with=self.cnf)
        count = 0

        while solver.solve(assumptions=assumptions):
            count += 1

            # Block current solution to find next one
            model = solver.get_model()
            solver.add_clause([-lit for lit in model if abs(lit) < self.next_var])

        return count

    def count_models_with_temporary_clause(self, clause_literals: FrozenSet[str]) -> int:
        """
        Count models with a temporary clause added (Phase 2 enhancement).

        Adds clause, counts models, removes clause - no deep copy needed!
        Much faster than cnf.copy() for temporary constraints.

        Args:
            clause_literals: Set of literal strings for the clause
                           (e.g., frozenset({'on(?x,?y)', '¬clear(?x)'}))

        Returns:
            Number of satisfying models with temporary clause
        """
        if not clause_literals:
            return self.count_solutions()

        # Convert literals to variable IDs for PySAT clause
        clause = []
        for literal in clause_literals:
            if literal.startswith('¬'):
                # Negative literal
                positive = literal[1:]
                var_id = self.add_fluent(positive)
                clause.append(-var_id)
            else:
                # Positive literal
                var_id = self.add_fluent(literal)
                clause.append(var_id)

        # Add clause temporarily
        self.cnf.clauses.append(clause)

        try:
            # Count models with new clause (creates fresh solver from self.cnf)
            count = self.count_solutions()
            return count
        finally:
            # Remove temporary clause (restore original state)
            self.cnf.clauses.pop()