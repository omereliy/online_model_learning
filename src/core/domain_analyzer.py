"""
Domain Analyzer for Algorithm Compatibility
Checks PDDL domains for features that may not be supported by certain algorithms.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from unified_planning.io import PDDLReader
from unified_planning.model import Problem, Action, Fluent, FNode
from unified_planning.shortcuts import Not

logger = logging.getLogger(__name__)


class DomainAnalyzer:
    """Analyzes PDDL domains for algorithm compatibility."""

    def __init__(self, domain_file: str, problem_file: str):
        """
        Initialize domain analyzer.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
        """
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.reader = PDDLReader()
        self.problem: Optional[Problem] = None
        self.analysis_results: Dict[str, any] = {}

    def analyze(self) -> Dict[str, any]:
        """
        Perform comprehensive domain analysis.

        Returns:
            Dictionary with analysis results including:
            - has_negative_preconditions: bool
            - has_conditional_effects: bool
            - has_disjunctive_preconditions: bool
            - has_equality_preconditions: bool
            - has_quantifiers: bool
            - has_derived_predicates: bool
            - max_action_arity: int
            - num_predicates: int
            - num_actions: int
            - num_objects: int
            - algorithm_compatibility: Dict[str, bool]
        """
        # Parse the domain and problem
        self.problem = self.reader.parse_problem(self.domain_file, self.problem_file)

        # Analyze domain features
        self.analysis_results = {
            'domain_name': Path(self.domain_file).stem,
            'problem_name': Path(self.problem_file).stem,
            'has_negative_preconditions': self._check_negative_preconditions(),
            'has_conditional_effects': self._check_conditional_effects(),
            'has_disjunctive_preconditions': self._check_disjunctive_preconditions(),
            'has_equality_preconditions': self._check_equality_preconditions(),
            'has_quantifiers': self._check_quantifiers(),
            'has_derived_predicates': self._check_derived_predicates(),
            'max_action_arity': self._get_max_action_arity(),
            'num_predicates': len(self.problem.fluents),
            'num_actions': len(self.problem.actions),
            'num_objects': len(list(self.problem.all_objects)),
            'predicates_with_negation': self._get_predicates_with_negation(),
        }

        # Determine algorithm compatibility
        self.analysis_results['algorithm_compatibility'] = self._check_algorithm_compatibility()

        return self.analysis_results

    def _check_negative_preconditions(self) -> bool:
        """Check if any action has negative preconditions."""
        for action in self.problem.actions:
            if self._has_negation_in_formula(action.preconditions):
                return True
        return False

    def _has_negation_in_formula(self, formula) -> bool:
        """Recursively check if formula contains negation."""
        # Handle list of preconditions
        if isinstance(formula, list):
            for f in formula:
                if self._has_negation_in_formula(f):
                    return True
            return False

        # Handle FNode
        if hasattr(formula, 'is_not') and formula.is_not():
            return True

        # Check children recursively
        if hasattr(formula, 'args'):
            for child in formula.args:
                if self._has_negation_in_formula(child):
                    return True

        return False

    def _check_conditional_effects(self) -> bool:
        """Check if any action has conditional effects."""
        for action in self.problem.actions:
            if hasattr(action, 'conditional_effects') and action.conditional_effects:
                return True
        return False

    def _check_disjunctive_preconditions(self) -> bool:
        """Check if any action has disjunctive (OR) preconditions."""
        for action in self.problem.actions:
            if self._has_disjunction_in_formula(action.preconditions):
                return True
        return False

    def _has_disjunction_in_formula(self, formula) -> bool:
        """Recursively check if formula contains disjunction."""
        # Handle list of preconditions
        if isinstance(formula, list):
            for f in formula:
                if self._has_disjunction_in_formula(f):
                    return True
            return False

        # Handle FNode
        if hasattr(formula, 'is_or') and formula.is_or():
            return True

        # Check children recursively
        if hasattr(formula, 'args'):
            for child in formula.args:
                if self._has_disjunction_in_formula(child):
                    return True

        return False

    def _check_equality_preconditions(self) -> bool:
        """Check if any action has equality preconditions."""
        for action in self.problem.actions:
            if self._has_equality_in_formula(action.preconditions):
                return True
        return False

    def _has_equality_in_formula(self, formula) -> bool:
        """Recursively check if formula contains equality."""
        # Handle list of preconditions
        if isinstance(formula, list):
            for f in formula:
                if self._has_equality_in_formula(f):
                    return True
            return False

        # Handle FNode
        if hasattr(formula, 'is_equals') and formula.is_equals():
            return True

        # Check children recursively
        if hasattr(formula, 'args'):
            for child in formula.args:
                if self._has_equality_in_formula(child):
                    return True

        return False

    def _check_quantifiers(self) -> bool:
        """Check if domain uses quantifiers (forall, exists)."""
        for action in self.problem.actions:
            if self._has_quantifier_in_formula(action.preconditions):
                return True
            if self._has_quantifier_in_formula(action.effects):
                return True
        return False

    def _has_quantifier_in_formula(self, formula) -> bool:
        """Recursively check if formula contains quantifiers."""
        # Handle list
        if isinstance(formula, list):
            for f in formula:
                if self._has_quantifier_in_formula(f):
                    return True
            return False

        # Handle FNode
        if hasattr(formula, 'is_forall') and formula.is_forall():
            return True
        if hasattr(formula, 'is_exists') and formula.is_exists():
            return True

        # Check children recursively
        if hasattr(formula, 'args'):
            for child in formula.args:
                if self._has_quantifier_in_formula(child):
                    return True

        return False

    def _check_derived_predicates(self) -> bool:
        """Check if domain has derived predicates (axioms)."""
        # UP doesn't directly expose derived predicates, check if problem has them
        return hasattr(self.problem, 'axioms') and len(self.problem.axioms) > 0

    def _get_max_action_arity(self) -> int:
        """Get maximum arity (number of parameters) among all actions."""
        max_arity = 0
        for action in self.problem.actions:
            max_arity = max(max_arity, len(action.parameters))
        return max_arity

    def _get_predicates_with_negation(self) -> List[str]:
        """Get list of predicates that appear negated in preconditions."""
        negated_predicates = set()

        for action in self.problem.actions:
            self._collect_negated_predicates(action.preconditions, negated_predicates)

        return sorted(list(negated_predicates))

    def _collect_negated_predicates(self, formula, negated_set: Set[str]):
        """Collect predicates that appear under negation."""
        # Handle list
        if isinstance(formula, list):
            for f in formula:
                self._collect_negated_predicates(f, negated_set)
            return

        # Handle FNode
        if hasattr(formula, 'is_not') and formula.is_not():
            # Get the negated content
            negated_content = formula.arg(0)
            if hasattr(negated_content, 'is_fluent_exp') and negated_content.is_fluent_exp():
                fluent = negated_content.fluent()
                negated_set.add(fluent.name)

        # Recurse on children
        if hasattr(formula, 'args'):
            for child in formula.args:
                self._collect_negated_predicates(child, negated_set)

    def _check_algorithm_compatibility(self) -> Dict[str, Dict[str, any]]:
        """
        Check compatibility with different algorithms.

        Returns:
            Dictionary mapping algorithm names to compatibility info
        """
        compatibility = {}

        # OLAM compatibility
        olam_issues = []
        if self.analysis_results['has_negative_preconditions']:
            olam_issues.append("Has negative preconditions (not supported)")
        if self.analysis_results['has_conditional_effects']:
            olam_issues.append("Has conditional effects (not supported)")
        if self.analysis_results['has_disjunctive_preconditions']:
            olam_issues.append("Has disjunctive preconditions (not supported)")
        if self.analysis_results['has_quantifiers']:
            olam_issues.append("Has quantifiers (not supported)")

        compatibility['olam'] = {
            'compatible': len(olam_issues) == 0,
            'issues': olam_issues,
            'recommendation': 'Use simple domains without negative preconditions' if olam_issues else 'Domain is compatible'
        }

        # Information Gain compatibility (more flexible)
        ig_issues = []
        if self.analysis_results['has_quantifiers']:
            ig_issues.append("Has quantifiers (may require special handling)")
        if self.analysis_results['has_derived_predicates']:
            ig_issues.append("Has derived predicates (not directly supported)")

        compatibility['information_gain'] = {
            'compatible': len(ig_issues) == 0,
            'issues': ig_issues,
            'recommendation': 'Domain is mostly compatible' if not ig_issues else 'May work with limitations'
        }

        # ModelLearner compatibility
        ml_issues = []
        if self.analysis_results['has_derived_predicates']:
            ml_issues.append("Has derived predicates (not supported)")
        if self.analysis_results['max_action_arity'] > 5:
            ml_issues.append(f"High action arity ({self.analysis_results['max_action_arity']}) may cause performance issues")

        compatibility['model_learner'] = {
            'compatible': len(ml_issues) == 0,
            'issues': ml_issues,
            'recommendation': 'Domain is compatible' if not ml_issues else 'May work with performance considerations'
        }

        return compatibility

    def is_compatible_with(self, algorithm: str) -> bool:
        """
        Quick check if domain is compatible with given algorithm.

        Args:
            algorithm: Algorithm name ('olam', 'information_gain', 'model_learner')

        Returns:
            True if domain is compatible with algorithm
        """
        if not self.analysis_results:
            self.analyze()

        alg_compat = self.analysis_results.get('algorithm_compatibility', {})
        alg_info = alg_compat.get(algorithm.lower(), {})
        return alg_info.get('compatible', False)

    def get_compatibility_report(self) -> str:
        """
        Generate human-readable compatibility report.

        Returns:
            Formatted compatibility report string
        """
        if not self.analysis_results:
            self.analyze()

        report = []
        report.append("=" * 60)
        report.append(f"Domain Analysis Report: {self.analysis_results['domain_name']}")
        report.append("=" * 60)

        # Domain features
        report.append("\nDomain Features:")
        report.append(f"  - Predicates: {self.analysis_results['num_predicates']}")
        report.append(f"  - Actions: {self.analysis_results['num_actions']}")
        report.append(f"  - Objects: {self.analysis_results['num_objects']}")
        report.append(f"  - Max action arity: {self.analysis_results['max_action_arity']}")

        # Advanced features
        report.append("\nAdvanced Features:")
        features = [
            ('Negative preconditions', 'has_negative_preconditions'),
            ('Conditional effects', 'has_conditional_effects'),
            ('Disjunctive preconditions', 'has_disjunctive_preconditions'),
            ('Equality preconditions', 'has_equality_preconditions'),
            ('Quantifiers', 'has_quantifiers'),
            ('Derived predicates', 'has_derived_predicates'),
        ]

        for name, key in features:
            status = "✓ Yes" if self.analysis_results[key] else "✗ No"
            report.append(f"  - {name}: {status}")

        if self.analysis_results['predicates_with_negation']:
            report.append(f"\nNegated predicates: {', '.join(self.analysis_results['predicates_with_negation'])}")

        # Algorithm compatibility
        report.append("\n" + "-" * 60)
        report.append("Algorithm Compatibility:")
        report.append("-" * 60)

        for alg_name, info in self.analysis_results['algorithm_compatibility'].items():
            status = "✓ COMPATIBLE" if info['compatible'] else "✗ INCOMPATIBLE"
            report.append(f"\n{alg_name.upper()}: {status}")
            if info['issues']:
                for issue in info['issues']:
                    report.append(f"  ⚠ {issue}")
            report.append(f"  → {info['recommendation']}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)