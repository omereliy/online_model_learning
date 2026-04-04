"""
PlanningExplorer: Planner-based state exploration for information gain learning.

Uses unified_planning's OneshotPlanner with the learned domain to plan toward
states where uncertain predicates can be tested. This enables the learner to escape
local optima where no action in the current state provides information gain.
"""

import itertools
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple

from information_gain_aml.core.grounding import ground_parameter_bound_literal
from information_gain_aml.core.lifted_domain import LiftedDomainKnowledge

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LearnerKnowledge(Protocol):
    """Protocol for what PlanningExplorer needs from a learner."""

    domain: LiftedDomainKnowledge

    def get_uncertain_action_names(self) -> List[str]: ...
    def get_certain_preconditions(self, action: str) -> Set[str]: ...
    def get_uncertain_preconditions(self, action: str) -> Set[str]: ...
    def get_all_action_names(self) -> List[str]: ...


@dataclass
class ExplorationPlan:
    """A plan to reach a state where uncertain predicates can be tested."""

    plan_actions: List[Tuple[str, List[str]]]  # (action_name, objects) sequence
    target_operator: str                        # Operator we're trying to learn about
    negated_precs: Set[str]                     # Uncertain precs we're testing


class PlanningExplorer:
    """
    Uses a planner with the learned domain to plan toward states where
    uncertain predicates can be tested.

    Algorithm:
    1. For each operator with uncertain preconditions:
       a. For n=1..max uncertain precs, try combinations of n precs to negate
       b. For each combination, try all groundings of the operator
       c. Formulate goal: certain precs satisfied, negated precs NOT satisfied
       d. Call planner with learned domain + this goal
       e. If plan found: return plan + target action
    2. If no plan found for any operator: return None
    """

    def __init__(self, learner: LearnerKnowledge, planner_timeout: float = 30.0):
        self.learner = learner
        self.planner_timeout = planner_timeout
        self._useless_negated: Dict[str, List[FrozenSet[str]]] = {}

    def find_exploration_plan(self, current_state: Set[str]) -> Optional[ExplorationPlan]:
        """
        Find a plan to reach a state where uncertain predicates can be tested.

        Returns ExplorationPlan or None if no exploration is possible.
        """
        candidates = sorted(self.learner.get_uncertain_action_names())

        for action_name in candidates:
            uncertain = sorted(self.learner.get_uncertain_preconditions(action_name))
            certain = self.learner.get_certain_preconditions(action_name)

            if not uncertain:
                continue

            for n in range(1, len(uncertain) + 1):
                for combo in itertools.combinations(uncertain, n):
                    negated = frozenset(combo)

                    if self._is_useless(action_name, negated):
                        continue

                    result = self._try_exploration(
                        current_state, action_name, certain, set(negated)
                    )
                    if result is not None:
                        plan_actions, target_objects = result
                        plan_actions.append((action_name, target_objects))
                        return ExplorationPlan(
                            plan_actions=plan_actions,
                            target_operator=action_name,
                            negated_precs=set(negated),
                        )

                    self.record_useless_negation(action_name, negated)

        return None

    def _try_exploration(self, current_state: Set[str], action_name: str,
                         certain_precs: Set[str], negated_precs: Set[str],
                         ) -> Optional[Tuple[List[Tuple[str, List[str]]], List[str]]]:
        """Try to plan for a specific operator + negated prec combination.

        Iterates all groundings. Returns (plan_actions, target_objects) or None.
        """
        action = self.learner.domain.get_action(action_name)
        if not action:
            return None

        param_names = LiftedDomainKnowledge._generate_parameter_names(
            len(action.parameters)
        )
        param_types = [p.type for p in action.parameters]

        for objects in self._get_object_bindings(param_types):
            goal_pos, goal_neg = self._compute_goal(
                certain_precs, negated_precs, objects, param_names
            )

            # Skip contradictory goals
            if goal_pos & goal_neg:
                continue

            # Check if goal is already satisfied in current state
            if goal_pos.issubset(current_state) and goal_neg.isdisjoint(current_state):
                return ([], objects)

            plan = self._plan_to_goal(current_state, goal_pos, goal_neg)
            if plan is not None:
                return (plan, objects)

        return None

    def _compute_goal(self, certain_precs: Set[str], negated_precs: Set[str],
                      objects: List[str], param_names: List[str],
                      ) -> Tuple[Set[str], Set[str]]:
        """Compute grounded positive and negative goal fluents.

        - Certain precs must be SATISFIED -> positive/negative goals
        - Negated precs must NOT be SATISFIED -> inverted goals
        """
        goal_pos: Set[str] = set()
        goal_neg: Set[str] = set()

        # Certain precs: must be satisfied
        for prec in certain_precs:
            grounded = ground_parameter_bound_literal(prec, objects, param_names)
            if grounded.startswith('¬'):
                goal_neg.add(grounded[1:])
            else:
                goal_pos.add(grounded)

        # Negated uncertain precs: must NOT be satisfied
        for prec in negated_precs:
            grounded = ground_parameter_bound_literal(prec, objects, param_names)
            if grounded.startswith('¬'):
                # ¬f NOT satisfied means f IS in state
                goal_pos.add(grounded[1:])
            else:
                # f NOT satisfied means f NOT in state
                goal_neg.add(grounded)

        return goal_pos, goal_neg

    def _plan_to_goal(self, current_state: Set[str],
                      goal_pos: Set[str], goal_neg: Set[str],
                      ) -> Optional[List[Tuple[str, List[str]]]]:
        """Use planner to find plan from current_state to goal."""
        from information_gain_aml.algorithms.model_export import to_pddl_string

        domain_pddl = to_pddl_string(self.learner, mode="safe")  # type: ignore[arg-type]
        problem_pddl = self._build_problem_pddl(current_state, goal_pos, goal_neg)
        return self._solve_pddl(domain_pddl, problem_pddl)

    def _build_problem_pddl(self, current_state: Set[str],
                             goal_pos: Set[str], goal_neg: Set[str]) -> str:
        """Build exploration problem PDDL string."""
        domain = self.learner.domain
        lines = ["(define (problem exploration)"]
        lines.append(f"  (:domain {domain.name})")

        # Objects grouped by type
        type_objects: Dict[str, List[str]] = {}
        for obj in domain.objects.values():
            type_objects.setdefault(obj.type, []).append(obj.name)

        all_object_type = all(obj.type == 'object' for obj in domain.objects.values())
        obj_parts = []
        if all_object_type:
            obj_parts.append(' '.join(sorted(domain.objects.keys())))
        else:
            for t in sorted(type_objects.keys()):
                names = sorted(type_objects[t])
                obj_parts.append(f"{' '.join(names)} - {t}")
        lines.append(f"  (:objects {' '.join(obj_parts)})")

        # Init state
        init_atoms = []
        for fluent in sorted(current_state):
            atom = self._fluent_to_pddl_atom(fluent)
            if atom:
                init_atoms.append(atom)
        if init_atoms:
            lines.append(f"  (:init {' '.join(init_atoms)})")
        else:
            lines.append("  (:init)")

        # Goal
        goal_atoms = []
        for fluent in sorted(goal_pos):
            atom = self._fluent_to_pddl_atom(fluent)
            if atom:
                goal_atoms.append(atom)
        for fluent in sorted(goal_neg):
            atom = self._fluent_to_pddl_atom(fluent)
            if atom:
                goal_atoms.append(f"(not {atom})")

        if not goal_atoms:
            lines.append("  (:goal (and))")
        elif len(goal_atoms) == 1:
            lines.append(f"  (:goal {goal_atoms[0]})")
        else:
            lines.append(f"  (:goal (and {' '.join(goal_atoms)}))")

        lines.append(")")
        return '\n'.join(lines)

    def _fluent_to_pddl_atom(self, fluent: str) -> Optional[str]:
        """Convert grounded fluent string to PDDL atom.

        e.g. 'on_a_b' -> '(on a b)', 'handempty' -> '(handempty)'
        Uses domain predicates to disambiguate parsing.
        """
        for pred_name, pred_sig in self.learner.domain.predicates.items():
            arity = len(pred_sig.parameters)
            if arity == 0:
                if fluent == pred_name:
                    return f"({pred_name})"
            else:
                prefix = pred_name + '_'
                if fluent.startswith(prefix):
                    args_str = fluent[len(prefix):]
                    args = args_str.split('_')
                    if (len(args) == arity
                            and all(a in self.learner.domain.objects for a in args)):
                        return f"({pred_name} {' '.join(args)})"
        return None

    def _solve_pddl(self, domain_pddl: str, problem_pddl: str,
                    ) -> Optional[List[Tuple[str, List[str]]]]:
        """Write PDDL to temp files and solve."""
        domain_path = None
        problem_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.pddl', delete=False
            ) as df:
                df.write(domain_pddl)
                domain_path = df.name

            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.pddl', delete=False
            ) as pf:
                pf.write(problem_pddl)
                problem_path = pf.name

            return self._call_planner(domain_path, problem_path)
        except Exception as e:
            logger.debug(f"PDDL solve failed: {e}")
            return None
        finally:
            for path in (domain_path, problem_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def _call_planner(self, domain_path: str, problem_path: str,
                      ) -> Optional[List[Tuple[str, List[str]]]]:
        """Call unified_planning OneshotPlanner on PDDL files.

        Returns list of (action_name, objects) or None if no plan found.
        """
        try:
            import unified_planning as up
            from unified_planning.io import PDDLReader as UPReader
            from unified_planning.shortcuts import OneshotPlanner

            reader = UPReader()
            problem = reader.parse_problem(domain_path, problem_path)

            with OneshotPlanner(problem_kind=problem.kind) as planner:
                result = planner.solve(problem, timeout=self.planner_timeout)

                if result.status in (
                    up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING,
                    up.engines.PlanGenerationResultStatus.SOLVED_OPTIMALLY,
                ):
                    plan_actions = []
                    for ai in result.plan.actions:
                        action_name = ai.action.name
                        objects = [str(p) for p in ai.actual_parameters]
                        plan_actions.append((action_name, objects))
                    return plan_actions

                return None
        except Exception as e:
            logger.debug(f"Planner call failed: {e}")
            return None

    def _get_object_bindings(self, param_types: List[str]):
        """Generate all valid injective object bindings for parameter types."""
        domain = self.learner.domain
        object_lists = []
        for t in param_types:
            matching = [
                obj.name for obj in domain.objects.values()
                if obj.type == t or t == 'object' or domain.is_subtype(obj.type, t)
            ]
            object_lists.append(sorted(matching))

        for combo in itertools.product(*object_lists):
            if len(set(combo)) == len(combo):  # injective
                yield list(combo)

    def _is_useless(self, action_name: str, negated_precs: FrozenSet[str]) -> bool:
        """Check if this negated prec combination has already failed planning."""
        useless = self._useless_negated.get(action_name, [])
        return negated_precs in useless

    def record_useless_negation(self, action_name: str, negated_precs):
        """Mark a negated precondition combination as infeasible."""
        fs = frozenset(negated_precs) if not isinstance(negated_precs, frozenset) else negated_precs
        self._useless_negated.setdefault(action_name, [])
        if fs not in self._useless_negated[action_name]:
            self._useless_negated[action_name].append(fs)

    def reset(self):
        """Clear all useless negation tracking."""
        self._useless_negated.clear()
