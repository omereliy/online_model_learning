"""
OLAM Knowledge Reconstructor for Post-Processing Analysis.

This module reconstructs OLAM's learned knowledge by replaying its execution trace
and applying OLAM's learning rules at each step. This enables checkpoint-based
model reconstruction without running OLAM in real-time.

Author: OLAM Refactor Implementation
Date: 2025
"""

from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import re

from .olam_trace_parser import OLAMTraceStep

logger = logging.getLogger(__name__)


@dataclass
class OLAMKnowledge:
    """
    OLAM's internal knowledge representation.

    This mirrors OLAM's internal data structures for learned models.
    """
    # Preconditions (operator -> set of predicates)
    certain_precs: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    uncertain_precs: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    neg_precs: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    useless_precs: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))  # Ruled out predicates

    # Effects (operator -> set of predicates)
    add_effects: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    del_effects: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Uncertain effects (operator -> set of predicates)
    uncertain_add_effects: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    uncertain_del_effects: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Tracking
    observation_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successful_observations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_observations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # History for learning
    successful_states: Dict[str, List[Set[str]]] = field(default_factory=lambda: defaultdict(list))


class OLAMKnowledgeReconstructor:
    """
    Reconstruct OLAM's learned knowledge by replaying execution trace.

    This class implements OLAM's learning rules as described in:
    Lamanna et al. "Online Learning of Action Models for PDDL Planning" (IJCAI 2021)
    """

    def __init__(self, domain_file: Optional[Path] = None):
        """
        Initialize the knowledge reconstructor.

        Args:
            domain_file: Optional domain file to extract operator schemas
        """
        self.operators = set()
        self.domain_file = domain_file

        if domain_file and domain_file.exists():
            self.operators = self._extract_operators(domain_file)
            logger.info(f"Extracted {len(self.operators)} operators from domain")

    def _extract_operators(self, domain_file: Path) -> Set[str]:
        """
        Extract operator names from PDDL domain file.

        Args:
            domain_file: Path to domain PDDL file

        Returns:
            Set of operator names
        """
        operators = set()

        with open(domain_file, 'r') as f:
            content = f.read()

        # Find all action definitions
        action_pattern = r':action\s+(\w+)'
        matches = re.findall(action_pattern, content)
        # Normalize action names to lowercase for consistent comparison
        operators.update([m.lower() for m in matches])

        logger.debug(f"Found operators: {sorted(operators)}")
        return operators

    def replay_trace(self,
                    trace: List[OLAMTraceStep],
                    until_iteration: Optional[int] = None) -> OLAMKnowledge:
        """
        Replay execution trace to reconstruct knowledge.

        Applies OLAM's learning rules at each step to build up the model.

        Args:
            trace: List of execution trace steps
            until_iteration: Stop at this iteration (None = replay all)

        Returns:
            Reconstructed knowledge at specified point
        """
        logger.info(f"Replaying trace with {len(trace)} steps")

        # Initialize empty knowledge
        knowledge = OLAMKnowledge()

        # Initialize operators if known
        for op in self.operators:
            knowledge.certain_precs[op] = set()
            knowledge.uncertain_precs[op] = set()
            knowledge.neg_precs[op] = set()
            knowledge.add_effects[op] = set()
            knowledge.del_effects[op] = set()

        # Track first observations for initial precondition hypothesis
        first_observation = defaultdict(bool)

        # Replay trace
        for step in trace:
            if until_iteration is not None and step.iteration > until_iteration:
                logger.debug(f"Stopping replay at iteration {until_iteration}")
                break

            # Extract operator name from action
            operator = self._extract_operator(step.action)

            # Update observation counts
            knowledge.observation_count[operator] += 1

            if step.success:
                knowledge.successful_observations[operator] += 1

                # Learn from successful execution
                self._learn_from_success(
                    knowledge=knowledge,
                    operator=operator,
                    action=step.action,
                    state_before=step.state_before,
                    state_after=step.state_after,
                    is_first=not first_observation[operator]
                )

                # Record successful state for future negative precondition learning
                knowledge.successful_states[operator].append(step.state_before)
                first_observation[operator] = True

            else:
                knowledge.failed_observations[operator] += 1

                # Learn from failed execution
                self._learn_from_failure(
                    knowledge=knowledge,
                    operator=operator,
                    action=step.action,
                    failed_state=step.state_before
                )

        # Post-process to finalize uncertain preconditions
        self._finalize_uncertain_preconditions(knowledge)

        logger.info(f"Replay complete. Learned {len(knowledge.certain_precs)} operators")
        return knowledge

    def _extract_operator(self, action: str) -> str:
        """
        Extract operator name from grounded action.

        Args:
            action: Grounded action like "pick-up(a)" or "stack(a,b)"

        Returns:
            Operator name like "pick-up" or "stack"
        """
        # Handle both formats: "operator(args)" and "operator"
        if '(' in action:
            operator = action.split('(')[0]
        else:
            operator = action
        # Normalize to lowercase for consistent comparison
        return operator.lower()

    def _extract_parameters(self, action: str) -> List[str]:
        """
        Extract parameter values from grounded action.

        Args:
            action: Grounded action like "pick-up(a)" or "stack(a,b)"

        Returns:
            List of parameters like ["a"] or ["a", "b"]
        """
        if '(' not in action:
            return []

        # Extract content between parentheses
        match = re.match(r'[^(]+\(([^)]*)\)', action)
        if not match:
            return []

        params_str = match.group(1)
        if not params_str:
            return []

        # Split by comma and strip whitespace
        return [p.strip() for p in params_str.split(',')]

    def _learn_from_success(self,
                           knowledge: OLAMKnowledge,
                           operator: str,
                           action: str,
                           state_before: Set[str],
                           state_after: Optional[Set[str]],
                           is_first: bool) -> None:
        """
        Apply OLAM's learning rules for successful action execution.

        OLAM's approach:
        1. First observation: All state predicates are potential preconditions
        2. Subsequent observations: Intersect with previous preconditions
        3. Effects: Difference between before and after states

        Args:
            knowledge: Knowledge structure to update
            operator: Operator name
            action: Grounded action
            state_before: State before execution
            state_after: State after execution
            is_first: Whether this is the first observation
        """
        # Convert to lifted predicates
        lifted_state = self._ground_to_lifted(state_before, action)

        # Learn preconditions
        if is_first:
            # First observation: all predicates are candidates
            knowledge.certain_precs[operator] = lifted_state.copy()
            logger.debug(f"Initial preconditions for {operator}: {len(lifted_state)} predicates")
        else:
            # Subsequent observations: intersection
            if knowledge.certain_precs[operator]:
                old_count = len(knowledge.certain_precs[operator])
                knowledge.certain_precs[operator] &= lifted_state
                new_count = len(knowledge.certain_precs[operator])

                if old_count != new_count:
                    removed = old_count - new_count
                    logger.debug(f"Refined {operator} preconditions: removed {removed} predicates")

                    # Move removed predicates to uncertain
                    # (This is a simplification - OLAM has more complex logic)

        # Learn effects if we have the after state
        if state_after is not None:
            # Add effects: predicates that appear
            add_effects = state_after - state_before
            if add_effects:
                lifted_add = self._ground_to_lifted(add_effects, action)
                knowledge.add_effects[operator].update(lifted_add)

            # Delete effects: predicates that disappear
            del_effects = state_before - state_after
            if del_effects:
                lifted_del = self._ground_to_lifted(del_effects, action)
                knowledge.del_effects[operator].update(lifted_del)

    def _learn_from_failure(self,
                           knowledge: OLAMKnowledge,
                           operator: str,
                           action: str,
                           failed_state: Set[str]) -> None:
        """
        Apply OLAM's learning rules for failed action execution.

        OLAM learns negative preconditions by comparing failed states
        with previously successful states.

        Args:
            knowledge: Knowledge structure to update
            operator: Operator name
            action: Grounded action that failed
            failed_state: State where action failed
        """
        # If we have no successful observations, we can't learn much
        if operator not in knowledge.successful_states or not knowledge.successful_states[operator]:
            logger.debug(f"No successful states for {operator}, cannot learn from failure")
            return

        # Find predicates that are missing compared to successful executions
        # (This is a simplified version of OLAM's algorithm)
        lifted_failed = self._ground_to_lifted(failed_state, action)

        # Find predicates that appear in all successful states but not in failed state
        common_in_successful = None
        for successful_state in knowledge.successful_states[operator]:
            lifted_successful = self._ground_to_lifted(successful_state, action)
            if common_in_successful is None:
                common_in_successful = lifted_successful
            else:
                common_in_successful &= lifted_successful

        if common_in_successful:
            # Predicates that are always present in successful states but missing in failed
            missing_predicates = common_in_successful - lifted_failed
            if missing_predicates:
                knowledge.neg_precs[operator].update(missing_predicates)
                logger.debug(f"Learned {len(missing_predicates)} negative preconditions for {operator}")

    def _ground_to_lifted(self, predicates: Set[str], action: str) -> Set[str]:
        """
        Convert grounded predicates to lifted (parameterized) form.

        OLAM uses ?param_1, ?param_2, ... convention for parameters.

        Args:
            predicates: Set of grounded predicates like "(holding a)"
            action: Grounded action like "pick-up(a)"

        Returns:
            Set of lifted predicates like "(holding ?param_1)"
        """
        # Extract parameter bindings from action
        parameters = self._extract_parameters(action)
        if not parameters:
            # No parameters, predicates remain as-is
            return predicates.copy()

        lifted = set()
        for pred in predicates:
            lifted_pred = pred

            # Replace each parameter with its variable
            for i, param in enumerate(parameters):
                # Use OLAM's convention: ?param_1, ?param_2, ...
                param_var = f"?param_{i+1}"

                # Replace the parameter in the predicate
                # Handle different formats
                # Standard PDDL: (predicate obj1 obj2)
                lifted_pred = re.sub(
                    rf'\b{re.escape(param)}\b',
                    param_var,
                    lifted_pred
                )

            lifted.add(lifted_pred)

        return lifted

    def _lifted_to_ground(self, predicates: Set[str], action: str) -> Set[str]:
        """
        Convert lifted predicates to grounded form (reverse of ground_to_lifted).

        Args:
            predicates: Set of lifted predicates like "(holding ?param_1)"
            action: Grounded action like "pick-up(a)"

        Returns:
            Set of grounded predicates like "(holding a)"
        """
        parameters = self._extract_parameters(action)
        if not parameters:
            return predicates.copy()

        grounded = set()
        for pred in predicates:
            grounded_pred = pred

            # Replace each variable with its grounded value
            for i, param in enumerate(parameters):
                param_var = f"?param_{i+1}"
                grounded_pred = grounded_pred.replace(param_var, param)

            grounded.add(grounded_pred)

        return grounded

    def _finalize_uncertain_preconditions(self, knowledge: OLAMKnowledge) -> None:
        """
        Post-process to identify uncertain preconditions.

        Uncertain preconditions are those that appear in some but not all
        successful executions. This is a simplification of OLAM's approach.

        Args:
            knowledge: Knowledge structure to update
        """
        # This is a placeholder - OLAM's actual logic is more complex
        # and involves tracking predicate frequency across observations
        pass

    def replay_to_checkpoint(self,
                            trace: List[OLAMTraceStep],
                            checkpoint: int) -> OLAMKnowledge:
        """
        Replay trace up to a specific checkpoint iteration.

        Args:
            trace: Complete execution trace
            checkpoint: Target iteration number

        Returns:
            Knowledge state at checkpoint
        """
        return self.replay_trace(trace, until_iteration=checkpoint)

    def export_snapshot(self,
                       knowledge: OLAMKnowledge,
                       iteration: int,
                       algorithm: str = "olam") -> Dict:
        """
        Export knowledge in format compatible with ModelReconstructor.

        This creates a snapshot that can be processed by the existing
        model reconstruction infrastructure.

        Args:
            knowledge: OLAM knowledge to export
            iteration: Current iteration number
            algorithm: Algorithm name for metadata

        Returns:
            Snapshot dictionary in standard format
        """
        snapshot = {
            'iteration': iteration,
            'algorithm': algorithm,
            'actions': {},
            'metadata': {
                'reconstructed_from_trace': True,
                'total_observations': sum(knowledge.observation_count.values()),
                'successful_observations': sum(knowledge.successful_observations.values()),
                'failed_observations': sum(knowledge.failed_observations.values())
            }
        }

        # Export each operator's learned knowledge
        all_operators = set()
        all_operators.update(knowledge.certain_precs.keys())
        all_operators.update(knowledge.add_effects.keys())
        all_operators.update(self.operators)

        for operator in all_operators:
            # Convert sets to sorted lists for JSON serialization
            snapshot['actions'][operator] = {
                'parameters': [],  # Would need domain info to extract
                'certain_preconditions': sorted(list(knowledge.certain_precs.get(operator, set()))),
                'uncertain_preconditions': sorted(list(knowledge.uncertain_precs.get(operator, set()))),
                'negative_preconditions': sorted(list(knowledge.neg_precs.get(operator, set()))),
                'useless_preconditions': sorted(list(knowledge.useless_precs.get(operator, set()))),
                'certain_add_effects': sorted(list(knowledge.add_effects.get(operator, set()))),
                'certain_del_effects': sorted(list(knowledge.del_effects.get(operator, set()))),
                'uncertain_add_effects': sorted(list(knowledge.uncertain_add_effects.get(operator, set()))),
                'uncertain_del_effects': sorted(list(knowledge.uncertain_del_effects.get(operator, set()))),
                'observations': knowledge.observation_count.get(operator, 0),
                'successes': knowledge.successful_observations.get(operator, 0),
                'failures': knowledge.failed_observations.get(operator, 0)
            }

        return snapshot

    def reconstruct_from_exports(self, exports: Dict[str, Dict]) -> OLAMKnowledge:
        """
        Reconstruct knowledge from OLAM's JSON exports.

        This is used when we have OLAM's final exported model
        and want to convert it to our format.

        Args:
            exports: Dictionary with OLAM's exported JSON files

        Returns:
            Reconstructed knowledge
        """
        knowledge = OLAMKnowledge()

        # Load certain preconditions
        if 'certain_precs' in exports:
            for operator, precs in exports['certain_precs'].items():
                knowledge.certain_precs[operator] = set(precs) if precs else set()

        # Load uncertain preconditions
        if 'uncertain_precs' in exports:
            for operator, precs in exports['uncertain_precs'].items():
                knowledge.uncertain_precs[operator] = set(precs) if precs else set()

        # Load negative preconditions
        if 'neg_precs' in exports:
            for operator, precs in exports['neg_precs'].items():
                knowledge.neg_precs[operator] = set(precs) if precs else set()

        # Load certain add effects
        if 'add_effects' in exports:
            for operator, effects in exports['add_effects'].items():
                knowledge.add_effects[operator] = set(effects) if effects else set()

        # Load certain delete effects
        if 'del_effects' in exports:
            for operator, effects in exports['del_effects'].items():
                knowledge.del_effects[operator] = set(effects) if effects else set()

        # Load uncertain add effects
        if 'uncertain_add_effects' in exports:
            for operator, effects in exports['uncertain_add_effects'].items():
                knowledge.uncertain_add_effects[operator] = set(effects) if effects else set()

        # Load uncertain delete effects
        if 'uncertain_del_effects' in exports:
            for operator, effects in exports['uncertain_del_effects'].items():
                knowledge.uncertain_del_effects[operator] = set(effects) if effects else set()

        # Load useless (ruled out) preconditions
        if 'useless_pos_precs' in exports:
            for operator, precs in exports['useless_pos_precs'].items():
                knowledge.useless_precs[operator] = set(precs) if precs else set()

        logger.info(f"Reconstructed knowledge for {len(knowledge.certain_precs)} operators from exports")
        return knowledge

    def compare_knowledge(self,
                         knowledge1: OLAMKnowledge,
                         knowledge2: OLAMKnowledge) -> Dict[str, float]:
        """
        Compare two knowledge states for validation.

        Used to verify that trace replay produces the same results
        as the original OLAM execution.

        Args:
            knowledge1: First knowledge state
            knowledge2: Second knowledge state

        Returns:
            Dictionary with similarity metrics
        """
        metrics = {}

        # Get all operators
        all_operators = set()
        all_operators.update(knowledge1.certain_precs.keys())
        all_operators.update(knowledge2.certain_precs.keys())

        if not all_operators:
            return {'similarity': 1.0}

        # Compare each component
        similarities = []

        for operator in all_operators:
            # Compare certain preconditions
            set1 = knowledge1.certain_precs.get(operator, set())
            set2 = knowledge2.certain_precs.get(operator, set())
            if set1 or set2:
                similarity = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 1.0
                similarities.append(similarity)
                metrics[f'{operator}_certain_prec_similarity'] = similarity

            # Compare effects
            add1 = knowledge1.add_effects.get(operator, set())
            add2 = knowledge2.add_effects.get(operator, set())
            if add1 or add2:
                similarity = len(add1 & add2) / len(add1 | add2) if (add1 | add2) else 1.0
                similarities.append(similarity)
                metrics[f'{operator}_add_effect_similarity'] = similarity

            del1 = knowledge1.del_effects.get(operator, set())
            del2 = knowledge2.del_effects.get(operator, set())
            if del1 or del2:
                similarity = len(del1 & del2) / len(del1 | del2) if (del1 | del2) else 1.0
                similarities.append(similarity)
                metrics[f'{operator}_del_effect_similarity'] = similarity

        # Overall similarity
        metrics['overall_similarity'] = sum(similarities) / len(similarities) if similarities else 1.0

        return metrics