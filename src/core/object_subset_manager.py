"""
Object Subset Manager for Information Gain algorithm.

Manages selection and rotation of object subsets to reduce grounding space
while maintaining complete learning through lifted knowledge transfer.

Key concepts:
- Type requirements: Minimum objects per type needed for action parameter coverage
- Subset selection: Choose minimal objects covering requirements + spare
- Subset rotation: Dismiss used objects, select new subset when converged
- Exhaustion: When no more subsets can be formed, signal termination
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.lifted_domain import LiftedDomainKnowledge

logger = logging.getLogger(__name__)


@dataclass
class TypeRequirement:
    """
    Type requirement for action parameter coverage.

    Attributes:
        type_name: Name of the type
        min_objects: Minimum objects needed (max across all action params)
        spare_objects: Additional spare objects to include
    """
    type_name: str
    min_objects: int
    spare_objects: int

    @property
    def total_required(self) -> int:
        """Total objects to select for this type."""
        return self.min_objects + self.spare_objects


class ObjectSubsetManager:
    """
    Manages object subset selection and rotation for Information Gain learning.

    This component enables computational efficiency by limiting the grounding
    space to a minimal subset of objects that covers all action parameter
    type requirements. Since learned knowledge is lifted (parameterized),
    it transfers across subset rotations.

    Responsibilities:
    1. Compute type requirements from action schemas
    2. Select minimal subsets covering requirements
    3. Track used/available objects per type
    4. Handle subset rotation when converged or iteration limit reached

    Example:
        manager = ObjectSubsetManager(domain, spare_objects_per_type=1)
        active_objects = manager.get_active_object_names()
        # ... use active_objects for grounding ...
        if converged:
            if manager.rotate_subset():
                # Continue with new subset
            else:
                # All objects exhausted
    """

    def __init__(self,
                 domain: 'LiftedDomainKnowledge',
                 spare_objects_per_type: int = 1,
                 random_seed: Optional[int] = None,
                 defer_initial_selection: bool = False):
        """
        Initialize the subset manager.

        Args:
            domain: Domain knowledge with actions, types, and objects
            spare_objects_per_type: Extra objects per type beyond minimum requirement
            random_seed: Optional seed for reproducible subset selection
            defer_initial_selection: If True, don't select initial subset (wait for state-aware selection)
        """
        self.domain = domain
        self.spare_objects_per_type = spare_objects_per_type

        if random_seed is not None:
            random.seed(random_seed)

        # Computed requirements: {type_name: TypeRequirement}
        self.type_requirements: Dict[str, TypeRequirement] = {}

        # Object tracking per type (using declared type, not including subtypes)
        self.all_objects_by_type: Dict[str, List[str]] = {}
        self.available_objects_by_type: Dict[str, List[str]] = {}
        self.used_objects_by_type: Dict[str, Set[str]] = {}

        # Current active subset: {type_name: [obj_names]}
        self.active_subset: Dict[str, List[str]] = {}

        # Rotation tracking
        self.subset_rotation_count = 0
        self._exhausted = False
        self._initial_selection_deferred = defer_initial_selection
        self._max_possible_subsets = 0  # Computed after initialization

        # Initialize
        self._initialize(defer_initial_selection)

    def _initialize(self, defer_initial_selection: bool = False) -> None:
        """Initialize type requirements and object pools."""
        self._compute_type_requirements()
        self._group_objects_by_type()
        self._compute_max_possible_subsets()
        if not defer_initial_selection:
            self.select_new_subset()

    def _compute_max_possible_subsets(self) -> None:
        """
        Compute maximum number of subsets that can be formed.

        This is the minimum of (total_objects / objects_per_subset) across all types.
        Used to determine if fallback to all objects is appropriate.
        """
        if not self.type_requirements:
            self._max_possible_subsets = 0
            return

        max_subsets = float('inf')
        for type_name, requirement in self.type_requirements.items():
            total_objects = len(self._get_all_objects_for_type(type_name))
            objects_per_subset = requirement.total_required
            if objects_per_subset > 0:
                type_max_subsets = total_objects // objects_per_subset
                max_subsets = min(max_subsets, type_max_subsets)

        self._max_possible_subsets = int(max_subsets) if max_subsets != float('inf') else 0
        logger.info(f"Max possible subsets: {self._max_possible_subsets}")

    def _compute_type_requirements(self) -> None:
        """
        Compute minimum objects needed per type from action parameter signatures.

        Algorithm:
        1. For each action, count objects needed per parameter type
        2. Take global max across all actions for each type
        3. Add spare objects (capped by availability later)
        """
        type_max_counts: Dict[str, int] = {}

        for action_name, action in self.domain.lifted_actions.items():
            # Count objects needed per type for this action
            type_counts: Dict[str, int] = {}

            for param in action.parameters:
                param_type = param.type
                type_counts[param_type] = type_counts.get(param_type, 0) + 1

            # Update global max
            for type_name, count in type_counts.items():
                type_max_counts[type_name] = max(
                    type_max_counts.get(type_name, 0), count
                )

            logger.debug(f"Action {action_name}: type counts = {type_counts}")

        # Create TypeRequirement for each type
        for type_name, min_count in type_max_counts.items():
            self.type_requirements[type_name] = TypeRequirement(
                type_name=type_name,
                min_objects=min_count,
                spare_objects=self.spare_objects_per_type
            )

        logger.info(
            f"Computed type requirements: "
            f"{[(t, r.min_objects, r.total_required) for t, r in self.type_requirements.items()]}"
        )

    def _group_objects_by_type(self) -> None:
        """Group all domain objects by their declared type."""
        for obj_name, obj_info in self.domain.objects.items():
            obj_type = obj_info.type

            if obj_type not in self.all_objects_by_type:
                self.all_objects_by_type[obj_type] = []
                self.available_objects_by_type[obj_type] = []
                self.used_objects_by_type[obj_type] = set()

            self.all_objects_by_type[obj_type].append(obj_name)
            self.available_objects_by_type[obj_type].append(obj_name)

        logger.debug(
            f"Objects by type: "
            f"{[(t, len(objs)) for t, objs in self.all_objects_by_type.items()]}"
        )

    def _get_available_objects_for_type(self, type_name: str) -> List[str]:
        """
        Get available objects matching a type (including subtypes).

        Args:
            type_name: Type name to match

        Returns:
            List of available object names
        """
        available = []

        # Get objects of exact type
        available.extend(self.available_objects_by_type.get(type_name, []))

        # Get objects of subtypes
        type_info = self.domain.get_type(type_name)
        if type_info:
            for child_type in type_info.children:
                available.extend(self._get_available_objects_for_type(child_type))

        return available

    def _get_all_objects_for_type(self, type_name: str) -> List[str]:
        """
        Get all objects matching a type (including subtypes).

        Args:
            type_name: Type name to match

        Returns:
            List of all object names (used and available)
        """
        all_objs = []

        # Get objects of exact type
        all_objs.extend(self.all_objects_by_type.get(type_name, []))

        # Get objects of subtypes
        type_info = self.domain.get_type(type_name)
        if type_info:
            for child_type in type_info.children:
                all_objs.extend(self._get_all_objects_for_type(child_type))

        return all_objs

    def select_new_subset(self) -> Dict[str, List[str]]:
        """
        Select a new subset of objects covering type requirements.

        Returns:
            Dict mapping type names to selected object lists
        """
        new_subset: Dict[str, List[str]] = {}

        for type_name, requirement in self.type_requirements.items():
            # Get available objects for this type (including subtypes)
            available = self._get_available_objects_for_type(type_name)
            total_available = len(available)

            needed = requirement.total_required
            to_select = min(needed, total_available)

            if to_select < requirement.min_objects:
                if self.subset_rotation_count == 0:
                    # First subset - use all available with warning
                    logger.warning(
                        f"Type {type_name}: Only {total_available} objects available, "
                        f"but {requirement.min_objects} required. Using all available."
                    )
                    to_select = total_available
                else:
                    # Subsequent subsets - not enough objects
                    logger.info(
                        f"Type {type_name} exhausted - not enough for new subset "
                        f"(need {requirement.min_objects}, have {total_available})"
                    )
                    self._exhausted = True
                    return self.active_subset

            if to_select > 0:
                selected = random.sample(available, to_select)
                new_subset[type_name] = selected

                # Mark selected objects as used
                for obj in selected:
                    # Get the actual type of this object
                    obj_type = self.domain.objects[obj].type
                    if obj in self.available_objects_by_type.get(obj_type, []):
                        self.available_objects_by_type[obj_type].remove(obj)
                    self.used_objects_by_type.setdefault(obj_type, set()).add(obj)
            else:
                new_subset[type_name] = []

        self.active_subset = new_subset
        self.subset_rotation_count += 1

        logger.info(
            f"Selected subset {self.subset_rotation_count}: "
            f"{[(t, objs) for t, objs in new_subset.items()]}"
        )
        return new_subset

    def rotate_subset(self) -> bool:
        """
        Rotate to a new object subset.

        Returns:
            True if rotation succeeded, False if all objects exhausted
        """
        if self._exhausted:
            return False

        # Check if we can form a new subset
        for type_name, requirement in self.type_requirements.items():
            available = self._get_available_objects_for_type(type_name)
            if len(available) < requirement.min_objects:
                self._exhausted = True
                logger.info(
                    f"All objects exhausted - cannot rotate to new subset "
                    f"(type {type_name}: need {requirement.min_objects}, have {len(available)})"
                )
                return False

        self.select_new_subset()
        return True

    def get_active_object_names(self) -> Set[str]:
        """
        Get flat set of all active object names.

        Returns:
            Set of object names in current active subset
        """
        names = set()
        for obj_list in self.active_subset.values():
            names.update(obj_list)
        return names

    def all_objects_exhausted(self) -> bool:
        """
        Check if all objects have been used and no more subsets can be formed.

        Returns:
            True if exhausted
        """
        return self._exhausted

    def should_fallback_to_all_objects(self) -> bool:
        """
        Check if fallback to all objects is appropriate when exhausted.

        Fallback is only appropriate when there are fewer than 2 possible subsets,
        meaning subset selection doesn't provide meaningful benefit.

        Returns:
            True if fallback to all objects should be used
        """
        return self._max_possible_subsets < 2

    def get_max_possible_subsets(self) -> int:
        """
        Get the maximum number of subsets that can be formed.

        Returns:
            Maximum number of non-overlapping subsets
        """
        return self._max_possible_subsets

    def get_status(self) -> Dict:
        """
        Get current status for logging/debugging.

        Returns:
            Dict with rotation count, exhaustion state, active subset, available counts
        """
        return {
            'rotation_count': self.subset_rotation_count,
            'exhausted': self._exhausted,
            'active_subset': {t: list(objs) for t, objs in self.active_subset.items()},
            'available_counts': {
                t: len(objs) for t, objs in self.available_objects_by_type.items()
            },
            'type_requirements': {
                t: {'min': r.min_objects, 'total': r.total_required}
                for t, r in self.type_requirements.items()
            }
        }

    def _extract_objects_from_state(self, state: Set[str]) -> Set[str]:
        """
        Extract object names from grounded fluents in state.

        Parses fluents like 'on_b3_b9' to extract objects ['b3', 'b9'].

        Args:
            state: Current state as set of grounded fluent strings

        Returns:
            Set of object names found in state predicates
        """
        objects = set()
        for fluent in state:
            # Handle negated fluents
            if fluent.startswith('¬'):
                fluent = fluent[1:]

            # Parse fluent: "on_b3_b9" -> ["on", "b3", "b9"]
            parts = fluent.split('_')
            if len(parts) > 1:
                # All parts after predicate name could be objects
                for part in parts[1:]:
                    if part in self.domain.objects:
                        objects.add(part)

        return objects

    def _partition_objects_by_type(self, objects: Set[str]) -> Dict[str, List[str]]:
        """
        Partition objects by their declared type.

        Args:
            objects: Set of object names

        Returns:
            Dict mapping type names to lists of objects
        """
        by_type: Dict[str, List[str]] = {}
        for obj in objects:
            if obj in self.domain.objects:
                obj_type = self.domain.objects[obj].type
                if obj_type not in by_type:
                    by_type[obj_type] = []
                by_type[obj_type].append(obj)
        return by_type

    def select_state_aware_subset(self, state: Set[str]) -> Dict[str, List[str]]:
        """
        Select subset prioritizing objects from current state predicates.

        This ensures objects that are "active" in the current state (appearing
        in true predicates) are included in the subset, making applicable
        actions more likely to be generated.

        Args:
            state: Current state as set of grounded fluents (e.g., {'on_b3_b9', 'clear_b3'})

        Returns:
            Dict mapping type names to selected object lists
        """
        # 1. Extract objects from state predicates
        state_objects = self._extract_objects_from_state(state)
        logger.debug(f"State-aware selection: Found {len(state_objects)} objects in state: {state_objects}")

        # 2. Partition state objects by type
        state_objects_by_type = self._partition_objects_by_type(state_objects)

        # 3. Build subset: state objects first, then fill with available
        new_subset: Dict[str, List[str]] = {}

        for type_name, requirement in self.type_requirements.items():
            # Start with state objects of this type
            type_state_objects = state_objects_by_type.get(type_name, [])
            available = self._get_available_objects_for_type(type_name)

            selected = []

            # Add state objects first (prioritized) - even if they exceed total_required
            for obj in type_state_objects:
                if obj in available and obj not in selected:
                    selected.append(obj)

            # Fill remaining with other available objects (up to total_required)
            remaining_needed = max(0, requirement.total_required - len(selected))
            other_available = [o for o in available if o not in selected]

            if remaining_needed > 0 and other_available:
                additional = random.sample(
                    other_available,
                    min(remaining_needed, len(other_available))
                )
                selected.extend(additional)

            # Check minimum requirement
            if len(selected) < requirement.min_objects:
                if self.subset_rotation_count == 0:
                    # First subset - use all available with warning
                    logger.warning(
                        f"Type {type_name}: Only {len(selected)} objects available "
                        f"(including {len(type_state_objects)} from state), "
                        f"but {requirement.min_objects} required."
                    )
                else:
                    # Subsequent subsets - not enough objects
                    logger.info(
                        f"Type {type_name} exhausted for state-aware selection "
                        f"(need {requirement.min_objects}, have {len(selected)})"
                    )
                    self._exhausted = True
                    return self.active_subset

            new_subset[type_name] = selected

            # Mark selected objects as used
            for obj in selected:
                obj_type = self.domain.objects[obj].type
                if obj in self.available_objects_by_type.get(obj_type, []):
                    self.available_objects_by_type[obj_type].remove(obj)
                self.used_objects_by_type.setdefault(obj_type, set()).add(obj)

        self.active_subset = new_subset
        self.subset_rotation_count += 1

        logger.info(
            f"Selected state-aware subset {self.subset_rotation_count}: "
            f"{[(t, objs) for t, objs in new_subset.items()]} "
            f"(prioritized {len(state_objects)} state objects)"
        )
        return new_subset

    def augment_with_state_objects(self, state: Set[str]) -> bool:
        """
        Augment current subset with objects from current state.

        Ensures objects relevant to current state are always included,
        without a full rotation. This handles state changes during learning.

        Args:
            state: Current state as set of grounded fluents

        Returns:
            True if subset was augmented, False if no changes needed
        """
        state_objects = self._extract_objects_from_state(state)
        current_active = self.get_active_object_names()

        augmented = False
        for obj in state_objects:
            if obj not in current_active and obj in self.domain.objects:
                obj_type = self.domain.objects[obj].type
                # Add to active subset
                if obj_type in self.active_subset:
                    self.active_subset[obj_type].append(obj)
                else:
                    self.active_subset[obj_type] = [obj]
                augmented = True
                logger.debug(f"Augmented subset with state object: {obj} (type: {obj_type})")

        if augmented:
            logger.info(f"Augmented subset with state objects. New active: {self.get_active_object_names()}")

        return augmented

    def rotate_state_aware(self, state: Set[str]) -> bool:
        """
        Rotate to a new object subset with state awareness.

        Args:
            state: Current state as set of grounded fluents

        Returns:
            True if rotation succeeded, False if all objects exhausted
        """
        if self._exhausted:
            return False

        # Check if we can form a new subset
        for type_name, requirement in self.type_requirements.items():
            available = self._get_available_objects_for_type(type_name)
            if len(available) < requirement.min_objects:
                self._exhausted = True
                logger.info(
                    f"All objects exhausted - cannot rotate to new subset "
                    f"(type {type_name}: need {requirement.min_objects}, have {len(available)})"
                )
                return False

        self.select_state_aware_subset(state)
        return True

    def reset(self) -> None:
        """Reset the manager to initial state (all objects available)."""
        logger.info("Resetting ObjectSubsetManager")

        # Reset available objects
        for type_name in self.all_objects_by_type:
            self.available_objects_by_type[type_name] = list(
                self.all_objects_by_type[type_name]
            )
            self.used_objects_by_type[type_name] = set()

        self.active_subset = {}
        self.subset_rotation_count = 0
        self._exhausted = False

        # Select first subset
        self.select_new_subset()
