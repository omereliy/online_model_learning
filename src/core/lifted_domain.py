"""
Lifted Domain Knowledge: Intermediate domain representation.

Represents domain knowledge in lifted (parameter-bound) form, supporting the
spectrum from sound to complete knowledge for learning algorithms.

Key concepts:
- Lifted actions: action schemas with parameters
- Parameter-bound literals: predicates with parameter variables (e.g., on(?x,?y))
- Partial knowledge: support for uncertain preconditions/effects (for learning)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from unified_planning.model import Problem

logger = logging.getLogger(__name__)


@dataclass
class Parameter:
    """
    Action or predicate parameter.

    Attributes:
        name: Parameter name (e.g., '?x', '?y')
        type: Type name (e.g., 'block', 'location')
    """
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.name} - {self.type}"


@dataclass
class PredicateSignature:
    """
    Predicate signature (name + parameter types).

    Attributes:
        name: Predicate name (e.g., 'on', 'clear')
        parameters: List of parameters
        arity: Number of parameters
    """
    name: str
    parameters: List[Parameter]

    @property
    def arity(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def __str__(self) -> str:
        if self.arity == 0:
            return f"{self.name}()"
        param_str = ', '.join(str(p) for p in self.parameters)
        return f"{self.name}({param_str})"


@dataclass
class LiftedAction:
    """
    Lifted action schema with parameter-bound preconditions and effects.

    Supports both complete knowledge (known preconditions/effects) and
    partial knowledge (uncertain/maybe conditions) for learning algorithms.

    Attributes:
        name: Action name
        parameters: List of parameters
        preconditions: Set of parameter-bound literals (e.g., 'clear(?x)')
        add_effects: Set of parameter-bound literals to add
        del_effects: Set of parameter-bound literals to delete

        # For learning algorithms - partial knowledge
        uncertain_preconditions: Preconditions we're uncertain about
        maybe_add_effects: Possible add effects (not determined yet)
        maybe_del_effects: Possible delete effects (not determined yet)
    """
    name: str
    parameters: List[Parameter]
    preconditions: Set[str] = field(default_factory=set)
    add_effects: Set[str] = field(default_factory=set)
    del_effects: Set[str] = field(default_factory=set)

    # Partial knowledge support for learning
    uncertain_preconditions: Optional[Set[str]] = None
    maybe_add_effects: Optional[Set[str]] = None
    maybe_del_effects: Optional[Set[str]] = None

    @property
    def arity(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def __str__(self) -> str:
        if self.arity == 0:
            return f"{self.name}()"
        param_str = ', '.join(p.name for p in self.parameters)
        return f"{self.name}({param_str})"

    def has_partial_knowledge(self) -> bool:
        """Check if action has uncertain/partial knowledge."""
        return (self.uncertain_preconditions is not None or
                self.maybe_add_effects is not None or
                self.maybe_del_effects is not None)


@dataclass
class ObjectInfo:
    """
    Object information.

    Attributes:
        name: Object name
        type: Type name
    """
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.name} - {self.type}"


@dataclass
class TypeInfo:
    """
    Type information with hierarchy.

    Attributes:
        name: Type name
        parent: Parent type name (None for root 'object')
        children: Set of child type names
    """
    name: str
    parent: Optional[str] = None
    children: Set[str] = field(default_factory=set)

    def __str__(self) -> str:
        if self.parent:
            return f"{self.name} - {self.parent}"
        return self.name


class LiftedDomainKnowledge:
    """
    Represents domain knowledge in lifted form.

    Supports the spectrum from sound (minimal knowledge) to complete
    (full knowledge) for learning algorithms.

    This is the central data structure for domain representation, used by:
    - Learning algorithms (to store learned models)
    - PDDL I/O (to read/write domains)
    - Grounding utilities (to generate grounded actions)
    """

    def __init__(self, name: str):
        """
        Initialize lifted domain knowledge.

        Args:
            name: Domain name
        """
        self.name = name
        self.types: Dict[str, TypeInfo] = {}
        self.lifted_actions: Dict[str, LiftedAction] = {}
        self.predicates: Dict[str, PredicateSignature] = {}
        self.objects: Dict[str, ObjectInfo] = {}

        # Cache for parameter-bound literals (La) per action
        self._La_cache: Dict[str, Set[str]] = {}

        logger.debug(f"Initialized LiftedDomainKnowledge: {name}")

    # ========== Query Methods ==========

    def get_action(self, name: str) -> Optional[LiftedAction]:
        """
        Get lifted action by name.

        Args:
            name: Action name

        Returns:
            LiftedAction or None if not found
        """
        return self.lifted_actions.get(name)

    def get_predicate(self, name: str) -> Optional[PredicateSignature]:
        """
        Get predicate signature by name.

        Args:
            name: Predicate name

        Returns:
            PredicateSignature or None if not found
        """
        return self.predicates.get(name)

    def get_object(self, name: str) -> Optional[ObjectInfo]:
        """
        Get object info by name.

        Args:
            name: Object name

        Returns:
            ObjectInfo or None if not found
        """
        return self.objects.get(name)

    def get_type(self, name: str) -> Optional[TypeInfo]:
        """
        Get type info by name.

        Args:
            name: Type name

        Returns:
            TypeInfo or None if not found
        """
        return self.types.get(name)

    # ========== Type Hierarchy Methods ==========

    def is_subtype(self, child: str, parent: str) -> bool:
        """
        Check if child is a subtype of parent.

        Args:
            child: Child type name
            parent: Parent type name

        Returns:
            True if child is subtype of parent
        """
        # Same type is subtype of itself
        if child == parent:
            return True

        # Everything is subtype of 'object'
        if parent == 'object':
            return True

        # Traverse up the hierarchy
        current = self.types.get(child)
        while current:
            if current.parent == parent:
                return True
            current = self.types.get(current.parent) if current.parent else None

        return False

    def get_type_ancestors(self, type_name: str) -> List[str]:
        """
        Get all ancestors of a type.

        Args:
            type_name: Type name

        Returns:
            List of ancestor type names (immediate parent to root)
        """
        ancestors = []
        current = self.types.get(type_name)

        while current and current.parent:
            ancestors.append(current.parent)
            current = self.types.get(current.parent)

        return ancestors

    def get_objects_of_type(self, type_name: str, include_subtypes: bool = True) -> List[ObjectInfo]:
        """
        Get all objects of a given type.

        Args:
            type_name: Type name
            include_subtypes: If True, include objects of subtypes

        Returns:
            List of matching objects
        """
        if include_subtypes:
            # Get all subtypes
            valid_types = {type_name}
            self._collect_subtypes(type_name, valid_types)

            return [obj for obj in self.objects.values() if obj.type in valid_types]
        else:
            return [obj for obj in self.objects.values() if obj.type == type_name]

    def _collect_subtypes(self, type_name: str, result: Set[str]) -> None:
        """Recursively collect all subtypes."""
        type_info = self.types.get(type_name)
        if type_info:
            for child in type_info.children:
                if child not in result:
                    result.add(child)
                    self._collect_subtypes(child, result)

    # ========== Parameter-Bound Literals ==========

    def get_parameter_bound_literals(self, action_name: str) -> Set[str]:
        """
        Get all parameter-bound literals (La) for an action.

        This generates all possible lifted fluents using the action's parameters,
        including both positive and negative literals. Only generates type-safe
        literals where parameter types match predicate signatures.

        Args:
            action_name: Action name

        Returns:
            Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})

        Example:
            domain.get_parameter_bound_literals('stack')
            # → {'on(?x,?y)', '¬on(?x,?y)', 'clear(?x)', '¬clear(?x)', ...}
        """
        # Check cache first
        if action_name in self._La_cache:
            return self._La_cache[action_name]

        action = self.get_action(action_name)
        if not action:
            logger.warning(f"Action {action_name} not found")
            return set()

        La = set()

        # Get parameter names using standard naming
        num_params = action.arity
        param_names = self._generate_parameter_names(num_params)

        # Create mapping from parameter names to types
        param_types = {param_names[i]: action.parameters[i].type for i in range(num_params)}

        # For each predicate, generate type-safe parameter combinations
        import itertools
        for pred_name, pred_sig in self.predicates.items():
            if pred_sig.arity == 0:
                # Propositional - no parameters
                La.add(pred_name)
                La.add(f"¬{pred_name}")
            else:
                # Generate combinations of parameters
                for combo in itertools.combinations_with_replacement(param_names, pred_sig.arity):
                    if len(combo) == pred_sig.arity:
                        # Check type compatibility: each parameter's type must match predicate's expected type
                        types_match = all(
                            self.is_subtype(param_types[param], pred_sig.parameters[i].type)
                            for i, param in enumerate(combo)
                        )

                        if types_match:
                            # Create positive literal
                            literal = f"{pred_name}({','.join(combo)})"
                            La.add(literal)
                            # Create negative literal
                            La.add(f"¬{literal}")

        # Cache result
        self._La_cache[action_name] = La

        logger.debug(f"Generated {len(La)} type-safe parameter-bound literals for {action_name}")
        return La

    @staticmethod
    def _generate_parameter_names(count: int) -> List[str]:
        """
        Generate standard parameter names.

        Args:
            count: Number of parameters

        Returns:
            List of parameter names ['?x', '?y', '?z', ...]
        """
        names = 'xyzuvwpqrst'
        return [f"?{names[i]}" if i < len(names) else f"?p{i}" for i in range(count)]

    # ========== Factory Methods ==========

    @staticmethod
    def _extract_literals_from_expression(expr, action_parameters, adapter) -> Set[str]:
        """
        Extract literals from potentially compound expression.

        Handles AND, OR, NOT, and atomic fluent expressions.

        Args:
            expr: UP FNode expression
            action_parameters: Action parameters list
            adapter: UPAdapter for conversions

        Returns:
            Set of parameter-bound literal strings
        """
        literals = set()

        # Handle AND expressions - extract all conjuncts
        if hasattr(expr, 'is_and') and expr.is_and():
            for arg in expr.args:
                literals.update(
                    LiftedDomainKnowledge._extract_literals_from_expression(
                        arg, action_parameters, adapter
                    )
                )

        # Handle NOT expressions - add negation prefix
        elif hasattr(expr, 'is_not') and expr.is_not():
            if expr.args:
                inner = expr.args[0]
                inner_literals = LiftedDomainKnowledge._extract_literals_from_expression(
                    inner, action_parameters, adapter
                )
                # Add negation to each literal
                literals.update(f"¬{lit}" for lit in inner_literals)

        # Handle atomic fluent expressions
        elif hasattr(expr, 'is_fluent_exp') and expr.is_fluent_exp():
            lit_str = adapter.up_expression_to_parameter_bound(expr, action_parameters)
            if lit_str:
                literals.add(lit_str)

        # Handle OR expressions (less common in preconditions, but possible)
        elif hasattr(expr, 'is_or') and expr.is_or():
            # For ORs, we could represent as CNF, but for now just extract all disjuncts
            for arg in expr.args:
                literals.update(
                    LiftedDomainKnowledge._extract_literals_from_expression(
                        arg, action_parameters, adapter
                    )
                )

        return literals

    @classmethod
    def from_up_problem(cls, up_problem: Problem, adapter) -> 'LiftedDomainKnowledge':
        """
        Create LiftedDomainKnowledge from UP Problem.

        Args:
            up_problem: Unified Planning Problem
            adapter: UPAdapter instance for conversions

        Returns:
            LiftedDomainKnowledge instance
        """
        domain = cls(up_problem.name)

        # Extract types
        for user_type in up_problem.user_types:
            type_info = TypeInfo(
                name=str(user_type.name),
                parent=str(user_type.father.name) if user_type.father else 'object'
            )
            domain.types[type_info.name] = type_info

        # Build type hierarchy (parent -> children)
        for type_info in domain.types.values():
            if type_info.parent:
                parent = domain.types.get(type_info.parent)
                if parent:
                    parent.children.add(type_info.name)

        # Extract objects
        for obj in up_problem.all_objects:
            obj_info = ObjectInfo(
                name=obj.name,
                type=str(obj.type.name) if hasattr(obj.type, 'name') else str(obj.type)
            )
            domain.objects[obj_info.name] = obj_info

        # Extract predicates (from fluents)
        for fluent in up_problem.fluents:
            params = [
                Parameter(
                    name=f"?{i}",  # Generic parameter names
                    type=str(param.type.name) if hasattr(param.type, 'name') else str(param.type)
                )
                for i, param in enumerate(fluent.signature)
            ]
            pred_sig = PredicateSignature(
                name=fluent.name,
                parameters=params
            )
            domain.predicates[pred_sig.name] = pred_sig

        # Extract actions
        for action in up_problem.actions:
            # Convert parameters
            params = [
                Parameter(
                    name=param.name,
                    type=str(param.type.name) if hasattr(param.type, 'name') else str(param.type)
                )
                for param in action.parameters
            ]

            # Convert preconditions to parameter-bound strings
            preconditions = set()
            for precond in action.preconditions:
                # Handle compound expressions (AND, OR, NOT) by extracting literals
                literals = cls._extract_literals_from_expression(precond, action.parameters, adapter)
                preconditions.update(literals)

            # Convert effects to parameter-bound strings
            add_effects = set()
            del_effects = set()
            for effect in action.effects:
                # Effects are simpler - just fluent expressions
                literals = cls._extract_literals_from_expression(
                    effect.fluent, action.parameters, adapter
                )
                for lit in literals:
                    if effect.value.bool_constant_value():
                        add_effects.add(lit)
                    else:
                        del_effects.add(lit)

            lifted_action = LiftedAction(
                name=action.name,
                parameters=params,
                preconditions=preconditions,
                add_effects=add_effects,
                del_effects=del_effects
            )
            domain.lifted_actions[lifted_action.name] = lifted_action

        logger.info(f"Created LiftedDomainKnowledge from UP: {len(domain.lifted_actions)} actions, "
                   f"{len(domain.predicates)} predicates, {len(domain.objects)} objects")

        return domain

    # ========== Summary Methods ==========

    def summary(self) -> Dict[str, int]:
        """
        Get summary statistics.

        Returns:
            Dictionary with counts of domain components
        """
        return {
            'types': len(self.types),
            'objects': len(self.objects),
            'predicates': len(self.predicates),
            'actions': len(self.lifted_actions),
            'actions_with_partial_knowledge': sum(
                1 for a in self.lifted_actions.values() if a.has_partial_knowledge()
            )
        }

    def __str__(self) -> str:
        """String representation."""
        return f"LiftedDomainKnowledge({self.name}): {len(self.lifted_actions)} actions, {len(self.predicates)} predicates"

    def __repr__(self) -> str:
        """Detailed representation."""
        stats = self.summary()
        return f"LiftedDomainKnowledge(name={self.name}, {stats})"
