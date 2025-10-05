# PDDLHandler Refactoring Plan - Step-by-Step Execution Guide

**Created**: 2025-10-05
**Purpose**: Improve code quality, type safety, and maintainability in `pddl_handler.py`
**Methodology**: TDD (Test-Driven Development) - Write tests first, then implement
**Status**: Ready for execution

---

## Critical Terminology (READ FIRST)

### Three Literal Representations

Understanding these distinctions is **essential** to avoid hallucinations during refactoring:

1. **Action Schema** (PDDL definition)
   ```
   (:action pick-up
     :parameters (?x - block)
     :precondition (and (clear ?x) (ontable ?x) (handempty))
     :effect (and (holding ?x) (not (ontable ?x)) (not (handempty))))
   ```

2. **Parameter-Bound Literal** (uses action's specific parameter names)
   ```
   clear(?x)        ← Uses ?x from pick-up(?x)
   ontable(?x)      ← Uses ?x from pick-up(?x)
   handempty        ← Propositional (no parameters)
   ¬clear(?x)       ← Negated parameter-bound literal
   ```
   **Key point**: Parameter names **MUST** match the action's parameter names

3. **Grounded Fluent** (fully instantiated with objects)
   ```
   clear_a          ← Object 'a' replaces ?x
   ontable_a        ← Object 'a' replaces ?x
   handempty        ← Propositional (unchanged)
   ¬clear_a         ← Negated grounded fluent
   ```

### Current Code Evidence

**File**: `src/core/pddl_handler.py`

**Lines 498-517** - `_expression_to_lifted_string()`:
```python
def _expression_to_lifted_string(self, expr: Any, parameters: List) -> Optional[str]:
    """Convert UP expression to lifted string representation."""
    if hasattr(expr, 'fluent'):
        fluent = expr.fluent()
        if expr.args:
            # Has parameters - create lifted representation
            param_strs = []
            for arg in expr.args:
                # Find parameter name
                for param in parameters:  # ← Uses action's parameters
                    if str(arg) == str(param):
                        param_strs.append(f"?{param.name}")  # ← Action's param name
                        break
                else:
                    # Not a parameter, use object name
                    param_strs.append(str(arg))
            return f"{fluent.name}({','.join(param_strs)})"
        else:
            return fluent.name
    return None
```
**Key**: `parameters` argument ensures we use **action's parameter names**

**Lines 656-663** - `get_action_preconditions(lifted=True)`:
```python
if lifted and action_str in self._lifted_actions:
    # Return lifted preconditions
    action = self._lifted_actions[action_str]
    preconditions = set()
    for precond in action.preconditions:
        # Convert to lift fluent string representation
        lifted_str = self._expression_to_lifted_string(precond, action.parameters)
        # ↑ Passes action.parameters to preserve param names
        if lifted_str:
            preconditions.add(lifted_str)
    return preconditions
```

**Key**: Always passes `action.parameters` to preserve parameter-bound semantics

### Information Gain Algorithm Context

**File**: `src/algorithms/information_gain.py`

The algorithm uses **parameter-bound literals** (La):
- **La**: Set of all parameter-bound literals for action `a`
- For `pick-up(?x)`: La = {clear(?x), ontable(?x), handempty, ¬clear(?x), ...}
- These literals use the action's parameter name `?x`

**Operations**:
- `bindP⁻¹(F, O)`: Ground parameter-bound → grounded fluents
- `bindP(f, O)`: Lift grounded fluents → parameter-bound

**Lines 137-149** in `information_gain.py`:
```python
def _get_parameter_bound_literals(self, action_name: str) -> Set[str]:
    """Get all parameter-bound literals (La) for an action."""
    return self.pddl_handler.get_parameter_bound_literals(action_name)
```

---

## Refactoring Overview

### Goals
1. ✅ Introduce type-safe data classes
2. ✅ Reduce code duplication (~200 lines)
3. ✅ Improve method clarity (split complex methods)
4. ✅ Centralize expression conversion logic
5. ✅ Better separation of concerns

### Impact
- **pddl_handler.py**: 1200 lines → ~900 lines (25% reduction)
- **New files**: 3 (pddl_types.py, expression_converter.py, binding_operations.py)
- **Tests**: 3 new files, 5 updated files
- **Documentation**: 4 files updated

### Validation Criteria (ALL PHASES)
```bash
# After EACH phase, run:
make test                    # Must pass all 165 tests
pytest tests/ -v             # Check for new failures
python -m pytest tests/core/test_pddl_handler.py -v  # Specific validation
```

---

## Phase 1: Create Type-Safe Data Classes

### Context
Currently, the codebase uses primitive types everywhere:
- `Dict[str, Object]` for parameter bindings
- `str` for both parameter-bound and grounded literals
- No type safety or validation

### Goal
Create domain-specific types that:
1. Provide type safety
2. Make code more self-documenting
3. Enable better IDE support
4. Validate data at creation time

### Current Code References

**Parameter Bindings** (currently `Dict[str, Object]`):
- `pddl_handler.py:129-168` - `_get_parameter_bindings()` returns `List[Dict[str, Object]]`
- `pddl_handler.py:338-364` - `parse_grounded_action()` returns `Tuple[Action, Dict[str, Object]]`
- `information_gain.py:486` - `binding.values()` used to get objects

**Parameter-Bound Literals** (currently `str`):
- `pddl_handler.py:898-945` - `get_parameter_bound_literals()` returns `Set[str]`
- `information_gain.py:124` - `self.pre[action_name] = La.copy()` stores `Set[str]`

**Grounded Fluents** (currently `str`):
- `pddl_handler.py:188-212` - `get_grounded_fluents()` returns `List[str]`
- `pddl_environment.py:59-97` - `get_state()` returns `Set[str]`

### Step 1.1: Write Tests FIRST

**File to create**: `tests/core/test_pddl_types.py`

```python
"""
Tests for PDDL type-safe data classes.
These tests are written BEFORE implementation (TDD).
"""

import pytest
from unified_planning.model import Object
from src.core.pddl_types import (
    ParameterBinding,
    ParameterBoundLiteral,
    GroundedFluent
)


class TestParameterBinding:
    """Test ParameterBinding class."""

    def test_initialization_with_dict(self):
        """Test creating ParameterBinding from dict."""
        obj_a = Object('a', 'block')
        bindings = {'x': obj_a}

        pb = ParameterBinding(bindings)

        assert pb.get_object('x') == obj_a
        assert pb.object_names() == ['a']

    def test_object_names_preserves_order(self):
        """Test object_names() preserves parameter order."""
        obj_a = Object('a', 'block')
        obj_b = Object('b', 'block')
        bindings = {'x': obj_a, 'y': obj_b}

        pb = ParameterBinding(bindings)

        # Should preserve order from parameter names
        assert pb.object_names() == ['a', 'b']

    def test_to_dict(self):
        """Test converting back to dict."""
        obj_a = Object('a', 'block')
        bindings = {'x': obj_a}

        pb = ParameterBinding(bindings)

        assert pb.to_dict() == bindings


class TestParameterBoundLiteral:
    """Test ParameterBoundLiteral class."""

    def test_initialization_positive_literal(self):
        """Test creating positive parameter-bound literal."""
        lit = ParameterBoundLiteral('clear', ['?x'], is_negative=False)

        assert lit.predicate == 'clear'
        assert lit.parameters == ['?x']
        assert lit.is_negative is False

    def test_to_string_positive(self):
        """Test converting positive literal to string."""
        lit = ParameterBoundLiteral('clear', ['?x'])

        assert lit.to_string() == 'clear(?x)'

    def test_to_string_negative(self):
        """Test converting negative literal to string."""
        lit = ParameterBoundLiteral('clear', ['?x'], is_negative=True)

        assert lit.to_string() == '¬clear(?x)'

    def test_to_string_multi_parameter(self):
        """Test multi-parameter literal."""
        lit = ParameterBoundLiteral('on', ['?x', '?y'])

        assert lit.to_string() == 'on(?x,?y)'

    def test_to_string_propositional(self):
        """Test propositional literal (no parameters)."""
        lit = ParameterBoundLiteral('handempty', [])

        assert lit.to_string() == 'handempty'

    def test_from_string_positive(self):
        """Test parsing positive literal from string."""
        lit = ParameterBoundLiteral.from_string('clear(?x)')

        assert lit.predicate == 'clear'
        assert lit.parameters == ['?x']
        assert lit.is_negative is False

    def test_from_string_negative(self):
        """Test parsing negative literal from string."""
        lit = ParameterBoundLiteral.from_string('¬clear(?x)')

        assert lit.predicate == 'clear'
        assert lit.parameters == ['?x']
        assert lit.is_negative is True

    def test_from_string_multi_parameter(self):
        """Test parsing multi-parameter literal."""
        lit = ParameterBoundLiteral.from_string('on(?x,?y)')

        assert lit.predicate == 'on'
        assert lit.parameters == ['?x', '?y']

    def test_from_string_propositional(self):
        """Test parsing propositional literal."""
        lit = ParameterBoundLiteral.from_string('handempty')

        assert lit.predicate == 'handempty'
        assert lit.parameters == []


class TestGroundedFluent:
    """Test GroundedFluent class."""

    def test_initialization(self):
        """Test creating grounded fluent."""
        fluent = GroundedFluent('clear', ['a'])

        assert fluent.predicate == 'clear'
        assert fluent.objects == ['a']

    def test_to_string_single_object(self):
        """Test converting to string with single object."""
        fluent = GroundedFluent('clear', ['a'])

        assert fluent.to_string() == 'clear_a'

    def test_to_string_multiple_objects(self):
        """Test converting to string with multiple objects."""
        fluent = GroundedFluent('on', ['a', 'b'])

        assert fluent.to_string() == 'on_a_b'

    def test_to_string_propositional(self):
        """Test propositional fluent."""
        fluent = GroundedFluent('handempty', [])

        assert fluent.to_string() == 'handempty'

    def test_from_string_single_object(self):
        """Test parsing from string."""
        fluent = GroundedFluent.from_string('clear_a')

        assert fluent.predicate == 'clear'
        assert fluent.objects == ['a']

    def test_from_string_multiple_objects(self):
        """Test parsing multi-object fluent."""
        fluent = GroundedFluent.from_string('on_a_b')

        assert fluent.predicate == 'on'
        assert fluent.objects == ['a', 'b']

    def test_from_string_propositional(self):
        """Test parsing propositional fluent."""
        fluent = GroundedFluent.from_string('handempty')

        assert fluent.predicate == 'handempty'
        assert fluent.objects == []


class TestCriticalSemantics:
    """Test critical semantic distinctions."""

    def test_parameter_bound_uses_action_param_names(self):
        """CRITICAL: Parameter-bound literals use action's parameter names.

        For action pick-up(?x), precondition clear(?x) uses ?x, NOT ?y or ?a.
        """
        # This represents clear(?x) for pick-up(?x)
        lit = ParameterBoundLiteral('clear', ['?x'])

        # Must preserve parameter name from action
        assert lit.to_string() == 'clear(?x)'
        assert '?x' in lit.to_string()  # Specific param name

    def test_different_actions_different_param_names(self):
        """Different actions can have different parameter names."""
        # pick-up(?x) has precondition clear(?x)
        lit1 = ParameterBoundLiteral('clear', ['?x'])

        # stack(?x,?y) has precondition clear(?y)
        lit2 = ParameterBoundLiteral('clear', ['?y'])

        # Different parameter names for different actions
        assert lit1.to_string() == 'clear(?x)'
        assert lit2.to_string() == 'clear(?y)'
        assert lit1.to_string() != lit2.to_string()
```

**Expected outcome**: Tests should FAIL (classes don't exist yet)
```bash
pytest tests/core/test_pddl_types.py
# Expected: ImportError or ModuleNotFoundError
```

### Step 1.2: Implement Classes

**File to create**: `src/core/pddl_types.py`

```python
"""
Type-safe data classes for PDDL representations.

This module provides domain-specific types to replace primitive types
throughout the codebase, improving type safety and code clarity.

Critical Semantic Distinction:
- ParameterBoundLiteral: Uses action's specific parameter names (e.g., clear(?x))
- GroundedFluent: Fully instantiated with objects (e.g., clear_a)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from unified_planning.model import Object


@dataclass
class ParameterBinding:
    """Type-safe wrapper for parameter → object mappings.

    Represents binding of action parameters to concrete objects.

    Example:
        For action pick-up(?x) with object 'a':
        ParameterBinding({'x': Object('a', 'block')})

    Attributes:
        bindings: Dict mapping parameter names (without ?) to Object instances
    """
    bindings: Dict[str, Object]

    def get_object(self, param_name: str) -> Object:
        """Get object bound to parameter.

        Args:
            param_name: Parameter name (without '?')

        Returns:
            Object bound to this parameter

        Raises:
            KeyError: If parameter not in bindings
        """
        return self.bindings[param_name]

    def object_names(self) -> List[str]:
        """Get list of object names in parameter order.

        Returns:
            List of object names (e.g., ['a', 'b'])
        """
        return [obj.name for obj in self.bindings.values()]

    def to_dict(self) -> Dict[str, Object]:
        """Convert to plain dict.

        Returns:
            Dictionary mapping parameter names to Objects
        """
        return self.bindings


@dataclass
class ParameterBoundLiteral:
    """Literal using action's parameter names.

    CRITICAL: These literals use the action's specific parameter names,
    not arbitrary variable names.

    Example:
        For action pick-up(?x):
        - Correct: ParameterBoundLiteral('clear', ['?x'])
        - WRONG: ParameterBoundLiteral('clear', ['?a']) or ['?obj']

    Attributes:
        predicate: Predicate name (e.g., 'clear', 'on')
        parameters: List of parameter names (e.g., ['?x', '?y'])
        is_negative: True if negated literal (¬)
    """
    predicate: str
    parameters: List[str]
    is_negative: bool = False

    def to_string(self) -> str:
        """Convert to string representation.

        Returns:
            String like "clear(?x)" or "¬on(?x,?y)"
        """
        if self.parameters:
            param_str = ','.join(self.parameters)
            base = f"{self.predicate}({param_str})"
        else:
            base = self.predicate

        return f"¬{base}" if self.is_negative else base

    @classmethod
    def from_string(cls, s: str) -> 'ParameterBoundLiteral':
        """Parse parameter-bound literal from string.

        Args:
            s: String like "clear(?x)" or "¬on(?x,?y)"

        Returns:
            ParameterBoundLiteral instance
        """
        # Check for negation
        is_negative = s.startswith('¬')
        if is_negative:
            s = s[1:]  # Remove negation symbol

        # Parse predicate and parameters
        if '(' in s:
            # Has parameters
            predicate = s[:s.index('(')]
            param_str = s[s.index('(') + 1:s.rindex(')')]
            parameters = [p.strip() for p in param_str.split(',')]
        else:
            # Propositional
            predicate = s
            parameters = []

        return cls(predicate, parameters, is_negative)


@dataclass
class GroundedFluent:
    """Fully grounded fluent with concrete objects.

    Example:
        GroundedFluent('clear', ['a']) → "clear_a"
        GroundedFluent('on', ['a', 'b']) → "on_a_b"

    Attributes:
        predicate: Predicate name
        objects: List of object names
    """
    predicate: str
    objects: List[str]

    def to_string(self) -> str:
        """Convert to string representation.

        Returns:
            String like "clear_a" or "on_a_b"
        """
        if self.objects:
            return f"{self.predicate}_{'_'.join(self.objects)}"
        else:
            return self.predicate

    @classmethod
    def from_string(cls, s: str) -> 'GroundedFluent':
        """Parse grounded fluent from string.

        Args:
            s: String like "clear_a" or "on_a_b"

        Returns:
            GroundedFluent instance
        """
        parts = s.split('_')

        if len(parts) == 1:
            # Propositional
            return cls(parts[0], [])
        else:
            # Has objects
            return cls(parts[0], parts[1:])
```

### Step 1.3: Validate Implementation

```bash
# Run tests - should now PASS
pytest tests/core/test_pddl_types.py -v

# Expected output:
# test_pddl_types.py::TestParameterBinding::test_initialization_with_dict PASSED
# test_pddl_types.py::TestParameterBoundLiteral::test_to_string_positive PASSED
# ... (all tests should pass)

# Run full test suite - should still pass
make test
```

### Step 1.4: Update Documentation

**File to update**: `docs/UNIFIED_PLANNING_GUIDE.md`

Add after line 92 (after "Lifted vs Grounded vs Parameter-Bound" section):

```markdown
### Type-Safe Representations (New)

The codebase provides type-safe classes for PDDL representations:

```python
from src.core.pddl_types import ParameterBinding, ParameterBoundLiteral, GroundedFluent

# Parameter binding
binding = ParameterBinding({'x': Object('a', 'block')})
binding.object_names()  # ['a']

# Parameter-bound literal (uses action's parameter names)
lit = ParameterBoundLiteral('clear', ['?x'])
lit.to_string()  # "clear(?x)"

# Grounded fluent
fluent = GroundedFluent('clear', ['a'])
fluent.to_string()  # "clear_a"
```

**Key point**: `ParameterBoundLiteral` preserves action's parameter names.
```

### Phase 1 Completion Checklist

- [ ] Tests created in `tests/core/test_pddl_types.py`
- [ ] Tests initially fail (classes don't exist)
- [ ] Implementation created in `src/core/pddl_types.py`
- [ ] All tests in `test_pddl_types.py` pass
- [ ] Full test suite passes (`make test`)
- [ ] Documentation updated in `UNIFIED_PLANNING_GUIDE.md`
- [ ] Commit changes: `git add tests/core/test_pddl_types.py src/core/pddl_types.py docs/UNIFIED_PLANNING_GUIDE.md`
- [ ] Commit message: "Phase 1: Add type-safe PDDL data classes"

---

## Phase 2: Extract Expression Conversion Logic

### Context
Currently, expression conversion logic (FNode → string) is scattered throughout `pddl_handler.py`:
- `_expression_to_lifted_string()` (lines 498-517)
- `_ground_expression_to_string()` (lines 519-555)
- `_extract_clauses_from_expression()` (lines 783-814)

This logic should be centralized for:
1. Reusability across different components
2. Easier testing in isolation
3. Single source of truth for conversion rules

### Goal
Create `ExpressionConverter` class that handles all FNode → string conversions while preserving parameter-bound semantics.

### Current Code Analysis

**Method 1**: `_expression_to_lifted_string()` (lines 498-517)
- **Purpose**: Convert FNode to parameter-bound literal string
- **Key**: Uses `parameters` list to get action's parameter names
- **Input**: FNode, List[Parameter]
- **Output**: String like "clear(?x)" or "on(?x,?y)"

**Method 2**: `_ground_expression_to_string()` (lines 519-555)
- **Purpose**: Convert FNode to grounded fluent string
- **Key**: Uses parameter binding to substitute objects
- **Input**: FNode, Dict[str, Object]
- **Output**: String like "clear_a" or "on_a_b"

**Method 3**: `_extract_clauses_from_expression()` (lines 783-814)
- **Purpose**: Convert FNode to CNF clauses
- **Key**: Recursively traverses AND/OR/NOT expressions
- **Input**: FNode, List[Parameter]
- **Output**: List[List[str]] CNF clauses

### Usage Analysis

**Who calls these methods?**

1. `get_action_preconditions()` (line 661):
   ```python
   lifted_str = self._expression_to_lifted_string(precond, action.parameters)
   ```

2. `get_action_effects()` (line 704):
   ```python
   lifted_str = self._expression_to_lifted_string(effect.fluent, action.parameters)
   ```

3. `get_action_preconditions()` (line 676):
   ```python
   grounded_str = self._ground_expression_to_string(arg, binding)
   ```

4. `extract_lifted_preconditions_cnf()` (line 778):
   ```python
   clauses = self._extract_clauses_from_expression(precond, action.parameters)
   ```

### Step 2.1: Write Tests FIRST

**File to create**: `tests/core/test_expression_converter.py`

```python
"""
Tests for FNode expression conversion logic.
These tests are written BEFORE implementation (TDD).
"""

import pytest
from unified_planning.model import Fluent, Action, Problem, Object
from unified_planning.shortcuts import BoolType, UserType
from src.core.expression_converter import ExpressionConverter
from src.core.pddl_types import ParameterBinding


class TestExpressionConverter:
    """Test ExpressionConverter class."""

    @pytest.fixture
    def simple_problem(self):
        """Create simple blocksworld problem for testing."""
        problem = Problem("test")

        # Add types
        block_type = UserType("block")
        problem.add_type(block_type)

        # Add objects
        a = Object("a", block_type)
        b = Object("b", block_type)
        problem.add_object(a)
        problem.add_object(b)

        # Add fluents
        clear = Fluent("clear", BoolType(), block=block_type)
        on = Fluent("on", BoolType(), block1=block_type, block2=block_type)
        handempty = Fluent("handempty", BoolType())

        problem.add_fluent(clear, default_initial_value=False)
        problem.add_fluent(on, default_initial_value=False)
        problem.add_fluent(handempty, default_initial_value=True)

        # Add action
        pickup = Action("pick-up", x=block_type)
        x = pickup.parameter("x")

        # Preconditions: clear(x) AND ontable(x) AND handempty
        problem.add_action(pickup)

        return problem, pickup, clear, on, handempty, a, b

    def test_to_parameter_bound_string_single_param(self, simple_problem):
        """Test converting fluent expression to parameter-bound string."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)  # FNode for clear(?x)

        result = ExpressionConverter.to_parameter_bound_string(expr, pickup.parameters)

        # Must use action's parameter name (?x from pick-up(?x))
        assert result == "clear(?x)"
        assert "?x" in result  # Verify parameter name preserved

    def test_to_parameter_bound_string_multi_param(self, simple_problem):
        """Test multi-parameter fluent."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        # Create stack action with two parameters
        stack = Action("stack", x=problem.user_type("block"), y=problem.user_type("block"))
        x = stack.parameter("x")
        y = stack.parameter("y")

        expr = on(x, y)  # FNode for on(?x, ?y)

        result = ExpressionConverter.to_parameter_bound_string(expr, stack.parameters)

        assert result == "on(?x,?y)"
        assert "?x" in result and "?y" in result

    def test_to_parameter_bound_string_propositional(self, simple_problem):
        """Test propositional fluent (no parameters)."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        expr = handempty()  # FNode for handempty

        result = ExpressionConverter.to_parameter_bound_string(expr, pickup.parameters)

        assert result == "handempty"

    def test_to_grounded_string_single_param(self, simple_problem):
        """Test converting to grounded string with binding."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)  # FNode for clear(?x)

        binding = ParameterBinding({"x": a})
        result = ExpressionConverter.to_grounded_string(expr, binding)

        assert result == "clear_a"

    def test_to_grounded_string_multi_param(self, simple_problem):
        """Test grounding multi-parameter fluent."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        stack = Action("stack", x=problem.user_type("block"), y=problem.user_type("block"))
        x = stack.parameter("x")
        y = stack.parameter("y")

        expr = on(x, y)

        binding = ParameterBinding({"x": a, "y": b})
        result = ExpressionConverter.to_grounded_string(expr, binding)

        assert result == "on_a_b"

    def test_to_grounded_string_propositional(self, simple_problem):
        """Test grounding propositional fluent."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        expr = handempty()

        binding = ParameterBinding({})
        result = ExpressionConverter.to_grounded_string(expr, binding)

        assert result == "handempty"

    def test_to_cnf_clauses_single_literal(self, simple_problem):
        """Test converting single literal to CNF."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)

        result = ExpressionConverter.to_cnf_clauses(expr, pickup.parameters)

        # Single literal becomes single-element clause
        assert result == [["clear(?x)"]]

    def test_to_cnf_clauses_and_expression(self, simple_problem):
        """Test AND expression becomes multiple clauses."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        # This test requires creating an AND expression
        # Will be implemented based on UP's expression builder
        pass  # TODO: Implement when we have AND expression

    def test_preserves_action_parameter_names(self, simple_problem):
        """CRITICAL: Verify parameter names from action are preserved."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)

        # pick-up has parameter ?x
        result = ExpressionConverter.to_parameter_bound_string(expr, pickup.parameters)

        # Must use ?x, not ?y or any other name
        assert result == "clear(?x)"
        assert result != "clear(?y)"
        assert result != "clear(?a)"
```

**Expected outcome**: Tests should FAIL (class doesn't exist yet)

### Step 2.2: Implement ExpressionConverter

**File to create**: `src/core/expression_converter.py`

```python
"""
FNode expression converter for PDDL representations.

This module centralizes conversion logic from Unified Planning FNode
expressions to various string representations.

CRITICAL: Parameter-bound conversions preserve action's parameter names.
"""

from typing import List, Optional, Tuple
from unified_planning.model.fnode import FNode
from unified_planning.model import Parameter
from src.core.pddl_types import ParameterBinding


class ExpressionConverter:
    """Converts UP FNode expressions to string representations.

    This class provides static methods for converting FNode expressions
    to different string formats while preserving semantic distinctions
    between parameter-bound and grounded representations.
    """

    @staticmethod
    def to_parameter_bound_string(expr: FNode,
                                   action_parameters: List[Parameter]) -> Optional[str]:
        """Convert FNode to parameter-bound literal string.

        CRITICAL: Uses action's parameter names, not arbitrary variables.

        Args:
            expr: FNode expression (typically fluent expression)
            action_parameters: Action's Parameter objects (preserves names)

        Returns:
            String like "clear(?x)" using action's parameter name

        Example:
            For action pick-up(?x):
            - expr = clear(x) FNode
            - action_parameters = [Parameter('x', block_type)]
            - Returns: "clear(?x)" ← Uses ?x from action
        """
        if not hasattr(expr, 'fluent'):
            return None

        fluent = expr.fluent()

        if expr.args:
            # Has parameters - create parameter-bound representation
            param_strs = []
            for arg in expr.args:
                # Find matching parameter in action's parameters
                param_name = None
                for param in action_parameters:
                    if str(arg) == str(param):
                        param_name = f"?{param.name}"
                        break

                if param_name:
                    param_strs.append(param_name)
                else:
                    # Not a parameter, use object name (shouldn't happen in normal case)
                    param_strs.append(str(arg))

            return f"{fluent.name}({','.join(param_strs)})"
        else:
            # Propositional fluent
            return fluent.name

    @staticmethod
    def to_grounded_string(expr: FNode,
                          binding: ParameterBinding) -> Optional[str]:
        """Convert FNode to grounded fluent string.

        Args:
            expr: FNode expression
            binding: Parameter binding (param → object mapping)

        Returns:
            String like "clear_a" or "on_a_b"
        """
        # Handle AND expressions recursively
        if hasattr(expr, 'is_and') and expr.is_and():
            # Shouldn't happen in normal precondition extraction
            return None

        # Handle OR expressions recursively
        if hasattr(expr, 'is_or') and expr.is_or():
            return None

        # Handle NOT expressions
        if hasattr(expr, 'is_not') and expr.is_not():
            inner = expr.args[0] if expr.args else expr
            inner_str = ExpressionConverter.to_grounded_string(inner, binding)
            if inner_str:
                return f"-{inner_str}"  # Negative prefix for CNF
            return None

        # Handle fluent expressions
        if hasattr(expr, 'is_fluent_exp') and expr.is_fluent_exp():
            fluent = expr.fluent()

            if expr.args:
                # Has parameters - ground them
                grounded_args = []
                for arg in expr.args:
                    param_name = str(arg)
                    if param_name in binding.bindings:
                        grounded_args.append(binding.bindings[param_name].name)
                    else:
                        # Try without quotes
                        param_name_clean = param_name.replace("'", "")
                        if param_name_clean in binding.bindings:
                            grounded_args.append(binding.bindings[param_name_clean].name)
                        else:
                            grounded_args.append(param_name_clean)

                return f"{fluent.name}_{'_'.join(grounded_args)}"
            else:
                # Propositional
                return fluent.name

        return None

    @staticmethod
    def to_cnf_clauses(expr: FNode,
                       action_parameters: List[Parameter]) -> List[List[str]]:
        """Extract CNF clauses from FNode expression.

        Args:
            expr: FNode expression (can be AND/OR/NOT/FLUENT)
            action_parameters: Action's parameters (for parameter-bound literals)

        Returns:
            List of CNF clauses (each clause is list of literals)
        """
        clauses = []

        # Handle different expression types
        if hasattr(expr, 'is_and') and expr.is_and():
            # AND: each operand becomes a separate clause
            for arg in expr.args:
                sub_clauses = ExpressionConverter.to_cnf_clauses(arg, action_parameters)
                clauses.extend(sub_clauses)

        elif hasattr(expr, 'is_or') and expr.is_or():
            # OR: combine operands into single clause
            clause = []
            for arg in expr.args:
                sub_clauses = ExpressionConverter.to_cnf_clauses(arg, action_parameters)
                if sub_clauses and sub_clauses[0]:
                    clause.extend(sub_clauses[0])
            if clause:
                clauses.append(clause)

        elif hasattr(expr, 'is_not') and expr.is_not():
            # NOT: negate the inner expression
            inner = expr.args[0] if expr.args else expr
            param_bound_str = ExpressionConverter.to_parameter_bound_string(
                inner, action_parameters
            )
            if param_bound_str:
                clauses.append([f"-{param_bound_str}"])

        else:
            # Base case: fluent expression
            param_bound_str = ExpressionConverter.to_parameter_bound_string(
                expr, action_parameters
            )
            if param_bound_str:
                clauses.append([param_bound_str])

        return clauses
```

### Step 2.3: Refactor PDDLHandler to Use ExpressionConverter

**File to modify**: `src/core/pddl_handler.py`

**Add import** (after line 10):
```python
from src.core.expression_converter import ExpressionConverter
```

**Replace `_expression_to_lifted_string()`** (lines 498-517):
```python
# BEFORE (lines 498-517):
def _expression_to_lifted_string(self, expr: Any, parameters: List) -> Optional[str]:
    """Convert UP expression to lifted string representation."""
    if hasattr(expr, 'fluent'):
        fluent = expr.fluent()
        if expr.args:
            # Has parameters - create lifted representation
            param_strs = []
            for arg in expr.args:
                # Find parameter name
                for param in parameters:
                    if str(arg) == str(param):
                        param_strs.append(f"?{param.name}")
                        break
                else:
                    # Not a parameter, use object name
                    param_strs.append(str(arg))
            return f"{fluent.name}({','.join(param_strs)})"
        else:
            return fluent.name
    return None

# AFTER (replace with delegation):
def _expression_to_lifted_string(self, expr: Any, parameters: List) -> Optional[str]:
    """Convert UP expression to parameter-bound string representation.

    DEPRECATED: Use ExpressionConverter.to_parameter_bound_string() directly.
    This method kept for backward compatibility.
    """
    return ExpressionConverter.to_parameter_bound_string(expr, parameters)
```

**Replace `_ground_expression_to_string()`** (lines 519-555):
```python
# AFTER (replace entire method):
def _ground_expression_to_string(self, expr: Any, binding: Dict[str, Object]) -> Optional[str]:
    """Convert UP expression to grounded string with binding.

    DEPRECATED: Use ExpressionConverter.to_grounded_string() directly.
    This method kept for backward compatibility.
    """
    from src.core.pddl_types import ParameterBinding
    pb = ParameterBinding(binding)
    return ExpressionConverter.to_grounded_string(expr, pb)
```

**Replace `_extract_clauses_from_expression()`** (lines 783-814):
```python
# AFTER (replace entire method):
def _extract_clauses_from_expression(self, expr: Any, parameters: List) -> List[List[str]]:
    """Extract CNF clauses from a UP expression.

    DEPRECATED: Use ExpressionConverter.to_cnf_clauses() directly.
    This method kept for backward compatibility.
    """
    return ExpressionConverter.to_cnf_clauses(expr, parameters)
```

### Step 2.4: Validate Implementation

```bash
# Run new tests
pytest tests/core/test_expression_converter.py -v

# Run PDDLHandler tests (should still pass with delegation)
pytest tests/core/test_pddl_handler.py -v

# Run full test suite
make test
```

### Step 2.5: Update Documentation

**File to update**: `docs/UNIFIED_PLANNING_GUIDE.md`

Update "String Conversion and Simplification" section (around line 212):

```markdown
### String Conversion with ExpressionConverter

The `ExpressionConverter` class centralizes all FNode → string conversions:

```python
from src.core.expression_converter import ExpressionConverter
from src.core.pddl_types import ParameterBinding

# Convert to parameter-bound literal (uses action's parameter names)
param_bound = ExpressionConverter.to_parameter_bound_string(
    expr, action.parameters  # ← Action's parameters preserve names
)
# Returns: "clear(?x)" for pick-up(?x) action

# Convert to grounded fluent
binding = ParameterBinding({'x': Object('a', 'block')})
grounded = ExpressionConverter.to_grounded_string(expr, binding)
# Returns: "clear_a"

# Extract CNF clauses
clauses = ExpressionConverter.to_cnf_clauses(expr, action.parameters)
# Returns: [["clear(?x)"], ["ontable(?x)"]] for AND expression
```

**Key point**: Always pass `action.parameters` to preserve parameter names.
```

### Phase 2 Completion Checklist

- [ ] Tests created in `tests/core/test_expression_converter.py`
- [ ] Tests initially fail (class doesn't exist)
- [ ] Implementation created in `src/core/expression_converter.py`
- [ ] PDDLHandler refactored to delegate to ExpressionConverter
- [ ] All tests in `test_expression_converter.py` pass
- [ ] All tests in `test_pddl_handler.py` still pass
- [ ] Full test suite passes (`make test`)
- [ ] Documentation updated in `UNIFIED_PLANNING_GUIDE.md`
- [ ] Commit changes

---

## Phase 3: Extract Grounding/Lifting Operations

### Context
The grounding and lifting operations (bindP, bindP⁻¹) are currently embedded in `pddl_handler.py`:
- `ground_literals()` (lines 946-980) - bindP⁻¹
- `lift_fluents()` (lines 982-1016) - bindP
- `_ground_lifted_literal_internal()` (lines 1018-1060)
- `_lift_grounded_fluent_internal()` (lines 1062-1100)

These operations are central to the Information Gain algorithm and should be:
1. Clearly separated as bindP/bindP⁻¹ operations
2. Easier to test in isolation
3. Well-documented with algorithm context

### Goal
Create `FluentBinder` class that handles all grounding/lifting operations with clear algorithm semantics.

### Current Code Analysis

**bindP⁻¹** (grounding): `ground_literals()` (lines 946-980)
- **Input**: Set[str] parameter-bound literals, List[str] objects
- **Output**: Set[str] grounded fluents
- **Example**: `{"clear(?x)"}` + `["a"]` → `{"clear_a"}`

**bindP** (lifting): `lift_fluents()` (lines 982-1016)
- **Input**: Set[str] grounded fluents, List[str] objects
- **Output**: Set[str] parameter-bound literals
- **Example**: `{"clear_a"}` + `["a"]` → `{"clear(?x)"}`

**Internal operations**:
- `_ground_lifted_literal_internal()` - single literal grounding
- `_lift_grounded_fluent_internal()` - single fluent lifting
- `_get_parameter_index_internal()` - parameter index lookup
- `_get_parameter_name_internal()` - index → parameter name

### Usage Analysis

**Who uses these methods?**

1. `information_gain.py` (lines 151-181):
   ```python
   def bindP_inverse(self, literals: Set[str], objects: List[str]) -> Set[str]:
       return self.pddl_handler.ground_literals(literals, objects)

   def bindP(self, fluents: Set[str], objects: List[str]) -> Set[str]:
       return self.pddl_handler.lift_fluents(fluents, objects)
   ```

### Step 3.1: Write Tests FIRST

**File to create**: `tests/core/test_binding_operations.py`

```python
"""
Tests for grounding and lifting operations (bindP, bindP⁻¹).
These tests are written BEFORE implementation (TDD).
"""

import pytest
from src.core.binding_operations import FluentBinder
from src.core.pddl_handler import PDDLHandler


class TestFluentBinder:
    """Test FluentBinder class for bindP/bindP⁻¹ operations."""

    @pytest.fixture
    def pddl_handler(self, temp_dir, blocksworld_domain, blocksworld_problem):
        """Create PDDLHandler for testing."""
        domain_file = temp_dir / "domain.pddl"
        problem_file = temp_dir / "problem.pddl"

        domain_file.write_text(blocksworld_domain)
        problem_file.write_text(blocksworld_problem)

        handler = PDDLHandler()
        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        return handler

    @pytest.fixture
    def binder(self, pddl_handler):
        """Create FluentBinder for testing."""
        return FluentBinder(pddl_handler)

    def test_ground_literal_single_param(self, binder):
        """Test grounding parameter-bound literal with single parameter."""
        literal = "clear(?x)"
        objects = ["a"]

        result = binder.ground_literal(literal, objects)

        assert result == "clear_a"

    def test_ground_literal_multi_param(self, binder):
        """Test grounding multi-parameter literal."""
        literal = "on(?x,?y)"
        objects = ["a", "b"]

        result = binder.ground_literal(literal, objects)

        assert result == "on_a_b"

    def test_ground_literal_propositional(self, binder):
        """Test grounding propositional literal (no params)."""
        literal = "handempty"
        objects = []

        result = binder.ground_literal(literal, objects)

        assert result == "handempty"

    def test_ground_literal_negative(self, binder):
        """Test grounding negative literal."""
        literal = "¬clear(?x)"
        objects = ["a"]

        result = binder.ground_literal(literal, objects)

        assert result == "¬clear_a"

    def test_lift_fluent_single_param(self, binder):
        """Test lifting grounded fluent to parameter-bound."""
        fluent = "clear_a"
        objects = ["a"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "clear(?x)"

    def test_lift_fluent_multi_param(self, binder):
        """Test lifting multi-parameter fluent."""
        fluent = "on_a_b"
        objects = ["a", "b"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "on(?x,?y)"

    def test_lift_fluent_propositional(self, binder):
        """Test lifting propositional fluent."""
        fluent = "handempty"
        objects = []

        result = binder.lift_fluent(fluent, objects)

        assert result == "handempty"

    def test_lift_fluent_negative(self, binder):
        """Test lifting negative fluent."""
        fluent = "¬clear_a"
        objects = ["a"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "¬clear(?x)"

    def test_ground_literals_batch(self, binder):
        """Test batch grounding (bindP⁻¹ operation)."""
        literals = {"clear(?x)", "ontable(?x)", "handempty"}
        objects = ["a"]

        result = binder.ground_literals(literals, objects)

        assert result == {"clear_a", "ontable_a", "handempty"}

    def test_lift_fluents_batch(self, binder):
        """Test batch lifting (bindP operation)."""
        fluents = {"clear_a", "ontable_a", "handempty"}
        objects = ["a"]

        result = binder.lift_fluents(fluents, objects)

        assert result == {"clear(?x)", "ontable(?x)", "handempty"}

    def test_ground_then_lift_inverse(self, binder):
        """Test that lift(ground(x)) = x (inverse operations)."""
        original = {"clear(?x)", "on(?x,?y)"}
        objects = ["a", "b"]

        grounded = binder.ground_literals(original, objects)
        lifted = binder.lift_fluents(grounded, objects)

        assert lifted == original

    def test_parameter_index_mapping(self, binder):
        """Test parameter index correctly maps to objects."""
        # ?x → index 0 → 'a'
        # ?y → index 1 → 'b'
        literal = "on(?x,?y)"
        objects = ["a", "b"]

        result = binder.ground_literal(literal, objects)

        assert result == "on_a_b"
        # Verify order: first param (?x) → first object (a)
        assert "_a_" in result
```

**Expected outcome**: Tests should FAIL (class doesn't exist yet)

### Step 3.2: Implement FluentBinder

**File to create**: `src/core/binding_operations.py`

```python
"""
Grounding and lifting operations for PDDL literals and fluents.

This module implements the bindP and bindP⁻¹ operations from the
Information Gain algorithm, converting between parameter-bound and
grounded representations.

Algorithm Context:
- bindP⁻¹(F, O): Ground parameter-bound literals with objects
- bindP(f, O): Lift grounded fluents to parameter-bound form
"""

from typing import Set, List
import logging
from src.core.pddl_handler import PDDLHandler

logger = logging.getLogger(__name__)


class FluentBinder:
    """Handles conversion between parameter-bound and grounded representations.

    This class implements the binding operations from the Information Gain
    algorithm, providing clear semantics for grounding and lifting.

    Operations:
    - ground_literal: Convert clear(?x) + [a] → clear_a
    - lift_fluent: Convert clear_a + [a] → clear(?x)
    - ground_literals: Batch bindP⁻¹ operation
    - lift_fluents: Batch bindP operation
    """

    def __init__(self, pddl_handler: PDDLHandler):
        """Initialize FluentBinder.

        Args:
            pddl_handler: PDDLHandler instance for parameter name generation
        """
        self.pddl_handler = pddl_handler

    def ground_literal(self, param_bound_literal: str, objects: List[str]) -> str:
        """Ground a parameter-bound literal with concrete objects.

        Implements single-literal version of bindP⁻¹.

        Args:
            param_bound_literal: e.g., "clear(?x)" or "¬on(?x,?y)"
            objects: Ordered list matching parameters, e.g., ['a'] or ['a', 'b']

        Returns:
            Grounded fluent, e.g., "clear_a" or "¬on_a_b"

        Example:
            ground_literal("clear(?x)", ["a"]) → "clear_a"
            ground_literal("on(?x,?y)", ["a", "b"]) → "on_a_b"
        """
        # Handle negation
        is_negative = param_bound_literal.startswith('¬')
        if is_negative:
            param_bound_literal = param_bound_literal[1:]

        # Parse literal: predicate(param1,param2,...)
        if '(' not in param_bound_literal:
            # Propositional literal
            grounded = param_bound_literal
        else:
            predicate = param_bound_literal[:param_bound_literal.index('(')]
            params_str = param_bound_literal[param_bound_literal.index('(') + 1:param_bound_literal.rindex(')')]

            if not params_str:
                # No parameters
                grounded = predicate
            else:
                params = [p.strip() for p in params_str.split(',')]

                # Replace each parameter with corresponding object
                grounded_params = []
                for param in params:
                    if param.startswith('?'):
                        # Extract parameter index from name
                        param_idx = PDDLHandler.parameter_index_from_name(param)
                        if param_idx < len(objects):
                            grounded_params.append(objects[param_idx])
                        else:
                            logger.warning(f"Parameter {param} index {param_idx} out of bounds for objects {objects}")
                            grounded_params.append(param)
                    else:
                        # Already grounded
                        grounded_params.append(param)

                # Create grounded fluent string
                grounded = '_'.join([predicate] + grounded_params)

        # Add back negation if needed
        return f"¬{grounded}" if is_negative else grounded

    def lift_fluent(self, grounded_fluent: str, objects: List[str]) -> str:
        """Lift grounded fluent to parameter-bound form.

        Implements single-fluent version of bindP.

        Args:
            grounded_fluent: e.g., "clear_a" or "¬on_a_b"
            objects: Ordered list used in grounding, e.g., ['a'] or ['a', 'b']

        Returns:
            Parameter-bound literal, e.g., "clear(?x)" or "¬on(?x,?y)"

        Example:
            lift_fluent("clear_a", ["a"]) → "clear(?x)"
            lift_fluent("on_a_b", ["a", "b"]) → "on(?x,?y)"
        """
        # Handle negation
        is_negative = grounded_fluent.startswith('¬')
        if is_negative:
            grounded_fluent = grounded_fluent[1:]

        # Parse fluent: predicate_obj1_obj2_...
        parts = grounded_fluent.split('_')

        if len(parts) == 1:
            # Propositional fluent
            lifted = parts[0]
        else:
            # First part is predicate, rest are object names
            predicate = parts[0]
            obj_names = parts[1:]

            # Replace each object with its parameter
            params = []
            for obj_name in obj_names:
                try:
                    obj_idx = objects.index(obj_name)
                    param_name = PDDLHandler.generate_parameter_names(obj_idx + 1)[obj_idx]
                    params.append(param_name)
                except ValueError:
                    logger.warning(f"Object {obj_name} not found in objects list {objects}")
                    params.append(obj_name)

            # Create lifted literal
            if params:
                lifted = f"{predicate}({','.join(params)})"
            else:
                lifted = predicate

        # Add back negation if needed
        return f"¬{lifted}" if is_negative else lifted

    def ground_literals(self, literals: Set[str], objects: List[str]) -> Set[str]:
        """Ground parameter-bound literals with concrete objects.

        Implements bindP⁻¹(F, O) operation from Information Gain algorithm.

        Args:
            literals: Set of parameter-bound literals (e.g., {'clear(?x)', '¬on(?x,?y)'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of grounded literals (e.g., {'clear_a', '¬on_a_b'})

        Example:
            ground_literals({'clear(?x)', 'handempty'}, ['a'])
            → {'clear_a', 'handempty'}
        """
        grounded = set()

        for literal in literals:
            grounded_literal = self.ground_literal(literal, objects)
            grounded.add(grounded_literal)

        return grounded

    def lift_fluents(self, fluents: Set[str], objects: List[str]) -> Set[str]:
        """Lift grounded fluents to parameter-bound literals.

        Implements bindP(f, O) operation from Information Gain algorithm.

        Args:
            fluents: Set of grounded fluent strings (e.g., {'clear_a', '¬on_a_b'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of parameter-bound literals (e.g., {'clear(?x)', '¬on(?x,?y)'})

        Example:
            lift_fluents({'clear_a', 'handempty'}, ['a'])
            → {'clear(?x)', 'handempty'}
        """
        lifted = set()

        for fluent in fluents:
            lifted_literal = self.lift_fluent(fluent, objects)
            lifted.add(lifted_literal)

        return lifted
```

### Step 3.3: Update PDDLHandler to Use FluentBinder

**File to modify**: `src/core/pddl_handler.py`

**Add import** (after line 10):
```python
from src.core.binding_operations import FluentBinder
```

**Add instance variable** in `__init__()` (after line 40):
```python
# Binding operations helper
self._binder = None  # Initialized lazily
```

**Add lazy initialization helper**:
```python
def _get_binder(self) -> 'FluentBinder':
    """Get FluentBinder instance (lazy initialization)."""
    if self._binder is None:
        from src.core.binding_operations import FluentBinder
        self._binder = FluentBinder(self)
    return self._binder
```

**Update `ground_literals()`** (lines 946-980):
```python
def ground_literals(self, literals: Set[str], objects: List[str]) -> Set[str]:
    """Ground parameter-bound literals with concrete objects.

    Implements bindP⁻¹(F, O) - converts lifted literals to grounded form.

    Args:
        literals: Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})
        objects: Ordered list of objects (e.g., ['a', 'b'])

    Returns:
        Set of grounded literals (e.g., {'on_a_b', '¬clear_a'})
    """
    return self._get_binder().ground_literals(literals, objects)
```

**Update `lift_fluents()`** (lines 982-1016):
```python
def lift_fluents(self, fluents: Set[str], objects: List[str]) -> Set[str]:
    """Lift grounded fluents to parameter-bound literals.

    Implements bindP(f, O) - converts grounded fluents to lifted form.

    Args:
        fluents: Set of grounded fluent strings (e.g., {'on_a_b', '¬clear_a'})
        objects: Ordered list of objects (e.g., ['a', 'b'])

    Returns:
        Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})
    """
    return self._get_binder().lift_fluents(fluents, objects)
```

**Mark internal methods as deprecated** (lines 1018-1100):
Add deprecation comments to `_ground_lifted_literal_internal()` and `_lift_grounded_fluent_internal()`:
```python
def _ground_lifted_literal_internal(self, literal: str, objects: List[str]) -> str:
    """DEPRECATED: Use FluentBinder.ground_literal() instead.

    Kept for backward compatibility during refactoring.
    """
    return self._get_binder().ground_literal(literal, objects)

def _lift_grounded_fluent_internal(self, fluent: str, objects: List[str]) -> str:
    """DEPRECATED: Use FluentBinder.lift_fluent() instead.

    Kept for backward compatibility during refactoring.
    """
    return self._get_binder().lift_fluent(fluent, objects)
```

### Step 3.4: Validate Implementation

```bash
# Run new tests
pytest tests/core/test_binding_operations.py -v

# Run PDDLHandler tests (should still pass)
pytest tests/core/test_pddl_handler.py -v

# Run Information Gain tests (uses bindP/bindP⁻¹)
pytest tests/algorithms/test_information_gain.py -v

# Run full test suite
make test
```

### Step 3.5: Update Documentation

**File to update**: `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md`

Add section about binding operations:

```markdown
### Binding Operations Implementation

The algorithm uses two key operations for converting between representations:

**bindP⁻¹(F, O)**: Ground parameter-bound literals
```python
from src.core.binding_operations import FluentBinder

binder = FluentBinder(pddl_handler)
grounded = binder.ground_literals({'clear(?x)', 'on(?x,?y)'}, ['a', 'b'])
# Returns: {'clear_a', 'on_a_b'}
```

**bindP(f, O)**: Lift grounded fluents
```python
lifted = binder.lift_fluents({'clear_a', 'on_a_b'}, ['a', 'b'])
# Returns: {'clear(?x)', 'on(?x,?y)'}
```

These operations are inverse:
- `bindP(bindP⁻¹(F, O), O) = F`
- `bindP⁻¹(bindP(f, O), O) = f`
```

### Phase 3 Completion Checklist

- [ ] Tests created in `tests/core/test_binding_operations.py`
- [ ] Tests initially fail (class doesn't exist)
- [ ] Implementation created in `src/core/binding_operations.py`
- [ ] PDDLHandler refactored to delegate to FluentBinder
- [ ] All tests in `test_binding_operations.py` pass
- [ ] All tests in `test_pddl_handler.py` still pass
- [ ] Information Gain tests still pass
- [ ] Full test suite passes (`make test`)
- [ ] Documentation updated
- [ ] Commit changes

---

## Phases 4-6: Summary

Due to length constraints, the remaining phases are summarized here. Full details will be provided when executing each phase.

### Phase 4: Refactor Complex Methods in PDDLHandler

**Goals**:
1. Split `_get_parameter_bindings()` into smaller methods
2. Split `state_to_fluent_set()` into dispatcher + handlers
3. Split `get_action_preconditions()` into lifted/grounded extractors

**Outcome**: Simpler, more testable methods (~200 lines reduced)

### Phase 5: Update Dependent Files

**Files to update**:
1. `src/algorithms/information_gain.py` - Use new types
2. `src/environments/pddl_environment.py` - Type-safe bindings
3. `src/algorithms/olam_adapter.py` - Minimal changes
4. All affected test files

**Outcome**: Type-safe interfaces throughout

### Phase 6: Documentation Updates

**Files to update**:
1. `docs/UNIFIED_PLANNING_GUIDE.md` - Complete rewrite of examples
2. `docs/LIFTED_SUPPORT.md` - Reference new modules
3. `docs/QUICK_REFERENCE.md` - Update code patterns
4. `docs/IMPLEMENTATION_TASKS.md` - Record refactoring

**Outcome**: Comprehensive, accurate documentation

---

## Execution Tracking

Use this section to track progress through phases:

**Phase 1**: [ ] Not started | [ ] In progress | [X] Complete
**Phase 1B**: [ ] Not started | [ ] In progress | [X] Complete (GroundedAction + LiftedAction)
**Phase 2**: [ ] Not started | [ ] In progress | [X] Complete
**Phase 3**: [ ] Not started | [ ] In progress | [X] Complete
**Phase 4**: [ ] Not started | [ ] In progress | [X] Complete
**Phase 5**: [ ] Not started | [ ] In progress | [X] Complete
**Phase 6**: [ ] Not started | [ ] In progress | [ ] Complete

---

## Emergency Rollback

If any phase causes issues:

```bash
# Rollback last commit
git reset --hard HEAD~1

# Or create rollback branch
git checkout -b rollback-phase-N
git reset --hard <commit-before-phase-N>
```

---

## Final Validation

After completing ALL phases:

```bash
# Full test suite
make test
pytest tests/ -v

# Coverage check
make coverage

# Quick smoke test
python -c "from src.core.pddl_handler import PDDLHandler; print('✅ Import successful')"

# Information Gain algorithm test
pytest tests/algorithms/test_information_gain.py -v
```

**Success criteria**:
- ✅ All 165 curated tests pass
- ✅ No new test failures
- ✅ Code coverage maintained or improved
- ✅ Documentation is accurate and complete
- ✅ No breaking changes to public API
