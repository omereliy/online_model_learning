# Test Suite for Online Model Learning Framework

## Overview

This test suite provides comprehensive validation for the Phase 1 implementation of the Online Model Learning Framework, specifically testing the `CNFManager` and `PDDLHandler` core components.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py             # Pytest fixtures and configuration
├── test_cnf_manager.py     # CNFManager comprehensive tests
├── test_pddl_handler.py    # PDDLHandler comprehensive tests
├── pytest.ini             # Pytest configuration
├── requirements-test.txt   # Testing dependencies
├── Makefile               # Test automation commands
└── README.md              # This file
```

## Key Testing Principles

### 1. Predefined Expected Outcomes
All tests use predefined expected results rather than evaluating outputs dynamically:

```python
# GOOD: Predefined expected outcomes
expected_solutions = [{'a', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}]
assert actual_solutions == expected_solutions

# AVOIDED: Dynamic evaluation
assert count_solutions() == len(get_all_solutions())  # Don't do this
```

### 2. Proper Assertions
Tests validate correctness through specific assertions:

```python
# Test exact values
assert cnf.count_solutions() == 4
assert prob_a == pytest.approx(0.5, abs=1e-10)
assert initial_state == {'clear_a', 'on_a_b', 'handempty'}
```

### 3. Comprehensive Coverage
Tests cover:
- **Normal operations**: Standard use cases
- **Edge cases**: Empty inputs, large formulas, invalid data
- **Error conditions**: Malformed inputs, non-existent files
- **State consistency**: Roundtrip conversions, mapping consistency

## Test Categories

### CNFManager Tests (`test_cnf_manager.py`)

1. **TestCNFManagerBasics** - Initialization and fluent management
2. **TestCNFManagerClauses** - Clause addition and manipulation
3. **TestCNFManagerSatisfiability** - SAT solving with known formulas
4. **TestCNFManagerSolutionCounting** - Model counting with expected counts
5. **TestCNFManagerProbabilities** - Probability calculations
6. **TestCNFManagerEntropy** - Information-theoretic measures
7. **TestCNFManagerOperations** - Copy, merge, and utility operations
8. **TestCNFManagerStringRepresentation** - Human-readable output
9. **TestCNFManagerEdgeCases** - Error handling and edge cases

### PDDLHandler Tests (`test_pddl_handler.py`)

1. **TestPDDLHandlerInitialization** - Basic setup and properties
2. **TestPDDLHandlerParsing** - PDDL file parsing with UP
3. **TestPDDLHandlerGroundedFluents** - Fluent grounding with exact counts
4. **TestPDDLHandlerGroundedActions** - Action grounding and parsing
5. **TestPDDLHandlerStateConversion** - State format conversions
6. **TestPDDLHandlerActionProperties** - Action precondition/effect extraction
7. **TestPDDLHandlerExportFeatures** - PDDL export functionality
8. **TestPDDLHandlerEdgeCases** - Error handling
9. **TestPDDLHandlerGoalHandling** - Goal state processing
10. **TestPDDLHandlerFeatureSupport** - PDDL feature support detection

## Key Test Fixtures

### CNF Formula Fixtures
Predefined CNF formulas with known properties:

- **simple_sat**: `(a OR b) AND (NOT a OR c)` - 4 solutions
- **unsat**: `(a) AND (NOT a)` - Unsatisfiable
- **single_solution**: `(a) AND (NOT b)` - Exactly 1 solution
- **blocksworld_precond**: Domain-specific uncertainty formula

### PDDL Domain Fixtures
Complete blocksworld domain with:
- 3 objects (a, b, c)
- 5 fluents (on, ontable, clear, handempty, holding)
- 4 actions (pick-up, put-down, stack, unstack)
- Expected counts: 19 grounded fluents, 24 grounded actions

## Running Tests

### Basic Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cnf_manager.py -v

# Run specific test class
pytest tests/test_cnf_manager.py::TestCNFManagerBasics -v
```

### Using Makefile
```bash
# All tests
make test

# Verbose output
make test-verbose

# Only CNF tests
make test-cnf

# Only PDDL tests
make test-pddl

# With conda environment
make test-conda
```

### Test Categories
```bash
# Only unit tests
pytest -m "unit"

# Exclude slow tests
pytest -m "not slow"

# CNF-specific tests
pytest -m "cnf"

# PDDL-specific tests
pytest -m "pddl"
```

## Validation Results

✅ **All 54 tests pass** with comprehensive coverage:

- **31 CNFManager tests**: Full SAT solver integration validation
- **23 PDDLHandler tests**: Complete UP framework integration validation

### Key Validated Behaviors

#### CNFManager
- Correct SAT solving for known formulas
- Accurate model counting (4 solutions for `(a OR b) AND (NOT a OR c)`)
- Precise probability calculations (P(a)=0.5, P(b)=0.75, P(c)=0.75)
- Proper entropy computation for information gain
- Consistent variable mapping and formula operations

#### PDDLHandler
- Successful PDDL parsing with Unified Planning Framework
- Exact grounding: 19 fluents and 24 actions for blocksworld
- Correct initial state extraction: `{clear_a, on_a_b, on_b_c, ontable_c, handempty}`
- Bidirectional state conversions (fluent sets ↔ state dictionaries)
- Proper action parameter binding and validation

## Dependencies

Core testing dependencies:
- `pytest>=7.0.0` - Testing framework
- `python-sat>=1.8.dev22` - SAT solver integration
- `unified-planning[fast-downward,tamer]>=1.2.0` - PDDL handling

Install with:
```bash
pip install -r requirements-test.txt
```

## Test Philosophy

This test suite follows the principle of **validation through assertion** rather than **verification through evaluation**. Each test:

1. **Defines expected outcomes upfront** based on theoretical analysis
2. **Uses specific assertions** to validate actual vs expected behavior
3. **Tests real functionality** using live SAT solvers and UP framework
4. **Covers edge cases** to ensure robustness
5. **Maintains independence** - each test can run in isolation

This approach ensures that the implementation correctly handles the complex logic of CNF formula manipulation and PDDL processing required for the information-theoretic action model learning algorithm.