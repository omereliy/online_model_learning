"""
Pytest configuration and fixtures for the test suite
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def blocksworld_domain():
    """Sample blocksworld domain PDDL content."""
    return """(define (domain blocksworld)
  (:requirements :strips :typing)

  (:types block)

  (:predicates
    (on ?x - block ?y - block)
    (ontable ?x - block)
    (clear ?x - block)
    (handempty)
    (holding ?x - block)
  )

  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x))
                 (not (clear ?x))
                 (not (handempty))
                 (holding ?x))
  )

  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (not (holding ?x))
                 (clear ?x)
                 (handempty)
                 (ontable ?x))
  )

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x))
                 (not (clear ?y))
                 (clear ?x)
                 (handempty)
                 (on ?x ?y))
  )

  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x)
                 (clear ?y)
                 (not (clear ?x))
                 (not (handempty))
                 (not (on ?x ?y)))
  )
)"""


@pytest.fixture
def blocksworld_problem():
    """Sample blocksworld problem PDDL content."""
    return """(define (problem blocksworld-p01)
  (:domain blocksworld)

  (:objects
    a b c - block
  )

  (:init
    (clear a)
    (on a b)
    (on b c)
    (ontable c)
    (handempty)
  )

  (:goal
    (and (on c b)
         (on b a))
  )
)"""


@pytest.fixture
def sample_cnf_formulas():
    """Predefined CNF formulas with known properties for testing."""
    return {
        # Simple satisfiable formula: (a OR b) AND (NOT a OR c)
        'simple_sat': {
            'clauses': [['a', 'b'], ['-a', 'c']],
            'expected_satisfiable': True,
            'expected_solutions': [
                {'b'},           # a=F, b=T, c=F
                {'b', 'c'},      # a=F, b=T, c=T
                {'a', 'c'},      # a=T, b=F, c=T
                {'a', 'b', 'c'}  # a=T, b=T, c=T
            ],
            'expected_count': 4,
            'expected_prob_a': 2 / 4,  # a is true in 2 out of 4 solutions
            'expected_prob_b': 3 / 4,  # b is true in 3 out of 4 solutions
            'expected_prob_c': 3 / 4   # c is true in 3 out of 4 solutions
        },

        # Unsatisfiable formula: (a) AND (NOT a)
        'unsat': {
            'clauses': [['a'], ['-a']],
            'expected_satisfiable': False,
            'expected_solutions': [],
            'expected_count': 0
        },

        # Single solution formula: (a) AND (NOT b)
        'single_solution': {
            'clauses': [['a'], ['-b']],
            'expected_satisfiable': True,
            'expected_solutions': [{'a'}],
            'expected_count': 1,
            'expected_prob_a': 1.0,
            'expected_prob_b': 0.0
        },

        # Complex formula for blocksworld scenario
        'blocksworld_precond': {
            'clauses': [['clear_a'], ['on_table_a', 'handempty']],
            'expected_satisfiable': True,
            'expected_count': 4,  # clear_a must be true, on_table_a OR handempty
            'expected_solutions': [
                {'clear_a', 'on_table_a'},
                {'clear_a', 'handempty'},
                {'clear_a', 'on_table_a', 'handempty'}
            ]
        }
    }


@pytest.fixture
def expected_grounded_fluents():
    """Expected grounded fluents for blocksworld domain with objects a,b,c."""
    return {
        # Expected fluents in specific order (sorted)
        'all_fluents': [
            'clear_a', 'clear_b', 'clear_c',
            'handempty',
            'holding_a', 'holding_b', 'holding_c',
            'on_a_a', 'on_a_b', 'on_a_c',
            'on_b_a', 'on_b_b', 'on_b_c',
            'on_c_a', 'on_c_b', 'on_c_c',
            'ontable_a', 'ontable_b', 'ontable_c'
        ],
        'total_count': 19
    }


@pytest.fixture
def expected_grounded_actions():
    """Expected grounded actions for blocksworld domain."""
    return {
        'all_actions': [
            'pick-up_a', 'pick-up_b', 'pick-up_c',
            'put-down_a', 'put-down_b', 'put-down_c',
            'stack_a_a', 'stack_a_b', 'stack_a_c',
            'stack_b_a', 'stack_b_b', 'stack_b_c',
            'stack_c_a', 'stack_c_b', 'stack_c_c',
            'unstack_a_a', 'unstack_a_b', 'unstack_a_c',
            'unstack_b_a', 'unstack_b_b', 'unstack_b_c',
            'unstack_c_a', 'unstack_c_b', 'unstack_c_c'
        ],
        'total_count': 24
    }


@pytest.fixture
def expected_initial_state():
    """Expected initial state from blocksworld problem p01."""
    return {
        'true_fluents': {'clear_a', 'on_a_b', 'on_b_c', 'ontable_c', 'handempty'},
        'false_fluents_count': 14  # Total 19 fluents - 5 true fluents
    }
