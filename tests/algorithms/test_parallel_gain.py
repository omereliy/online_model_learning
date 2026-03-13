"""
Tests for parallel action gain computation.

Verifies that:
1. ActionGainContext can be serialized (pickled)
2. CNFManager reconstruction produces correct results
3. Parallel computation matches sequential computation
4. Worker mode selection (parallel vs sequential)
5. Worker exception handling works
"""

import pickle
import pytest
import logging
from typing import Set

from information_gain_aml.algorithms.information_gain import InformationGainLearner
from information_gain_aml.algorithms.parallel_gain import (
    ActionGainContext, ActionGainTask, ActionGainResult,
    WorkerCNFCache, _compute_action_gains_chunk, _worker_init
)
from information_gain_aml.core.cnf_manager import CNFManager
from information_gain_aml.environments.active_environment import ActiveEnvironment

logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def blocksworld_learner():
    """Create a blocksworld learner for testing."""
    return InformationGainLearner(
        domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
        problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
        max_iterations=10,
        num_workers=0,  # Disable parallel by default
    )


@pytest.fixture
def blocksworld_env():
    """Create blocksworld environment."""
    return ActiveEnvironment(
        "benchmarks/olam-compatible/blocksworld/domain.pddl",
        "benchmarks/olam-compatible/blocksworld/p01.pddl"
    )


class TestContextSerialization:
    """Test that ActionGainContext can be pickled."""

    def test_empty_context_can_be_pickled(self):
        """Empty context should be serializable."""
        context = ActionGainContext(
            pre={},
            pre_constraints={},
            eff_add={},
            eff_del={},
            eff_maybe_add={},
            eff_maybe_del={},
            cnf_clauses={},
            cnf_fluent_to_var={},
            cnf_var_to_fluent={},
            cnf_next_var={},
            cnf_solution_cache={},
            parameter_bound_literals={},
            state=set()
        )
        pickled = pickle.dumps(context)
        unpickled = pickle.loads(pickled)
        assert unpickled.state == set()
        assert unpickled.pre == {}

    def test_real_context_can_be_pickled(self, blocksworld_learner, blocksworld_env):
        """Real learner context should be serializable."""
        state = blocksworld_env.get_state()
        context = blocksworld_learner._create_parallel_context(state)

        # Should not raise
        pickled = pickle.dumps(context)
        unpickled = pickle.loads(pickled)

        # Verify key data survived
        assert unpickled.state == state
        assert len(unpickled.pre) == len(blocksworld_learner.pre)
        assert len(unpickled.parameter_bound_literals) > 0

    def test_context_with_constraints_can_be_pickled(self, blocksworld_learner, blocksworld_env):
        """Context with CNF constraints should be serializable."""
        state = blocksworld_env.get_state()

        # Add a failure to create constraints
        blocksworld_learner.observe(
            state=state,
            action='pick-up',
            objects=['b1'],
            success=False,
            next_state=None
        )

        context = blocksworld_learner._create_parallel_context(state)
        pickled = pickle.dumps(context)
        unpickled = pickle.loads(pickled)

        # Verify constraints survived
        assert len(unpickled.pre_constraints.get('pick-up', set())) > 0


class TestCNFReconstruction:
    """Test CNFManager reconstruction from serialized state."""

    def test_reconstruct_empty_cnf(self, blocksworld_learner, blocksworld_env):
        """Empty CNF should reconstruct correctly."""
        state = blocksworld_env.get_state()
        context = blocksworld_learner._create_parallel_context(state)

        cache = WorkerCNFCache(context)
        cnf = cache.get_cnf('pick-up')

        assert cnf is not None
        assert isinstance(cnf, CNFManager)

    def test_reconstructed_cnf_has_same_solutions(self, blocksworld_learner, blocksworld_env):
        """Reconstructed CNF should produce same solution count."""
        state = blocksworld_env.get_state()

        # Add constraint to make CNF non-trivial
        blocksworld_learner.observe(
            state=state,
            action='pick-up',
            objects=['b1'],
            success=False,
            next_state=None
        )

        # Build CNF in original learner
        original_cnf = blocksworld_learner.cnf_managers['pick-up']
        if original_cnf.has_clauses():
            original_count = original_cnf.count_solutions()

            # Create context and reconstruct
            context = blocksworld_learner._create_parallel_context(state)
            cache = WorkerCNFCache(context)
            reconstructed_cnf = cache.get_cnf('pick-up')

            reconstructed_count = reconstructed_cnf.count_solutions()
            assert reconstructed_count == original_count

    def test_cache_reuses_cnf_managers(self, blocksworld_learner, blocksworld_env):
        """WorkerCNFCache should reuse CNF managers for same action type."""
        state = blocksworld_env.get_state()
        context = blocksworld_learner._create_parallel_context(state)

        cache = WorkerCNFCache(context)
        cnf1 = cache.get_cnf('pick-up')
        cnf2 = cache.get_cnf('pick-up')

        assert cnf1 is cnf2  # Same object


class TestParallelMatchesSequential:
    """Test that parallel computation matches sequential."""

    def test_gains_match_for_simple_state(self, blocksworld_env):
        """Parallel and sequential should produce identical gains."""
        state = blocksworld_env.get_state()

        # Sequential learner
        learner_seq = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            max_iterations=5,
            num_workers=0,  # Force sequential
        )

        # Parallel learner
        learner_par = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            max_iterations=5,
            num_workers=2,
        )

        # Get action selections
        action_seq, objects_seq = learner_seq.select_action(state)
        action_par, objects_par = learner_par.select_action(state)

        # Top action should be the same
        assert action_seq == action_par

    def test_gains_match_after_observations(self, blocksworld_env):
        """Parallel and sequential should match after observations."""
        state = blocksworld_env.get_state()

        learner_seq = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            max_iterations=10,
            num_workers=0,
        )

        learner_par = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            max_iterations=10,
            num_workers=2,
        )

        # Add same observations to both
        for learner in [learner_seq, learner_par]:
            learner.observe(
                state=state,
                action='pick-up',
                objects=['b1'],
                success=False,
                next_state=None
            )

        # Selections should still match
        action_seq, _ = learner_seq.select_action(state)
        action_par, _ = learner_par.select_action(state)
        assert action_seq == action_par


class TestWorkerModes:
    """Test parallel vs sequential worker modes."""

    def test_parallel_with_multiple_workers(self, blocksworld_env):
        """Should complete using parallel when num_workers > 1."""
        state = blocksworld_env.get_state()

        learner = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            max_iterations=5,
            num_workers=2,
        )

        action, objects = learner.select_action(state)
        assert action is not None

    def test_disabled_with_zero_workers(self, blocksworld_env):
        """Should use sequential when num_workers=0."""
        state = blocksworld_env.get_state()

        learner = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            max_iterations=5,
            num_workers=0,  # Disabled
        )

        action, objects = learner.select_action(state)
        assert action is not None


class TestWorkerHelperFunctions:
    """Test worker helper functions directly."""

    def test_compute_chunk_returns_results(self, blocksworld_learner, blocksworld_env):
        """_compute_action_gains_chunk should return results."""
        state = blocksworld_env.get_state()
        context = blocksworld_learner._create_parallel_context(state)

        # Initialize worker cache manually
        _worker_init(context)

        tasks = [
            ActionGainTask('pick-up', ['b1']),
            ActionGainTask('pick-up', ['b2']),
        ]

        results = _compute_action_gains_chunk(tasks)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, ActionGainResult)
            assert r.expected_gain >= 0.0

    def test_worker_handles_errors_gracefully(self, blocksworld_learner, blocksworld_env):
        """Worker should handle errors and return results with error field."""
        state = blocksworld_env.get_state()
        context = blocksworld_learner._create_parallel_context(state)

        _worker_init(context)

        # Task with invalid action - should not crash
        tasks = [
            ActionGainTask('nonexistent-action', ['x']),
        ]

        results = _compute_action_gains_chunk(tasks)

        assert len(results) == 1
        assert results[0].error is not None or results[0].expected_gain == 0.0


class TestOptimizations:
    """Test optimization features for parallel computation."""

    def test_base_model_counts_precomputed(self, blocksworld_learner, blocksworld_env):
        """Context should include pre-computed base model counts."""
        state = blocksworld_env.get_state()
        context = blocksworld_learner._create_parallel_context(state)

        # Verify base_model_counts is populated
        assert hasattr(context, 'base_model_counts')
        assert len(context.base_model_counts) > 0

        # All actions should have counts
        for action_name in blocksworld_learner.pre.keys():
            assert action_name in context.base_model_counts
            assert context.base_model_counts[action_name] >= 1

    def test_worker_cache_update_context(self, blocksworld_learner, blocksworld_env):
        """WorkerCNFCache.update_context should clear cached managers."""
        state = blocksworld_env.get_state()
        context1 = blocksworld_learner._create_parallel_context(state)

        cache = WorkerCNFCache(context1)
        # Access an action to populate cache
        cache.get_cnf('pick-up')
        assert 'pick-up' in cache._managers

        # Create new context (simulating iteration change)
        context2 = blocksworld_learner._create_parallel_context(state)
        cache.update_context(context2)

        # Cache should be cleared
        assert len(cache._managers) == 0

    def test_persistent_pool_created_once(self, blocksworld_env):
        """Persistent pool should be reused across iterations."""
        learner = InformationGainLearner(
            domain_file="benchmarks/olam-compatible/blocksworld/domain.pddl",
            problem_file="benchmarks/olam-compatible/blocksworld/p01.pddl",
            num_workers=2,
        )

        state = blocksworld_env.get_state()

        # First call creates pool
        learner.select_action(state)
        first_pool = learner._pool

        # Second call should reuse pool
        learner.select_action(state)
        second_pool = learner._pool

        assert first_pool is second_pool

        # Cleanup
        learner._cleanup_pool()
