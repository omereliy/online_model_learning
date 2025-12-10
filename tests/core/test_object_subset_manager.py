"""
Unit tests for ObjectSubsetManager class.

Tests verify correct behavior of:
- Type requirement computation from action schemas
- Object subset selection covering type requirements
- Subset rotation and object dismissal
- Exhaustion detection
"""

import pytest
from pathlib import Path
import tempfile

from src.core.object_subset_manager import ObjectSubsetManager, TypeRequirement
from src.core.pddl_io import PDDLReader


class TestTypeRequirement:
    """Test TypeRequirement dataclass."""

    def test_total_required(self):
        """Test total_required property calculation."""
        req = TypeRequirement(type_name='block', min_objects=2, spare_objects=1)
        assert req.total_required == 3

    def test_total_required_zero_spare(self):
        """Test total_required with zero spare objects."""
        req = TypeRequirement(type_name='truck', min_objects=1, spare_objects=0)
        assert req.total_required == 1


class TestObjectSubsetManagerBasics:
    """Test basic ObjectSubsetManager functionality."""

    @pytest.fixture
    def blocksworld_domain_large(self):
        """Blocksworld domain PDDL with many blocks for subset testing."""
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
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))
  )
  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))
  )
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))
  )
  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))
  )
)"""

    @pytest.fixture
    def blocksworld_problem_8_blocks(self):
        """Blocksworld problem with 8 blocks."""
        return """(define (problem blocksworld-p8)
  (:domain blocksworld)
  (:objects a b c d e f g h - block)
  (:init (clear a) (ontable a) (clear b) (ontable b) (clear c) (ontable c)
         (clear d) (ontable d) (clear e) (ontable e) (clear f) (ontable f)
         (clear g) (ontable g) (clear h) (ontable h) (handempty))
  (:goal (on a b))
)"""

    @pytest.fixture
    def loaded_domain_8_blocks(self, blocksworld_domain_large, blocksworld_problem_8_blocks, tmp_path):
        """Load domain with 8 blocks."""
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        domain_file.write_text(blocksworld_domain_large)
        problem_file.write_text(blocksworld_problem_8_blocks)

        reader = PDDLReader()
        domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
        return domain

    def test_type_requirements_computed_correctly(self, loaded_domain_8_blocks):
        """Test that type requirements are computed from action signatures."""
        manager = ObjectSubsetManager(loaded_domain_8_blocks, spare_objects_per_type=0)

        # Blocksworld: max 2 blocks needed (stack, unstack have 2 block params)
        assert 'block' in manager.type_requirements
        assert manager.type_requirements['block'].min_objects == 2

    def test_spare_objects_added(self, loaded_domain_8_blocks):
        """Test that spare objects are added to requirements."""
        manager = ObjectSubsetManager(loaded_domain_8_blocks, spare_objects_per_type=1)

        # With 1 spare: 2 + 1 = 3 total required
        assert manager.type_requirements['block'].total_required == 3

    def test_initial_subset_selected(self, loaded_domain_8_blocks):
        """Test that initial subset is selected on initialization."""
        manager = ObjectSubsetManager(loaded_domain_8_blocks, spare_objects_per_type=1)

        active = manager.get_active_object_names()

        # Should have 3 blocks (2 required + 1 spare)
        assert len(active) == 3
        assert manager.subset_rotation_count == 1

    def test_subset_objects_are_from_domain(self, loaded_domain_8_blocks):
        """Test that selected objects are valid domain objects."""
        manager = ObjectSubsetManager(loaded_domain_8_blocks, spare_objects_per_type=1)

        active = manager.get_active_object_names()
        all_domain_objects = set(loaded_domain_8_blocks.objects.keys())

        assert active.issubset(all_domain_objects)

    def test_rotation_selects_different_objects(self, loaded_domain_8_blocks):
        """Test that rotation selects new objects."""
        manager = ObjectSubsetManager(
            loaded_domain_8_blocks,
            spare_objects_per_type=1,
            random_seed=42
        )

        first_subset = manager.get_active_object_names().copy()

        # Rotate to new subset
        success = manager.rotate_subset()
        assert success

        second_subset = manager.get_active_object_names()

        # Subsets should be different (no overlap)
        assert first_subset.isdisjoint(second_subset)
        assert manager.subset_rotation_count == 2

    def test_exhaustion_after_all_objects_used(self, loaded_domain_8_blocks):
        """Test that exhaustion is detected when no more subsets possible."""
        # With 8 blocks and 3 per subset:
        # Init: uses 3, leaves 5
        # First rotation: uses 3, leaves 2
        # Second rotation: 2 >= 2 (min), succeeds, uses 2, leaves 0
        # Third rotation: 0 < 2 (min), fails
        manager = ObjectSubsetManager(
            loaded_domain_8_blocks,
            spare_objects_per_type=1,  # 2 min + 1 spare = 3
            random_seed=42
        )

        assert not manager.all_objects_exhausted()

        # First rotation (leaves 2)
        success = manager.rotate_subset()
        assert success
        assert not manager.all_objects_exhausted()

        # Second rotation (uses remaining 2, leaves 0)
        success = manager.rotate_subset()
        assert success
        assert not manager.all_objects_exhausted()

        # Third rotation - should exhaust (0 < 2 min)
        success = manager.rotate_subset()
        assert not success
        assert manager.all_objects_exhausted()

    def test_get_status_returns_correct_info(self, loaded_domain_8_blocks):
        """Test that get_status returns useful information."""
        manager = ObjectSubsetManager(loaded_domain_8_blocks, spare_objects_per_type=1)

        status = manager.get_status()

        assert 'rotation_count' in status
        assert status['rotation_count'] == 1
        assert 'exhausted' in status
        assert status['exhausted'] is False
        assert 'active_subset' in status
        assert 'available_counts' in status
        assert 'type_requirements' in status


class TestObjectSubsetManagerEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def small_domain_pddl(self):
        """Domain with only 2 blocks (minimum for stack action)."""
        return """(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (clear ?x - block))
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (clear ?y)
    :effect (on ?x ?y)
  )
)"""

    @pytest.fixture
    def small_problem_pddl(self):
        """Problem with only 2 blocks."""
        return """(define (problem small)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (clear a) (clear b))
  (:goal (on a b))
)"""

    @pytest.fixture
    def loaded_small_domain(self, small_domain_pddl, small_problem_pddl, tmp_path):
        """Load domain with 2 blocks."""
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        domain_file.write_text(small_domain_pddl)
        problem_file.write_text(small_problem_pddl)

        reader = PDDLReader()
        domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
        return domain

    def test_insufficient_objects_uses_all_available(self, loaded_small_domain):
        """Test that insufficient objects triggers warning and uses all available."""
        # 2 blocks available, but 2 required + 1 spare = 3 needed
        manager = ObjectSubsetManager(loaded_small_domain, spare_objects_per_type=1)

        active = manager.get_active_object_names()

        # Should use all 2 available blocks
        assert len(active) == 2
        assert active == {'a', 'b'}

    def test_immediate_exhaustion_with_exact_objects(self, loaded_small_domain):
        """Test that exact number of objects exhausts after first subset."""
        # 2 blocks, 2 required, 0 spare
        manager = ObjectSubsetManager(loaded_small_domain, spare_objects_per_type=0)

        # First subset uses all objects
        assert len(manager.get_active_object_names()) == 2

        # Cannot rotate - no more objects
        success = manager.rotate_subset()
        assert not success
        assert manager.all_objects_exhausted()

    def test_reset_restores_all_objects(self, loaded_small_domain):
        """Test that reset makes all objects available again."""
        manager = ObjectSubsetManager(loaded_small_domain, spare_objects_per_type=0)

        # Use all objects
        first_subset = manager.get_active_object_names()
        manager.rotate_subset()  # Should exhaust
        assert manager.all_objects_exhausted()

        # Reset
        manager.reset()

        # Should have objects again
        assert not manager.all_objects_exhausted()
        assert manager.subset_rotation_count == 1
        assert len(manager.get_active_object_names()) == 2


class TestObjectSubsetManagerTypeHierarchy:
    """Test handling of type hierarchies."""

    @pytest.fixture
    def hierarchical_domain_pddl(self):
        """Domain with type hierarchy (depots-like)."""
        return """(define (domain depots-simple)
  (:requirements :strips :typing)
  (:types
    surface - object
    pallet crate - surface
    truck - object
  )
  (:predicates
    (on ?x - crate ?y - surface)
  )
  (:action move-crate
    :parameters (?c - crate ?from - surface ?to - surface)
    :precondition (on ?c ?from)
    :effect (and (not (on ?c ?from)) (on ?c ?to))
  )
)"""

    @pytest.fixture
    def hierarchical_problem_pddl(self):
        """Problem with objects of hierarchical types."""
        return """(define (problem depots-p1)
  (:domain depots-simple)
  (:objects
    t1 t2 - truck
    c1 c2 c3 c4 - crate
    p1 p2 - pallet
  )
  (:init (on c1 p1) (on c2 p2))
  (:goal (on c1 c2))
)"""

    @pytest.fixture
    def loaded_hierarchical_domain(self, hierarchical_domain_pddl, hierarchical_problem_pddl, tmp_path):
        """Load hierarchical domain."""
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        domain_file.write_text(hierarchical_domain_pddl)
        problem_file.write_text(hierarchical_problem_pddl)

        reader = PDDLReader()
        domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
        return domain

    def test_type_requirements_include_parent_types(self, loaded_hierarchical_domain):
        """Test that type requirements are computed for action parameter types."""
        manager = ObjectSubsetManager(loaded_hierarchical_domain, spare_objects_per_type=0)

        # move-crate needs: 1 crate, 2 surfaces
        # crate is subtype of surface, so objects of type crate can fill surface slots
        assert 'crate' in manager.type_requirements
        assert 'surface' in manager.type_requirements

        # 1 crate parameter
        assert manager.type_requirements['crate'].min_objects == 1
        # 2 surface parameters
        assert manager.type_requirements['surface'].min_objects == 2


class TestObjectSubsetManagerDeterminism:
    """Test deterministic behavior with seeds."""

    @pytest.fixture
    def domain_for_seed_test(self, tmp_path):
        """Create domain for seed testing."""
        domain_content = """(define (domain test)
  (:requirements :strips :typing)
  (:types item)
  (:predicates (has ?x - item))
  (:action use :parameters (?x - item ?y - item) :precondition (has ?x) :effect (has ?y))
)"""
        problem_content = """(define (problem test-p)
  (:domain test)
  (:objects i1 i2 i3 i4 i5 i6 - item)
  (:init (has i1))
  (:goal (has i6))
)"""
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        domain_file.write_text(domain_content)
        problem_file.write_text(problem_content)

        reader = PDDLReader()
        domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
        return domain

    def test_same_seed_produces_same_subset(self, domain_for_seed_test):
        """Test that same seed produces identical subset selection."""
        manager1 = ObjectSubsetManager(domain_for_seed_test, spare_objects_per_type=1, random_seed=12345)
        manager2 = ObjectSubsetManager(domain_for_seed_test, spare_objects_per_type=1, random_seed=12345)

        assert manager1.get_active_object_names() == manager2.get_active_object_names()

    def test_different_seeds_may_produce_different_subsets(self, domain_for_seed_test):
        """Test that different seeds can produce different subsets."""
        manager1 = ObjectSubsetManager(domain_for_seed_test, spare_objects_per_type=1, random_seed=111)
        manager2 = ObjectSubsetManager(domain_for_seed_test, spare_objects_per_type=1, random_seed=222)

        # Not guaranteed to be different, but very likely with different seeds
        # Just check they both have valid subsets
        assert len(manager1.get_active_object_names()) == 3  # 2 + 1 spare
        assert len(manager2.get_active_object_names()) == 3
