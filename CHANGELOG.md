# Changelog

All notable changes to `information-gain-aml` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-03-15

### Added
- `BoundedLookaheadSelector` — depth-limited lookahead action selection with discounted future gain estimation
- `IGMCTSSelector` — full UCT-based MCTS action selection (selection, expansion, greedy rollout, backpropagation)
- `simulate_action()` — transition simulation using partial model (confirmed effects + stochastic applicability)
- `selection_strategy='lookahead'` option with configurable depth, top_k, discount
- `selection_strategy='mcts'` option with configurable iterations and rollout depth
- Selection strategy and MCTS/lookahead CLI flags for AMLGym experiment script

### Changed
- Extract `_apply_injective_binding_filter()` to deduplicate binding filter logic in sequential/parallel gain computation

### Known Issues
- Full UCT MCTS (`selection_strategy='mcts'`) is significantly slower than other strategies; performance optimization needed before practical use
- `run_full_experiments.py` missing `--mcts-iterations` / `--mcts-rollout-depth` CLI args (MCTS uses defaults; configurable via YAML)

---

## [0.2.1] — 2026-03-14

### Fixed
- 58 mypy errors across the codebase: real bugs, null-safety issues, and type annotations
- Zero-handling bug in `max_count` checks (`is not None` instead of truthiness)
- `_invalidate_cache()` called per-iteration instead of once after loop
- Empty-list handling for `assumptions` parameter
- `__repr__` triggering SAT solve on print

### Changed
- Extract `_literals_to_var_clause` helper to deduplicate literal conversion across 4 call sites
- Extract `_enumerate_models` helper to deduplicate SAT model counting loop
- Modernize type hints to Python 3.10+ style (`list[str]`, `dict[str, int]`, `X | None`)
- Added mypy to CI pipeline as a dev dependency

### Removed
- ~280 lines of dead code from `cnf_manager.py`: `minimize_qm`, `_rebuild_from_solutions`, `minimize_espresso`, `get_variable`, `add_var_clause`, `remove_clause`, `merge`
- Obsolete test-only methods superseded by assumptions API: `create_with_state_constraints`, `add_unit_constraint`, `add_constraint_from_unsatisfied`
- Unused imports (`itertools`, `FrozenSet`) and pointless try/except import block

---

## [0.2.0] — 2026-03-13

### Added
- `learn_negative_preconditions` flag to skip negative precondition candidates
- `--no-negative-preconditions` CLI option for local experiment script
- GitHub Actions CI/CD workflow for automated PyPI publishing via Trusted Publishers (OIDC)
- Local AMLGym experiment runner — run AMLGym benchmarks directly from this repo

### Changed
- Split experiment skills for finer-grained control

### Removed
- Unused OLAM-related code and outdated documentation
- Dead `parallel_threshold` parameter and `_should_use_parallel` method

### Fixed
- 28+ failing tests: stale APIs, predicate format, `MockEnvironment` constructor
- Test failures in checkpoints, model validator, and metrics
- CI: install `experiments` extras for test dependencies
- CI: make mypy non-blocking in publish workflow
- Gripper domain removed from tests due to `ObjectSubsetManager` recursion bug

---

## [0.1.1] — 2026-03-11

### Fixed
- Export `InformationGainLearner` from `algorithms` package `__init__.py`

---

## [0.1.0] — 2026-03-11

### Added
- Initial PyPI release of `information-gain-aml`
- CNF/SAT-based information-theoretic online action model learning
- Core algorithm: `InformationGainLearner` with SAT-backed precondition learning
- `SATKnowledgeBase` (via `cnf_manager.py`) for CNF formula management
- PDDL export and AMLGym adapter support (`OnlineAlgorithmAdapter`)
- Local experiment framework with configurable domains
- Unified Planning integration for PDDL parsing and domain representation

### Changed
- Renamed `src/` to `information_gain_aml/` for pip-installable package structure
- Migrated build tooling to `uv`

### Removed
- Legacy OLAM code, scripts, and documentation
- Broken/unused scripts and orphaned configs

---

## Roadmap

Planned features and improvements for upcoming releases.

### 0.2.1 — CNF Manager Cleanup

| Type | Description | Status |
|------|-------------|--------|
| Cleanup | Remove obsolete test-only methods from `cnf_manager.py` | Done |
| Cleanup | Remove dead code and fix imports | Done |
| Refactor | Extract literal conversion helper and deduplicate model counting | Done |
| Quality | Fix mypy errors and modernize type annotations | Done |

### 0.3.0-beta.1 — MCTS Phase 1: Bounded Lookahead

| Type | Description | Status |
|------|-------------|--------|
| Feature | `BoundedLookaheadSelector` — depth-limited lookahead action selection using discounted information gain | Done |
| Feature | `simulate_action()` — transition simulation using partial model (confirmed effects + stochastic applicability) | Done |
| Feature | `selection_strategy='lookahead'` option with configurable depth, top_k, discount | Done |

### 0.3.0-beta.2 — MCTS Phase 2: Full UCT

| Type | Description | Status |
|------|-------------|--------|
| Feature | `IGMCTSSelector` with UCT selection, expansion, greedy rollout, backpropagation | Done |
| Feature | `selection_strategy='mcts'` with configurable iterations and rollout depth | Done |

### 0.3.0 — MCTS (Stable)

| Type | Description | Status |
|------|-------------|--------|
| Feature | Stable MCTS-based action selection with lookahead and full UCT variants | Done |

### 0.4.0 — Planned Action Sequences

| Type | Description | Status |
|------|-------------|--------|
| Feature | Action selection from an ordered sequence generated by planning from the current state to an "informative state" using the learned model | Planned |

> **Note:** Definition of "informative state" is still under exploration.

### Future (unscheduled)

| Type | Description |
|------|-------------|
| Refactor | Framework-agnostic core — remove Unified Planning imports from algorithm code |
| Refactor | Consolidate to 3-class architecture: `SATKnowledgeBase`, `InformationGainLearner`, `Agent` |
| Bug | Fix `ObjectSubsetManager` recursion bug (gripper domain) |
| Milestone | **1.0.0** — Stable public API |

---

[Unreleased]: https://github.com/omereliy/online_model_learning/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/omereliy/online_model_learning/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/omereliy/online_model_learning/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/omereliy/online_model_learning/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/omereliy/online_model_learning/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/omereliy/online_model_learning/releases/tag/v0.1.0
