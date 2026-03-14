# Changelog

All notable changes to `information-gain-aml` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_No unreleased changes._

---

## [0.2.0] — 2026-03-13

### Added
- `learn_negative_preconditions` flag to skip negative precondition candidates
- `--no-negative-preconditions` CLI option for local experiment script
- GitHub Actions CI/CD workflow for automated PyPI publishing via Trusted Publishers (OIDC)

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
| Cleanup | Remove obsolete test-only methods from `cnf_manager.py` | Planned |
| Cleanup | Remove dead code and fix imports | Planned |
| Refactor | Extract literal conversion helper and deduplicate model counting | Planned |
| Quality | Fix mypy errors and modernize type annotations | Planned |

### 0.3.0-beta.N — MCTS Pre-releases

| Type | Description | Status |
|------|-------------|--------|
| Feature | MCTS-based action selection (unstable, iterating toward 0.3.0) | Planned |

### 0.3.0 — MCTS

| Type | Description | Status |
|------|-------------|--------|
| Feature | Full MCTS-based action selection (stable) | Planned |

### Future (unscheduled)

| Type | Description |
|------|-------------|
| Feature | AMLGym-native environment integration — replace `active_environment.py` and `mock_environment.py` |
| Refactor | Framework-agnostic core — remove Unified Planning imports from algorithm code |
| Refactor | Consolidate to 3-class architecture: `SATKnowledgeBase`, `InformationGainLearner`, `Agent` |
| Bug | Fix `ObjectSubsetManager` recursion bug (gripper domain) |
| Milestone | **1.0.0** — Stable public API, fully importable from AMLGym |

---

[Unreleased]: https://github.com/omereliy/online_model_learning/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/omereliy/online_model_learning/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/omereliy/online_model_learning/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/omereliy/online_model_learning/releases/tag/v0.1.0
