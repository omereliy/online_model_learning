# Claude Code Context Guide - Navigation Index

## 🚨 MANDATORY: Read First
**[DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md)** - ALWAYS read this file first in every conversation
- Project conventions, architecture, implementation rules
- Testing approach (`make test` vs `pytest`)
- Docker usage and rationale
- Markdown editing guidelines
- GitHub MCP safety rules

## 📍 Quick Navigation by Topic

| Topic | Go To | Purpose |
|-------|-------|----------|
| **Project Status** | [IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md) | Current progress, todos |
| **Commands & Patterns** | [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | All commands, code snippets |
| **Test Analysis** | [TEST_IMPLEMENTATION_REVIEW.md](docs/TEST_IMPLEMENTATION_REVIEW.md) | Test quality assessment |
| **UP Expression Trees** | [UNIFIED_PLANNING_GUIDE.md](docs/UNIFIED_PLANNING_GUIDE.md) | FNode traversal patterns |
| **Lifted Support** | [LIFTED_SUPPORT.md](docs/LIFTED_SUPPORT.md) | Parameterized actions/fluents |
| **CNF/SAT** | [CNF_SAT_INTEGRATION.md](docs/information_gain_algorithm/CNF_SAT_INTEGRATION.md) | PySAT integration |
| **Info Gain Algorithm** | [INFORMATION_GAIN_ALGORITHM.md](docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md) | CNF-based learning |
| **OLAM Interface** | [OLAM_interface.md](docs/external_repos/OLAM_interface.md) | OLAM adapter guide |
| **ModelLearner** | [ModelLearner_interface.md](docs/external_repos/ModelLearner_interface.md) | Optimistic exploration |
| **Integration Pattern** | [integration_guide.md](docs/external_repos/integration_guide.md) | Adapter implementation |

## 🎯 Quick Navigation by Task

| If You're... | Read This | Location |
|--------------|-----------|----------|
| **Working on PDDL parsing** | UNIFIED_PLANNING_GUIDE | `src/core/pddl_handler.py` |
| **Implementing CNF formulas** | CNF_SAT_INTEGRATION | `src/core/cnf_manager.py` |
| **Adding algorithm adapters** | integration_guide | `src/algorithms/` |
| **Running experiments** | IMPLEMENTATION_TASKS | `configs/` |
| **Debugging tests** | TEST_IMPLEMENTATION_REVIEW | `tests/` |
| **Looking for commands** | QUICK_REFERENCE | Commands section |
| **Checking project status** | IMPLEMENTATION_TASKS | Status section |
| **Understanding architecture** | DEVELOPMENT_RULES | Architecture section |

## 📂 Key File Locations

| Component | Path | Status |
|-----------|------|--------|
| **CNF Manager** | `src/core/cnf_manager.py` | ✅ Complete |
| **PDDL Handler** | `src/core/pddl_handler.py` | ✅ Complete |
| **OLAM Adapter** | `src/algorithms/olam_adapter.py` | ✅ Complete |
| **Experiment Runner** | `src/experiments/runner.py` | ✅ Complete |
| **Metrics Collector** | `src/experiments/metrics.py` | ✅ Complete |
| **Info Gain Learner** | `src/algorithms/information_gain.py` | ✅ Complete (Phase 3) |
| **ModelLearner Adapter** | `src/algorithms/optimistic_adapter.py` | ⏳ TODO (blocked - repo unavailable) |
| **PDDL Environment** | `src/environments/pddl_environment.py` | ✅ Complete |

## 🔗 External Resources

| Resource | Path |
|----------|------|
| **OLAM** | `/home/omer/projects/OLAM/` |
| **ModelLearner** | `/home/omer/projects/ModelLearner/` (⚠️ Repo unavailable) |
| **Fast Downward** | `/home/omer/projects/fast-downward/` |
| **VAL Validator** | `/home/omer/projects/Val/` |

## 📋 Quick Checklist

Before starting any work:
- [ ] Read DEVELOPMENT_RULES.md
- [ ] Check IMPLEMENTATION_TASKS.md for current status
- [ ] Review relevant documentation for your task
- [ ] Run `make test` to verify baseline
- [ ] Use appropriate test approach (`make test` for stable, `pytest` for all)

## 🛡️ Pre-Commit Test Protection

A git pre-commit hook runs `make test` before every commit.
- Blocks commits if tests fail (enforces TDD methodology)
- Bypass if needed: `git commit --no-verify` (emergency only)
- Check status: `python .claude/agents/test_guardian.py --status`
- **Important**: Must be in `action-learning` conda environment for tests to pass

Implementation: `.claude/hooks/pre-commit` installed via `.claude/install.sh`

## 📝 Documentation Maintenance Rules

**Single Source of Truth**: Each fact appears in exactly ONE file

### Content Ownership
| File | Owns |
|------|------|
| README.md | Installation, basic usage |
| DEVELOPMENT_RULES.md | Architecture, rules, conventions |
| QUICK_REFERENCE.md | All commands, code snippets |
| IMPLEMENTATION_TASKS.md | Current status, TODOs |

### Rules
- ✅ Keep each fact in ONE document only
- ✅ Use cross-references via links
- ✅ Update only the authoritative source
- ❌ Don't duplicate information across files
- ❌ Don't copy content instead of linking

### Maintenance Protocol
1. Quarterly review for duplicate content
2. Update only authoritative location
3. Add links instead of copying
4. Remove duplicates when found
5. Track changes in git history