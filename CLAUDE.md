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
| **Experiment Planning** | [experiment_readiness_assessment.md](docs/validation/experiment_readiness_assessment.md) | Paper-ready experiment gaps |
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
| **Starting a new task** | WORKFLOW_GUIDE | Use `/plan` then `/implement` |
| **Working on PDDL parsing** | UNIFIED_PLANNING_GUIDE | `src/core/pddl_handler.py` |
| **Implementing CNF formulas** | CNF_SAT_INTEGRATION | `src/core/cnf_manager.py` |
| **Adding algorithm adapters** | integration_guide | `src/algorithms/` |
| **Running experiments** | IMPLEMENTATION_TASKS | `configs/` |
| **Planning experiments** | experiment_readiness_assessment | `docs/validation/` |
| **Looking for commands** | QUICK_REFERENCE | Commands section |
| **Checking project status** | IMPLEMENTATION_TASKS | Status section |
| **Understanding architecture** | DEVELOPMENT_RULES | Architecture section |

## 🚀 TDD Workflow Commands

**→ See [WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md) for complete workflow examples and best practices**

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/docs-sync` | Synchronize all documentation | After major changes, before releases |
| `/plan` | Plan task with TDD methodology | Before implementing any feature/fix |
| `/implement` | Execute implementation with strict TDD | After planning, writes tests FIRST |
| `/validate-theory` | Validate algorithm correctness | Before implementing complex algorithms |
| `/inspect-refactor` | Analyze code quality and identify issues | Regular reviews, before refactoring |
| `/engineer-prompt` | Transform free text to optimized prompt | Before starting work in new/other sessions |

**Common Patterns**:
- **Simple Task**: `/plan` → `/implement` → `/docs-sync`
- **Algorithm**: `/validate-theory` → `/plan` → `/implement` → `/docs-sync`
- **Refactoring**: `/inspect-refactor` → `/plan` → `/implement` → `/docs-sync`
- **From Task List**: `/plan "Phase X"` → `/implement "Phase X"` → `/docs-sync`

## 📂 Component Status & File Locations

**→ See [IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md#completed-components) for current component status and file locations**

## 🔗 External Resources

**→ See [DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md#external-tool-paths) for external tool paths and availability**

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

## 📝 Documentation Rules

**→ See [DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md#markdown-documentation-rules) for documentation maintenance rules and single source of truth guidelines**