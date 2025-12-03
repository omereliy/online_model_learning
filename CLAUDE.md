# Claude Code Context Guide - Navigation Index

- Dont over engineer solutions/architecture

## 🔴 CRITICAL UPDATE: OLAM Refactor (Nov 2025)
**OLAM integration has been completely refactored from real-time control to post-processing**
- **OLD**: `src/algorithms/olam_adapter.py` (REMOVED - 3,200 lines)
- **NEW**: Post-processing approach (1,000 lines total)
  - `src/algorithms/olam_external_runner.py` - Run OLAM as subprocess
  - `src/core/olam_trace_parser.py` - Parse execution logs
  - `src/core/olam_knowledge_reconstructor.py` - Replay learning rules
  - `scripts/analyze_olam_results.py` - Complete pipeline
- **IMPORTANT**: Use system Python 3.10+ (`/usr/bin/python3`) for OLAM
  - ✅ System Python 3.10.12: All domains work (blocksworld 100% success)
  - ❌ Conda Python 3.9.23: Blocksworld fails with IndexError
- **Documentation**: See [OLAM_interface_new.md](docs/external_repos/OLAM_interface_new.md)
- **68% code reduction**, no Java dependency in our code

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
| **Lifted Support** | [LIFTED_SUPPORT.md](docs/LIFTED_SUPPORT.md) | Parameterized actions/fluents, grounding |
| **CNF/SAT** | [CNF_SAT_INTEGRATION.md](docs/information_gain_algorithm/CNF_SAT_INTEGRATION.md) | PySAT integration |
| **Info Gain Algorithm** | [INFORMATION_GAIN_ALGORITHM.md](docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md) | CNF-based learning |
| **OLAM Interface (NEW)** | [OLAM_interface_new.md](docs/external_repos/OLAM_interface_new.md) | Post-processing approach |
| **ModelLearner** | [ModelLearner_interface.md](docs/external_repos/ModelLearner_interface.md) | Optimistic exploration |
| **Integration Pattern** | [integration_guide.md](docs/external_repos/integration_guide.md) | Adapter implementation |

## 🎯 Quick Navigation by Task

| If You're... | Read This | Location |
|--------------|-----------|----------|
| **Starting a new task** | WORKFLOW_GUIDE | Use `/plan` then `/implement` |
| **Working on PDDL/domain representation** | LIFTED_SUPPORT | `src/core/lifted_domain.py`, `src/core/grounding.py` |
| **Implementing CNF formulas** | CNF_SAT_INTEGRATION | `src/core/cnf_manager.py` |
| **Adding algorithm adapters** | integration_guide | `src/algorithms/` |
| **Running OLAM experiments** | OLAM_interface_new | `scripts/analyze_olam_results.py` |
| **Running experiments** | IMPLEMENTATION_TASKS | `configs/` |
| **Planning experiments** | experiment_readiness_assessment | `docs/validation/` |
| **Looking for commands** | QUICK_REFERENCE | Commands section |
| **Checking project status** | IMPLEMENTATION_TASKS | Status section |
| **Understanding architecture** | DEVELOPMENT_RULES | Architecture section |

## 🚀 Streamlined Commands (85% Smaller!)

**→ See [WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md) for complete workflow examples**

### Core Development
| Command | Purpose | Output |
|---------|---------|--------|
| `/tdd` | Complete TDD workflow (plan + implement) | Brief confirmation |
| `/refactor` | Quick code quality analysis | Actionable issues only |
| `/check` | Validate correctness | Pass/fail summary |
| `/sync` | Synchronize documentation | Changed files list |

### Research Workflow
| Command | Purpose | Output |
|---------|---------|--------|
| `/status` | Quick project status check | Current state |
| `/continue` | Resume from previous session | Pick up where left |
| `/experiment` | Run and analyze experiments | Results summary |

### Utility
| Command | Purpose | Output |
|---------|---------|--------|
| `/engineer_prompt` | Transform text to prompt | Structured prompt |

**Common Patterns**:
- **New Task**: `/status` → `/tdd "task"` → `/sync`
- **Resume Work**: `/continue` → `/sync`
- **Code Review**: `/refactor "files"` → `/tdd "fix issues"`
- **Run Tests**: `/experiment "config"` → `/status`

## 📂 Component Status & File Locations

**→ See [IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md#completed-components) for current component status and file locations**

## 🔗 External Resources

**→ See [DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md#external-tool-paths) for external tool paths and availability**

## 🔧 OLAM Experiments (Post-Refactor Workflow)

### OLAM Results Directory Structure
**IMPORTANT**: OLAM is run externally (by user), and results are stored in a separate directory.
Our code **reads** these results (READ-ONLY) for analysis.

Expected structure (provided by user):
```
/home/omer/projects/olam_results/
├── <domain_name>/                    # e.g., depots, blocksworld, etc.
│   ├── trace_complete.json           # Concatenated trace of all problems
│   ├── 1_p00_<domain>_gen/           # Problem directory (number_problemID_domain_gen)
│   │   ├── trace.json                # JSON Lines format (one JSON per line)
│   │   ├── checkpoints/
│   │   │   ├── iter_1/
│   │   │   │   ├── operator_certain_predicates.json
│   │   │   │   ├── operator_uncertain_precs.json
│   │   │   │   ├── operator_certain_positive_effects.json
│   │   │   │   ├── operator_certain_negative_effects.json
│   │   │   │   ├── operator_uncertain_positive_effects.json
│   │   │   │   ├── operator_uncertain_negative_effects.json
│   │   │   │   ├── operator_useless_negated_precs.json
│   │   │   │   ├── operator_useless_possible_precs.json
│   │   │   │   ├── domain_learned.pddl
│   │   │   │   └── domain_learned_certain.pddl
│   │   │   ├── iter_2/
│   │   │   ├── iter_3/
│   │   │   └── ... (checkpoints may end earlier than max iterations)
│   │   ├── final/                    # Final checkpoint (same structure as iter_N)
│   │   ├── domain_learned.pddl       # Final learned domain
│   │   └── ... (other final outputs)
│   ├── 2_p01_<domain>_gen/
│   └── ... (all problems for this domain)
```

**Key Notes**:
- Checkpoints are numbered dynamically (iter_1, iter_2, ..., iter_N)
- Not all problems reach the same max iteration (may terminate early)
- Each checkpoint has 8-10 JSON files with model components
- trace.json is JSON Lines format: `{"domain": "...", "iter": 1, "action": "...", "success": true, ...}`

### Processing OLAM Results
```bash
# Process all problems in a domain
python scripts/process_olam_results.py \
    --olam-results /home/omer/projects/olam_results/depots \
    --ground-truth benchmarks/olam-compatible/depots/domain.pddl \
    --output-dir results/olam_processed

# This will:
# 1. Auto-detect all problem directories (1_p00_*, 2_p01_*, etc.)
# 2. Auto-detect checkpoints per problem (iter_1, iter_2, ...)
# 3. Load JSON exports from each checkpoint
# 4. Compute metrics (precision/recall) vs ground truth
# 5. Aggregate across problems to domain level
```

### Key Changes from Old Approach
- **No more OLAMAdapter** - removed entirely
- **No Java management** - OLAM handles its own Java
- **No state synchronization** - works via checkpoint JSON exports
- **Simpler debugging** - can analyze results offline
- **User runs OLAM** - we only read results (READ-ONLY)

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
- dont create excessive amount of scripts and markdowns for each task i give you