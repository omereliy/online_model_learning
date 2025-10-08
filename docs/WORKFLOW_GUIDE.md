# TDD Workflow Guide - Streamlined Commands

## Overview

This guide demonstrates the streamlined slash commands optimized for research workflows. Commands are now 80-90% shorter, focusing on essential functionality without verbose output.

## Available Commands

### Core Development
| Command | Purpose | Lines | Output Style |
|---------|---------|-------|--------------|
| `/tdd` | Complete TDD workflow | 50 | Brief confirmation |
| `/refactor` | Code quality analysis | 40 | Actionable issues only |
| `/check` | Validate correctness | 30 | Pass/fail summary |
| `/sync` | Documentation sync | 45 | Changed files list |

### Research Workflow
| Command | Purpose | Lines | Output Style |
|---------|---------|-------|--------------|
| `/status` | Project status | 25 | Current state summary |
| `/continue` | Resume work | 35 | Pick up where left off |
| `/experiment` | Run experiments | 40 | Results and insights |

### Utility
| Command | Purpose | Lines | Output Style |
|---------|---------|-------|--------------|
| `/engineer_prompt` | Optimize prompts | 99 | Structured prompt |

## Key Improvements

- **85% reduction** in command file sizes (2,330 → 350 lines)
- **No verbose templates** - just actionable output
- **Cross-session continuity** - `/continue` command for resuming
- **Research-focused** - removed enterprise security warnings
- **Shared context** - all commands load from `_project.md`

## Core Workflows

### Workflow 1: New Feature Implementation

```bash
# Check current status
/status

# Implement with TDD (combines old /plan + /implement)
/tdd "Implement convergence detection for Information Gain"

# Sync documentation
/sync
```

### Workflow 2: Resuming Work (Cross-Session)

```bash
# Start new session - check where you left off
/status

# Continue previous work
/continue
# or with context hint:
/continue "working on state converter refactoring"
```

### Workflow 3: Code Quality Review

```bash
# Quick refactoring check
/refactor "src/algorithms/*.py"

# Validate specific component
/check "OLAMAdapter"

# Apply fixes with TDD
/tdd "Fix identified issues in OLAMAdapter"
```

### Workflow 4: Running Experiments

```bash
# Run from config
/experiment "configs/blocksworld_comparison.yaml"

# Or describe experiment
/experiment "compare OLAM and InfoGain on gripper domain"

# Check results
/status
```

### Workflow 5: Quick Task from IMPLEMENTATION_TASKS.md

```bash
# Reference phase directly
/tdd "Phase 2"

# Or specific task
/tdd "convergence detection validation"
```

## Model and Mode Recommendations

For the streamlined commands, simpler configuration:

| Command | Recommended Model | Notes |
|---------|------------------|-------|
| `/tdd` | Current model | Fast execution needed |
| `/refactor` | Current model | Pattern matching |
| `/check` | Current model | Logic validation |
| `/sync` | Current model | File operations |
| `/status` | Current model | Simple queries |
| `/continue` | Current model | Context awareness |
| `/experiment` | Current model | Execution and analysis |
| `/engineer_prompt` | Current model | Text transformation |

No extended thinking needed - commands are now focused enough that standard mode works well.

## Example: Complete Development Cycle

```bash
# Monday - Start new feature
/status                                    # See current state
/tdd "Add model stability tracking"        # Implement with tests
/sync                                      # Update docs

# Tuesday - New session
/continue                                  # Resume automatically
/refactor "src/algorithms/information_gain.py"  # Check code quality
/tdd "Address refactoring issues"         # Fix issues found

# Wednesday - Testing
/check "InformationGain"                  # Validate correctness
/experiment "test convergence detection"   # Run experiments
/sync                                      # Final doc update
```

## Migration Notes

### From Legacy Commands

| If you previously used... | Now use... |
|--------------------------|------------|
| `/plan` then `/implement` | `/tdd` |
| `/inspect_refactor` | `/refactor` |
| `/validate_theory` | `/check` |
| `/docs_sync` | `/sync` |
| Starting fresh each session | `/continue` |

### Key Differences

1. **Output**: Much more concise, no verbose templates
2. **Speed**: Faster execution due to less output
3. **Context**: Shared `_project.md` reduces redundancy
4. **Workflow**: `/continue` enables cross-session work
5. **Focus**: Research-oriented, no enterprise features

## Tips

- Use `/status` at session start to orient yourself
- Use `/continue` to resume work seamlessly
- Use `/tdd` for all implementation (it includes planning)
- Use `/sync` after major changes to keep docs current
- Commands output only essential information - no fluff

## Legacy Commands

Previous verbose commands remain available in `.claude/commands/legacy/` but are deprecated. The new streamlined versions are recommended for all work.