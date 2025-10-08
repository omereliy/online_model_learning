# Slash Commands - Streamlined for Research

## Overview

These commands are optimized for the Online Model Learning research project. They prioritize:
- **Brevity**: ~40-100 lines per command (vs 300-550 previously)
- **Cross-session continuity**: Built for resuming work across conversations
- **Research focus**: No enterprise bloat or security warnings
- **Clean output**: Minimal templates, concise responses

## Available Commands

### Core Development (4 commands)
- `/tdd "task"` - Complete TDD workflow (plan + implement combined)
- `/refactor "files"` - Quick code quality analysis
- `/check "component"` - Validate correctness
- `/sync` - Synchronize all documentation

### Research Workflow (3 commands)
- `/status` - Quick project status check
- `/continue ["context"]` - Resume from previous session
- `/experiment "config"` - Run and analyze experiments

### Utility (1 command)
- `/engineer_prompt "text"` - Transform ideas into structured prompts

## Command Structure

All commands follow this pattern:
1. Load shared context from `_project.md`
2. Execute focused task
3. Provide concise output

## Migration from Legacy Commands

| Old Command (lines) | New Command (lines) | Key Changes |
|-------------------|-------------------|-------------|
| /plan (267) + /implement (420) | /tdd (50) | Combined, removed verbose steps |
| /inspect_refactor (551) | /refactor (40) | Removed templates and security checks |
| /validate_theory (481) | /check (30) | Focus on pass/fail, not theory |
| /docs_sync (120) | /sync (45) | Enhanced with consistency checks |
| /engineer_prompt (491) | /engineer_prompt (99) | Removed theory, kept essentials |

## Usage Examples

### Starting new work
```bash
/status                    # Check where you are
/tdd "implement Phase 2"   # Start implementation
```

### Resuming work
```bash
/continue                  # Auto-detect from git/docs
/continue "refactoring"    # With context hint
```

### Code quality
```bash
/refactor "src/core/*.py"  # Analyze files
/check "InformationGain"   # Validate algorithm
```

### Experiments
```bash
/experiment "configs/test.yaml"     # Run from config
/experiment "compare OLAM vs InfoGain"  # Run from description
```

## Legacy Commands

Previous verbose commands are archived in `legacy/` directory. They remain accessible but are deprecated in favor of the streamlined versions.

## Philosophy

**Less is more**: These commands generate focused, actionable output rather than comprehensive documentation. They're designed for a researcher who knows the project and needs quick, efficient assistance across multiple sessions.