# Claude Code Context Guide

## 🚨 MANDATORY: Start Here
**ALWAYS review this first:** [Development Rules](docs/DEVELOPMENT_RULES.md)
- Contains project structure, tech stack, implementation rules
- GitHub MCP safety guidelines
- Common pitfalls to avoid
- **IMPORTANT**: Update IMPLEMENTATION_TASKS.md after each task (with user approval!)

## 📚 Quick Documentation Reference

### Core Understanding
- **[Unified Planning Guide](docs/UNIFIED_PLANNING_GUIDE.md)** - Critical for understanding how UP handles expression trees, NOT simple sets
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Fast lookup for common patterns
- **[Lifted Support](docs/LIFTED_SUPPORT.md)** - Lifted vs grounded fluents/actions

### Implementation Planning
- **[Implementation Tasks](docs/IMPLEMENTATION_TASKS.md)** - When implementing new features, check task breakdown here
- **[Information Gain Algorithm](docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md)** - For CNF-based learning approach
- **[CNF/SAT Integration](docs/information_gain_algorithm/CNF_SAT_INTEGRATION.md)** - PySAT usage patterns

### External Integrations
- **[OLAM Interface](docs/external_repos/OLAM_interface.md)** - When working with OLAM adapter
- **[ModelLearner Interface](docs/external_repos/ModelLearner_interface.md)** - When working with Optimistic Exploration
- **[Integration Guide](docs/external_repos/integration_guide.md)** - Adapter pattern implementation

## 🎯 Context Hints by Task

### Working on PDDL Parsing?
→ Read: [Unified Planning Guide](docs/UNIFIED_PLANNING_GUIDE.md)
- UP uses expression trees (FNode), not sets
- Preconditions need recursive traversal
- See `src/core/pddl_handler.py`

### Implementing CNF Formulas?
→ Read: [CNF/SAT Integration](docs/information_gain_algorithm/CNF_SAT_INTEGRATION.md)
- PySAT/python-sat import handling
- Variable mapping strategies
- See `src/core/cnf_manager.py`

### Adding Algorithm Adapters?
→ Read: [Integration Guide](docs/external_repos/integration_guide.md)
- BaseActionModelLearner interface
- State/action format conversion
- See `src/algorithms/`

### Debugging Type Hierarchy?
→ Check: `pddl_handler.get_type_hierarchy()`
- 'object' is implicit root
- Single parent constraint
- See tests in `tests/test_pddl_handler.py`

### Running Experiments?
→ Reference: [Implementation Tasks](docs/IMPLEMENTATION_TASKS.md)
- YAML config structure
- Metrics to collect
- See `configs/` directory

## 🔧 Key Technical Notes

1. **UP Expression Trees**: Preconditions are FNode trees, not sets. Always traverse recursively.

2. **Import Paths**:
   ```python
   # External algorithms need sys.path
   sys.path.append('/home/omer/projects/OLAM')
   sys.path.append('/home/omer/projects/ModelLearner/src')
   ```

3. **PySAT Import Fix**:
   ```python
   try:
       from pysat.solvers import Minisat22
   except ImportError:
       from pysat.solvers import Minisat22  # python-sat package
   ```

4. **Type Hierarchy**: Always check `is_subtype_of('object')` returns True for all types.

5. **Lifted → Grounded**: Three stages: Lifted (schema) → Parameter-bound → Fully grounded

## 💡 Common Issues & Solutions

| Issue | Solution | Reference |
|-------|----------|-----------|
| UP preconditions not visible as sets | Use recursive traversal of FNode tree | [UP Guide](docs/UNIFIED_PLANNING_GUIDE.md#expression-tree-structure) |
| PySAT import errors | Use try/except for pysat vs python-sat | `src/core/cnf_manager.py:6-12` |
| Type hierarchy missing 'object' | Special case in `is_subtype_of()` | `src/core/pddl_handler.py` |
| Lifted fluent grounding | Use `instantiate_lifted_clause()` | `src/core/cnf_manager.py:393` |

## 🚀 Quick Commands

```bash
# Run tests
pytest tests/test_pddl_handler.py -v
pytest tests/test_cnf_manager.py -v

# Check implementation status
grep -r "TODO" src/ --include="*.py"
grep -r "NotImplementedError" src/ --include="*.py"
```

## 📝 Remember
- Don't modify external repos (OLAM, ModelLearner)
- Always convert state/action formats between systems
- Test with multiple PDDL domains (blocksworld, logistics, etc.)
- Use YAML configs for experiments, not hardcoded values