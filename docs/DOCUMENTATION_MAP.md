# Documentation Map - Single Source of Truth

## Document Purposes

Each document has a specific purpose. Information appears in exactly ONE authoritative location:

| Document | Purpose | Content Owner |
|----------|---------|--------------|
| **[CLAUDE.md](../CLAUDE.md)** | Navigation index | Links only, no content duplication |
| **[README.md](../README.md)** | Project overview | Installation, basic usage |
| **[DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md)** | Project conventions | Architecture, rules, testing approach |
| **[IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md)** | Progress tracking | Current status, completed work, TODOs |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Commands & patterns | All commands, code snippets |

## Specialized Documentation

### Algorithm Interfaces
| Document | Purpose |
|----------|---------|
| **[OLAM_interface.md](external_repos/OLAM_interface.md)** | OLAM algorithm specifics |
| **[ModelLearner_interface.md](external_repos/ModelLearner_interface.md)** | ModelLearner specifics |
| **[integration_guide.md](external_repos/integration_guide.md)** | Adapter pattern guide |

### Technical Guides
| Document | Purpose |
|----------|---------|
| **[UNIFIED_PLANNING_GUIDE.md](UNIFIED_PLANNING_GUIDE.md)** | UP Framework usage |
| **[LIFTED_SUPPORT.md](LIFTED_SUPPORT.md)** | Parameterized actions/fluents |
| **[CNF_SAT_INTEGRATION.md](information_gain_algorithm/CNF_SAT_INTEGRATION.md)** | CNF/SAT implementation |
| **[INFORMATION_GAIN_ALGORITHM.md](information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md)** | Info gain algorithm design |

## Content Rules

### ✅ DO
- Keep each fact in ONE document only
- Use cross-references via links
- Update only the authoritative source
- Keep CLAUDE.md as pure navigation

### ❌ DON'T
- Duplicate information across files
- Copy content instead of linking
- Add verbose explanations
- Include implementation details in multiple places

## Finding Information

### By Topic
- **Project rules** → DEVELOPMENT_RULES.md
- **Current status** → IMPLEMENTATION_TASKS.md
- **Commands** → QUICK_REFERENCE.md
- **Navigation** → CLAUDE.md
- **Algorithm details** → external_repos/*_interface.md

### By Task
- **Starting work** → Read CLAUDE.md, then DEVELOPMENT_RULES.md
- **Running tests** → QUICK_REFERENCE.md (commands section)
- **Adding adapter** → integration_guide.md
- **Checking progress** → IMPLEMENTATION_TASKS.md

## Maintenance Protocol

1. **Quarterly review** for duplicate content
2. **Update only** authoritative location
3. **Add links** instead of copying
4. **Remove duplicates** when found
5. **Track changes** in git history