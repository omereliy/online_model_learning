---
description: Synchronize all project documentation
args: []
---

# Documentation Synchronization Command

## Purpose
Scan and synchronize all project documentation to ensure consistency, accuracy, and adherence to single source of truth principles.

## Execution Steps

### Step 1: Documentation Discovery
Scan for all documentation files:
- [ ] Find all `.md` files in project root
- [ ] Find all `.md` files in `docs/` directory
- [ ] Identify documentation in code comments (docstrings)
- [ ] List all discovered documentation files

**Expected Outcome**: Complete inventory of documentation

### Step 2: Content Validation
For each documentation file, check:
- [ ] Broken internal links (references to files/sections)
- [ ] Broken external links (URLs)
- [ ] References to deleted files or features
- [ ] Outdated version numbers or dates
- [ ] Inconsistent terminology across documents

**Expected Outcome**: List of validation issues

### Step 3: Single Source of Truth Enforcement
Apply DEVELOPMENT_RULES.md principles:
- [ ] Verify each fact appears in exactly ONE file
- [ ] Check for duplicate content across files
- [ ] Validate file purposes match content (README vs CLAUDE.md vs IMPLEMENTATION_TASKS.md)
- [ ] Ensure cross-references used instead of duplicated content

**Expected Outcome**: Identified violations of single source of truth

### Step 4: Consistency Checks
- [ ] Verify all code examples are syntactically correct
- [ ] Check command examples are up-to-date
- [ ] Validate file paths and directory structures
- [ ] Ensure consistent formatting (headers, lists, code blocks)
- [ ] Check for outdated API references

**Expected Outcome**: List of consistency issues

### Step 5: Missing Documentation Detection
Identify gaps:
- [ ] Scan `src/` for undocumented modules
- [ ] Check for public functions without docstrings
- [ ] Identify features in code but not in docs
- [ ] Find configuration options not documented
- [ ] Detect new dependencies not in README

**Expected Outcome**: List of missing documentation

### Step 6: Update Execution
For each issue found:
- [ ] Fix broken links
- [ ] Remove references to deleted features
- [ ] Update outdated content
- [ ] Add missing documentation
- [ ] Consolidate duplicate content to single source
- [ ] Update cross-references

**Expected Outcome**: All documentation synchronized

### Step 7: Validation Report
Generate comprehensive report:
```markdown
# Documentation Sync Report

## Files Scanned
- [List of all .md files]

## Issues Found
### Broken Links (count)
- file.md:line - [description]

### Outdated Content (count)
- file.md:line - [description]

### Missing Documentation (count)
- [description]

### Duplicates Removed (count)
- [description]

## Changes Made
- [Summary of updates]

## Recommendations
- [Suggestions for documentation improvements]
```

**Expected Outcome**: Complete sync report

## Validation Criteria
- [ ] All internal links functional
- [ ] No duplicate content (single source of truth maintained)
- [ ] All files match their designated purpose
- [ ] Code examples verified
- [ ] Cross-references accurate
- [ ] No references to deleted features

## Output Format
Present findings and changes in structured report showing:
1. Issues detected (categorized)
2. Fixes applied
3. Remaining manual actions needed (if any)
4. Documentation quality score

## Integration Points
- **DEVELOPMENT_RULES.md**: Enforces documentation rules
- **IMPLEMENTATION_TASKS.md**: Updates task documentation
- **CLAUDE.md**: Maintains navigation accuracy
- **README.md**: Keeps installation/usage current
