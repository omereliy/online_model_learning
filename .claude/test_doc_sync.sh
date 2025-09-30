#!/bin/bash
# Test Documentation Sync Agent

echo "Testing Documentation Sync Agent..."
echo "==================================="

# Test 1: Check for duplicates
echo ""
echo "1. Checking for duplicate content..."
python .claude/agents/documentation_sync.py --duplicates

# Test 2: Check ownership violations
echo ""
echo "2. Checking ownership violations..."
python .claude/agents/documentation_sync.py --check

# Test 3: Test suggestions for changed files
echo ""
echo "3. Testing update suggestions..."
# Simulate changed files
echo "src/algorithms/olam_adapter.py" > /tmp/changed_files.txt
echo "tests/test_olam_adapter.py" >> /tmp/changed_files.txt

cat /tmp/changed_files.txt | xargs python .claude/agents/documentation_sync.py --auto-sync --files

# Test 4: Generate report
echo ""
echo "4. Generating sync report..."
python .claude/agents/documentation_sync.py --report | head -20

echo ""
echo "Documentation Sync Agent test complete!"