#!/bin/bash
# Test Docker compatibility of Test Guardian

echo "Testing Docker environment detection..."

# Simulate Docker environment
export DOCKER_CONTAINER=1
export HOSTNAME="test-container-abc123"

# Run status check
python .claude/agents/test_guardian.py --status | grep -A5 "environment"

# Clean up
unset DOCKER_CONTAINER
unset HOSTNAME

echo "Docker compatibility test complete"