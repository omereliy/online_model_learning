#!/bin/bash
# Installation script for Test Guardian Agent

echo "🛡️ Installing Test Guardian Agent..."
echo "===================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Detect Python command based on environment
echo "Detecting Python environment..."

# Check if in conda environment
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ Conda environment detected: $CONDA_DEFAULT_ENV"
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version 2>&1)
# Check if in Docker
elif [ -f /.dockerenv ] || [ ! -z "$DOCKER_CONTAINER" ]; then
    echo "✓ Docker environment detected"
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
# Check for python3 first
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version 2>&1)
# Check for python
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version 2>&1)
else
    echo "Error: Python not found"
    exit 1
fi

echo "✓ Python found: $PYTHON_CMD ($PYTHON_VERSION)"

# Make scripts executable
chmod +x .claude/agents/test_guardian.py
chmod +x .claude/hooks/pre-commit
echo "✓ Made scripts executable"

# Install git hook
if [ -f .git/hooks/pre-commit ]; then
    echo "Backing up existing pre-commit hook..."
    mv .git/hooks/pre-commit .git/hooks/pre-commit.backup
fi

ln -sf ../../.claude/hooks/pre-commit .git/hooks/pre-commit
echo "✓ Installed pre-commit hook"

# Test the installation
echo ""
echo "Testing Test Guardian..."
$PYTHON_CMD .claude/agents/test_guardian.py --status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Test Guardian is working"
else
    echo "⚠️ Test Guardian may have issues"
fi

# Install Documentation Sync Agent if present
if [ -f .claude/agents/documentation_sync.py ]; then
    chmod +x .claude/agents/documentation_sync.py
    echo "✓ Documentation Sync Agent installed"
fi

# Install Experiment Validator Agent if present
if [ -f .claude/agents/experiment_validator.py ]; then
    chmod +x .claude/agents/experiment_validator.py
    echo "✓ Experiment Validator Agent installed"
fi

echo ""
echo "Installation complete! Agents are now active."
echo ""
echo "Test Guardian Usage:"
echo "  - Pre-commit validation: runs automatically"
echo "  - Manual validation: $PYTHON_CMD .claude/agents/test_guardian.py"
echo "  - Check status: $PYTHON_CMD .claude/agents/test_guardian.py --status"
echo ""
echo "Bypass (emergency): git commit --no-verify"
echo ""