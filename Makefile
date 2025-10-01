.PHONY: test test-quick test-full test-metrics test-integration clean help run-experiment analyze fix-deadlock coverage coverage-html coverage-report benchmark

# Default target
help:
	@echo "Available targets:"
	@echo "  make test           - Run full test suite"
	@echo "  make test-quick     - Run quick critical tests only"
	@echo "  make test-metrics   - Test metrics module"
	@echo "  make test-integration - Run integration tests"
	@echo "  make run-experiment - Run a quick experiment with mock environment"
	@echo "  make analyze        - Analyze latest experiment results"
	@echo "  make clean          - Remove generated files and caches"
	@echo "  make coverage       - Run tests with code coverage"
	@echo "  make coverage-html  - Generate HTML coverage report"
	@echo "  make benchmark      - Run performance benchmarks"
	@echo "  make fix-deadlock   - Apply RLock fix (already applied)"

# Quick test of critical functionality
test-quick:
	@echo "Running quick critical tests..."
	@python3 scripts/run_test_suite.py --quick

# Run full test suite
test-full:
	@echo "Running complete test suite..."
	@python3 scripts/run_test_suite.py

# Default test target runs full suite
test: test-full

# Test specific modules
test-metrics:
	@echo "Testing metrics module..."
	@python3 -m pytest tests/test_metrics.py -v

test-integration:
	@echo "Running integration tests..."
	@python3 -m pytest tests/test_experiment_integration.py -v

# Run a quick experiment
run-experiment:
	@echo "Running quick mock experiment..."
	@python3 scripts/test_mock_experiment.py

# Analyze results
analyze:
	@echo "Analyzing latest experiment results..."
	@python3 scripts/analyze_results.py --latest

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache
	@rm -rf results/*_test_*
	@echo "Cleaned!"

# Check if deadlock fix is applied
check-deadlock:
	@echo "Checking if RLock fix is applied..."
	@grep -q "threading.RLock()" src/experiments/metrics.py && \
		echo "✓ RLock fix is applied" || \
		echo "✗ RLock fix not found - Lock may cause deadlocks"

# Apply deadlock fix (for reference - already applied)
fix-deadlock:
	@echo "RLock fix already applied to metrics.py"
	@make check-deadlock

# Development workflow helpers
.PHONY: tdd verify commit

# TDD workflow: write test -> implement -> verify
tdd:
	@echo "Test-Driven Development Workflow:"
	@echo "1. Write tests first"
	@echo "2. Run 'make test-quick' frequently"
	@echo "3. Run 'make test' before committing"
	@echo "4. Only mark tasks complete when all tests pass"

# Verify implementation is complete
verify: test
	@echo "Verification complete!"

# Pre-commit check
commit: test
	@echo "All tests passed - ready to commit!"

# Docker commands
docker-build:
	@echo "Building Docker images..."
	@docker-compose build

docker-test:
	@echo "Running tests in Docker..."
	@docker-compose run test

# Single command to build and test
docker-setup-and-test:
	@echo "Setting up Docker and running tests (single command)..."
	@docker-compose build test && docker-compose run --rm test

docker-test-quick:
	@echo "Running quick tests in Docker..."
	@docker-compose run test-quick

docker-dev:
	@echo "Starting development environment..."
	@docker-compose run dev

docker-shell:
	@echo "Opening shell in development container..."
	@docker-compose run dev /bin/bash

docker-experiment:
	@echo "Running experiment in Docker..."
	@docker-compose run experiment

docker-notebook:
	@echo "Starting Jupyter notebook..."
	@docker-compose up notebook

docker-clean:
	@echo "Cleaning Docker containers and volumes..."
	@docker-compose down -v
	@docker system prune -f

# Code coverage commands
coverage:
	@echo "Running test coverage analysis..."
	@python3 scripts/simple_coverage_report.py

coverage-detailed:
	@echo "Running detailed coverage analysis (requires coverage package)..."
	@python3 scripts/run_coverage.py

coverage-report: coverage

# Performance benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	@python3 scripts/benchmark_performance.py --domains blocksworld gripper logistics rover depots --repetitions 10

benchmark-quick:
	@echo "Running quick performance benchmark..."
	@python3 scripts/benchmark_performance.py --domains blocksworld gripper --repetitions 3

# CI/CD commands
ci-local:
	@echo "Running local CI pipeline..."
	@echo "1. Linting..."
	@black --check src/ tests/ || echo "Format issues found"
	@flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203 || echo "Lint issues found"
	@echo "2. Unit tests..."
	@pytest tests/test_cnf_manager.py tests/test_pddl_handler.py tests/test_metrics.py -v
	@echo "3. Integration tests..."
	@pytest tests/test_experiment_runner.py tests/test_olam_adapter.py -v
	@echo "CI pipeline complete!"