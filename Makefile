# Makefile for Online Model Learning Framework

.PHONY: help test test-verbose test-coverage test-fast clean install install-dev lint format check

# Help target
help:
	@echo "Available targets:"
	@echo "  test          - Run all tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  test-fast     - Run tests excluding slow ones"
	@echo "  test-unit     - Run only unit tests"
	@echo "  test-cnf      - Run only CNF-related tests"
	@echo "  test-pddl     - Run only PDDL-related tests"
	@echo "  clean         - Clean up temporary files"
	@echo "  install       - Install main dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code"
	@echo "  check         - Run all checks (lint, test)"

# Test targets
test:
	pytest

test-verbose:
	pytest -v -s

test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest -m "not slow"

test-unit:
	pytest -m "unit"

test-cnf:
	pytest -m "cnf" -v

test-pddl:
	pytest -m "pddl" -v

test-core:
	pytest tests/test_cnf_manager.py tests/test_pddl_handler.py -v

# Development targets
install:
	pip install python-sat unified-planning[fast-downward,tamer]

install-dev: install
	pip install -r requirements-test.txt
	pip install black flake8 isort mypy

install-test:
	pip install -r requirements-test.txt

# Code quality targets
lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

check: lint test

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -f .coverage

# Conda environment setup
setup-conda:
	source ~/miniconda3/etc/profile.d/conda.sh && \
	conda activate action-learning && \
	pip install -r requirements-test.txt

# Run tests with conda environment
test-conda:
	source ~/miniconda3/etc/profile.d/conda.sh && \
	conda activate action-learning && \
	pytest