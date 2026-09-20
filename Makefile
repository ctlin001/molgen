.PHONY: help env install lint format test check clean

help:
	@echo "env      create the conda environment"
	@echo "install  install the package in editable mode"
	@echo "lint     ruff check"
	@echo "format   ruff format + fix"
	@echo "test     pytest (skips slow/vina/llm markers)"
	@echo "check    molgen environment check"
	@echo "clean    remove caches and build artefacts"

env:
	conda env create -f environment.yaml

install:
	pip install -e ".[dev,viz,cif]"

lint:
	ruff check src tests

format:
	ruff format src tests && ruff check --fix src tests

test:
	pytest -m "not slow and not requires_vina and not requires_llm"

test-all:
	pytest

check:
	molgen check

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
